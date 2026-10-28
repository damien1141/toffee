"""
CPU-only synthetic test for the refined ``rco.py`` and ``gsq.py``.

This script does NOT need a llama.cpp binary, a real GGUF, or a GPU. It
loads the refined modules from ``/home/z/my-project/download/refined/``
by setting up ``toffee.quant`` as a virtual package, then exercises both
on synthetic data and reports the before/after accuracy.

Run:
    python /home/z/my-project/scripts/test_refined.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np

# ---------------------------------------------------------------------------
# Make ``toffee.quant`` importable from the refined directory.
# ---------------------------------------------------------------------------
REFINED_DIR = "/home/z/my-project/download/refined"

toffee_mod = types.ModuleType("toffee")
toffee_mod.__path__ = []
sys.modules["toffee"] = toffee_mod

quant_mod = types.ModuleType("toffee.quant")
quant_mod.__path__ = [REFINED_DIR]
sys.modules["toffee.quant"] = quant_mod


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gguf_kquant = _load(
    "toffee.quant.gguf_kquant",
    os.path.join(REFINED_DIR, "gguf_kquant.py"),
)
rco = _load(
    "toffee.quant.rco",
    os.path.join(REFINED_DIR, "rco.py"),
)
gsq = _load(
    "toffee.quant.gsq",
    os.path.join(REFINED_DIR, "gsq.py"),
)

TypeSpec = gguf_kquant.TypeSpec
BlockRaw = gguf_kquant.BlockRaw
spec_for = gguf_kquant.spec_for
decode_blocks = gguf_kquant.decode_blocks
encode_blocks = gguf_kquant.encode_blocks
dequantize_blocks = gguf_kquant.dequantize_blocks
effective_scales = gguf_kquant.effective_scales
effective_mins = gguf_kquant.effective_mins


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weighted_mse(recon, target, weights):
    diff = (recon.astype(np.float64) - target.astype(np.float64))
    return float(
        (diff * diff * weights).sum()
        / max(float(weights.sum()), 1e-12)
    )


def _naive_quantize_q8_0(values, rng):
    """Naive Q8_0 quantizer with a deliberately suboptimal d (1.1x optimal)."""
    n_blocks, block_elems = values.shape
    spec = spec_for("Q8_0")
    codes = np.zeros((n_blocks, block_elems), dtype=np.int32)
    raws = []
    for b in range(n_blocks):
        block = values[b].astype(np.float64)
        amax = float(np.abs(block).max())
        d_naive = float(np.float16(1.1 * amax / 128 + 1e-12)) or 1e-6
        codes[b] = np.clip(np.round(block / d_naive), -128, 127).astype(np.int32)
        raws.append(BlockRaw(d=d_naive))
    return codes, raws


def _naive_quantize_q4_k(values, rng, d_factor=1.0):
    """Naive Q4_K quantizer with optimal-ish d (multiplied by d_factor).

    Q4_K value map:  value = (code - 0) * (d * sc_raw) - (dmin * m_raw)
    So:              code = (value + dmin * m_raw) / (d * sc_raw)
    """
    n_blocks, block_elems = values.shape
    spec = spec_for("Q4_K")
    n_groups = spec.n_groups
    gs = spec.group_size
    codes = np.zeros((n_blocks, block_elems), dtype=np.int32)
    raws = []
    for b in range(n_blocks):
        block = values[b].astype(np.float64)
        bmin = float(block.min())
        bmax = float(block.max())
        span = bmax - bmin
        # Optimal d ≈ span / 15; perturb by d_factor.
        d_naive = float(np.float16(d_factor * span / 15 + 1e-12)) or 1e-6
        dmin_naive = float(np.float16(abs(bmin) + 1e-12)) or 1e-6
        sc_raw = np.full(n_groups, 1, dtype=np.int64)
        m_raw = np.full(n_groups, 1, dtype=np.int64)
        eff_s = float(d_naive)
        eff_m = float(dmin_naive)
        c = np.clip(
            np.round((block + eff_m) / max(eff_s, 1e-12)),
            spec.code_min, spec.code_max,
        ).astype(np.int32)
        codes[b] = c
        raws.append(BlockRaw(
            d=d_naive, dmin=dmin_naive,
            sc_raw=sc_raw.copy(), m_raw=m_raw.copy(),
        ))
    return codes, raws


def _perturb_d(raws, factor):
    """Multiply every block's fp16 super-scale d by ``factor`` (after the
    codes have been generated). This simulates a llama.cpp output where
    the codes are RTN-optimal for a different scale than the one stored.
    """
    new_raws = []
    for r in raws:
        new_d = float(np.float16(r.d * factor))
        if new_d == 0:
            new_d = float(np.float16(1e-6))
        new_raws.append(BlockRaw(
            d=new_d,
            dmin=r.dmin,
            sc_raw=r.sc_raw.copy(),
            m_raw=r.m_raw.copy() if r.m_raw.size else r.m_raw,
        ))
    return new_raws


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rco_optimizer_reduces_soft_loss():
    """The optimizer's objective (soft expected distortion) must decrease.

    The optimizer minimises L(alpha) = sum_i w_i sum_k p_ik d_ik, so the
    soft loss is the right metric to verify it works.
    """
    print("\n=== RCO: optimize_projected_gumbel_recon (soft loss) ===")
    rng = np.random.default_rng(42)
    n_groups, n_options = 12, 3
    costs = np.array([8.0, 5.5, 4.5], dtype=np.float64)
    weights = rng.dirichlet(np.ones(n_groups) * 2.0).astype(np.float64)
    weights /= weights.sum()
    distortion = rng.uniform(0.001, 0.05, size=(n_groups, n_options))
    distortion[:, 0] *= 0.1  # high-precision is best
    # Some groups have especially bad cheap option.
    for g in [0, 3, 7]:
        distortion[g, 2] *= 5.0
    allowed = np.ones((n_groups, n_options), dtype=bool)
    target = 5.5

    # Initial alpha: uniform (no distortion info), then retract to budget.
    alpha = np.zeros((n_groups, n_options), dtype=np.float64)
    alpha[~allowed] = -1e9
    rco.retraction(alpha, costs, target, weights, allowed=allowed)
    soft_before = rco.expected_distortion(alpha, distortion, weights)

    # Optimize in place.
    rco.optimize_projected_gumbel_recon(
        alpha, distortion, costs, target,
        weights=weights, n_steps=200, lr=0.05, allowed=allowed,
    )
    soft_after = rco.expected_distortion(alpha, distortion, weights)

    improvement = (soft_before - soft_after) / max(soft_before, 1e-12)
    print(f"  soft loss before: {soft_before:.6f}")
    print(f"  soft loss after:  {soft_after:.6f}")
    print(f"  improvement:      {improvement*100:+.2f}%")

    assert soft_after <= soft_before + 1e-9, (
        f"soft loss went UP: {soft_before:.6f} -> {soft_after:.6f}"
    )
    print(f"  PASS: optimizer reduced the soft expected distortion.")
    return improvement


def test_rco_optimizer_reduces_real_distortion():
    """The DP assignment from the optimized alpha should have lower *real*
    distortion than the DP assignment from the un-optimized alpha.

    Setup: 8 groups, 2 options (high-precision 8.0 bpw, cheap 4.5 bpw).
    Budget = 6.0 bpw (so roughly half the groups go on each). The
    distortion profile makes group B's cheap option *catastrophically*
    bad, so the optimal assignment puts B on high-precision. With a
    *uniform* (non-distortion-aware) initial alpha, the DP picks
    arbitrarily; with the optimizer, the DP picks the distortion-optimal
    assignment.
    """
    print("\n=== RCO: real-distortion of DP assignment ===")
    n_groups = 8
    n_options = 2
    costs = np.array([8.0, 4.5], dtype=np.float64)
    weights = np.full(n_groups, 1.0 / n_groups, dtype=np.float64)
    # Distortion: high-precision is best for all; cheap is catastrophic
    # for half the groups.
    distortion = np.zeros((n_groups, n_options), dtype=np.float64)
    distortion[:, 0] = 0.001  # high-precision is always low distortion
    distortion[:, 1] = 0.005  # cheap is moderately bad
    # Make groups 0, 2, 4, 6 catastrophically bad on cheap.
    distortion[::2, 1] = 0.05
    allowed = np.ones((n_groups, n_options), dtype=bool)
    target = 6.0  # average; (8+4.5)/2 = 6.25 too much, so 4 of 8 on high.

    # 1) One-shot DP from a uniform (non-distortion-aware) alpha.
    alpha_uniform = np.zeros((n_groups, n_options), dtype=np.float64)
    alpha_uniform[~allowed] = -1e9
    rco.retraction(alpha_uniform, costs, target, weights, allowed=allowed)
    assign_uniform = rco.budget_constrained_argmax(
        alpha_uniform, weights, costs, target, allowed=allowed,
    )
    real_d_uniform = float(
        (weights * distortion[np.arange(n_groups), assign_uniform]).sum()
    )
    bpw_uniform = float((weights * costs[assign_uniform]).sum())

    # 2) Optimized alpha.
    alpha_opt = np.zeros((n_groups, n_options), dtype=np.float64)
    alpha_opt[~allowed] = -1e9
    rco.retraction(alpha_opt, costs, target, weights, allowed=allowed)
    rco.optimize_projected_gumbel_recon(
        alpha_opt, distortion, costs, target,
        weights=weights, n_steps=200, lr=0.05, allowed=allowed,
    )
    assign_opt = rco.budget_constrained_argmax(
        alpha_opt, weights, costs, target, allowed=allowed,
    )
    real_d_opt = float(
        (weights * distortion[np.arange(n_groups), assign_opt]).sum()
    )
    bpw_opt = float((weights * costs[assign_opt]).sum())

    # Count how many "catastrophic" groups (even-indexed) ended up on
    # high-precision.
    catastrophic_on_high_uniform = int(
        ((assign_uniform == 0) & (np.arange(n_groups) % 2 == 0)).sum()
    )
    catastrophic_on_high_opt = int(
        ((assign_opt == 0) & (np.arange(n_groups) % 2 == 0)).sum()
    )

    print(f"  uniform-alpha DP:  bpw={bpw_uniform:.4f}  "
          f"real_d={real_d_uniform:.6f}  "
          f"catastrophic-on-high: {catastrophic_on_high_uniform}/{n_groups//2}")
    print(f"  optimized DP:      bpw={bpw_opt:.4f}  "
          f"real_d={real_d_opt:.6f}  "
          f"catastrophic-on-high: {catastrophic_on_high_opt}/{n_groups//2}")
    improvement = (real_d_uniform - real_d_opt) / max(real_d_uniform, 1e-12)
    print(f"  improvement:        {improvement*100:+.2f}%")

    assert real_d_opt <= real_d_uniform + 1e-9, (
        f"optimized real distortion is WORSE: "
        f"{real_d_uniform:.6f} -> {real_d_opt:.6f}"
    )
    if real_d_opt < real_d_uniform:
        print(f"  PASS: optimized DP picks a lower-distortion assignment.")
    else:
        print(f"  PASS (tie): optimized DP matched the uniform DP's "
              f"distortion; both found an equivalent assignment.")
    return improvement


def test_gsq_optimal_eff_scales_mins_closed_form():
    """The closed-form helper must recover the known optimum."""
    print("\n=== GSQ: _optimal_eff_scales_mins closed-form check ===")

    # Symmetric case (Q8_0): the optimal scale is closed form.
    spec8 = spec_for("Q8_0")
    true_scale = 0.07
    rng = np.random.default_rng(7)
    block8 = rng.normal(0, 1, size=spec8.block_elems).astype(np.float32)
    codes8 = np.clip(np.round(block8 / true_scale), -128, 127).astype(np.int32)
    values8 = block8.reshape(1, -1)
    weights8 = np.ones_like(values8)
    eff_s8, _ = gsq._optimal_eff_scales_mins(
        codes8.reshape(1, -1), values8, weights8, spec8
    )
    print(f"  Q8_0: expected scale≈{true_scale:.4f}  "
          f"recovered={float(eff_s8[0, 0]):.4f}")
    assert abs(float(eff_s8[0, 0]) - true_scale) < 0.10 * true_scale + 1e-3, \
        f"Q8_0 optimal scale off: got {eff_s8[0, 0]} vs {true_scale}"

    # Asymmetric case (Q4_K): the optimal (scale, min) is the 2x2 system.
    # Use a value distribution that does NOT clip the codes, so the closed
    # form recovers the (scale, min) we used to generate the codes.
    # If codes are clipped, the closed form finds the (scale, min) that
    # minimises the reconstruction error for the *clipped* codes, which is
    # a different problem and gives different numbers.
    spec4 = spec_for("Q4_K")
    true_eff_scale = 0.10
    true_eff_min = 0.50
    rng = np.random.default_rng(7)
    # block values in [-0.5, 1.0], so codes round((value + 0.5) / 0.10)
    # are in [0, 15] with no clipping (the max code 15 maps to value 1.0).
    block4 = rng.uniform(-0.5, 1.0, size=spec4.block_elems).astype(np.float32)
    codes4 = np.clip(
        np.round((block4 + true_eff_min) / true_eff_scale),
        spec4.code_min, spec4.code_max,
    ).astype(np.int32)
    n_clipped = int(((codes4 == spec4.code_min) | (codes4 == spec4.code_max)).sum())
    print(f"  Q4_K: {n_clipped}/{spec4.block_elems} codes clipped")
    values4 = block4.reshape(1, -1)
    weights4 = np.ones_like(values4)
    eff_s4, eff_m4 = gsq._optimal_eff_scales_mins(
        codes4.reshape(1, -1), values4, weights4, spec4
    )
    mean_s = float(eff_s4.mean())
    mean_m = float(eff_m4.mean())
    print(f"  Q4_K: expected scale≈{true_eff_scale:.4f}  "
          f"recovered≈{mean_s:.4f}")
    print(f"  Q4_K: expected min≈{true_eff_min:.4f}  "
          f"recovered≈{mean_m:.4f}")
    # Allow 5% tolerance (the integer codes introduce some quantization
    # noise that the closed form compensates for).
    assert abs(mean_s - true_eff_scale) < 0.05 * true_eff_scale + 1e-3, \
        f"Q4_K optimal scale off: got {mean_s} vs {true_eff_scale}"
    assert abs(mean_m - true_eff_min) < 0.05 * true_eff_min + 1e-3, \
        f"Q4_K optimal min off: got {mean_m} vs {true_eff_min}"
    print("  PASS: closed-form helper recovers the known optimum.")
    return mean_s


def test_gsq_refine_tensor_q8_0():
    """End-to-end: refine_tensor on a deliberately-suboptimal Q8_0 block."""
    print("\n=== GSQ: refine_tensor on Q8_0 (deliberately suboptimal d) ===")
    rng = np.random.default_rng(123)
    n_blocks = 16
    block_elems = 32  # Q8_0
    values = rng.normal(0, 1.0, size=(n_blocks, block_elems)).astype(np.float32)
    weights = np.ones_like(values)

    codes, raws = _naive_quantize_q8_0(values, rng)
    eff_scales = np.array([[r.d] for r in raws], dtype=np.float32)
    eff_mins = np.zeros((n_blocks, 1), dtype=np.float32)
    recon = dequantize_blocks(codes, eff_scales, eff_mins, "Q8_0")
    initial_error = _weighted_mse(recon, values, weights)

    cfg = gsq.GsqConfig(epochs=0, search_super_scale=True)
    result = gsq.refine_tensor(
        values, "Q8_0", imatrix=None, config=cfg,
        initial_codes=codes, initial_raw=raws,
    )
    final_error = result.final_error

    print(f"  initial_error: {initial_error:.6f}")
    print(f"  final_error:   {final_error:.6f}")
    improvement = (initial_error - final_error) / max(initial_error, 1e-12)
    print(f"  improvement: {improvement*100:+.2f}%")

    assert final_error <= initial_error + 1e-9, \
        f"refine increased the error: {initial_error:.6f} -> {final_error:.6f}"
    return improvement


def test_gsq_refine_tensor_q4_k():
    """End-to-end: refine_tensor on Q4_K with a perturbed super-scale.

    Generate codes with the optimal d, then *perturb* d to 1.5x AFTER
    generation. The codes no longer match the stored d, so the
    reconstruction error is large; the refiner should recover a d close
    to the original (optimal-for-the-codes) d.
    """
    print("\n=== GSQ: refine_tensor on Q4_K (perturbed super-scale) ===")
    rng = np.random.default_rng(456)
    n_blocks = 16
    block_elems = 256  # Q4_K
    values = rng.normal(0, 1.0, size=(n_blocks, block_elems)).astype(np.float32)
    weights = np.ones_like(values)

    # Generate codes with the optimal d (d_factor=1.0).
    codes, raws = _naive_quantize_q4_k(values, rng, d_factor=1.0)
    # Perturb d to 1.5x AFTER generation -- the codes no longer match.
    raws_perturbed = _perturb_d(raws, factor=1.5)

    spec = spec_for("Q4_K")
    eff_scales, eff_mins = gsq._effective_from_raw(raws_perturbed, spec)
    recon = dequantize_blocks(codes, eff_scales, eff_mins, "Q4_K")
    initial_error = _weighted_mse(recon, values, weights)

    cfg = gsq.GsqConfig(epochs=0, search_super_scale=True)
    result = gsq.refine_tensor(
        values, "Q4_K", imatrix=None, config=cfg,
        initial_codes=codes, initial_raw=raws_perturbed,
    )
    final_error = result.final_error

    print(f"  initial_error (perturbed d): {initial_error:.6f}")
    print(f"  final_error   (after refine): {final_error:.6f}")
    improvement = (initial_error - final_error) / max(initial_error, 1e-12)
    print(f"  improvement: {improvement*100:+.2f}%")

    assert final_error <= initial_error + 1e-9, \
        f"refine increased the error: {initial_error:.6f} -> {final_error:.6f}"
    return improvement


def test_gsq_relaxation_no_regression():
    """GSQ: with epochs>0 and the bug-fixed schedule, the relaxation should
    never INCREASE the error (the previous bug silently overwrote codes with
    worse values)."""
    print("\n=== GSQ: Gumbel-Softmax relaxation (epochs=4, no regression) ===")
    rng = np.random.default_rng(789)
    n_blocks = 8
    block_elems = 256  # Q4_K
    # Heavy-tailed distribution: the codes are far from optimal under any
    # single scale, so the relaxation has signal.
    values = rng.standard_cauchy(size=(n_blocks, block_elems))
    values = values.clip(-5, 5).astype(np.float32)
    weights = np.ones_like(values)

    codes, raws = _naive_quantize_q4_k(values, rng)
    spec = spec_for("Q4_K")
    eff_scales, eff_mins = gsq._effective_from_raw(raws, spec)
    recon = dequantize_blocks(codes, eff_scales, eff_mins, "Q4_K")
    initial_error = _weighted_mse(recon, values, weights)

    # Run with the bug-fixed schedule (the new defaults).
    cfg = gsq.GsqConfig(
        epochs=4,
        # Match upstream GSQ's schedule (these are the new defaults).
        temperature_init=2.0, temperature_end=0.05,
        logit_scale_init=100.0, logit_scale_end=500.0,
        search_super_scale=True,
    )
    result = gsq.refine_tensor(
        values, "Q4_K", imatrix=None, config=cfg,
        initial_codes=codes, initial_raw=raws,
    )
    final_error = result.final_error

    print(f"  initial_error: {initial_error:.6f}")
    print(f"  final_error:   {final_error:.6f}  (epochs=4)")
    improvement = (initial_error - final_error) / max(initial_error, 1e-12)
    print(f"  improvement: {improvement*100:+.2f}%")

    # The relaxation should never increase the error.
    assert final_error <= initial_error + 1e-9, \
        f"relaxation increased the error: {initial_error:.6f} -> {final_error:.6f}"
    print(f"  PASS: relaxation did not increase the error.")
    return improvement


def test_gsq_super_scale_vs_no_super_scale():
    """Direct comparison: turning ON search_super_scale should give a
    lower (or equal) error than turning it OFF, on the same Q4_K block
    with a perturbed super-scale."""
    print("\n=== GSQ: search_super_scale ON vs OFF (Q4_K, perturbed d) ===")
    rng = np.random.default_rng(2024)
    n_blocks = 16
    block_elems = 256
    values = rng.normal(0, 1.0, size=(n_blocks, block_elems)).astype(np.float32)
    weights = np.ones_like(values)
    codes, raws = _naive_quantize_q4_k(values, rng, d_factor=1.0)
    raws_perturbed = _perturb_d(raws, factor=1.5)

    spec = spec_for("Q4_K")
    eff_s, eff_m = gsq._effective_from_raw(raws_perturbed, spec)
    recon = dequantize_blocks(codes, eff_s, eff_m, "Q4_K")
    initial_error = _weighted_mse(recon, values, weights)

    cfg_off = gsq.GsqConfig(epochs=0, search_super_scale=False)
    res_off = gsq.refine_tensor(
        values, "Q4_K", imatrix=None, config=cfg_off,
        initial_codes=codes, initial_raw=raws_perturbed,
    )

    cfg_on = gsq.GsqConfig(epochs=0, search_super_scale=True)
    res_on = gsq.refine_tensor(
        values, "Q4_K", imatrix=None, config=cfg_on,
        initial_codes=codes, initial_raw=raws_perturbed,
    )

    print(f"  initial_error:           {initial_error:.6f}")
    print(f"  final (super_scale OFF): {res_off.final_error:.6f}")
    print(f"  final (super_scale ON):  {res_on.final_error:.6f}")
    delta = (res_off.final_error - res_on.final_error) / max(res_off.final_error, 1e-12)
    print(f"  super_scale gain: {delta*100:+.2f}% (positive is better)")

    assert res_on.final_error <= res_off.final_error + 1e-9, (
        f"super_scale ON is WORSE than OFF: "
        f"{res_on.final_error:.6f} > {res_off.final_error:.6f}"
    )
    print(f"  PASS: super_scale search is at least as good as the baseline.")
    return delta


def main():
    print("=" * 72)
    print("Refined RCO + GSQ synthetic test (CPU-only, no llama.cpp binary)")
    print("=" * 72)

    rco_soft = test_rco_optimizer_reduces_soft_loss()
    rco_real = test_rco_optimizer_reduces_real_distortion()
    test_gsq_optimal_eff_scales_mins_closed_form()
    q8_imp = test_gsq_refine_tensor_q8_0()
    q4_imp = test_gsq_refine_tensor_q4_k()
    relax_imp = test_gsq_relaxation_no_regression()
    super_delta = test_gsq_super_scale_vs_no_super_scale()

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  RCO soft-loss reduction:              {rco_soft*100:+.2f}%")
    print(f"  RCO real-distortion reduction:         {rco_real*100:+.2f}%")
    print(f"  GSQ refine_tensor Q8_0:                {q8_imp*100:+.2f}%")
    print(f"  GSQ refine_tensor Q4_K:                {q4_imp*100:+.2f}%")
    print(f"  GSQ relaxation (Q4_K, epochs=4):      {relax_imp*100:+.2f}%")
    print(f"  GSQ super_scale gain over baseline:   {super_delta*100:+.2f}%")
    print()
    print("All tests passed.")


if __name__ == "__main__":
    main()
