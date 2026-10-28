"""
Benchmark the refined GSQ on a large synthetic Q4_K tensor.

Goal: find the actual bottleneck before optimising. The user reported
11 hours projected on a 12 GB GPU for a 36B MoE; on a CPU-only laptop
the same workload would be much slower. We want to know which stage
(super-scale search, scale coordinate descent, Gumbel-Softmax relaxation)
eats the wall-clock so we can target the optimisation.

Run:
    python /home/z/my-project/scripts/bench_gsq.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import types

import numpy as np

# ---------------------------------------------------------------------------
# Load the refined modules.
# ---------------------------------------------------------------------------
REFINED_DIR = "/home/z/my-project/download/refined"
toffee_mod = types.ModuleType("toffee")
toffee_mod.__path__ = []
sys.modules["toffee"] = toffee_mod
quant_mod = types.ModuleType("toffee.quant")
quant_mod.__path__ = [REFINED_DIR]
sys.modules["toffee.quant"] = quant_mod


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gguf_kquant = _load("toffee.quant.gguf_kquant",
                    os.path.join(REFINED_DIR, "gguf_kquant.py"))
gsq = _load("toffee.quant.gsq",
            os.path.join(REFINED_DIR, "gsq.py"))

spec_for = gguf_kquant.spec_for
dequantize_blocks = gguf_kquant.dequantize_blocks
BlockRaw = gguf_kquant.BlockRaw


def _naive_quantize_q4_k(values, d_factor=1.0):
    n_blocks, block_elems = values.shape
    spec = spec_for("Q4_K")
    n_groups = spec.n_groups
    codes = np.zeros((n_blocks, block_elems), dtype=np.int32)
    raws = []
    for b in range(n_blocks):
        block = values[b].astype(np.float64)
        bmin = float(block.min())
        bmax = float(block.max())
        span = bmax - bmin
        d_naive = float(np.float16(d_factor * span / 15 + 1e-12)) or 1e-6
        dmin_naive = float(np.float16(abs(bmin) + 1e-12)) or 1e-6
        sc_raw = np.full(n_groups, 1, dtype=np.int64)
        m_raw = np.full(n_groups, 1, dtype=np.int64)
        eff_s = float(d_naive)
        eff_m = float(dmin_naive)
        codes[b] = np.clip(
            np.round((block + eff_m) / max(eff_s, 1e-12)),
            spec.code_min, spec.code_max,
        ).astype(np.int32)
        raws.append(BlockRaw(
            d=d_naive, dmin=dmin_naive,
            sc_raw=sc_raw.copy(), m_raw=m_raw.copy(),
        ))
    return codes, raws


def _perturb_d(raws, factor):
    new_raws = []
    for r in raws:
        new_d = float(np.float16(r.d * factor)) or 1e-6
        new_raws.append(BlockRaw(
            d=new_d, dmin=r.dmin,
            sc_raw=r.sc_raw.copy(),
            m_raw=r.m_raw.copy() if r.m_raw.size else r.m_raw,
        ))
    return new_raws


def _weighted_mse(recon, target, weights):
    diff = (recon.astype(np.float64) - target.astype(np.float64))
    return float(
        (diff * diff * weights).sum()
        / max(float(weights.sum()), 1e-12)
    )


def time_block(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"  {label:40s} {dt*1000:8.1f} ms")
    return dt, out


def bench_q4_k(n_blocks, label, epochs=0, search_super_scale=True):
    """Benchmark one Q4_K tensor of ``n_blocks`` blocks."""
    print(f"\n--- {label}: n_blocks={n_blocks}, epochs={epochs}, "
          f"super_scale={search_super_scale} ---")
    rng = np.random.default_rng(0)
    block_elems = 256  # Q4_K
    values = rng.normal(0, 1.0, size=(n_blocks, block_elems)).astype(np.float32)
    weights = np.ones_like(values)
    codes, raws = _naive_quantize_q4_k(values, d_factor=1.0)
    raws_perturbed = _perturb_d(raws, factor=1.5)

    spec = spec_for("Q4_K")
    eff_s, eff_m = gsq._effective_from_raw(raws_perturbed, spec)
    recon0 = dequantize_blocks(codes, eff_s, eff_m, "Q4_K")
    initial_error = _weighted_mse(recon0, values, weights)
    print(f"  initial_error (perturbed d): {initial_error:.6f}")

    # Time each stage in isolation.
    # Stage 2a: _refine_super_scale (the new pass).
    if search_super_scale:
        dt_super, _ = time_block(
            "_refine_super_scale",
            lambda: gsq._refine_super_scale(
                codes.copy(), list(raws_perturbed), values, weights,
                spec, "Q4_K", gsq.GsqConfig(),
            )
        )
    else:
        dt_super = 0.0

    # Stage 2b: _refine_scales (the existing coordinate descent).
    # Pass the output of stage 2a as the starting point.
    if search_super_scale:
        codes_after_super, raws_after_super, err_after_super = \
            gsq._refine_super_scale(
                codes.copy(), list(raws_perturbed), values, weights,
                spec, "Q4_K", gsq.GsqConfig(),
            )
    else:
        codes_after_super, raws_after_super = codes.copy(), list(raws_perturbed)
        eff_s2, eff_m2 = gsq._effective_from_raw(raws_after_super, spec)
        err_after_super = _weighted_mse(
            dequantize_blocks(codes_after_super, eff_s2, eff_m2, "Q4_K"),
            values, weights,
        )
    dt_scales, _ = time_block(
        "_refine_scales (coord descent)",
        lambda: gsq._refine_scales(
            codes_after_super.copy(), list(raws_after_super),
            values, weights, spec, "Q4_K", gsq.GsqConfig(),
        )
    )

    # Stage 1: Gumbel-Softmax relaxation (only if epochs > 0).
    if epochs > 0:
        cfg = gsq.GsqConfig(
            epochs=epochs, search_super_scale=search_super_scale,
            chunk_blocks=max(1, min(n_blocks, 2048)),
        )
        dt_relax, result = time_block(
            f"refine_tensor (epochs={epochs})",
            lambda: gsq.refine_tensor(
                values, "Q4_K", imatrix=None, config=cfg,
                initial_codes=codes, initial_raw=raws_perturbed,
            )
        )
        final_error = result.final_error
    else:
        dt_relax = 0.0
        # Just run the full refine_tensor once for the end-to-end number.
        cfg = gsq.GsqConfig(
            epochs=0, search_super_scale=search_super_scale,
        )
        dt_full, result = time_block(
            "refine_tensor (end-to-end)",
            lambda: gsq.refine_tensor(
                values, "Q4_K", imatrix=None, config=cfg,
                initial_codes=codes, initial_raw=raws_perturbed,
            )
        )
        final_error = result.final_error
        print(f"  final_error:   {final_error:.6f}  "
              f"improvement: {(initial_error - final_error) / max(initial_error, 1e-12) * 100:+.2f}%")

    total = dt_super + dt_scales + dt_relax
    print(f"  total: {total*1000:.1f} ms "
          f"(super={dt_super*1000:.1f}, scales={dt_scales*1000:.1f}, "
          f"relax={dt_relax*1000:.1f})")
    return total


def main():
    print("=" * 72)
    print("GSQ benchmark (CPU-only)")
    print("=" * 72)
    # Small case first.
    bench_q4_k(n_blocks=64, label="small (64 blocks)")
    # Medium: one transformer-block-sized tensor.
    bench_q4_k(n_blocks=2048, label="medium (2048 blocks)")
    # Large: a big expert tensor.
    bench_q4_k(n_blocks=16384, label="large (16384 blocks)")
    # Very large: a 36B-MoE-sized expert tensor (simulated).
    # bench_q4_k(n_blocks=131072, label="xlarge (131072 blocks)")  # too slow


if __name__ == "__main__":
    main()
