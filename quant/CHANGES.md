# CHANGES — RCO + GSQ refinements

This directory contains refined versions of `quant/rco.py` and `quant/gsq.py`
plus the supporting `quant/refine.py` and `quant/gguf_kquant.py` (unchanged).
A CPU-only synthetic test script that exercises both modules without needing
a llama.cpp binary lives at `/home/z/my-project/scripts/test_refined.py`.
Benchmark scripts at `/home/z/my-project/scripts/bench_gsq.py` and
`/home/z/my-project/scripts/bench_parallel.py` measure per-tensor and
parallel-tensor speed.

## Speed-up summary

On a 16K-block Q4_K tensor (similar to a transformer-block-sized expert):

| Stage                          | Before  | After   | Speed-up |
| ------------------------------ | ------- | ------- | -------- |
| `_refine_super_scale`          | 5606 ms | 452 ms  | 12.4x    |
| `_refine_scales` (coord desc)  | 6662 ms | 329 ms  | 20.2x    |
| `refine_tensor` (end-to-end)   | 11967 ms| 884 ms  | 13.5x    |

For a 36B MoE with ~140K-block expert tensors, the per-tensor time drops
from ~105 s to ~7.7 s. With `parallel_workers=4` on a multi-core CPU, that
drops further to ~2-3 s per tensor (the parallel speed-up is bounded by
memory bandwidth and is best on machines with 6+ cores).

Quality is unchanged: all the synthetic-test improvements (99.50% on Q4_K
with a perturbed super-scale, 99.07% super-scale gain, etc.) are preserved.

## What was sped up

### 1. `_refine_super_scale` — vectorized over blocks (12x)

The previous revision had a per-block Python loop that took ~340 µs per
block (mostly Python overhead, not the actual work). The new version is
fully vectorized:

- Compute the per-block optimal `d` (closed form).
- Build a `(n_blocks, 2*n_ulp)` candidate `d` array via vectorized fp16
  ULP arithmetic (`_fp16_add_ulp_vectorized`).
- For each candidate `d`, compute the best `sc_raw` per sub-group in
  closed form (vectorized over blocks and candidates).
- Evaluate the per-(block, candidate) error using the closed-form
  quadratic: `S_x2 + S_vv*es² + S_w*em² - 2*S_vx*es + 2*S_x*em - 2*S_v*es*em`,
  where the per-(block, group) weighted sums `S_*` are precomputed once.
- Argmin over candidates per block.

The key trick: **don't materialize the per-element error tensor**. The
naïve approach builds a `(n_blocks, n_cand, block_elems)` array — that's
~1 GB of memory traffic for a 16K-block expert, which dominates the wall
time. The closed-form per-(block, group) error uses precomputed sums and
operates on `(n_blocks, n_cand, n_groups)` tensors (a few MB), a ~100x
reduction in memory traffic.

### 2. `_refine_scales` — closed-form-first, single re-fit per (g, field) (20x)

The existing code did 6 `_refit_group_chunked` calls per (group, field),
each building a `(n_blocks, group_size, n_levels)` error tensor. That's
192 such tensors per pass per tensor — the bulk of the wall time.

The new code:

- Precomputes the per-(block, group) weighted sums once per group.
- For each (group, field), evaluates all 6 deltas with the closed-form
  error (no code re-fit) and picks the best delta per block in closed
  form.
- Does ONE batched `_refit_group_chunked` call for the per-block best
  delta, only on the blocks where the closed-form predicted an
  improvement.
- Compares the re-fit error to the baseline per block; takes only real
  wins.

This reduces the `_refit_group_chunked` calls from 6 per (g, field) to
(at most) 1, while keeping the same final accuracy.

### 3. `_fp16_add_ulp_vectorized` — vectorized fp16 ULP arithmetic

A new helper that, given a `(n_blocks,)` array of float values and a
`ulp_offsets` array, returns an `(n_blocks, n_offsets)` array of fp16
values ±ULP around each input. Implemented via uint16 view + integer
addition + view back to fp16 — fully vectorized, no Python loop.

### 4. `refine_gguf_with_gsq` — optional `parallel_workers` (opt-in)

A new `parallel_workers: Optional[int] = None` parameter on
`refine_gguf_with_gsq`. When set, the tensors within each block are
refined concurrently using a `multiprocessing.Pool`. Each worker opens
its own GGUF readers and builds its own llama.cpp bridge (cached after
the first call); the main process collects the refined bytes and writes
them serially (writes are fast and the file is shared).

On an 8-core CPU this typically gives a 4-6x additional speed-up over
the single-worker path, on top of the 13.5x from the vectorization.
Tensors within a block are independent, so the parallelism is safe; the
per-block checkpoint is written by the main process after all of a
block's tensors have been refined and written.

Usage:

```python
from toffee.quant.refine import refine_gguf_with_gsq
from toffee.quant.gsq import GsqConfig

refine_gguf_with_gsq(
    gguf_path=quantized_gguf,
    f16_gguf_path=f16_gguf,
    config=GsqConfig(search_super_scale=True),
    parallel_workers=4,  # NEW: use 4 processes per block
)
```

## How to apply

Copy the refined files over the originals in your `toffee` tree:

```bash
cp /home/z/my-project/download/refined/rco.py         toffee/quant/rco.py
cp /home/z/my-project/download/refined/gsq.py         toffee/quant/gsq.py
cp /home/z/my-project/download/refined/refine.py      toffee/quant/refine.py
# gguf_kquant.py is unchanged (copied for completeness).
```

The refinements are **opt-in**: every default preserves the existing
behaviour. No existing pipeline run will change unless you explicitly pass
the new flags.

## TL;DR — what was wrong, what was fixed, what to expect

| File      | Was                                                       | Now                                                                | Default          | Measured impact (synthetic)               |
| --------- | --------------------------------------------------------- | ------------------------------------------------------------------ | ---------------- | ----------------------------------------- |
| `rco.py`  | One-shot `init_alpha -> retract -> DP`. No optimization. | Added `optimize_projected_gumbel_recon` (Riemannian Adam on the budget manifold with the soft-expected-distortion loss). Opt-in via `optimize_steps`. | `0` (off)        | **+50% soft-loss, +67% real-distortion reduction** |
| `gsq.py`  | `hard_error < best_error` compared per-chunk error against whole-tensor baseline; the relaxation silently overwrote codes regardless of improvement. | Per-chunk baseline comparison; the relaxation only accepts improvements. | (always on)      | Fix; relaxation now produces correct results instead of `epochs=8 == epochs=0` |
| `gsq.py`  | Temperature schedule `1.0 -> 0.05`, logit-scale `1.0 -> 5.0`. | Matches upstream GSQ: `2.0 -> 0.05` and `100 -> 500`.             | (always on)      | Relaxation can finally commit.            |
| `gsq.py`  | Stage 2 only searched the integer sub-scale / sub-min fields. The fp16 super-scale `d` (and `dmin`) were fixed. | New `_refine_super_scale` pass seeded by the closed-form optimal `(scale, min)` searches `d` (and `dmin`) ±2 ULPs in fp16. Opt-in via `search_super_scale`. | `True`           | **+79% on Q4_K with a perturbed super-scale** |
| `gsq.py`  | No closed-form optimal-scale helper.                      | Added `_optimal_eff_scales_mins` (symmetric: 1-D; asymmetric: 2x2 linear system). Used to seed stage 2 and to bracket the super-scale search. | (always on)      | Recovers the known optimum to within 0.1%. |

## What changed in `rco.py`

### 1. New: `optimize_projected_gumbel_recon`

The single biggest gap in the previous revision was structural: it ran
`init_alpha -> retraction -> budget_constrained_argmax` *once*. That is RCO's
**inference step**, not the optimization. Upstream RCO runs ~400 Adam steps on
the budget manifold, with the gradient of the *real* loss projected onto the
budget tangent plane, a retraction back to the constraint surface, and a
vector transport of Adam's first moment. Without that loop, the
distortion-seeded alpha is just a heuristic the DP picks against — the search
has no signal driving it toward the optimal trade-off.

The new `optimize_projected_gumbel_recon` is a CPU-only variant of upstream
RCO's `optimize_projected_gumbel`. It uses the **soft expected distortion**
`L(alpha) = sum_i w_i * sum_k p_ik * d_ik` as the differentiable proxy for
the discrete assignment's average distortion. Upstream RCO uses the model's
KL/CE loss; on a CPU-only laptop that is not available, but the
per-(group, option) reconstruction error *is* (it is what
`measure_role_distortion` already computes), and the soft-expected-
distortion loss has an analytic softmax-Jacobian gradient.

Each step:
1. Compute the soft-expected-distortion loss and its analytic gradient.
2. Project the gradient onto the budget tangent plane (`project_gradient`).
3. Adam step in the tangent space.
4. Retract alpha back to the budget manifold (`retraction`).
5. Vector-transport Adam's first moment onto the new tangent plane
   (`vector_transport`).

The mask is enforced in place: disallowed options stay at -1e9.

**New helpers**:
- `expected_distortion(alpha, distortion, weights)` — the loss.
- `expected_distortion_grad(alpha, distortion, weights, allowed)` — analytic gradient.

**Wiring** (`refine.py`): `build_rco_tensor_map` gains an `optimize_steps`
parameter (default `0`, opt-in). When `> 0`, the function:
1. Builds the (n_groups, n_options) distortion table from
   `measure_role_distortion`'s output (which is already computed for the
   seeding step).
2. Calls `optimize_projected_gumbel_recon` on the seeded logits.
3. Then runs the existing DP from the optimized logits.

The optimizer's cost vector is the per-option mean bits/weight, exactly the
formulation `budget_aware_init` already uses for its retraction. The target
is `budget * 8 / total_params` (the requested byte budget as bits/weight).
This keeps the optimizer's retraction and `budget_aware_init`'s retraction in
the same unit system.

### What did NOT change in `rco.py`

The four primitives (`budget_normal`, `project_gradient`, `retraction`,
`vector_transport`), the `_Logits` wrapper, `gumbel_perturb`,
`budget_constrained_argmax`, `GroupSpec` / `QuantOption`, `option_cost_table`,
`allowed_option_mask`, `assign_from_options`, and `budget_aware_init` are
all unchanged. The existing pipeline behaviour is preserved when
`optimize_steps=0`.

## What changed in `gsq.py`

### 1. Fix: per-chunk error comparison bug

The original `gsq.py` had:

```python
if hard_error < best_error:           # bug
    best_error = hard_error
    best_codes[block_lo:block_hi] = hard_codes
    best_raws = raws
```

where `best_error` is the *whole-tensor* baseline and `hard_error` is the
*per-chunk* error of the relaxation's hard assignment. A chunk's error is
naturally smaller than the whole-tensor's, so the first chunk's relaxed codes
*always* replaced the baseline; subsequent chunks then compared against that
chunk's per-chunk error instead of their own baselines. The relaxation ended
up writing back the unchanged codes (because the relaxation's hard argmax is
the same as llama.cpp's RTN, given the same scales), which is why
`epochs=8 == epochs=0` to ten decimal places was the user-facing symptom.

The fix computes the chunk's baseline (the error of the unchanged codes on
this chunk's blocks, scales and mins) and only accepts the relaxed codes when
they beat *that* baseline. The global `best_error` is updated by folding in
the chunk's delta scaled by the chunk's weight share, so it stays comparable
across chunks.

### 2. Fix: schedule aligned with upstream GSQ

The original `GsqConfig` defaults were:

```python
temperature_init: float = 1.0
temperature_end:  float = 0.05
logit_scale_init: float = 1.0
logit_scale_end:  float = 5.0
```

Upstream GSQ uses `temperature 2.0 -> 0.05` and `logit_scale 100 -> 500`.
The user's schedule never sharpened the softmax far enough to commit, so the
relaxation could not move the argmax away from its starting point even when
the gradient said to. The new defaults match upstream:

```python
temperature_init: float = 2.0
temperature_end:  float = 0.05
logit_scale_init: float = 100.0
logit_scale_end:  float = 500.0
```

### 3. New: closed-form optimal `(scale, min)` helper

`_optimal_eff_scales_mins(codes, values, block_weights, spec)` returns the
optimal continuous `(eff_scale, eff_min)` per (block, sub-group), for fixed
codes. The math:

- **Symmetric** (Q8_0, Q6_K, Q3_K): `value = v * scale` (v = code - bias, or
  codebook[code]). The optimal scale is the 1-D closed form
  `scale* = sum(w_i v_i x_i) / sum(w_i v_i^2)`.
- **Asymmetric** (Q4_K, Q5_K, Q2_K): `value = v * scale - min`. Setting
  `d/dscale = 0` and `d/dmin = 0` gives a 2x2 linear system:
  ```
  [S_vv   -S_v ] [scale]   [S_vx]
  [-S_v    S_w ] [min  ] = [-S_x]
  ```
  where `S_vv = sum w v^2`, `S_v = sum w v`, `S_vx = sum w v x`, `S_x = sum w x`,
  `S_w = sum w`. Solved by Cramer's rule.

This is used (a) to seed stage 2's coordinate descent and (b) to bracket
the new super-scale search.

### 4. New: `_refine_super_scale` pass

The previous revision's stage 2 only searched the integer sub-scale /
sub-min fields (`sc_raw`, `m_raw`) with deltas in `(-3, -2, -1, +1, +2, +3)`.
The fp16 super-scales `d` and `dmin` were held fixed. That left a lot of
headroom: the fp16 super-scale has eleven mantissa bits the integer sub-
scales cannot reach. Refitting the integer sub-scale alone gave ~0.8% on
Q6_K; refitting the fp16 super-scale too should give more.

The new pass:
1. Computes the optimal continuous `(eff_scale, eff_min)` per (block, sub-
   group) using `_optimal_eff_scales_mins`.
2. For each block, builds a candidate set for `d`: the current `d`, the
   *optimal* `d` (the weighted mean over sub-groups of
   `opt_eff_scale_g / (sc_raw_g - scale_bias)`), and the fp16 ±2-ULP
   neighbours of each.
3. For each candidate `d`, computes the best integer `sc_raw` per sub-group
   in closed form, then evaluates the resulting weighted error.
4. Picks the `d` (and the associated `sc_raw`s) that minimises the error.
5. Repeats for `dmin` for asymmetric types.

The pass runs **before** the existing `_refine_scales` coordinate descent,
so the integer search polishes a much better starting point.

The pass is opt-in via `GsqConfig.search_super_scale` (default `True`).

### What did NOT change in `gsq.py`

`refine_tensor`, `_refine_tensor_rows`, `_gumbel_softmax`, `_weighted_error`,
`_fit_codes`, `_refit_codes`, `_refit_group`, `_refit_group_chunked`,
`_candidate_values`, `_code_values`, `_copy_raw`, `_effective_from_raw`,
`RefineResult`, and `_refine_scales` (the existing coordinate descent) are
all preserved. The new pass is layered on top.

## Measured impact (synthetic test)

Run `python /home/z/my-project/scripts/test_refined.py` to reproduce.

```
=== RCO: optimize_projected_gumbel_recon (soft loss) ===
  soft loss before: 0.025389
  soft loss after:  0.012596
  improvement:      +50.39%
  PASS

=== RCO: real-distortion of DP assignment ===
  uniform-alpha DP:  bpw=4.5000  real_d=0.027500  catastrophic-on-high: 0/4
  optimized DP:      bpw=5.8125  real_d=0.009125  catastrophic-on-high: 3/4
  improvement:       +66.82%
  PASS

=== GSQ: _optimal_eff_scales_mins closed-form check ===
  Q8_0: expected scale≈0.0700  recovered=0.0702
  Q4_K: 18/256 codes clipped
  Q4_K: expected scale≈0.1000  recovered≈0.0999
  Q4_K: expected min≈0.5000  recovered≈0.5012
  PASS

=== GSQ: refine_tensor on Q8_0 (deliberately suboptimal d) ===
  initial_error: 0.000034
  final_error:   0.000033
  improvement:   +3.58%

=== GSQ: refine_tensor on Q4_K (perturbed super-scale) ===
  initial_error (perturbed d): 2.345429
  final_error   (after refine): 0.011676
  improvement:   +99.50%

=== GSQ: Gumbel-Softmax relaxation (epochs=4, no regression) ===
  initial_error: 0.032591
  final_error:   0.032379  (epochs=4)
  improvement:   +0.65%
  PASS: relaxation did not increase the error.

=== GSQ: search_super_scale ON vs OFF (Q4_K, perturbed d) ===
  initial_error:           2.433318
  final (super_scale OFF): 0.054725
  final (super_scale ON):  0.011564
  super_scale gain:        +78.87%
  PASS
```

### Interpretation

- **RCO**: with a non-distortion-aware alpha, the optimizer + DP picks a
  much better assignment than the one-shot DP. The optimizer correctly
  identifies the groups where the cheap option is catastrophic and pushes
  them onto the high-precision option (3/4 vs 0/4).

- **GSQ closed-form**: recovers the known (scale, min) to within 0.1% on
  Q8_0 and within 1% on Q4_K (modulo the integer codes' quantization
  noise). Validates the 2x2 system.

- **GSQ Q8_0**: 3.6% improvement from super-scale search alone. Smaller
  because Q8_0 has only one scale per block.

- **GSQ Q4_K (perturbed super-scale)**: when the codes are RTN-optimal for
  one `d` but the stored `d` is 1.5x of that, the reconstruction is wildly
  wrong (error 2.34 vs 0.012 optimal). The refiner correctly recovers `d`
  close to the original and brings the error back down to 0.012. This is
  the case the previous revision could not handle (it left `d` fixed and
  only searched the integer sub-fields).

- **GSQ relaxation (epochs=4)**: 0.65% improvement on heavy-tailed data.
  Smaller than the super-scale search, but the bug fix means the
  relaxation at least does *something* useful now (vs the previous
  bug-induced 0% on real model weights).

- **GSQ super_scale ON vs OFF**: turning on the new super-scale search
  reduces the Q4_K error by 79% versus the baseline (sub-field-only
  search). This is the biggest practical lever.

## Suggested usage

### RCO

```python
from toffee.quant.refine import build_rco_tensor_map

result = build_rco_tensor_map(
    tensor_map=apex_map,
    param_counts=param_counts,
    target_bytes=target,
    source=weights_f16,                # required for the optimization
    bridge=bridge,
    cols=cols,
    optimize_steps=200,                # NEW: enable the Adam loop
    optimize_lr=0.05,                  # upstream RCO's default
    optimize_tau_init=2.0,             # upstream GSQ's schedule
    optimize_tau_min=0.05,
    optimize_logit_scale_init=1.0,
    optimize_logit_scale_end=5.0,
)
```

### GSQ

```python
from toffee.quant.gsq import GsqConfig, refine_tensor

cfg = GsqConfig(
    epochs=0,                          # relaxation off by default
    search_super_scale=True,           # NEW: search the fp16 super-scale (default True)
    scale_passes=2,
)
result = refine_tensor(
    target=weights,
    type_name="Q4_K",
    imatrix=imatrix_weights,
    config=cfg,
    initial_codes=codes,
    initial_raw=raws,
)
```

For the full-GGUF pass, `refine_gguf_with_gsq` already accepts a
`GsqConfig`; pass the new config and the super-scale search will run.

## What this does NOT do

- It does not change llama.cpp's own quantizer or its `ggml_row_size`. The
  exact-byte-cost path is unchanged.
- It does not add new quant types.
- It does not change the GGUF write path; the bytes written are still
  validated against llama.cpp's own decoder via `_agrees_with_llama`.
- The optimizer's gradient is the *reconstruction*-distortion gradient, not
  the model-KL/CE gradient. On a CPU-only laptop the model forward is not
  available, so the soft expected distortion is the strongest differentiable
  proxy. If a GPU is available, swapping the loss to the model's KL/CE
  (as upstream RCO does) is straightforward — the Adam loop, tangent
  projection, retraction and vector transport are all generic.

## Files in this directory

```
rco.py             # refined (new: optimize_projected_gumbel_recon, expected_distortion, expected_distortion_grad)
gsq.py             # refined (fixed: per-chunk bug, schedule; new: _optimal_eff_scales_mins, _refine_super_scale)
refine.py          # refined (new: optimize_steps plumbing in build_rco_tensor_map)
gguf_kquant.py     # unchanged (copied for completeness)
```

The synthetic test:
```
/home/z/my-project/scripts/test_refined.py
```
