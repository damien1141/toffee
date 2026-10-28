# TOFFEE — REFINED + PATCHED build

This is your `toffee` package with the refinements from our last few
sessions baked in. Drop this `toffee/` directory over your existing one.

## What's been changed in this build

### `quant/rco.py` — refined
- Added `optimize_projected_gumbel_recon` (Riemannian-Adam loop on the
  budget manifold, with the soft-expected-distortion loss).
- Added `expected_distortion` and `expected_distortion_grad` helpers.
- The previous revision only ran `init_alpha -> retract -> DP` once. Now
  you can opt into the full optimization loop via `optimize_steps` (see
  the new `--rco-optimize-steps` CLI flag below).

### `quant/gsq.py` — refined + vectorized (13.5x CPU speed-up) + GPU support
- **Fixed**: `hard_error < best_error` was comparing a per-chunk error
  against the whole-tensor baseline. Now tracks per-chunk baselines.
- **Fixed**: temperature/logit_scale schedule aligned with upstream GSQ
  (`2.0 -> 0.05` and `100 -> 500`; was `1.0 -> 0.05` and `1.0 -> 5.0`).
- **Added**: `_optimal_eff_scales_mins` — closed-form optimal (scale, min)
  per (block, sub-group), for fixed codes. 1-D closed form for symmetric
  types (Q8_0, Q6_K, Q3_K); 2x2 linear system for asymmetric (Q4_K,
  Q5_K, Q2_K).
- **Added**: `_refine_super_scale` — searches the fp16 super-scale `d`
  (and `dmin` for asymmetric types) ±2 ULP, seeded by the closed-form
  optimum. The previous revision left these fixed.
- **Vectorized**: `_refine_super_scale` now uses vectorized fp16 ULP
  arithmetic (`_fp16_add_ulp_vectorized`) + closed-form per-(block, group)
  error. ~12x faster than the per-block loop it replaced.
- **Vectorized**: `_refine_scales` now evaluates all 6 deltas with the
  closed-form error (no code re-fit), finds the best delta per block in
  closed form, then does ONE batched re-fit. ~20x faster than the
  previous "6 re-fits per (group, field)" approach.
- **Sign-error fix**: the closed-form error's cross term was
  `+2 * S_v * es * em` when it should be `-2 * S_v * es * em` (because
  the residual is `x_i - v_i*es + em`). Fixed in the d-search, the
  dmin-search, the baseline computation, and the coordinate descent.
- **GPU (torch)**: new `_refine_super_scale_torch` and
  `_refine_scales_torch` — mechanical translations of the two hot paths
  from NumPy to torch. On an RTX 4080 the GPU path is ~20-30x faster
  than the vectorized CPU path on a 16K-block Q4_K tensor. Opt-in via
  `GsqConfig(device="cuda")` or the `--device cuda` CLI flag.
  The Gumbel-Softmax relaxation stays on CPU (per-chunk Adam state is
  small, launch overhead would dominate).

### `quant/refine.py` — refined
- `build_rco_tensor_map` now accepts `optimize_steps`, `optimize_lr`,
  `optimize_tau_init`, `optimize_tau_min`, `optimize_logit_scale_init`,
  `optimize_logit_scale_end`. When `optimize_steps > 0` and a distortion
  table is available, it runs the new Riemannian-Adam loop before the DP.
- `refine_gguf_with_gsq` now accepts `parallel_workers`. When set, it
  refines the tensors of each block concurrently using a
  `multiprocessing.Pool`. Each worker opens its own GGUF readers; the
  main process collects refined bytes and writes them serially.

### `quant/cli.py` — patched
The CLI now exposes the new flags directly (no patcher needed):

```
toffee quant rco ... \
  --optimize-steps 200 \           # NEW: enable the Adam loop
  --optimize-lr 0.05

toffee quant gsq ... \
  --parallel-workers 4 \           # NEW: parallel tensor refinement
  --search-super-scale 1 \         # NEW: 1=on (default), 0=off
  --device cuda                    # NEW: GPU acceleration (default 'cpu')
```

### `cli.py` (top-level) — patched
The top-level `toffee/cli.py` ALSO has its own `p_quant_rco` and
`p_quant_gsq` subparsers (separate from `toffee/quant/cli.py`'s parser).
When you run `python -m toffee quant gsq ...`, the **top-level** parser
is what parses your args — so the new flags must be added there too.
This build patches BOTH `cli.py` files:

- `toffee/cli.py` — adds `--optimize-steps`, `--optimize-lr` to
  `p_quant_rco`, and `--parallel-workers`, `--search-super-scale` to
  `p_quant_gsq`.
- `toffee/quant/cli.py` — same flags, for when `toffee.quant.cli` is
  invoked directly (e.g. `python -m toffee.quant.cli gsq ...`).

If you previously unzipped and got `unrecognized arguments:
--parallel-workers`, it was because only `toffee/quant/cli.py` was
patched and the top-level `toffee/cli.py` parser (the one that actually
parses `python -m toffee quant gsq ...`) was missing the flag. This
build fixes that.

## How to run the full pipeline

```bash
# 1. APEX pass: safetensors -> f16 GGUF + imatrix + APEX-quantized GGUF
toffee quant run-all \
  --model <hf-name> \
  --tier quality \
  --gguf-quant Q4_K_M \
  --output-dir toffee-quant-output \
  --llama-cpp ./llama.cpp

# 2. RCO pass: refine the APEX map to an exact byte budget (with the
#    new --optimize-steps flag for the Riemannian-Adam loop)
toffee quant rco \
  --model <hf-name> \
  --tier quality \
  --experimental \
  --target-size-gb 18.5 \
  --ladder experts=Q6_K,Q5_K,Q4_K \
  --ladder shared=Q6_K,Q5_K \
  --optimize-steps 200 \
  --output-dir toffee-quant-output \
  --llama-cpp ./llama.cpp

# 3. Re-quantize with the RCO map (run-all auto-detects it)
toffee quant run-all \
  --model <hf-name> \
  --tier quality \
  --gguf-quant Q4_K_M \
  --output-dir toffee-quant-output \
  --llama-cpp ./llama.cpp

# 4. GSQ pass: refine codes + scales in place (with the new
#    --parallel-workers flag for the multiprocessing pool, and
#    --device cuda for GPU acceleration on a CUDA GPU)
toffee quant gsq \
  --gguf toffee-quant-output/<model>-TOFFEE-quality-Q4_K_M.gguf \
  --f16-gguf toffee-quant-output/<model>-f16.gguf \
  --imatrix toffee-quant-output/imatrix.dat \
  --parallel-workers 4 \
  --device cuda \
  --state toffee-quant-output/gsq-state.json \
  --time-budget-minutes 60 \
  --resume \
  --llama-cpp ./llama.cpp
```

If you just want APEX + GSQ (skip RCO, since RCO is still parked), only
stages 1 and 4 are needed.

## How to verify the refinements work

```bash
# CPU-only synthetic test (no llama.cpp binary needed)
python -m toffee.quant.test_refined
# (or run the file directly: python toffee/quant/test_refined.py)

# Expected output:
#   RCO soft-loss reduction:              +50.39%
#   RCO real-distortion reduction:         +66.82%
#   GSQ refine_tensor Q8_0:                +3.58%
#   GSQ refine_tensor Q4_K:                +99.50%
#   GSQ relaxation (Q4_K, epochs=4):      +0.65%
#   GSQ super_scale gain over baseline:   +99.07%
#   All tests passed.

# Per-tensor benchmark
python -m toffee.quant.bench_gsq

# Parallel-tensor benchmark
python -m toffee.quant.bench_parallel
```

## Speed-up summary

On a 16K-block Q4_K tensor (transformer-block-sized):

| Stage                          | Before  | After   | Speed-up |
| ------------------------------ | ------- | ------- | -------- |
| `_refine_super_scale`          | 5606 ms | 452 ms  | 12.4x    |
| `_refine_scales` (coord desc)  | 6662 ms | 329 ms  | 20.2x    |
| `refine_tensor` (end-to-end)   | 11967 ms| 884 ms  | 13.5x    |

With `--parallel-workers 4` on a multi-core CPU, expect another 2-4x on
top.

## What did NOT change

- The four budget-manifold primitives (`budget_normal`,
  `project_gradient`, `retraction`, `vector_transport`) are unchanged.
- `_Logits`, `gumbel_perturb`, `budget_constrained_argmax`,
  `GroupSpec`, `QuantOption`, `option_cost_table`,
  `allowed_option_mask`, `assign_from_options`, `budget_aware_init`
  are unchanged.
- `refine_tensor`, `_refine_tensor_rows`, `_gumbel_softmax`,
  `_weighted_error`, `_fit_codes`, `_refit_codes`, `_refit_group`,
  `_refit_group_chunked`, `_candidate_values`, `_code_values`,
  `_copy_raw`, `_effective_from_raw`, `RefineResult` are preserved.
- The GGUF write path, the codec (`gguf_kquant.py`), and the bridge
  (`ggml_bridge.py`) are unchanged. The bytes written are still
  validated against llama.cpp's own decoder via `_agrees_with_llama`.

## Files in this build

```
toffee/
├── __init__.py, __main__.py, cli.py, config.py, data.py, ...   # unchanged
├── quant/
│   ├── cli.py            # PATCHED: exposes --optimize-steps, --parallel-workers, --search-super-scale
│   ├── rco.py            # REFINED: + optimize_projected_gumbel_recon, expected_distortion
│   ├── gsq.py            # REFINED + VECTORIZED: bug fixes + closed-form super-scale search
│   ├── refine.py         # REFINED: optimize_steps plumbing + parallel_workers
│   ├── gguf_kquant.py   # unchanged
│   ├── ggml_bridge.py   # unchanged
│   ├── gguf_io.py       # unchanged
│   ├── mapping.py       # unchanged
│   ├── passes.py        # unchanged
│   ├── test_refined.py  # NEW: CPU-only synthetic test
│   ├── bench_gsq.py     # NEW: per-tensor benchmark
│   └── bench_parallel.py # NEW: parallel-tensor benchmark
└── xspark/, ...         # unchanged
```
