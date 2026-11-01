# AMPUTATE — no-training MoEfication for VeriLoop-E2 (and friends)

`toffee amputate` turns a **dense** hybrid checkpoint into a **sparse MoE**
checkpoint (Qwen3.5-MoE layout) with **zero gradient training**: routers and
expert assignments are fit from activation statistics, calibration scales are
solved in closed form, and the output is a normal HF directory that the
existing TOFFEE quant pipeline consumes unchanged.

The locked preset for VeriLoop-E2 27B (**A12B**):

| | |
|---|---|
| layers kept | 64 (all; mixers untouched) |
| FFN 17408 → | **32 routed experts x 544 neurons** (complete partition: every dense neuron lands in exactly one expert) |
| shared expert | **4096 neurons**, duplicated from the global top-importance set |
| routing | **top-4** of 32 |
| active params | **10.94 B** per token (vs 27.5 B dense) — 64% FFN sparsity |
| stored params | ~27.2 B (routed pool keeps every neuron once; the shared expert is the duplication cost) |

`toffee amputate plan` prints the full bandwidth math and the three quant
dials (IQ2_XS speed / Q4_K_M quality / RCO split).

---

## The commands

All of them also work standalone: `python -m toffee.amputate <cmd>`.

```bash
# 0. sanity: the whole machinery on a synthetic model, no download, CPU, ~1 min
toffee amputate selftest                      # expect: 73/73 checks passed

# 1. inspect the checkpoint: what exactly lives in the shards?
toffee amputate report --model /path/to/VeriLoop-E2 --output-dir amp
#    -> amputate_registry.json + the classification table

# 2. the surgery statistics (two forward sweeps per layer, layer-streamed)
toffee amputate harvest --model /path/to/VeriLoop-E2 --output-dir amp \
    --dataset <your-domain-corpus> --samples 256 --max-length 2048 --vram-limit 10
#    -> amp/surgery_stats.pt  (+ the offline canary table printed at the end)
#    NOTE: the harvest corpus must NOT contain canary corpus text
#    (canary/SPEC.md section 2, the disjointness rule)

# 3. the amputation itself (disk-bound slicing; no GPU needed)
toffee amputate build --model /path/to/VeriLoop-E2 --output-dir amp
#    -> amp/amputated/  (qwen3_5_moe HF directory, ~4 GiB shards)
#
#    config.json mirrors the family WRAPPER layout that
#    convert_hf_to_gguf.py registers (architectures:
#    Qwen3_5MoeForConditionalGeneration + nested text_config /
#    model_type qwen3_5_moe_text).  If you built BEFORE v1.0.5 and only
#    need the config fixed (shards are config-independent), refresh it
#    in seconds instead of re-slicing ~55 GB:
toffee amputate build --model /path/to/VeriLoop-E2 --output-dir amp --config-only
#    (add --no-vision-config if your loader dislikes CLIP metadata on a
#     text-only GGUF)

# 4. verdict before quantizing
toffee amputate verify --output-dir amp \
    --ppl-dense veriloop-e2-iq2xs.gguf \
    --ppl-amputated amputated-iq2xs.gguf --ppl-corpus canary_corpus

# 5. quantize exactly like any other MoE (your existing pipeline)
toffee quant run-all --model amp/amputated --gguf-quant IQ2_XS ...

# 6. build + freeze the PPL canary corpus (once, ~2 min; do this BEFORE
#    the first A/B so results can never be argued after the fact)
toffee amputate canary-prep --tokenizer /path/to/VeriLoop-E2/tokenizer.json \
    --domain-pack ~/vgr-logs          # -> canary_corpus/ + sha256 manifest

# 7. the PPL verdict: pre-registered rules, mechanical decision
toffee amputate canary-run --llama-bin ~/llama.cpp/build/bin \
    --model dense_iq2xs=veriloop-e2-iq2xs.gguf \
    --model amp_iq2xs=amputated-iq2xs.gguf --repeat-baseline
#    exit 0 ACCEPT/CONDITIONAL, 1 INVESTIGATE, 2 KILL, 3 INVALID
```

## What `report` decides (the `mtp.veriloop_*` question)

The VeriLoop-E2 shards carry 34 tensors under `mtp.veriloop_*`
("veriloop_conditional_memory", semantic-gate routing,
`vllm_ignored_mtp_namespace`, "embedded_loader_cpu_verified_gpu_pending").
Policy, baked into `toffee/amputate/registry.py`:

| category | default policy | why |
|---|---|---|
| `mtp.veriloop_*` | **keep, copy verbatim** | semantics unverified on GPU; the Qwen3.5 loader skips `mtp.`, so they are inert for llama.cpp; deleting an unexplained memory bank from a *reasoning* model is not a call to make blind. `--drop-mtp` overrides. |
| other `mtp.*` (the real MTP head) | drop | was already skipped at GGUF load time (the 332 MiB in your VRAM overrun) |
| `model.visual.*` / vision towers | drop | text-only amputation |
| unknown tensors | keep + **build refuses** | `--drop-unknown` after inspecting `amputate_registry.json` |

## How the surgery works (design notes)

* **Importance is OUTPUT energy**, not activation magnitude:
  `sum_t act_t[i]^2 * ||W_down[:, i]||^2`. Loud neurons whose down column is
  ~0 are dead weight; energy weighting finds them for free (the layer is
  resident during harvest).
* **Router**: ridge regression from the FFN input (post-attention LN output)
  to per-expert aggregated **`log1p` output energy**. Target chosen by direct
  experiment on the synthetic model (`energy` vs `log1p` vs `indicator` vs
  per-neuron transforms): per-neuron `log1p` lifted top-4 Jaccard 0.13 → 0.23
  and selection mass 0.886 → 0.906 over raw energy, and unlike
  log1p-of-aggregate / indicator it stays computable from streaming
  sufficient statistics (no token is ever stored).
  **Temperature normalization** (2026-10-31, after the real 27B run failed
  the offline canary at `router_corr_mean 0.023`): the raw fit's logits are
  in the target's units -- a SUM of I/E = 544 per-neuron `log1p` energies,
  i.e. hundreds-to-thousands on a real FFN -- so llama.cpp's `softmax`
  degenerated to a hard argmax and the top-4 weights collapsed to
  `[1,0,0,0]` (3 of 4 selected experts dead, calibration scales clamped,
  correlation dead while Jaccard stayed at 0.35). The fit now rescales the
  emitted weights so the logits' DISCRIMINATIVE pooled std (what softmax
  actually sees) equals `router_logit_scale` (default 1.0 since v1.0.4;
  was 2.0), computed
  exactly from the sufficient statistics. Top-k order is untouched; only
  the weights are put back in range. The verify table now carries a `p1`
  column (mean top-1 probability; ~1.0 = saturated) and the canary fails
  on `router_top1_prob > 0.90`.
  **Attenuation vs blindness** (2026-10-31, after the fixed real run
  landed at `router_corr_mean 0.110` vs the 0.12 gate with healthy
  `p1 0.478`): the gated corr is computed in PROB space —
  Pearson(softmax(z), s) — and the softmax nonlinearity *attenuates* it
  far below the logit-space Pearson(z, s). Multi-seed synthetics: lz
  0.60–0.78 while prob-space corr is only 0.08–0.20, and dropping the
  temperature 2.0 → 1.0 raises the measured corr by +0.017…+0.038
  (selection, Jaccard, mass and calibrated cos are all invariant — a
  positive rescaling never reorders logits and the calibration absorbs
  the weight scale). Harvest now also accumulates the logit-space
  moments (no extra pass) and reports `router_logit_corr` per layer —
  the `lz` verify column, a temp-invariant diagnostic that is NOT
  gated: high lz + low corr = attenuation (re-harvest with a lower
  `--router-logit-scale` — the FAIL message suggests half the current
  value, e.g. 0.75 once at the 1.0 default); low lz = the router
  genuinely
  cannot rank experts (try `--model-impl real`, or revisit the preset).
  The verify FAIL message makes exactly this call.
  **v1.0.4 — default temperature lowered 2.0 → 1.0**: the first real
  64L/32E harvest (VeriLoop-E2) measured `router_corr_mean 0.1101` at
  2.0 — just under the 0.12 gate — with healthy `lz 0.830` and
  `p1 0.478`, i.e. the attenuation regime the diagnostic predicts.
  Real-code-path synthetic sweep (E=32/K=4): corr ×1.32 at scale 1.0,
  lz/jaccard/mass bit-identical, calibrated cos flat (0.9510 → 0.9515);
  two parametric regimes (heavy-tail logits, sparse-active experts)
  recover ≥ ×1.16. Selection is invariant to the scale, so lowering it
  only re-spreads the top-4 weights, which the calibration re-fits.
* **GGUF converter key, v1.0.5** (2026-10-31, after `toffee quant run-all`
  on the real amputated model died with `Model Qwen3_5MoeForCausalLM is
  not supported`): convert_hf_to_gguf.py resolves a checkpoint by
  `architectures[0]` and only accepts registered family class names, so
  the amputated config.json must mirror the family WRAPPER layout the
  real members use -- `architectures:
  ["Qwen3_5MoeForConditionalGeneration"]`, top-level `model_type:
  qwen3_5_moe`, text backbone NESTED under `text_config`
  (`model_type: qwen3_5_moe_text`) with the MoE overrides, plus the
  vocab-level vision token ids and (by default) the source
  `vision_config` block mirrored into the wrapper.  v1.0.4's invented
  `Qwen3_5MoeForCausalLM` + flat layout was never a registered key.
  The tensor set is config-independent, so `build --config-only`
  re-emits the config in seconds without re-slicing the ~55 GB of
  shards; `--no-vision-config` drops the mirrored CLIP block for loaders
  that dislike vision metadata on a text-only GGUF.
* **Calibration**: llama.cpp MoE forward is `softmax` over ALL experts, then
  top-k experts summed, each weighted by its softmax probability. Those
  weights are not what the dense FFN did, so per-expert scales `c_e` and a
  shared scale `s` are solved as an **exact joint (E+1)-dim least squares**
  from co-selection statistics accumulated in sweep B, then **baked into the
  expert `down_proj` columns** — zero runtime cost, zero format extensions.
* **Shared expert gate**: emitted as all-zero weights, so llama.cpp's
  `sigmoid(0) = 0.5` combines with the `2s` baked into the shared
  `down_proj` to realise the fitted scale `s` exactly.
* **Known limitations** (documented, deliberate):
  * no router bias (HF/llama.cpp MoE layout has none; the fitted bias is kept
    in `surgery_stats.pt` for diagnostics),
  * partition is importance-balanced round-robin, not co-activation
    clustering (ToMoE-style clustering is a possible v2; load balance is
    guaranteed by construction),
  * the harvest is one-shot per layer pair of sweeps; `surgery_stats.pt` is
    rewritten after every layer, but hidden-state propagation means a crash
    restarts the harvest (budget ~1.5-2.5 h on the 4080 laptop).
* **Dense-source enforcement**: the harvest trusts the checkpoint's TENSOR
  NAMES, not the config object. Config-family MoE defaults (transformers
  class defaults for the Qwen3.5 hybrid-MoE family, or the local
  qwen4_exp_text fallback's CLASS attributes `num_experts=180` /
  `moe_intermediate_size=512`, which plain `getattr` sees) are cleared
  before the model is built; an already-MoE checkpoint is rejected up
  front; and the built model is structurally verified dense (a real-impl
  build that still comes out MoE auto-falls-back to the minimal model).
* **Model implementations**: prefer `--model-impl real` (the true
  transformers classes). When transformers can't build the model (older
  releases without `qwen3_5`), the harvest falls back to the minimal
  stand-in, whose norms are now faithful to the real architecture
  (zero-centred `Qwen3_5RMSNorm` semantics, silu-gated RMSNorm on the
  GDN value path) -- the fallback used to emulate every norm as
  mean-subtracting `nn.LayerNorm`, which skewed the FFN-input
  distribution the router is fit on. The GDN recurrence inside the
  minimal model remains a crude stand-in, so `real` is still the
  higher-fidelity path for the final harvest.

## Runtime budget on the Creator 16 (RTX 4080 Mobile, 12 GB)

| stage | VRAM | time (estimate) |
|---|---|---|
| report | ~0 (header scan) | seconds |
| harvest (2 sweeps x 64 layers, 256x2048 tokens) | ≤ ~4 GB on top of one layer (~1.7 GB weights + ~1 GB accumulators) | ~1.5–2.5 h |
| build | CPU + disk | ~20–40 min (reads+writes ~55 GB) |
| quant run-all on the amputated dir | your usual envelope | ~your usual time |

Disk: source 54 GB + amputated dir ~55 GB (bf16) + F16 GGUF intermediates —
make room or use `--split-max-size` as you already do.

## Verification: offline canary first, PPL canary second

1. **Offline canary** (free, printed by `harvest`/`verify`): per-layer
   `cos_calibrated` vs `cos_shared_only` vs oracle mass. Gates the BUILD.
   `verify` exits non-zero if any layer is below threshold.
2. **PPL canary** (frozen corpus + pre-registered verdict,
   `toffee/amputate/canary/SPEC.md` — normative):
   * `canary-prep`: 8 domain slots (en/zh prose, tables+numbers, code,
     math, long-CoT reasoning, tool-calls, **your domain pack**) ×
     32,768 tokens, sha256-frozen manifest, disjoint from the harvest
     calibration corpus.
   * `canary-run`: llama-perplexity A/B (identical flags, same bytes,
     same tokenizer, token-parity gate), log-space pooled PPL, noise
     floor from a baseline rerun, `tau = max(0.5%, 2x noise)`, then the
     pre-registered verdict: `ACCEPT` (pooled ≤ +3%, worst slot ≤ +6%) /
     `CONDITIONAL` (daily driver only) / `INVESTIGATE` (3–8% or any slot
     > +10%, with a symptom→fix triage table) / `KILL` (> +8%: revert to
     dense IQ2_XS). Optional quant-matched pair (`dense Q4_K_M` vs
     `amp Q4_K_M`, identical `-ngl`) separates router damage from quant
     damage (> +2.5% pooled → refit the router, don't blame the quant).
3. 20 fixed temp-0 prompts diff + VGR repair-loop canaries before daily use.

## Selftest

`toffee amputate selftest` builds a tiny dense hybrid (hidden 64, 4 layers,
GDN + full attention, vision junk, `mtp.veriloop_*` tensors, untied lm_head)
with a **structured** FFN — rank-8 gate/up (learnable activations) and a
two-tier down-column energy structure — then runs report → harvest → build →
verify and asserts 51 properties, including:

* registry classification (ffn / vision / mtp_veriloop / mtp_head counts),
* dense-source robustness: the tensor-truth check accepts the dense
  checkpoint and rejects an MoE one; config pollution (the qwen4_exp_text
  fallback's class attributes AND transformers instance-level `__init__`
  defaults) still builds a dense FFN -- regression for the laptop bug
  where a dense checkpoint's config leaked `num_experts=180` and the
  harvest model came out MoE ("dense FFN not found"),
* the shared expert finds the heavy tier (≥ 80% overlap),
* selection mass recall ≥ 85% (oracle printed alongside),
* routing strictly beats the shared-only floor (calibrated cosine),
* expert slicing + calibration baking are bit-faithful: the **built** MoE
  forward under exact llama.cpp semantics equals the source-math forward at
  cos 1.00000, and down columns carry the scales at rel err 0.0000,
* the emitted config is a valid text-only `qwen3_5_moe` layout.

Last run: **73/73 passed** (CPU, torch 2.15, transformers 5.22; the four
config-wrapper checks pin the convert_hf_to_gguf key
(`Qwen3_5MoeForConditionalGeneration` + nested `text_config`) and the
`--no-vision-config` mode; the four dense-source checks reproduce the
config-pollution crash seen on a laptop whose transformers does not
register `qwen3_5`).

The canary module has its own offline selftests (no network, no models,
no llama.cpp needed):

```bash
toffee amputate canary-prep --selftest   # corpus build/freeze/verify machinery
toffee amputate canary-run --selftest    # ppl parser + verdict engine
```
