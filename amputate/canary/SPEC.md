# PPL Canary Corpus Spec — `toffee/amputate/canary`

**Status: registered before the first dense-vs-amputated run.**
Every threshold below is fixed *now*, on purpose. Once results exist,
changing a threshold means writing SPEC v2 and keeping the old verdict —
never quietly retuning until the answer flips. This document is
normative; `prep_corpus.py` / `run_canary.py` implement it exactly.

Purpose: decide **mechanically** whether the A12B amputated model
(dense→MoE, no training) is allowed to replace the dense IQ2_XS daily
driver, using `llama-perplexity` on a frozen corpus.

---

## 1. Corpus design

**8 slots × 32,768 evaluated tokens** (8 chunks × ctx 4096) =
**262,144 tokens**. Small enough for a ~4–5 min dense run on the
Creator 16, large enough that a +3% PPL delta sits far outside
measurement noise (per-slot NLL standard error at 32K tokens is ~0.1%;
the noise we actually care about is machine state, handled in §5).

| # | slot | critical | source (fallback chain) | why it exists |
|---|------|----------|-------------------------|---------------|
| 1 | `en_wiki` | – | `Salesforce/wikitext` wikitext-2-raw-v1 **test** | classic anchor; smooth prose the router must not break |
| 2 | `zh_general` | – | `wmt/wmt19` zh-en **val+test** (zh side) → `wmt/wmt20` | ZH prose; the 248K vocab + FFN treat ZH differently |
| 3 | `tables_nums` | – | `google-research-datasets/wikitablequestions` **test** | digits & rigid structure; most sensitive to low-bit + router damage |
| 4 | `code_py` | – | `HuggingFaceH4/CodeAlpaca_20K` train (responses) → `sahil2801/CodeAlpaca` | exact grammar; router collapse shows in code first |
| 5 | `math` | ✔ | `EleutherAI/hendrycks_math` all 7 configs **test** → `openai/gsm8k` test | symbol-dense; core domain of a hybrid reasoning model |
| 6 | `reasoning` | ✔ | `simplescaling/s1K-1.1` train (solutions) | long-chain CoT, the model's native mode — the slot that decides the verdict |
| 7 | `agent_tools` | ✔ | `NousResearch/hermes-function-calling-v1` test_gpt4o (+train pad) | tool-call grammar + JSON, VGR-adjacent |
| 8 | `domain_pack` | ✔ | **local only**: `--domain-pack DIR` (repair-loop logs, manuals, lab notes) | the only slot that measures the domain you actually run |

Properties:

* **Raw completion text** (no chat template, no vision tokens). This is
  a *relative* canary — absolute PPL is not comparable to any
  leaderboard or other tool's numbers, and is not meant to be.
* **Every slot is eval-split / non-calibration data.** See §2.
* Slots degrade, never crash: a moved HF repo or a missing domain pack
  marks the slot `SKIPPED`/`SHORT` in the manifest; the verdict math
  uses actual token counts.
* `reasoning` caveat, accepted deliberately: s1K traces are R1-style,
  not VeriLoop's own thinking format. It stays because long-horizon
  coherence is exactly where router damage concentrates.
* Swap any slot with `--local slot=file.txt`; the manifest records the
  substitution.

## 2. Disjointness rule (non-negotiable)

The canary corpus must **not** be part of the harvest calibration data
(`toffee amputate harvest --dataset`). A router calibrated on canary
text would make the PPL A/B optimistic and the verdict worthless.

* Default harvest dataset (`el4/bartowski-imatrix-v5-semantic-parquet`,
  train) is disjoint from every slot by construction. ✔
* If you point harvest at your own corpus: exclude every
  `domain_pack` file. `canary_manifest.json` records their sha256s —
  the check is a hash diff.
* If you add a domain to calibration to fix a collapsed slot (§6), use
  *train*-split or otherwise different text than the canary slot.

## 3. Building the corpus (once, ~2 min, a few MB)

```bash
pip install datasets tokenizers          # once

python -m toffee.amputate.canary prep \
    --tokenizer /path/to/VeriLoop-E2/tokenizer.json \
    --domain-pack ~/vgr-logs            # optional but recommended

python -m toffee.amputate.canary prep --verify   # re-hash anytime
```

* With `--tokenizer` (recommended: the VeriLoop-E2 repo's
  `tokenizer.json` — the amputation preserves embed/lm_head, so both
  models share it), each slot is trimmed to ≥ 34,000 **exact** tokens;
  without it, a +8%-margin chars-per-token estimate is used. Either
  way the runner pins `--chunks 8`, so the **evaluated** length is
  exactly 32,768 tokens per slot for every model.
* Output: `canary_corpus/<slot>.txt` + `canary_manifest.json`
  (sha256 of every file, resolved HF dataset shas, trim mode, token
  counts, domain-pack hashes, disjointness note).
* The corpus is **frozen**: rebuilding over an existing manifest
  requires `--force` (it invalidates every result computed against
  it); `prep --verify` re-hashes; the runner re-verifies before every
  A/B.

## 4. Run matrix

Same machine, same session, back-to-back, AC power, cool GPU, no other
GPU jobs. **All flags identical across runs** except `-m` (and `-ngl`
for the partial-offload pair — keep `-ngl` identical *within* a pair;
partial offload changes kernel placement, so never compare absolute
PPL across different `-ngl` settings).

| run | model | fits 12 GB VRAM? | purpose | time (est.) |
|-----|-------|------------------|---------|-------------|
| **E1** | dense IQ2_XS (~7.1 GB) | ✔ full offload | baseline = the bar to beat | ~4–5 min |
| **E1b** | E1 repeat (`--repeat-baseline`) | ✔ | measures the noise floor | ~4–5 min |
| **E3** | amputated IQ2_XS, dial 1 (~7.0 GB) | ✔ | the shipping candidate | ~2–3 min |
| **E2** | amputated Q4_K_M, dial 2 (~16 GB) | ✖ `-ngl ~40` | high-fidelity candidate | ~10–25 min, one-shot |
| **E0b** | dense Q4_K_M (~16.2 GB) | ✖ `-ngl ~40` (same as E2!) | E2's quant-matched partner: isolates amputation damage from quant damage | ~15–30 min, one-shot, optional but recommended |

Commands (E1/E1b/E3 minimum):

```bash
python -m toffee.amputate.canary run \
    --llama-bin ~/llama.cpp/build/bin \
    --model dense_iq2xs=/path/veriloop-e2-iq2xs.gguf \
    --model amp_iq2xs=/path/amputated-iq2xs.gguf \
    --repeat-baseline
```

Optional quant-matched pair:

```bash
python -m toffee.amputate.canary run ... \
    --model dense_q4km=/path/dense-q4km.gguf \
    --model amp_q4km=/path/amp-q4km.gguf \
    --ngl dense_q4km=40 --ngl amp_q4km=40 \
    --pair dense_q4km,amp_q4km
```

Pinned invocation per (model × slot): `llama-perplexity -m M -f S -c
4096 --chunks 8 -b 2048 -ub 512 -fa [-ngl N]`. KV cache stays fp16
(default) — no `-ctk/-ctv` in the canary; quant knobs are evaluated by
the dial comparison, not mixed into the amputation verdict.

Runtime budget: dense ≈ 262K tokens / ~961 t/s ≈ 4.5 min + model
loads (mmap + page cache makes the 8 per-slot loads cheap); amputated
at the forecast ~1400–1800 t/s ≈ 2.5–3 min.

## 5. Pre-registered decision rules

Definitions:

* `Δ(model)` per slot and pooled (token-weighted, **log space**:
  `pooled = exp(Σ tokensᵢ·ln PPLᵢ / Σ tokensᵢ)` — never average PPLs
  directly).
* **Noise floor**: `|Δ(E1b vs E1)|` pooled, from the baseline repeat.
* **Tie bar** `τ = max(0.5%, 2 × noise floor)`; `eff(x) = max(0, x − τ)`.

Sanity gates (before any quality reading; failure → verdict `INVALID`,
exit 3, fix and rerun):

1. **Token parity**: every slot must report the *same token count* for
   every model (same corpus bytes + same tokenizer). A mismatch is
   tokenizer/export drift in the amputated GGUF — a build bug, not a
   quality result.
2. **Parse**: every run yields a finite PPL (version-tolerant parser;
   raw stdout tails are kept in `results.json` for audit).
3. **Stability**: baseline rerun noise > 0.5% → unstable machine state
   (thermal/throttle/background GPU) → rerun both models back-to-back.

Then, per candidate model (E3, E2, …) vs the baseline (E1):

| verdict | condition (all in `eff` %) | meaning |
|---------|---------------------------|---------|
| **ACCEPT** | pooled ≤ 3 AND worst slot ≤ 6 | ship it, measure tps |
| **CONDITIONAL** | pooled ≤ 3 AND worst slot > 6 | daily driver yes; keep dense IQ2_XS for final artifacts |
| **INVESTIGATE** | 3 < pooled ≤ 8 OR any slot > 10 | fixable: triage table §6 |
| **KILL** | pooled > 8 | revert to dense IQ2_XS; this FFN is not MoEfiable at this budget on this data |

Critical slots (`math`, `reasoning`, `agent_tools`, `domain_pack`)
carry the +6% guard implicitly via the worst-slot rule; a non-critical
slot in (6, 10] also lands CONDITIONAL, not ACCEPT — with 8 slots,
one sloppy domain is a warning, not a pass.

**Quant-matched pair rule** (E0b vs E2, `--pair`): pooled Δ > +2.5% →
the damage is the **router fit** (amputation), not the quant → refit
before touching quant dials. Δ ≤ +2.5% while a cheaper candidate fails
→ the damage is **quant** → switch shipping config to the RCO split
dial (Q4 always-on / Q3 routed).

Exit codes: ACCEPT/CONDITIONAL → 0, INVESTIGATE → 1, KILL → 2,
INVALID → 3.

## 6. Triage table (symptom → diagnosis → fix)

| symptom | diagnosis | fix |
|---|---|---|
| pooled Δ > 8% | router fit under-trained / FFN not MoEfiable here | re-harvest with `--samples 512+` and a domain-mixed calibration corpus (disjoint!); last resort: top-k 4→6 or shared expert 4096→6144 (re-run the bandwidth math via `toffee amputate plan`) |
| one slot > 10%, others fine | that domain is OOD for the router | add *its* text (train split, not canary files) to harvest calibration, refit |
| `code_py` / `tables_nums` worst | expert granularity too coarse (544 neurons/expert) | `--num-experts 48` (verify partition), or top-k 6 on FFN-heavy layers |
| `reasoning` worst only | long-horizon routing drift | shared expert up, or keep the last 8 layers dense (~+0.9 GB at IQ2_XS) |
| pair read clean but E3 fails | quant damage, not amputation | ship the RCO split dial instead of uniform IQ2_XS |
| token parity violated | export/tokenizer bug in the amputated GGUF | hard stop; fix `build` before reading any PPL |
| noise floor > 0.5% | machine state | AC, cool-down, close other GPU jobs, rerun back-to-back |

## 7. Manifest & results schema

`canary_manifest.json`: `canary_version`, created UTC, trim target,
eval params (32,768 / ctx 4096 / 8 chunks), tokenizer path + exactness,
per-slot {file, sha256, bytes, tokens, trim_mode, blocks_used,
sources, critical, why, short_slot?, skipped?}, resolved HF shas,
domain-pack file hashes, disjointness rule text.

`results.json`: raw PPL + token count + wall time + full command +
stdout tail per (model × slot); pooled per model; noise floor; τ;
per-candidate deltas (raw and eff) per slot + pooled; verdicts with
the fired rule; pair reads.

## 8. Non-goals & follow-ups (after the PPL gate passes)

* **20 fixed temp-0 prompt diffs** — generation behavior (formatting,
  refusal drift, reasoning-mode triggering), PPL blind spots.
* **VGR repair-loop canaries** — the actual task: repair success on
  the lab's scripted failures vs dense.
* Long-context spot check — the canary is 4K-chunked; a one-off 32K-ctx
  comparison is cheap insurance once ACCEPT.
* This canary measures *relative* damage only. No leaderboard claims.

## 9. Selftest

```bash
python -m toffee.amputate.canary prep --selftest   # offline corpus machinery
python -m toffee.amputate.canary run --selftest    # parser + verdict engine
```

Both run without network, models, or llama.cpp (synthetic blocks,
canned llama-perplexity outputs, fake executor). They must pass before
the first real A/B — a verdict engine that was never executed is just
a hope.
