"""
@file toffee/amputate/verify.py
   Read ``surgery_stats.pt`` and turn it into a verdict.

Two layers of verification, cheapest first:

Offline (free, computed during harvest sweep B, rendered here):
  per-layer cosine of the amputated FFN reconstruction vs the dense FFN
  under exact llama.cpp semantics, selection-mass recall vs the
  same-budget oracle, router correlation and top-k Jaccard, calibration
  scale ranges. These numbers predict whether the surgery is sane
  BEFORE any GGUF is built or quantized.

Online (commands printed, run them on the laptop):
  the frozen PPL canary corpus (``toffee/amputate/canary``): 8 domain
  slots x 32,768 tokens, llama-perplexity dense IQ2_XS vs amputated
  GGUF, same bytes (sha256-frozen), same tokenizer, with pre-written
  kill criteria so the decision is mechanical, not vibes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

# Offline canary thresholds. These gate the BUILD, not the model: a
# layer whose reconstruction is worse than its shared-only floor has a
# broken router fit, not a broken model. (Values calibrated on the
# synthetic selftest; a real 27B FFN should clear them comfortably --
# MoEfication literature reports much stronger predictability on real
# activations than on random synthetic ones.)
DEFAULT_THRESHOLDS = {
    "cos_min": 0.80,          # per-layer calibrated cosine floor
    "mass_min": 0.80,         # selection-mass recall floor
    "corr_min": 0.12,         # mean router/truth correlation floor (log1p space)
    "jaccard_min": 0.10,      # ~1.5x chance (chance ~ K/E * ...~0.07)
    "top1_prob_max": 0.90,    # softmax saturation ceiling (1.0 = hard argmax;
                               # checked only when the metric exists, i.e.
                               # harvests with the temperature normalization)
}

KILL_CRITERIA = """
PPL canary decision rules (registered in canary/SPEC.md BEFORE the
first run; honor AFTER). eff(x) = max(0, x - tau),
tau = max(0.5%, 2 x baseline-rerun noise):

  ACCEPT       eff(pooled) <= +3%  and worst slot <= +6%   -> ship + measure tps
  CONDITIONAL  eff(pooled) <= +3%  and worst slot  > +6%   -> daily driver only,
                                                            keep dense for finals
  INVESTIGATE  +3% < eff(pooled) <= +8%, or any slot > +10%
                         -> per-layer table; re-harvest offender layers
                            (strategy=block / shared 6144) or fix the
                            collapsed slot's domain in calibration
  KILL         eff(pooled) > +8%  -> revert to dense IQ2_XS; the 27B
                            dense FFN is not MoEfiable at this budget
                            on this data

Sanity gates first (verdict INVALID, fix and rerun): token-count
parity per slot across models (export drift), parse success, baseline
rerun noise <= 0.5%.
Quant-matched pair (dense Q4_K_M vs amputated Q4_K_M, identical
-ngl): pooled delta > +2.5% -> damage is the router fit (refit),
not the quant.

Then the 20 fixed temp-0 prompts diff + the VGR repair-loop canaries
before calling it a day.
"""


def load_stats(stats_path: str | Path) -> Dict:
    return torch.load(stats_path, map_location="cpu", weights_only=False)


def per_layer_table(stats: Dict) -> str:
    layers = stats["layers"]
    head = (f"{'layer':>5} {'cos_cal':>8} {'cos_shr':>8} {'mass':>7} "
            f"{'oracle':>7} {'corr':>6} {'lz':>6} {'jac':>6} {'p1':>5} "
            f"{'c_min':>6} {'c_max':>6} {'s':>6}")
    lines = [head, "-" * len(head)]
    for lstr in sorted(layers, key=int):
        m = layers[lstr]["metrics"]
        p1 = m.get("router_top1_prob")
        p1_s = f"{p1:>5.2f}" if p1 is not None else "    -"
        lz = m.get("router_logit_corr")
        lz_s = f"{lz:>6.3f}" if lz is not None else "     -"
        lines.append(
            f"{lstr:>5} {m['cos_calibrated']:>8.4f} {m['cos_shared_only']:>8.4f} "
            f"{m['mass_recall']:>7.4f} {m['mass_oracle']:>7.4f} "
            f"{m['router_corr_mean']:>6.3f} {lz_s} {m['sel_jaccard']:>6.3f} {p1_s} "
            f"{m['scale_c_min']:>6.2f} {m['scale_c_max']:>6.2f} "
            f"{m['scale_s']:>6.2f}"
        )
    return "\n".join(lines)


def aggregate(stats: Dict) -> Dict[str, float]:
    layers = stats["layers"]
    n = len(layers)
    keys = ("cos_calibrated", "cos_shared_only", "mass_recall", "mass_oracle",
            "router_corr_mean", "sel_jaccard")
    agg = {k: sum(layers[l]["metrics"][k] for l in layers) / max(n, 1)
           for k in keys}
    p1s = [layers[l]["metrics"]["router_top1_prob"] for l in layers
           if layers[l]["metrics"].get("router_top1_prob") is not None]
    agg["router_top1_prob"] = (sum(p1s) / len(p1s)) if p1s else None
    lzs = [layers[l]["metrics"]["router_logit_corr"] for l in layers
           if layers[l]["metrics"].get("router_logit_corr") is not None]
    agg["router_logit_corr"] = (sum(lzs) / len(lzs)) if lzs else None
    agg["cos_min_layer"] = min(
        (layers[l]["metrics"]["cos_calibrated"] for l in layers), default=0.0)
    agg["mass_min_layer"] = min(
        (layers[l]["metrics"]["mass_recall"] for l in layers), default=0.0)
    return agg


def check_thresholds(stats: Dict,
                     thresholds: Optional[Dict[str, float]] = None) -> Dict:
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    agg = aggregate(stats)
    failures = []
    if agg["cos_min_layer"] < th["cos_min"]:
        failures.append(
            f"cos_min_layer {agg['cos_min_layer']:.4f} < {th['cos_min']} "
            "(router fit broken on some layer)")
    if agg["mass_min_layer"] < th["mass_min"]:
        failures.append(
            f"mass_min_layer {agg['mass_min_layer']:.4f} < {th['mass_min']} "
            "(selection is missing real energy)")
    if agg["router_corr_mean"] < th["corr_min"]:
        msg = (f"router_corr_mean {agg['router_corr_mean']:.4f} < "
               f"{th['corr_min']} (router cannot see expert importance)")
        lz = agg.get("router_logit_corr")
        if lz is not None:
            # disambiguate: prob-space corr is a softmax-attenuated
            # version of the logit-space correlation (lz). High lz + low
            # corr = attenuation (temperature), low lz = real blindness.
            if lz >= 2.0 * th["corr_min"]:
                # suggest half the temperature the harvest actually used
                # (0.75 once already at the 1.0 default); selection is
                # scale-invariant so only the weight spread changes
                cur = (stats.get("preset") or {}).get("router_logit_scale")
                nxt = 0.75 if (cur is None or cur <= 1.05) else cur / 2.0
                msg += (f" [logit-corr {lz:.3f} is healthy: this is softmax "
                        "ATTENUATION, not blindness -- re-harvest with a "
                        f"lower --router-logit-scale (e.g. {nxt:g}; "
                        "jaccard/mass are scale-invariant)]")
            else:
                msg += (f" [logit-corr {lz:.3f} is also low: the router "
                        "genuinely ranks experts poorly -- try "
                        "--model-impl real, or revisit the preset]")
        failures.append(msg)
    if agg["sel_jaccard"] < th["jaccard_min"]:
        failures.append(
            f"sel_jaccard {agg['sel_jaccard']:.4f} < {th['jaccard_min']} "
            "(selection ~ chance)")
    # saturation gate (only for stats files that carry the metric)
    if agg.get("router_top1_prob") is not None \
            and agg["router_top1_prob"] > th["top1_prob_max"]:
        failures.append(
            f"router_top1_prob {agg['router_top1_prob']:.3f} > "
            f"{th['top1_prob_max']} (softmax saturated to a hard argmax; "
            "top-k weights collapsed -- lower router_logit_scale)")
    return {"aggregate": agg, "thresholds": th, "failures": failures,
            "pass": not failures}


def ppl_commands(stats_path: str | Path, dense_gguf: str,
                 amputated_gguf: str, corpus: str = "canary_corpus",
                 ctx: int = 4096) -> str:
    corpus_dir = corpus if corpus and Path(corpus).is_dir() \
        else "canary_corpus"
    return f"""
# Build + freeze the canary corpus once (~2 min, a few MB):
#   (tokenizer = VeriLoop-E2's own tokenizer.json; the amputation
#    preserves embed/lm_head, so both models share it)
python -m toffee.amputate.canary prep \\
    --tokenizer /path/to/VeriLoop-E2/tokenizer.json \\
    --domain-pack ~/vgr-logs

# E1 dense baseline (+ repeat for the noise floor) vs E3 candidate:
python -m toffee.amputate.canary run \\
    --llama-bin ~/llama.cpp/build/bin \\
    --model dense_iq2xs={dense_gguf} \\
    --model amp_iq2xs={amputated_gguf} --repeat-baseline

# 8 slots x 32,768 tokens; per-slot + pooled PPL, token-parity check,
# pre-registered verdict + triage table; results in
# {corpus_dir}/results.json
{KILL_CRITERIA}"""


def render_report(stats_path: str | Path,
                  dense_gguf: Optional[str] = None,
                  amputated_gguf: Optional[str] = None,
                  corpus: Optional[str] = None) -> str:
    stats = load_stats(stats_path)
    meta = stats["meta"]
    preset = stats["preset"]
    res = check_thresholds(stats)
    agg = res["aggregate"]
    lines = [
        f"amputation stats: {stats_path}",
        f"model: {meta.get('model')}  layers: {meta.get('num_layers')} "
        f"hidden: {meta.get('hidden')} inter: {meta.get('inter')} "
        f"(impl: {'transformers' if meta.get('used_real_transformers') else 'minimal'})",
        f"preset: {preset}",
        "",
        per_layer_table(stats),
        "",
        f"mean cos_calibrated {agg['cos_calibrated']:.4f} | "
        f"mean cos_shared_only {agg['cos_shared_only']:.4f} | "
        f"mean mass_recall {agg['mass_recall']:.4f} "
        f"(oracle {agg['mass_oracle']:.4f})"
        + (f" | mean top1_prob {agg['router_top1_prob']:.3f}"
           if agg.get("router_top1_prob") is not None else "")
        + (f" | mean logit-corr {agg['router_logit_corr']:.3f}"
           if agg.get("router_logit_corr") is not None else ""),
        "",
        "OFFLINE CANARY: " + ("PASS" if res["pass"] else "FAIL"),
    ]
    for f_ in res["failures"]:
        lines.append(f"  FAIL {f_}")
    if not res["pass"]:
        lines.append("  -> fix harvest before building the GGUF")
    if dense_gguf and amputated_gguf and corpus:
        lines.append(ppl_commands(stats_path, dense_gguf, amputated_gguf, corpus))
    return "\n".join(lines)
