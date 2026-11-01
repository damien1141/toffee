"""
@file toffee/amputate/canary/run_canary.py
   Run llama-perplexity over the frozen canary corpus and emit the
   pre-registered verdict.

Usage (E1 baseline + repeat, E3 shipping candidate)::

    python -m toffee.amputate.canary run \\
        --model dense_iq2xs=veriloop-e2-iq2xs.gguf \\
        --model amp_iq2xs=amputated-iq2xs.gguf \\
        --repeat-baseline

Optional quant-matched pair (isolates amputation damage from quant
damage; keep ``-ngl`` identical within the pair)::

    ... --model dense_q4km=dense-q4km.gguf --model amp_q4km=amp-q4km.gguf \\
        --ngl dense_q4km=40 --ngl amp_q4km=40 --pair dense_q4km,amp_q4km

Decision rules (pre-registered BEFORE the first run; SPEC.md is
normative). With ``eff(x) = max(0, x - tau)`` and
``tau = max(0.5%, 2 x measured rerun noise)``:

  KILL         eff(pooled) > 8%
  INVESTIGATE  3% < eff(pooled) <= 8%   or   any eff(slot) > 10%
  CONDITIONAL  eff(pooled) <= 3% and some eff(slot) > 6%
  ACCEPT       eff(pooled) <= 3% and all eff(slot) <= 6%

Exit codes: ACCEPT/CONDITIONAL 0, INVESTIGATE 1, KILL 2, INVALID 3.

The pooled perplexity is computed in log space (never average PPLs
directly): ``pooled = exp(sum(tokens_i * ln PPL_i) / sum(tokens_i))``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from toffee.amputate.canary.slots import CRITICAL_SLOTS
from toffee.amputate.canary.prep_corpus import (
    MANIFEST_NAME,
    verify_corpus,
)

logger = logging.getLogger(__name__)

CANARY_VERSION = "1.0"
DEFAULT_TAU = 0.5           # % noise-floor when no rerun was done
NOISE_INVALID = 0.5         # % rerun noise above which we call INVALID

EXIT_CODES = {"ACCEPT": 0, "CONDITIONAL": 0, "INVESTIGATE": 1,
              "KILL": 2, "INVALID": 3}


# ---------------------------------------------------------------------------
# llama-perplexity output parsing (version-tolerant)
# ---------------------------------------------------------------------------

_PPL_RE = re.compile(
    r"perplexity\s*[:=]\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
    re.IGNORECASE)
_CHUNK_RE = re.compile(
    r"\[(\d+)\]\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")
_CHUNKS_TOKENS_RE = re.compile(
    r"([0-9][0-9,._]*)\s+chunks?\b[^0-9]*([0-9][0-9,._]*)\s+tokens?",
    re.IGNORECASE)
_TOKENS_RE = re.compile(r"([0-9][0-9,._]*)\s+tokens?", re.IGNORECASE)


def _int(s: str) -> int:
    return int(re.sub(r"[,._]", "", s))


def parse_output(stdout: str) -> Dict:
    """Extract (ppl, tokens) from llama-perplexity stdout.

    Tolerant across llama.cpp versions: prefers the last
    ``perplexity = X`` / ``perplexity : X`` / ``Final estimation:
    perplexity = X`` line; falls back to the per-chunk ``[i]X.XXXX``
    values (geometric mean); grabs the token count from the
    ``N chunks, M tokens`` banner.
    """
    tail = stdout[-1500:]
    ppl = None
    method = "none"
    matches = _PPL_RE.findall(stdout)
    if matches:
        try:
            ppl = float(matches[-1])
            method = "final-line"
        except ValueError:
            matches = []
    if ppl is None:
        vals = [float(v) for _, v in _CHUNK_RE.findall(stdout)
                if float(v) > 0]
        if vals:
            ppl = math.exp(sum(math.log(v) for v in vals) / len(vals))
            method = "chunk-lines-geomean"
    tokens = None
    m = _CHUNKS_TOKENS_RE.search(stdout)
    if m:
        tokens = _int(m.group(2))
    else:
        ms = _TOKENS_RE.findall(stdout)
        if ms:
            tokens = _int(ms[-1])
    return {"ppl": ppl, "tokens": tokens, "method": method,
            "raw_tail": tail}


# ---------------------------------------------------------------------------
# math + verdict engine (pure, unit-tested)
# ---------------------------------------------------------------------------

def pooled_ppl(slot_results: Dict[str, Dict]) -> Optional[float]:
    """Token-weighted pooled perplexity in log space."""
    num = 0.0
    den = 0
    for r in slot_results.values():
        if r.get("ppl") is None or not r.get("tokens"):
            return None
        num += r["tokens"] * math.log(r["ppl"])
        den += r["tokens"]
    if den == 0:
        return None
    return math.exp(num / den)


def delta_pct(a: float, b: float) -> float:
    """Percent change of b relative to a."""
    return (b / a - 1.0) * 100.0


def verdict_for(pooled_pct: float, slot_pcts: Dict[str, float],
                tau_pct: float,
                critical_slots=CRITICAL_SLOTS) -> Tuple[str, str]:
    """Apply the pre-registered decision rules to one candidate."""
    eff_p = max(0.0, pooled_pct - tau_pct)
    eff_slots = {k: max(0.0, v - tau_pct) for k, v in slot_pcts.items()}
    worst_slot, worst = max(eff_slots.items(), key=lambda kv: kv[1],
                            default=("<none>", 0.0))
    if eff_p > 8.0:
        return "KILL", (f"eff pooled {eff_p:+.2f}% > +8%")
    if eff_p > 3.0 or worst > 10.0:
        return "INVESTIGATE", (
            f"eff pooled {eff_p:+.2f}%, worst slot {worst_slot} "
            f"{worst:+.2f}%")
    if worst > 6.0:
        return "CONDITIONAL", (
            f"eff pooled {eff_p:+.2f}% ok but worst slot {worst_slot} "
            f"{worst:+.2f}% > +6%")
    return "ACCEPT", (f"eff pooled {eff_p:+.2f}%, worst slot "
                      f"{worst_slot} {worst:+.2f}%")


TRIAGE = {
    "KILL": [
        "revert to dense IQ2_XS (pre-registered rule)",
        "the 27B dense FFN is not MoEfiable at this budget on this data",
        "before giving up: retry harvest with --samples 512+ and a "
        "domain-mixed calibration corpus (DISJOINT from the canary)",
    ],
    "INVESTIGATE": [
        "print the per-layer table: python -m toffee.amputate verify "
        "--output-dir <amp>",
        "a minority of bad layers -> re-harvest those with "
        "--strategy block or --shared-intermediate 6144",
        "one slot collapsed (others fine) -> that domain is OOD for the "
        "router: add ITS text (train split, NOT the canary files) to the "
        "harvest calibration corpus and refit",
        "code/tables worst -> expert granularity too coarse: try "
        "--num-experts 48 (re-checks partition) or top-k 6",
        "reasoning worst -> shared expert up, or keep the last 8 layers "
        "dense (costs ~0.9 GB at IQ2_XS)",
    ],
    "CONDITIONAL": [
        "ship as the daily driver (2x speed), keep dense IQ2_XS for "
        "final artifacts",
        "watch the flagged slots in daily use; re-check after any "
        "router refit",
    ],
    "ACCEPT": [
        "ship dial 1 (IQ2_XS) and measure tps (forecast ~44-50)",
        "then run the 20 fixed temp-0 prompt diff + VGR repair-loop "
        "canaries (SPEC.md follow-ups)",
    ],
}


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def resolve_binary(llama_bin: Optional[str], binary: Optional[str]) -> str:
    if binary:
        return binary
    candidates = []
    if llama_bin:
        d = Path(llama_bin)
        candidates += [d / "llama-perplexity", d / "perplexity"]
    for name in ("llama-perplexity", "perplexity"):
        w = shutil.which(name)
        if w:
            candidates.append(Path(w))
    for rel in ("build/bin/llama-perplexity",
                "../llama.cpp/build/bin/llama-perplexity"):
        p = Path(rel)
        if p.exists():
            candidates.append(p)
    for c in candidates:
        if Path(c).exists():
            return str(c)
    raise SystemExit(
        "llama-perplexity not found; pass --llama-bin <dir> or "
        "--binary <path>")


def build_cmd(binary: str, model_path: str, slot_file: str, ctx: int,
              chunks: int, batch: int, ubatch: int, fa: bool,
              ngl: Optional[int], extra: List[str]) -> List[str]:
    cmd = [binary, "-m", model_path, "-f", slot_file,
           "-c", str(ctx), "--chunks", str(chunks),
           "-b", str(batch), "-ub", str(ubatch)]
    if fa:
        cmd.append("-fa")
    if ngl is not None:
        cmd += ["-ngl", str(ngl)]
    return cmd + list(extra)


def real_exec(cmd: List[str]) -> Tuple[str, float]:
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
    dt = time.time() - t0
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        out += f"\n[exit code {proc.returncode}]"
    return out, dt


def evaluate(models: List[Tuple[str, str]], ngl: Dict[str, int],
             corpus_dir: Path, binary: str, ctx: int, chunks: int,
             batch: int, ubatch: int, fa: bool, extra: List[str],
             repeat_baseline: bool, pairs: List[Tuple[str, str]],
             out_path: Path, exec_fn: Optional[Callable] = None,
             dry_run: bool = False) -> Dict:
    exec_fn = exec_fn or real_exec
    manifest = json.loads((corpus_dir / MANIFEST_NAME).read_text())
    slot_files = [(name, corpus_dir / m["file"])
                  for name, m in manifest["slots"].items()
                  if m.get("file")]

    def run_model(name: str, path: str) -> Dict[str, Dict]:
        results: Dict[str, Dict] = {}
        for slot, fpath in slot_files:
            cmd = build_cmd(binary, path, str(fpath), ctx, chunks, batch,
                            ubatch, fa, ngl.get(name), extra)
            if dry_run:
                print(" ".join(cmd))
                continue
            out, dt = exec_fn(cmd)
            parsed = parse_output(out)
            parsed["seconds"] = round(dt, 1)
            parsed["cmd"] = cmd
            results[slot] = parsed
            status = (f"ppl={parsed['ppl']:.4f}" if parsed["ppl"]
                      else "PPL PARSE FAIL")
            print(f"  {name:<16} {slot:<14} {status:<12} "
                  f"tokens={parsed['tokens']} ({dt:.0f}s)")
        return results

    if dry_run:
        for name, path in models:
            run_model(name, path)
        return {}

    runs: Dict[str, Dict[str, Dict]] = {}
    order: List[str] = []
    for name, path in models:
        print(f"[run] {name}: {path}")
        runs[name] = run_model(name, path)
        order.append(name)

    baseline = models[0][0]
    noise_pct = None
    if repeat_baseline and len(models) >= 1:
        print(f"[run] {baseline} (repeat, for the noise floor)")
        runs[f"{baseline}#2"] = run_model(f"{baseline}#2", models[0][1])
        p1 = pooled_ppl(runs[baseline])
        p2 = pooled_ppl(runs[f"{baseline}#2"])
        if p1 and p2:
            noise_pct = abs(delta_pct(p1, p2))
            print(f"noise floor (baseline rerun): {noise_pct:.3f}%")
    tau = max(DEFAULT_TAU, 2 * noise_pct) if noise_pct is not None \
        else DEFAULT_TAU

    # --- sanity: parse + token-count parity ------------------------------
    invalid_reason = None
    for name in order:
        for slot, r in runs[name].items():
            if r["ppl"] is None:
                invalid_reason = (f"PPL parse failed: {name}/{slot} "
                                  f"(raw tail in results.json)")
                break
        if invalid_reason:
            break
    if not invalid_reason:
        ref = runs[baseline]
        for name in order[1:]:
            for slot in ref:
                t_ref = ref[slot].get("tokens")
                t_other = runs[name].get(slot, {}).get("tokens")
                if t_ref != t_other:
                    invalid_reason = (
                        f"token-count mismatch {baseline}/{slot}="
                        f"{t_ref} vs {name}/{slot}={t_other}: "
                        "tokenizer/export drift, NOT a quality result")
                    break
            if invalid_reason:
                break
    if not invalid_reason and noise_pct is not None \
            and noise_pct > NOISE_INVALID:
        invalid_reason = (f"rerun noise {noise_pct:.2f}% > "
                          f"{NOISE_INVALID}%: unstable machine state "
                          "(thermal/throttle/other GPU jobs) -- rerun "
                          "both models back-to-back")

    # --- aggregate --------------------------------------------------------
    pooled = {n: pooled_ppl(runs[n]) for n in runs}
    deltas: Dict[str, Dict] = {}
    verdicts: Dict[str, Dict] = {}
    if invalid_reason:
        for name in order[1:]:
            verdicts[name] = {"verdict": "INVALID", "rule": invalid_reason,
                              "hint": ""}
    else:
        base_p = pooled[baseline]
        for name in order[1:]:
            slot_pcts = {s: delta_pct(runs[baseline][s]["ppl"],
                                      runs[name][s]["ppl"])
                         for s in runs[baseline]}
            pooled_pct = delta_pct(base_p, pooled[name])
            v, rule = verdict_for(pooled_pct, slot_pcts, tau)
            deltas[name] = {"pooled_pct": pooled_pct,
                            "slots_pct": slot_pcts}
            verdicts[name] = {"verdict": v, "rule": rule}

    # --- quant-matched pair reads ----------------------------------------
    pair_reads = []
    for a, b in pairs:
        if a in pooled and b in pooled and pooled[a] and pooled[b]:
            d = delta_pct(pooled[a], pooled[b])
            if d > 2.5:
                note = ("damage is the AMPUTATION (router fit), not the "
                        "quant: refit before touching quant dials")
            else:
                note = ("amputation is clean at matched quant; if a "
                        "cheaper-quant candidate fails, the damage is "
                        "QUANT -> switch to the RCO split dial")
            pair_reads.append({"pair": [a, b], "pooled_pct": d,
                               "rule": "pre-registered: > +2.5% => refit",
                               "read": note})

    results = {
        "canary_version": CANARY_VERSION,
        "corpus_dir": str(corpus_dir),
        "flags": {"ctx": ctx, "chunks": chunks, "batch": batch,
                  "ubatch": ubatch, "fa": fa, "extra": extra,
                  "ngl": ngl},
        "baseline": baseline,
        "noise_floor_pct": noise_pct,
        "tau_pct": tau,
        "pooled": pooled,
        "deltas": deltas,
        "verdicts": verdicts,
        "pair_reads": pair_reads,
        "runs": runs,
    }
    out_path.write_text(json.dumps(results, indent=2))

    # --- report ------------------------------------------------------------
    print()
    print("=" * 74)
    print("PPL canary -- raw perplexity per slot")
    print("=" * 74)
    hdr = f"{'model':<18}" + "".join(
        f"{s[:9]:>10}" for s, _ in slot_files) + f"{'pooled':>10}"
    print(hdr)
    for n in runs:
        row = f"{n:<18}"
        for s, _ in slot_files:
            r = runs[n].get(s, {})
            row += (f"{r['ppl']:>10.4f}" if r.get("ppl") else f"{'--':>10}")
        row += (f"{pooled[n]:>10.4f}" if pooled[n] else f"{'--':>10}")
        print(row)
    print()
    print(f"tau (tie bar) = {tau:.2f}%   "
          f"noise floor = {noise_pct if noise_pct is not None else 'n/a'}")
    for name, v in verdicts.items():
        print(f"\nverdict[{name}]: {v['verdict']}  ({v['rule']})")
        for line in TRIAGE.get(v["verdict"], []):
            print(f"  - {line}")
    for pr in pair_reads:
        print(f"\npair read {pr['pair'][0]} vs {pr['pair'][1]}: "
              f"{pr['pooled_pct']:+.2f}%  -> {pr['read']}")
    print(f"\nresults: {out_path}")
    return results


def worst_exit(results: Dict) -> int:
    codes = [EXIT_CODES.get(v["verdict"], 3)
             for v in results.get("verdicts", {}).values()]
    return max(codes) if codes else 0


# ---------------------------------------------------------------------------
# selftest (offline, no llama.cpp, no network)
# ---------------------------------------------------------------------------

_GOOD = """main: build = 42b17f
perplexity : calculating perplexity over 8 chunks, 32768 tokens
[1]4.5023,[2]4.4890,[3]4.5111,[4]4.4987,[5]4.5234,[6]4.5001,[7]4.5122,[8]4.4955,
final estimate:  PPL = 4.5040
perplexity : 4.5040
llama_perf_context_print: eval time = 12345.67 ms
"""

_CHUNKY = """perplexity : calculating perplexity over 8 chunks, 32768 tokens
[1]4.0000,[2]9.0000,[3]4.0000,[4]9.0000,[5]4.0000,[6]9.0000,[7]4.0000,[8]9.0000,
"""

_FINAL_ONLY = """perplexity : calculating perplexity over 8 chunks, 32768 tokens
Final estimation: perplexity = 6.1234
"""

_GARBAGE = """llama_model_load: error loading model
Segmentation fault (core dumped)
"""


def run_selftest() -> int:
    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))
        return cond

    # --- parser ------------------------------------------------------------
    r = parse_output(_GOOD)
    check("parse final-line ppl", abs(r["ppl"] - 4.5040) < 1e-9)
    check("parse tokens", r["tokens"] == 32768)
    r = parse_output(_CHUNKY)
    check("chunk-line geomean fallback",
          abs(r["ppl"] - 6.0) < 1e-9 and r["method"] == "chunk-lines-geomean")
    check("chunky tokens", r["tokens"] == 32768)
    r = parse_output(_FINAL_ONLY)
    check("final-estimation variant", abs(r["ppl"] - 6.1234) < 1e-9)
    r = parse_output(_GARBAGE)
    check("garbage -> None", r["ppl"] is None and r["tokens"] is None)

    # --- pooled math -------------------------------------------------------
    p = pooled_ppl({"a": {"ppl": 4.0, "tokens": 1000},
                    "b": {"ppl": 9.0, "tokens": 1000}})
    check("pooled log-space math", abs(p - 6.0) < 1e-9)
    p2 = pooled_ppl({"a": {"ppl": 4.0, "tokens": 3000},
                     "b": {"ppl": 9.0, "tokens": 1000}})
    check("pooled token weighting", abs(p2 - math.exp(
        (3 * math.log(4) + math.log(9)) / 4)) < 1e-9)

    # --- verdict engine ----------------------------------------------------
    v, _ = verdict_for(1.0, {"reasoning": 1.0, "code_py": 0.5}, 0.5)
    check("ACCEPT case", v == "ACCEPT")
    v, _ = verdict_for(1.0, {"reasoning": 7.0, "code_py": 0.5}, 0.5)
    check("CONDITIONAL on critical slot", v == "CONDITIONAL")
    v, _ = verdict_for(1.0, {"code_py": 7.0}, 0.5)
    check("CONDITIONAL on non-critical slot", v == "CONDITIONAL")
    v, _ = verdict_for(5.0, {"reasoning": 1.0}, 0.5)
    check("INVESTIGATE on pooled", v == "INVESTIGATE")
    v, _ = verdict_for(1.0, {"code_py": 11.0}, 0.5)
    check("INVESTIGATE on slot collapse", v == "INVESTIGATE")
    v, _ = verdict_for(12.0, {"reasoning": 1.0}, 0.5)
    check("KILL on pooled", v == "KILL")
    v, _ = verdict_for(3.2, {"reasoning": 1.0}, 0.5)
    check("tau absorbs near-threshold", v == "ACCEPT")
    v, _ = verdict_for(3.2, {"reasoning": 1.0}, 0.5)
    check("tau math: eff 2.7 <= 3", v == "ACCEPT")

    # --- end-to-end with fake exec_fn ---------------------------------------
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # fake corpus + manifest
        cdir = Path(td) / "corpus"
        cdir.mkdir()
        slots = ["en_wiki", "reasoning", "code_py"]
        manifest = {"canary_version": CANARY_VERSION,
                    "slots": {s: {"file": f"{s}.txt", "sha256": "x",
                                  "critical": s != "en_wiki"}
                              for s in slots}}
        for s in slots:
            (cdir / f"{s}.txt").write_text("synthetic " * 100)
        (cdir / MANIFEST_NAME).write_text(json.dumps(manifest))

        def fake_exec(cmd):
            slot = Path([a for a in cmd if a.endswith(".txt")][0]).stem
            model = Path(cmd[cmd.index("-m") + 1]).stem
            ppl = {"dense": 5.000, "amp": 5.050}.get(model, 5.0)
            if slot == "reasoning" and model == "amp":
                ppl = 5.360          # +7.2% raw, +6.7% eff > 6% guard
            out = (f"perplexity : calculating perplexity over 8 chunks, "
                   f"32768 tokens\nperplexity : {ppl:.4f}\n")
            return out, 1.0

        res = evaluate(
            models=[("dense", f"{td}/dense.gguf"),
                    ("amp", f"{td}/amp.gguf")],
            ngl={}, corpus_dir=cdir, binary="llama-perplexity",
            ctx=4096, chunks=8, batch=2048, ubatch=512, fa=True,
            extra=[], repeat_baseline=False, pairs=[],
            out_path=Path(td) / "results.json", exec_fn=fake_exec)
        check("e2e results written", (Path(td) / "results.json").exists())
        check("e2e verdict computed",
              res["verdicts"]["amp"]["verdict"] in EXIT_CODES)
        # +1.0% pooled, reasoning +6.4% (>6 after tau=0.5) -> CONDITIONAL
        check("e2e CONDITIONAL as designed",
              res["verdicts"]["amp"]["verdict"] == "CONDITIONAL")
        check("e2e pooled sane",
              abs(res["pooled"]["dense"] - 5.0) < 1e-6)

        # token mismatch -> INVALID
        def fake_exec_mismatch(cmd):
            slot = Path([a for a in cmd if a.endswith(".txt")][0]).stem
            model = Path(cmd[cmd.index("-m") + 1]).stem
            n = 32768 if model == "dense" else 32000
            return (f"perplexity : calculating perplexity over 8 chunks, "
                    f"{n} tokens\nperplexity : 5.0000\n"), 1.0

        res2 = evaluate(
            models=[("dense", f"{td}/dense.gguf"),
                    ("amp", f"{td}/amp.gguf")],
            ngl={}, corpus_dir=cdir, binary="llama-perplexity",
            ctx=4096, chunks=8, batch=2048, ubatch=512, fa=True,
            extra=[], repeat_baseline=False, pairs=[],
            out_path=Path(td) / "r2.json", exec_fn=fake_exec_mismatch)
        check("token mismatch -> INVALID",
              res2["verdicts"]["amp"]["verdict"] == "INVALID")
        check("INVALID exit code 3",
              worst_exit(res2) == 3)

        # pair read path
        def fake_exec_pair(cmd):
            model = Path(cmd[cmd.index("-m") + 1]).stem
            ppl = {"dense_q4": 4.500, "amp_q4": 4.700}.get(model, 5.0)
            return (f"perplexity : calculating perplexity over 8 chunks, "
                    f"32768 tokens\nperplexity : {ppl:.4f}\n"), 1.0

        res3 = evaluate(
            models=[("dense_q4", f"{td}/dense_q4.gguf"),
                    ("amp_q4", f"{td}/amp_q4.gguf")],
            ngl={}, corpus_dir=cdir, binary="llama-perplexity",
            ctx=4096, chunks=8, batch=2048, ubatch=512, fa=True,
            extra=[], repeat_baseline=False,
            pairs=[("dense_q4", "amp_q4")],
            out_path=Path(td) / "r3.json", exec_fn=fake_exec_pair)
        check("pair read fires",
              len(res3["pair_reads"]) == 1
              and res3["pair_reads"][0]["pooled_pct"] > 2.5)

    failed = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    print(f"run selftest: {len(checks) - len(failed)}/{len(checks)} passed")
    return 0 if not failed else 1


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="toffee amputate canary run",
        description="Run llama-perplexity over the frozen canary corpus, "
                    "apply the pre-registered verdict rules.")
    p.add_argument("--corpus", default="canary_corpus")
    p.add_argument("--llama-bin", default=None,
                   help="directory containing llama-perplexity")
    p.add_argument("--binary", default=None,
                   help="explicit path to the llama-perplexity binary")
    p.add_argument("--model", action="append", default=[],
                   metavar="NAME=GGUF",
                   help="repeatable; FIRST model is the baseline (E1); "
                        "required unless --selftest")
    p.add_argument("--ngl", action="append", default=[],
                   metavar="NAME=N",
                   help="per-model -ngl (keep identical within a "
                        "quant-matched pair)")
    p.add_argument("--pair", action="append", default=[],
                   metavar="A,B",
                   help="quant-matched pair read (e.g. "
                        "dense_q4km,amp_q4km)")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--chunks", type=int, default=8)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--ubatch", type=int, default=512)
    p.add_argument("--no-fa", action="store_true",
                   help="omit -fa (default: pass -fa to every run)")
    p.add_argument("--extra", default="",
                   help="extra flags appended to every invocation")
    p.add_argument("--repeat-baseline", action="store_true",
                   help="rerun the baseline at the end to measure the "
                        "noise floor")
    p.add_argument("--out", default=None,
                   help="results json (default <corpus>/results.json)")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the corpus hash check (NOT recommended)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands without running")
    p.add_argument("--selftest", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()

    if not args.selftest and not args.model:
        raise SystemExit("--model NAME=GGUF is required (at least the "
                         "baseline)")

    corpus_dir = Path(args.corpus)
    if not args.no_verify:
        if verify_corpus(corpus_dir) != 0:
            print("corpus verification failed: rebuild or fix before "
                  "comparing models (a changed corpus invalidates every "
                  "previous result)", file=sys.stderr)
            return 3

    models = []
    for item in args.model:
        if "=" not in item:
            raise SystemExit(f"--model expects NAME=GGUF, got {item!r}")
        n, g = item.split("=", 1)
        models.append((n.strip(), g.strip()))
    ngl = {}
    for item in args.ngl:
        if "=" not in item:
            raise SystemExit(f"--ngl expects NAME=N, got {item!r}")
        n, v = item.split("=", 1)
        ngl[n.strip()] = int(v)
    pairs = [tuple(x.split(",", 1)) for x in args.pair if "," in x]
    binary = resolve_binary(args.llama_bin, args.binary)
    out_path = Path(args.out) if args.out else corpus_dir / "results.json"

    results = evaluate(
        models=models, ngl=ngl, corpus_dir=corpus_dir, binary=binary,
        ctx=args.ctx, chunks=args.chunks, batch=args.batch,
        ubatch=args.ubatch, fa=not args.no_fa,
        extra=args.extra.split() if args.extra else [],
        repeat_baseline=args.repeat_baseline, pairs=pairs,
        out_path=out_path, dry_run=args.dry_run)
    if args.dry_run:
        return 0
    return worst_exit(results)


if __name__ == "__main__":
    raise SystemExit(main())
