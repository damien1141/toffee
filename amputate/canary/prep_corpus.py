"""
@file toffee/amputate/canary/prep_corpus.py
   Build (and freeze) the PPL canary corpus + manifest.

Usage::

    python -m toffee.amputate.canary prep \\
        --tokenizer /path/to/VeriLoop-E2/tokenizer.json \\
        --domain-pack ~/vgr-logs

    python -m toffee.amputate.canary prep --verify     # re-hash vs manifest
    python -m toffee.amputate.canary prep --selftest   # offline, no network

What "frozen" means here:

  * every slot file's sha256 goes into ``canary_manifest.json``;
  * the runner re-verifies the hashes before every A/B, so the dense
    and the amputated run are guaranteed to see identical bytes;
  * re-prepping an existing corpus requires ``--force`` (protects a
    corpus that results were already computed against);
  * remote source revisions (resolved HF shas) are recorded best-effort
    so drift is detectable later.

Trim policy: each slot file is grown block-by-block until it holds at
least ``--tokens-per-slot`` tokens (exact count when a tokenizer is
supplied, chars-per-token estimate otherwise, with +8% margin). The
runner evaluates exactly ``--chunks x ctx`` = 32,768 tokens of it, so
the evaluated length is identical across models regardless of the trim
precision. A slot that cannot reach 75% of budget is marked
``short_slot`` in the manifest (verdict math uses actual token counts,
so this degrades gracefully).

Disjointness rule (SPEC.md): the canary corpus must NOT be part of the
harvest calibration data. The manifest records domain-pack file hashes
so the check is mechanical: diff them against whatever you point
``toffee amputate harvest --dataset`` at.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from toffee.amputate.canary.slots import (
    SLOT_TABLE,
    synthetic_blocks,
)

logger = logging.getLogger(__name__)

CANARY_VERSION = "1.0"
MANIFEST_NAME = "canary_manifest.json"
DEFAULT_TOKENS_PER_SLOT = 34_000        # margin above the 32,768 evaluated
EVAL_TOKENS_PER_SLOT = 32_768           # 8 chunks x ctx 4096 (runner-side)
CHAR_MARGIN = 1.08                      # estimate-mode safety margin


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tokenizer(path: Optional[str]):
    """Load a HF `tokenizers` Tokenizer, or None (estimate mode)."""
    if not path:
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError:
        logger.warning(
            "--tokenizer given but the `tokenizers` package is missing; "
            "falling back to chars-per-token estimates "
            "(pip install tokenizers for exact counts)")
        return None
    tok = Tokenizer.from_file(str(path))
    logger.info("tokenizer loaded: %s (vocab %s)", path,
                tok.get_vocab_size())
    return tok


def count_tokens(tok, text: str) -> int:
    if tok is None:
        return -1
    return len(tok.encode(text, add_special_tokens=False).ids)


def trim_to_budget(blocks: List[str], tok, spec, tokens_per_slot: int
                   ) -> tuple[str, Dict]:
    """Concatenate blocks until the token (or char) budget is reached.

    Returns (text, info) where info records how the budget decision was
    made and the final (exact or estimated) token count.
    """
    if tok is not None:
        target = tokens_per_slot
        parts: List[str] = []
        n = 0
        for b in blocks:
            parts.append(b)
            n += count_tokens(tok, b)
            if n >= target:
                break
        text = "\n\n".join(parts)
        exact = count_tokens(tok, text)
        return text, {
            "trim_mode": "exact",
            "blocks_used": len(parts),
            "tokens": exact,
        }
    # estimate mode
    cpr = spec.chars_per_token
    target_chars = int(tokens_per_slot * cpr * CHAR_MARGIN)
    parts = []
    n_chars = 0
    for b in blocks:
        parts.append(b)
        n_chars += len(b)
        if n_chars >= target_chars:
            break
    text = "\n\n".join(parts)
    return text, {
        "trim_mode": "estimate",
        "blocks_used": len(parts),
        "tokens": int(len(text) / cpr),
        "chars": len(text),
    }


def resolved_sha(repo: str) -> Optional[str]:
    """Best-effort resolved commit sha of an HF dataset repo."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(repo)
        return getattr(info, "sha", None)
    except Exception:  # noqa: BLE001 - purely informational
        return None


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build_corpus(out_dir: Path,
                 tokenizer_path: Optional[str] = None,
                 tokens_per_slot: int = DEFAULT_TOKENS_PER_SLOT,
                 only_slots: Optional[List[str]] = None,
                 domain_pack: Optional[str] = None,
                 local_overrides: Optional[Dict[str, str]] = None,
                 force: bool = False,
                 allow_drift: bool = False,
                 record_shas: bool = True) -> Dict:
    """Build the corpus and write ``canary_manifest.json``. Returns it."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / MANIFEST_NAME
    if manifest_path.exists() and not force:
        raise SystemExit(
            f"{manifest_path} already exists (a frozen corpus). "
            "Pass --force to rebuild, or `prep --verify` to check it. "
            "Rebuilding invalidates every result computed against it.")
    if manifest_path.exists() and force and not allow_drift:
        try:
            old = json.loads(manifest_path.read_text())
            repos = sorted({s.split("|")[0]
                            for m in old.get("slots", {}).values()
                            for s in (m.get("sources") or []) if "|" in s})
            for r in repos:
                new_sha = resolved_sha(r)
                old_sha = (old.get("source_shas") or {}).get(r)
                if new_sha and old_sha and new_sha != old_sha:
                    logger.warning(
                        "source %s moved since the corpus was frozen "
                        "(%s -> %s); pass --allow-drift to accept",
                        r, old_sha[:8], new_sha[:8])
        except Exception:  # noqa: BLE001 - informational only
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(tokenizer_path)
    local_overrides = local_overrides or {}

    slots = [s for s in SLOT_TABLE
             if not only_slots or s.name in only_slots]
    unknown = (set(only_slots or []) - {s.name for s in SLOT_TABLE})
    if unknown:
        raise SystemExit(f"unknown slots: {sorted(unknown)}; "
                         f"valid: {[s.name for s in SLOT_TABLE]}")

    manifest: Dict = {
        "canary_version": CANARY_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokens_per_slot_trim": tokens_per_slot,
        "eval_tokens_per_slot": EVAL_TOKENS_PER_SLOT,
        "eval_ctx": 4096,
        "eval_chunks_per_slot": EVAL_TOKENS_PER_SLOT // 4096,
        "tokenizer": {"path": str(tokenizer_path) if tokenizer_path else None,
                      "exact_trim": tok is not None},
        "slots": {},
        "source_shas": {},
        "disjointness_rule": (
            "This corpus must NOT be part of the amputate-harvest "
            "calibration data. Domain-pack file hashes are recorded "
            "below; diff them against the harvest dataset."),
    }

    print(f"building canary corpus -> {out_dir}")
    print(f"{'slot':<14} {'blocks':>7} {'tokens':>9}  mode     status")
    print("-" * 60)

    for spec in slots:
        # 1) local override wins
        if spec.name in local_overrides:
            p = Path(local_overrides[spec.name])
            text = p.read_text(encoding="utf-8", errors="replace")
            info = {"trim_mode": "local_override",
                    "blocks_used": 1,
                    "tokens": count_tokens(tok, text) if tok else int(
                        len(text) / spec.chars_per_token)}
            sources = [f"local-override:{p}"]
            status = "ok"
        else:
            # 2) regular loader (remote or domain pack)
            import toffee.amputate.canary.slots as slots_mod
            loader = getattr(slots_mod, spec.loader)
            ctx = {"domain_pack": domain_pack, "spec": spec}
            try:
                blocks = loader(ctx)
            except Exception as e:  # noqa: BLE001 - degrade, never die
                print(f"{spec.name:<14} {'-':>7} {'-':>9}  "
                      f"{'-':<8} FAILED: {e}")
                manifest["slots"][spec.name] = {
                    "file": None, "skipped": True, "error": str(e),
                    "why": spec.why, "critical": spec.critical,
                    "sources": spec.sources}
                continue
            if not blocks:
                manifest["slots"][spec.name] = {
                    "file": None, "skipped": True,
                    "why": spec.why, "critical": spec.critical,
                    "sources": spec.sources}
                print(f"{spec.name:<14} {'0':>7} {'0':>9}  "
                      f"{'-':<8} SKIPPED (no data)")
                continue
            text, info = trim_to_budget(blocks, tok, spec, tokens_per_slot)
            sources = list(spec.sources)

        # 3) write + hash
        fpath = out_dir / f"{spec.name}.txt"
        fpath.write_text(text, encoding="utf-8")
        entry = {
            "file": fpath.name,
            "sha256": sha256_file(fpath),
            "bytes": fpath.stat().st_size,
            "why": spec.why,
            "critical": spec.critical,
            "sources": sources,
            **info,
        }
        short = (info.get("tokens", 0) <
                 int(EVAL_TOKENS_PER_SLOT * 0.75))
        if short:
            entry["short_slot"] = True
        manifest["slots"][spec.name] = entry
        status = "SHORT" if short else "ok"
        print(f"{spec.name:<14} {info.get('blocks_used', 0):>7} "
              f"{info.get('tokens', 0):>9}  {info.get('trim_mode', '-'):<8} "
              f"{status}")

        # record resolved shas best-effort (skipped in selftest: no
        # network assumptions there)
        if record_shas and "local-override" not in sources[0]:
            for s in sources:
                repo = s.split("|")[0].strip()
                if repo and repo not in manifest["source_shas"]:
                    sha = resolved_sha(repo)
                    if sha:
                        manifest["source_shas"][repo] = sha

    # domain-pack provenance (for the disjointness check)
    if domain_pack:
        dp = Path(domain_pack)
        if dp.is_dir():
            manifest["domain_pack_files"] = {
                str(p.relative_to(dp)): sha256_file(p)
                for p in sorted(dp.rglob("*"))
                if p.is_file() and p.suffix.lower()
                in (".txt", ".md", ".log", ".text", ".jsonl")}

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    n_ok = sum(1 for m in manifest["slots"].values() if m.get("file"))
    n_short = sum(1 for m in manifest["slots"].values() if m.get("short_slot"))
    total = sum(m.get("tokens", 0) for m in manifest["slots"].values()
                if m.get("file"))
    print("-" * 60)
    print(f"slots written: {n_ok}/{len(slots)}"
          + (f" ({n_short} short)" if n_short else ""))
    print(f"total corpus tokens (trim target): ~{total:,}")
    print(f"manifest: {manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def verify_corpus(out_dir: Path) -> int:
    out_dir = Path(out_dir)
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())
    ok = True
    for name, m in manifest["slots"].items():
        if not m.get("file"):
            print(f"{name:<14} SKIPPED (recorded)")
            continue
        p = out_dir / m["file"]
        if not p.exists():
            print(f"{name:<14} MISSING {p}")
            ok = False
            continue
        digest = sha256_file(p)
        if digest != m["sha256"]:
            print(f"{name:<14} HASH MISMATCH {p}")
            ok = False
        else:
            print(f"{name:<14} ok ({m.get('tokens', '?')} tokens)")
    print("corpus " + ("VERIFIED (bytes frozen)" if ok else "CHANGED"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# selftest (offline, no network, stdlib only)
# ---------------------------------------------------------------------------

def run_selftest() -> int:
    import tempfile

    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "corpus"
        # build with synthetic loaders: monkeypatch the remote loaders
        import toffee.amputate.canary.slots as slots_mod
        real = {s.loader: getattr(slots_mod, s.loader, None)
                for s in SLOT_TABLE}
        for s in SLOT_TABLE:
            if s.loader == "load_domain_pack":
                continue
            def fake(ctx, s=s):
                return synthetic_blocks(s)
            setattr(slots_mod, s.loader, fake)

        # tiny budget so it is fast; char-estimate mode (no tokenizer)
        m = build_corpus(out, tokens_per_slot=4000, record_shas=False)
        check("manifest written", (out / MANIFEST_NAME).exists())
        check("8 slots present", len(m["slots"]) == 8)
        check("domain_pack skipped w/o pack",
              m["slots"]["domain_pack"].get("skipped") is True)
        for name, e in m["slots"].items():
            if e.get("file"):
                p = out / e["file"]
                check(f"{name} file exists", p.exists())
                check(f"{name} hash matches",
                      sha256_file(p) == e["sha256"])
        # estimate mode must land near budget (within 2x)
        en = m["slots"]["en_wiki"]
        check("estimate trim near budget",
              2000 <= en["tokens"] <= 8000)

        # frozen protection
        try:
            build_corpus(out, tokens_per_slot=4000)
            check("refuses to overwrite frozen corpus", False)
        except SystemExit:
            check("refuses to overwrite frozen corpus", True)

        # force rebuild + verify
        m2 = build_corpus(out, tokens_per_slot=4000, force=True,
                          record_shas=False)
        check("force rebuild ok", len(m2["slots"]) == 8)
        check("verify passes", verify_corpus(out) == 0)

        # tamper detection
        victim = out / "en_wiki.txt"
        victim.write_text(victim.read_text() + " tampered")
        check("verify catches tampering", verify_corpus(out) == 1)

        # short-slot path: rebuild domain_pack with a tiny pack
        packdir = Path(td) / "pack"
        packdir.mkdir()
        (packdir / "log.txt").write_text(
            "[repair-loop 0] vision ok torque 1.10\n" * 40)
        m3 = build_corpus(out, tokens_per_slot=4000, force=True,
                          domain_pack=str(packdir), record_shas=False)
        dp = m3["slots"]["domain_pack"]
        check("domain_pack built from dir", dp.get("file") is not None)
        check("domain_pack hashes recorded",
              m3.get("domain_pack_files", {}).get("log.txt") is not None)
        check("short_slot flagged on tiny pack", dp.get("short_slot") is True)

        # restore real loaders
        for k, v in real.items():
            if v is not None:
                setattr(slots_mod, k, v)

    failed = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    print(f"prep selftest: {len(checks) - len(failed)}/{len(checks)} passed")
    return 0 if not failed else 1


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="toffee amputate canary prep",
        description="Build + freeze the PPL canary corpus (SPEC.md).")
    p.add_argument("--out", default="canary_corpus",
                   help="output directory (default ./canary_corpus)")
    p.add_argument("--tokenizer", default=None,
                   help="path to VeriLoop-E2 tokenizer.json for exact "
                        "token trimming (pip install tokenizers)")
    p.add_argument("--tokens-per-slot", type=int,
                   default=DEFAULT_TOKENS_PER_SLOT)
    p.add_argument("--slots", default=None,
                   help="comma list; default all")
    p.add_argument("--domain-pack", default=None,
                   help="directory with YOUR texts (repair-loop logs, "
                        "manuals); fills the domain_pack slot")
    p.add_argument("--local", action="append", default=[],
                   metavar="SLOT=FILE",
                   help="override a slot with a local text file")
    p.add_argument("--force", action="store_true",
                   help="rebuild over an existing frozen corpus")
    p.add_argument("--allow-drift", action="store_true",
                   help="allow rebuild when remote sources moved")
    p.add_argument("--verify", action="store_true",
                   help="re-hash an existing corpus vs its manifest")
    p.add_argument("--selftest", action="store_true",
                   help="offline machinery test (no network)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    out_dir = Path(args.out)
    if args.verify:
        return verify_corpus(out_dir)
    only = [s.strip() for s in args.slots.split(",")] if args.slots else None
    overrides = {}
    for item in args.local:
        if "=" not in item:
            raise SystemExit(f"--local expects SLOT=FILE, got {item!r}")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()
    build_corpus(out_dir, tokenizer_path=args.tokenizer,
                 tokens_per_slot=args.tokens_per_slot, only_slots=only,
                 domain_pack=args.domain_pack, local_overrides=overrides,
                 force=args.force, allow_drift=args.allow_drift)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
