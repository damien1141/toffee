"""
@file toffee/amputate/canary/slots.py
   The PPL canary corpus: slot table + deterministic text loaders.

Design (SPEC.md is the normative document, this is the implementation):

  * 8 slots x 32,768 evaluated tokens (8 chunks x ctx 4096) = 262,144
    tokens total -- small enough for a ~5 min dense run on the laptop,
    big enough that a 3% PPL delta is far outside measurement noise.
  * Every slot targets a different failure mode of MoEfication:
    general prose (router should be trivial), code / tables (exact
    grammar, router collapse shows first), math + long CoT (the model's
    native mode, the slot that actually matters), tool-call JSON, and a
    user-supplied domain pack.
  * Loaders are defensive: each slot has a fallback chain of HF
    sources; a moved repo degrades that slot, never the whole prep.
  * Everything is plain raw text (llama-perplexity completion mode, no
    chat template) -- this is a *relative* canary, absolute PPL is not
    comparable to any leaderboard.

Stdlib only at import time; `datasets` is imported lazily by the
loaders that need it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SlotSpec:
    """One canary slot.

    Attributes:
        name: slot id == output filename stem.
        why: one-line rationale (echoed in SPEC.md, manifest, verdicts).
        critical: critical slots get the tighter +6% guard in the
            pre-registered decision rules (reasoning / math / agent_tools).
        chars_per_token: fallback trim estimate when no tokenizer is
            supplied to `prep` (Qwen-class 248K vocab numbers).
        budget_blocks: rough block count hint for the synthetic selftest.
        sources: human-readable source chain, first that reaches the
            token budget wins; recorded in the manifest.
        loader: name of the loader in this module (resolved lazily so
            importing this module never pulls `datasets`).
    """

    name: str
    why: str
    critical: bool
    chars_per_token: float
    budget_blocks: int
    sources: List[str]
    loader: str


SLOT_TABLE: List[SlotSpec] = [
    SlotSpec(
        name="en_wiki",
        why="classic PPL anchor; smooth general prose the router must not break",
        critical=False,
        chars_per_token=3.4,
        budget_blocks=260,
        sources=["Salesforce/wikitext | wikitext-2-raw-v1 | test"],
        loader="load_en_wiki",
    ),
    SlotSpec(
        name="zh_general",
        why="ZH prose: the 248K vocab and the FFN treat ZH differently than EN",
        critical=False,
        chars_per_token=1.5,
        budget_blocks=2200,
        sources=[
            "wmt/wmt19 | zh-en | validation+test (zh side)",
            "wmt/wmt20 | zh-en | validation+test (zh side)",
        ],
        loader="load_zh_general",
    ),
    SlotSpec(
        name="tables_nums",
        why="digits and rigid structure; the most sensitive slot to low-bit "
            "and router damage",
        critical=False,
        chars_per_token=3.0,
        budget_blocks=240,
        sources=["google-research-datasets/wikitablequestions | test"],
        loader="load_tables_nums",
    ),
    SlotSpec(
        name="code_py",
        why="exact grammar; MoE router collapse shows up in code first",
        critical=False,
        chars_per_token=3.2,
        budget_blocks=320,
        sources=[
            "HuggingFaceH4/CodeAlpaca_20K | train (response field)",
            "sahil2801/CodeAlpaca | train (output field)",
        ],
        loader="load_code_py",
    ),
    SlotSpec(
        name="math",
        why="symbol-dense; a core domain for a hybrid reasoning model",
        critical=True,
        chars_per_token=2.8,
        budget_blocks=240,
        sources=[
            "EleutherAI/hendrycks_math | all 7 configs | test",
            "openai/gsm8k | main | test",
        ],
        loader="load_math",
    ),
    SlotSpec(
        name="reasoning",
        why="long-chain CoT, the model's native mode; the slot that "
            "actually decides the verdict",
        critical=True,
        chars_per_token=3.3,
        budget_blocks=14,
        sources=["simplescaling/s1K-1.1 | train (solution field)"],
        loader="load_reasoning",
    ),
    SlotSpec(
        name="agent_tools",
        why="tool-call grammar + JSON, VGR-adjacent; structured-output drift",
        critical=True,
        chars_per_token=3.2,
        budget_blocks=700,
        sources=[
            "NousResearch/hermes-function-calling-v1 | test_gpt4o (+train pad)",
        ],
        loader="load_agent_tools",
    ),
    SlotSpec(
        name="domain_pack",
        why="YOUR texts: repair-loop logs, hardware manuals, lab notes; "
            "the only slot that measures the domain you actually run",
        critical=True,
        chars_per_token=3.2,
        budget_blocks=140,
        sources=["local: --domain-pack DIR (optional, skipped if absent)"],
        loader="load_domain_pack",
    ),
]

SLOTS_BY_NAME: Dict[str, SlotSpec] = {s.name: s for s in SLOT_TABLE}

CRITICAL_SLOTS = frozenset(s.name for s in SLOT_TABLE if s.critical)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _datasets():
    """Import `datasets` lazily; raise a friendly error if missing."""
    try:
        import datasets  # noqa: F401  (returned for callers)
        return datasets
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "the `datasets` package is required for remote canary slots "
            "(pip install datasets); or pass --local slot=file.txt overrides"
        ) from e


def _try_sources(calls):
    """Run loader thunks in order, concatenating blocks.

    Each thunk returns (blocks, source_desc). Thunks that raise are
    logged and skipped -- a moved HF repo degrades one slot, never the
    whole prep. Returns (blocks, used_sources, errors).
    """
    import logging
    log = logging.getLogger(__name__)
    blocks: List[str] = []
    used: List[str] = []
    errors: List[str] = []
    for fn in calls:
        try:
            got, desc = fn()
        except Exception as e:  # noqa: BLE001 - deliberate wide net
            hint = getattr(fn, "source_hint", fn.__name__)
            errors.append(f"{hint}: {type(e).__name__}: {e}")
            log.warning("canary source failed: %s", errors[-1])
            continue
        if got:
            blocks.extend(got)
        if desc:
            used.append(desc)
    return blocks, used, errors


def _hint(desc: str):
    """Decorate a thunk with a human-readable source hint."""
    def wrap(fn):
        fn.source_hint = desc
        return fn
    return wrap


def _clean(blocks: List[str]) -> List[str]:
    return [b for b in blocks if b and b.strip()]


# ---------------------------------------------------------------------------
# slot loaders (remote)
# ---------------------------------------------------------------------------

def load_en_wiki(ctx: Dict) -> List[str]:
    ds = _datasets()

    def src():
        d = ds.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                            split="test")
        return _clean([ex["text"] for ex in d]), \
            "Salesforce/wikitext|wikitext-2-raw-v1|test"

    blocks, _, errors = _try_sources(
        [_hint("Salesforce/wikitext|wikitext-2-raw-v1|test")(src)])
    if not blocks:
        raise RuntimeError("all sources failed: " + "; ".join(errors))
    return blocks


def load_zh_general(ctx: Dict) -> List[str]:
    ds = _datasets()
    calls = []
    for repo, split in (("wmt/wmt19", "validation"), ("wmt/wmt19", "test"),
                        ("wmt/wmt20", "validation"), ("wmt/wmt20", "test")):
        def src(repo=repo, split=split):
            d = ds.load_dataset(repo, "zh-en", split=split)
            blocks = [ex["translation"]["zh"] for ex in d
                      if (ex.get("translation") or {}).get("zh")]
            return _clean(blocks), f"{repo}|zh-en|{split}"

        calls.append(_hint(f"{repo}|zh-en|{split}")(src))
    blocks, _, errors = _try_sources(calls)
    if not blocks:
        raise RuntimeError("all sources failed: " + "; ".join(errors))
    return blocks


def load_tables_nums(ctx: Dict) -> List[str]:
    ds = _datasets()

    def src():
        d = ds.load_dataset("google-research-datasets/wikitablequestions",
                            split="test")
        blocks = []
        for ex in d:
            table = (ex.get("table") or "").strip()
            q = (ex.get("question") or "").strip()
            answers = ex.get("answers") or []
            a = ", ".join(str(x) for x in answers)
            if not (table or q):
                continue
            blocks.append(f"{table}\nQuestion: {q}\nAnswer: {a}")
        return blocks, "google-research-datasets/wikitablequestions|test"

    blocks, _, errors = _try_sources(
        [_hint("google-research-datasets/wikitablequestions|test")(src)])
    if not blocks:
        raise RuntimeError("all sources failed: " + "; ".join(errors))
    return blocks


def load_code_py(ctx: Dict) -> List[str]:
    ds = _datasets()

    def src_h4():
        d = ds.load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")
        return _clean([ex.get("response") or "" for ex in d]), \
            "HuggingFaceH4/CodeAlpaca_20K|train"

    def src_orig():
        d = ds.load_dataset("sahil2801/CodeAlpaca", split="train")
        return _clean([ex.get("output") or "" for ex in d]), \
            "sahil2801/CodeAlpaca|train"

    blocks, _, errors = _try_sources([
        _hint("HuggingFaceH4/CodeAlpaca_20K|train")(src_h4),
        _hint("sahil2801/CodeAlpaca|train")(src_orig),
    ])
    if not blocks:
        raise RuntimeError("all sources failed: " + "; ".join(errors))
    return blocks


_MATH_CONFIGS = ["algebra", "counting_and_probability", "geometry",
                 "intermediate_algebra", "number_theory", "prealgebra",
                 "precalculus"]


def load_math(ctx: Dict) -> List[str]:
    ds = _datasets()
    calls = []
    for cfg in _MATH_CONFIGS:
        def src(cfg=cfg):
            d = ds.load_dataset("EleutherAI/hendrycks_math", cfg, split="test")
            blocks = []
            for ex in d:
                p = (ex.get("problem") or "").strip()
                s = (ex.get("solution") or "").strip()
                if p or s:
                    blocks.append(f"Problem: {p}\nSolution: {s}")
            return blocks, f"EleutherAI/hendrycks_math|{cfg}|test"

        calls.append(_hint(f"EleutherAI/hendrycks_math|{cfg}|test")(src))

    def src_gsm():
        d = ds.load_dataset("openai/gsm8k", "main", split="test")
        blocks = [f"Question: {ex['question']}\nAnswer: {ex['answer']}"
                  for ex in d]
        return blocks, "openai/gsm8k|main|test"

    calls.append(_hint("openai/gsm8k|main|test")(src_gsm))
    blocks, _, errors = _try_sources(calls)
    if not blocks:
        raise RuntimeError("all sources failed: " + "; ".join(errors))
    return blocks


def load_reasoning(ctx: Dict) -> List[str]:
    ds = _datasets()

    def make(repo):
        def src():
            d = ds.load_dataset(repo, split="train")
            blocks = []
            for ex in d:
                p = (ex.get("problem") or "").strip()
                s = (ex.get("solution") or "").strip()
                if s:
                    blocks.append(f"{p}\n\n{s}" if p else s)
            return blocks, f"{repo}|train"
        return src

    blocks, _, errors = _try_sources([
        _hint("simplescaling/s1K-1.1|train")(make("simplescaling/s1K-1.1")),
        _hint("simplescaling/s1K|train")(make("simplescaling/s1K")),
    ])
    if not blocks:
        raise RuntimeError("all sources failed: " + "; ".join(errors))
    return blocks


def load_agent_tools(ctx: Dict) -> List[str]:
    ds = _datasets()

    def textualize(ex) -> str:
        conv = ex.get("conversations")
        if conv:
            parts = []
            for turn in conv:
                v = turn.get("value") if isinstance(turn, dict) else None
                if v:
                    parts.append(str(v))
            if parts:
                return "\n".join(parts)
        q = ex.get("question") or ex.get("query") or ""
        a = ex.get("answer") or ""
        if q or a:
            return f"{q}\n{a}"
        return ""

    calls = []
    for split in ("test_gpt4o", "test_mini", "train"):
        def src(split=split):
            d = ds.load_dataset("NousResearch/hermes-function-calling-v1",
                                split=split)
            return _clean([textualize(ex) for ex in d]), \
                f"NousResearch/hermes-function-calling-v1|{split}"

        calls.append(
            _hint(f"NousResearch/hermes-function-calling-v1|{split}")(src))
    blocks, _, errors = _try_sources(calls)
    if not blocks:
        raise RuntimeError("all sources failed: " + "; ".join(errors))
    return blocks


# ---------------------------------------------------------------------------
# slot loader (local)
# ---------------------------------------------------------------------------

_DOMAIN_EXTS = (".txt", ".md", ".log", ".text", ".jsonl")
_LOCAL_BLOCK_CHARS = 8_192


def load_domain_pack(ctx: Dict) -> List[str]:
    """Read the user's own texts from ctx['domain_pack'] (a directory).

    Deterministic: files in sorted order, large files chunked to
    ~8 KB blocks. Returns [] when the directory is absent or empty --
    the slot is then SKIPPED (recorded in the manifest), never fatal.
    """
    from pathlib import Path
    pack = ctx.get("domain_pack")
    if not pack:
        return []
    root = Path(pack)
    if not root.is_dir():
        return []
    blocks: List[str] = []
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in _DOMAIN_EXTS)
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = text.strip()
        if not text:
            continue
        for i in range(0, len(text), _LOCAL_BLOCK_CHARS):
            blocks.append(text[i:i + _LOCAL_BLOCK_CHARS])
    return blocks


# ---------------------------------------------------------------------------
# synthetic generators (selftest, no network)
# ---------------------------------------------------------------------------

def synth_block(slot: SlotSpec, i: int) -> str:
    """Deterministic synthetic stand-in for slot `slot` (selftest only)."""
    n = slot.name
    if n == "en_wiki":
        return (f" The VeriLoop corpus section {i} discusses alignment of "
                f"sensor frames and calibration residue at station {i % 37}.")
    if n == "zh_general":
        return f"第{i}段：维修机器人在装配线{i % 9}完成了视觉引导的抓取。"
    if n == "tables_nums":
        return (f"| run | torque | error |\n|---|---|---|\n"
                f"| {i} | {i * 1.37:.2f} | {i * 0.021:.4f} |\n"
                f"Question: what is run {i}'s torque?\nAnswer: {i * 1.37:.2f}")
    if n == "code_py":
        return (f"def repair_step_{i}(state):\n"
                f"    torque = {i} * 0.25\n"
                f"    if torque > 3.0:\n        return state.fail()\n"
                f"    return state.advance(torque)\n")
    if n == "math":
        return (f"Problem: compute {2 * i + 3} mod {i % 7 + 2}.\n"
                f"Solution: by division, {2 * i + 3} = "
                f"{(i % 7 + 2) * ((2 * i + 3) // (i % 7 + 2))} + "
                f"{(2 * i + 3) % (i % 7 + 2)}, so the remainder is "
                f"{(2 * i + 3) % (i % 7 + 2)}.")
    if n == "reasoning":
        return (f"Problem {i}: the gantry drifts {i % 5} um per cycle.\n\n"
                + "Let me think carefully. " * 40
                + f"\n\nTherefore the drift after {i} cycles is "
                f"{(i % 5) * i} um.")
    if n == "agent_tools":
        return (f'{{"tool": "move_gripper", "args": {{"mm": {i}}}}}\n'
                f'{{"ok": true, "position_mm": {i}}}')
    # domain_pack
    return (f"[repair-loop {i}] vision ok, torque {i * 0.11:.2f} Nm, "
            f"grip={i % 2 == 0}, retry={i % 3}")


def synthetic_blocks(slot: SlotSpec, n: Optional[int] = None) -> List[str]:
    """N deterministic synthetic blocks for `slot` (selftest only)."""
    n = n if n is not None else slot.budget_blocks
    return [synth_block(slot, i) for i in range(n)]
