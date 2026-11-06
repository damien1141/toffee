"""
TOFFEE XSpark data preparation.

Fetches calibration text via the HF datasets-server API (whole-Parquet download
or /rows pagination — no `datasets` library and no jsonl involved) into local
Parquet shards that harvest consumes.

Source selection — 8 datasets, 2 per domain, probe-verified against
datasets-server (config + split + field names all confirmed live):

  chat (weight 0.30 — the dominant deployment domain)
  * allenai/tulu-3-sft-mixture            — 939k multi-turn chat/instruction
    SFT mixture (WildChat, personas, math/code-in-chat, ...)
  * HuggingFaceTB/smoltalk                — 1.1M-sample chat mixture
    (Magpie-Ultra, systemchats, rewrite/summarize, ...), config `all`

  math (0.25)
  * nvidia/OpenMathInstruct-2             — 14M GSM8K+MATH problems augmented
    with Llama-405B-generated solutions
  * openai/gsm8k                          — 7.5k original human-written
    grade-school problems + step-by-step solutions, config `main`

  code (0.25)
  * codeparrot/github-code-clean          — raw multi-language GitHub code,
    deduplicated (config `all-all`: every language, every license)
  * ise-uiuc/Magicoder-Evol-Instruct-110K — 111k evol-instruct code Q&A
    pairs (code in chat style)

  science (0.20)
  * nvidia/OpenScienceReasoning-2         — 308k physics/chemistry/biology
    problems with DeepSeek-R1-style reasoning traces (`input` embeds the
    MCQ options, `output` carries the <think> chain + boxed answer)
  * allenai/sciq                          — 11.7k science MCQs with support
    passages, rendered into natural Q&A text (see _render_sciq)

Licenses (per the HF dataset cards): tulu-3 ODC-By, smoltalk Apache-2.0,
OpenMathInstruct-2 CC-BY-4.0, gsm8k MIT, github-code-clean Apache-2.0,
Magicoder Apache-2.0, OpenScienceReasoning-2 CC-BY-4.0, sciq CC-BY-NC-3.0
(research/calibration use).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

logger = structlog.get_logger()


HF_DATASETS_SERVER = "https://datasets-server.huggingface.co"
PARQUET_ROOT = Path("xspark_data/parquet")

RenderFn = Callable[[Dict[str, Any]], Optional[str]]
FieldSpec = Union[str, Tuple[str, ...]]


def _render_sciq(row: Dict[str, Any]) -> Optional[str]:
    """Render a SciQ row (MCQ + support passage) as natural Q&A text.

    The correct answer's position is shuffled deterministically per question
    (md5 of the question text) so the calibration text doesn't teach a
    positional bias like "the answer is always D".
    """
    question = str(row.get("question") or "").strip()
    support = str(row.get("support") or "").strip()
    opts = [row.get("distractor1"), row.get("distractor2"),
            row.get("distractor3"), row.get("correct_answer")]
    if not question or not all(o and str(o).strip() for o in opts):
        return None
    opts = [str(o).strip() for o in opts]
    correct = opts[3]
    pos = int(hashlib.md5(question.encode("utf-8")).hexdigest()[:8], 16) % len(opts)
    opts[pos], opts[3] = opts[3], opts[pos]
    lines = []
    if support:
        lines.append(support)
    lines.append(f"Question: {question}")
    for letter, opt in zip("ABCD", opts):
        lines.append(f"{letter}. {opt}")
    lines.append(f"Answer: {'ABCD'[pos]}. {correct}")
    return "\n".join(lines)


@dataclass(frozen=True)
class SourceSpec:
    """One calibration source: where to fetch it and how to store it.

    fields   — column(s) to pull; a tuple is joined with newlines (or fed to
               `render` when set) and stored under "text"
    render   — optional {field: value} -> text mapping for structured rows
               (e.g. MCQ datasets); returning None skips the row
    weight   — share of harvest samples drawn from this source
    """

    hf_id: str
    fields: FieldSpec
    local_name: str
    config: Optional[str] = None
    split: Optional[str] = None
    weight: float = 0.0
    render: Optional[RenderFn] = None

    @property
    def stored_field(self) -> str:
        if isinstance(self.fields, str) and self.render is None:
            return self.fields
        return "text"

    def field_list(self) -> List[str]:
        return [self.fields] if isinstance(self.fields, str) else list(self.fields)


# Registry of calibration sources (probe-verified configs/splits/fields).
PREP_SOURCES: List[SourceSpec] = [
    # chat (0.30)
    SourceSpec("allenai/tulu-3-sft-mixture", "messages", "chat_tulu",
               "default", "train", weight=0.18),
    SourceSpec("HuggingFaceTB/smoltalk", "messages", "chat_smoltalk",
               "all", "train", weight=0.12),
    # math (0.25)
    SourceSpec("nvidia/OpenMathInstruct-2", ("problem", "generated_solution"),
               "math_openmath2", "default", "train", weight=0.15),
    SourceSpec("openai/gsm8k", ("question", "answer"), "math_gsm8k",
               "main", "train", weight=0.10),
    # code (0.25)
    SourceSpec("codeparrot/github-code-clean", "code", "github_code",
               "all-all", "train", weight=0.15),
    SourceSpec("ise-uiuc/Magicoder-Evol-Instruct-110K",
               ("instruction", "response"), "code_magicoder",
               "default", "train", weight=0.10),
    # science (0.20)
    SourceSpec("nvidia/OpenScienceReasoning-2", ("input", "output"),
               "science_osr2", "default", "train", weight=0.12),
    SourceSpec("allenai/sciq",
               ("support", "question", "distractor1", "distractor2",
                "distractor3", "correct_answer"),
               "science_sciq", "default", "train", weight=0.08,
               render=_render_sciq),
]

# Harvest reads from these LOCAL Parquet dirs: (local_dir, stored_field, weight).
DOMAIN_SOURCES = [
    (str(PARQUET_ROOT / s.local_name), s.stored_field, s.weight)
    for s in PREP_SOURCES
]

_PAGE = 100
_REQ_DELAY = 0.0
_MAX_RETRIES = 6
_MAX_PARQUET_WORKERS = 8


def _hfds_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """GET a datasets-server endpoint with retry/backoff on 429/5xx."""
    url = HF_DATASETS_SERVER + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "toffee-xspark/1.0"})
    delay = 0.5
    for attempt in range(_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise
            if attempt >= _MAX_RETRIES:
                raise
            logger.warning("hfds_retry", url=url, status=e.code, attempt=attempt + 1)
        except Exception:
            if attempt >= _MAX_RETRIES:
                raise
            logger.warning("hfds_retry_network", url=url, attempt=attempt + 1)
        time.sleep(delay)
        delay = min(delay * 2, 30.0)
    raise RuntimeError(f"exhausted retries for {url}")


def _hfds_row_count(hf_id: str, config: str = "default", split: str = "train") -> Optional[int]:
    """Best-effort total row count via the /info endpoint."""
    try:
        info = _hfds_get("/info", {"dataset": hf_id})
        ds_info = info.get("dataset_info", {}).get(config)
        if ds_info:
            si = ds_info.get("splits", {}).get(split)
            if si:
                return int(si.get("num_examples", 0))
    except Exception as e:
        logger.warning("hfds_info_failed", dataset=hf_id, error=str(e))
    return None


def _resolve_config_split(hf_id: str, config: Optional[str], split: Optional[str]) -> Tuple[str, str]:
    """Resolve a (config, split) pair for a dataset.

    HF datasets vary wildly: some have no 'default' config (github-code-clean
    is per-language, gsm8k has main/socratic), some have no plain 'train'
    split. An explicitly-configured pair is used as-is; otherwise the best
    candidate is discovered from /info.
    """
    if config and split:
        return config, split
    want_cfg, want_split = config or "default", split or "train"
    try:
        di = _hfds_get("/info", {"dataset": hf_id}).get("dataset_info", {})
    except Exception:
        return want_cfg, want_split
    candidates = [(c, s) for c, info in di.items() for s in info.get("splits", {})]
    candidates.sort(key=lambda cs: (cs[0] != want_cfg, cs[1] != want_split,
                                    not cs[1].startswith("train")))
    for cfg, sp in candidates:
        if (config is None or cfg == config) and (split is None or sp == split):
            return cfg, sp
    return want_cfg, want_split


def _row_to_value(spec: SourceSpec, row_values: Dict[str, Any]) -> Any:
    """Convert a {field: value} dict into the value to store (None skips)."""
    if spec.render is not None:
        try:
            text = spec.render(row_values)
        except Exception as e:
            logger.warning("render_failed", source=spec.local_name, error=str(e))
            return None
        return text if (isinstance(text, str) and text.strip()) else None
    fields = spec.field_list()
    if len(fields) == 1:
        return row_values.get(fields[0])
    parts = [row_values.get(f) for f in fields]
    if any(p is None for p in parts):
        return None
    return "\n".join(str(p) for p in parts)


def _parquet_urls_for(hf_id: str, config: str, split: str) -> List[Tuple[str, int]]:
    """Return [(url, size_bytes), ...] for the split's Parquet files."""
    try:
        data = _hfds_get("/parquet", {"dataset": hf_id, "config": config, "split": split})
        files = data.get("parquet_files", [])
        urls = [(f["url"], int(f.get("size", 0))) for f in files if f.get("url")]
        if urls:
            return urls
    except Exception as e:
        logger.warning("hfds_parquet_failed", dataset=hf_id, error=str(e))
    return []


def _read_parquet_values(url: str, spec: SourceSpec) -> List[Any]:
    """Download one Parquet file and return the stored values for `spec`."""
    table = pq.read_table(url)
    fields = spec.field_list()
    missing = [f for f in fields if f not in table.column_names]
    if missing:
        raise ValueError(f"missing columns {missing} in {url}")
    if len(fields) == 1 and spec.render is None:
        return [v for v in table.column(fields[0]).to_pylist() if v is not None]
    cols = {f: table.column(f).to_pylist() for f in fields}
    out = []
    for i in range(table.num_rows):
        val = _row_to_value(spec, {f: cols[f][i] for f in fields})
        if val is not None:
            out.append(val)
    return out


def _count_existing_rows(out_dir: Path, stored_field: str) -> int:
    """Count rows already written to local Parquet shards."""
    import glob
    total = 0
    for shard in sorted(glob.glob(str(out_dir / "*.parquet"))):
        try:
            total += pq.read_table(shard, columns=[stored_field]).num_rows
        except Exception:
            pass
    return total


def _write_shard(out_dir: Path, stored_field: str, rows: List[Any], shard_idx: int):
    """Write a batch of values as a Parquet shard."""
    table = pa.table({stored_field: rows})
    pq.write_table(table, out_dir / f"shard-{shard_idx:03d}.parquet")


def prepare_source(spec: SourceSpec, max_rows: Optional[int],
                   out_root: Path, resume: bool = True) -> int:
    """Fetch one HF dataset via datasets-server and write local Parquet shards.

    Returns the number of rows on disk for this source afterwards.
    """
    out_dir = out_root / spec.local_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stored_field = spec.stored_field

    config, split = _resolve_config_split(spec.hf_id, spec.config, spec.split)
    logger.info("prepare_source_start", dataset=spec.hf_id, config=config, split=split,
                max_rows=max_rows)

    existing = _count_existing_rows(out_dir, stored_field) if resume else 0
    if existing and max_rows is None:
        logger.info("prepare_resume", dataset=spec.hf_id, existing_rows=existing)

    # Sampling: prefer the precise /rows pager
    if max_rows is not None:
        _prepare_via_rows(spec, out_dir, stored_field, max_rows, resume, existing, config, split)
        return _count_existing_rows(out_dir, stored_field)

    # Full fetch: whole-file Parquet download
    try:
        parquet_written = _fetch_via_parquet(spec, out_dir, stored_field, max_rows, resume, config, split)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            logger.error("hfds_unauthorized", dataset=spec.hf_id,
                         hint="dataset is not served by datasets-server (gated/private); skip or replace")
            return _count_existing_rows(out_dir, stored_field)
        parquet_written = -1
    except Exception as e:
        logger.warning("hfds_parquet_path_failed", dataset=spec.hf_id, error=str(e))
        parquet_written = -1

    if parquet_written == -1:
        _prepare_via_rows(spec, out_dir, stored_field, max_rows, resume, existing, config, split)

    return _count_existing_rows(out_dir, stored_field)


def _fetch_via_parquet(spec: SourceSpec, out_dir: Path, stored_field: str,
                       max_rows: Optional[int], resume: bool, config: str, split: str):
    """Fast path: download whole Parquet files concurrently."""
    import concurrent.futures as _cf

    urls = _parquet_urls_for(spec.hf_id, config, split)
    if not urls:
        return -1

    if max_rows is None:
        total_gb = sum(size for _, size in urls) / 1e9
        if total_gb > 5.0:
            logger.warning("prepare_full_fetch_large", dataset=spec.hf_id,
                           total_gb=round(total_gb, 1), n_files=len(urls),
                           hint="this downloads the ENTIRE dataset — pass --max-rows "
                                "to fetch a bounded sample instead")

    fetched = 0
    existing_names = {p.name for p in out_dir.glob("*.parquet")} if resume else set()
    todo = []
    for u, _size in urls:
        base = u.rsplit("/", 1)[-1]
        if base in existing_names:
            continue
        todo.append((u, base))
    if not todo:
        logger.info("prepare_resume_complete", dataset=spec.hf_id, local_name=spec.local_name)
        return _count_existing_rows(out_dir, stored_field)

    shard_idx = len(sorted(out_dir.glob("*.parquet")))

    def _worker(item):
        url, base = item
        try:
            values = _read_parquet_values(url, spec)
        except Exception as e:
            logger.error("hfds_parquet_read_failed", dataset=spec.hf_id, url=url, error=str(e))
            return None
        return base, values

    with _cf.ThreadPoolExecutor(max_workers=_MAX_PARQUET_WORKERS) as ex:
        for result in ex.map(_worker, todo):
            if result is None:
                continue
            base, values = result
            if not values:
                continue
            if max_rows is not None and fetched >= max_rows:
                break
            if max_rows is not None:
                values = values[: max(0, max_rows - fetched)]
            _write_shard(out_dir, stored_field, values, shard_idx)
            shard_idx += 1
            fetched += len(values)
            logger.info("prepare_parquet_file", dataset=spec.hf_id, file=base, rows=len(values))
    return fetched


def _prepare_via_rows(spec: SourceSpec, out_dir: Path, stored_field: str,
                      max_rows: Optional[int], resume: bool, existing: int,
                      config: str, split: str):
    """Slow fallback: paginate the /rows endpoint."""
    total = _hfds_row_count(spec.hf_id, config, split)
    limit = total if (total is not None and max_rows is None) else (max_rows or total or 0)

    rows = []
    shard_idx = len(sorted(out_dir.glob("*.parquet")))
    offset = existing
    fetched = existing

    while True:
        if limit and fetched >= limit:
            break
        length = min(_PAGE, limit - fetched) if limit else _PAGE
        if offset >= (total or offset + 1) and total is not None:
            break
        try:
            page = _hfds_get("/rows", {
                "dataset": spec.hf_id, "config": config, "split": split,
                "offset": offset, "length": length,
            })
        except urllib.error.HTTPError as e:
            if e.code == 401:
                logger.error("hfds_unauthorized", dataset=spec.hf_id,
                             hint="dataset is not served by datasets-server (gated/private); skip or replace")
                break
            logger.error("hfds_rows_failed", dataset=spec.hf_id, offset=offset, error=str(e))
            break
        except Exception as e:
            logger.error("hfds_rows_failed", dataset=spec.hf_id, offset=offset, error=str(e))
            break

        batch = page.get("rows", [])
        if not batch:
            break
        fields = spec.field_list()
        for r in batch:
            inner = r.get("row", r) if isinstance(r, dict) else {}
            val = _row_to_value(spec, {f: inner.get(f) for f in fields})
            if val is None:
                continue
            rows.append(val)
            fetched += 1
            if limit and fetched >= limit:
                break
        offset += len(batch)
        if len(rows) >= 50_000:
            _write_shard(out_dir, stored_field, rows, shard_idx)
            shard_idx += 1
            rows.clear()
        if len(batch) < length:
            break
        time.sleep(_REQ_DELAY)

    if rows:
        _write_shard(out_dir, stored_field, rows, shard_idx)
        shard_idx += 1
        rows.clear()
    logger.info("prepare_source_done", dataset=spec.hf_id, local_name=spec.local_name,
                rows=_count_existing_rows(out_dir, stored_field))


def prepare_all(out_root: Path, only: Optional[str] = None, max_rows: Optional[int] = None,
                resume: bool = True):
    """Prepare all sources via the datasets-server API."""
    jobs = [s for s in PREP_SOURCES if not (only and only != s.local_name)]

    if not jobs:
        logger.warning("prepare_all_no_sources",
                       hint=f"unknown --only; valid names: "
                            ", ".join(s.local_name for s in PREP_SOURCES))
        return
    if len(jobs) <= 1:
        for spec in jobs:
            prepare_source(spec, max_rows, out_root, resume=resume)
        return
    import concurrent.futures as _cf
    with _cf.ThreadPoolExecutor(max_workers=min(len(jobs), 4)) as ex:
        list(ex.map(lambda s: prepare_source(s, max_rows, out_root, resume=resume), jobs))


def _normalize_field(field: str, text):
    """Flatten a raw field value into a single string."""
    if field == "messages" and isinstance(text, list):
        convo_text = []
        for msg in text:
            if isinstance(msg, dict) and "content" in msg:
                convo_text.append(msg["content"])
        return "\n".join(convo_text)
    if isinstance(text, list):
        return " ".join(str(t) for t in text)
    return text


def _stream_parquet_dir(parquet_dir: Path, field: str, n: int) -> List[str]:
    """Read local Parquet shards with pyarrow, yielding normalized strings."""
    import glob
    out = []
    for shard in sorted(glob.glob(str(parquet_dir / "*.parquet"))):
        table = pq.read_table(shard)
        if field not in table.column_names:
            logger.warning("parquet_missing_field", path=shard, field=field)
            continue
        for raw in table.column(field).to_pylist():
            if len(out) >= n:
                return out
            text = _normalize_field(field, raw)
            if text and isinstance(text, str) and len(text) > 50:
                out.append(text)
    return out


def _stream_text(ds_name: str, field: str, n: int) -> List[str]:
    """Stream text from a LOCAL Parquet directory or HF id (fallback)."""
    out: List[str] = []
    try:
        if Path(ds_name).is_dir():
            out = _stream_parquet_dir(Path(ds_name), field, n)
        else:
            from datasets import load_dataset
            ds = load_dataset(ds_name, split="train", streaming=True)
            for item in ds:
                if len(out) >= n:
                    break
                text = item.get(field) if isinstance(item, dict) else None
                text = _normalize_field(field, text)
                if text and isinstance(text, str) and len(text) > 50:
                    out.append(text)
    except Exception as e:
        logger.error("dataset_source_failed", source=ds_name, error=str(e))
    return out


def _has_shards(parquet_dir) -> bool:
    """A prepared source has at least one Parquet shard on disk."""
    p = Path(parquet_dir)
    return p.is_dir() and any(p.glob("*.parquet"))


def build_dataset(num_samples: int, balance_domains: bool = True) -> List[str]:
    """Fetch diverse, domain-balanced calibration data."""
    if not balance_domains:
        samples = _stream_text(DOMAIN_SOURCES[0][0], DOMAIN_SOURCES[0][1], num_samples)
        if not samples:
            samples = [f"Sample text for calibration {i}" for i in range(num_samples)]
        return samples

    spec_by_dir = {str(PARQUET_ROOT / s.local_name): s for s in PREP_SOURCES}

    total_w = sum(w for *_, w in DOMAIN_SOURCES)
    results = []
    for idx, (ds_name, field, w) in enumerate(DOMAIN_SOURCES):
        # Auto-prepare when the local dir is MISSING **OR EMPTY** — an empty
        # dir (e.g. left over from an aborted run) must not silently disable
        # the source. Bounded rows: harvest only needs ~num_samples * weight
        # rows per domain, so never auto-download an entire dataset.
        if not _has_shards(ds_name):
            spec = spec_by_dir.get(ds_name)
            if spec is not None:
                bounded = min(8192, max(1024, 2 * num_samples))
                logger.info("auto_preparing_source", dir=ds_name, max_rows=bounded)
                prepare_source(spec, bounded, PARQUET_ROOT)
            else:
                logger.error("source_unknown", dir=ds_name)
        n = max(1, int(round(num_samples * w / total_w)))
        got = _stream_text(ds_name, field, n)
        results.append((ds_name, got))
        if got:
            logger.info("dataset_source_ok", source=ds_name, n=len(got))
        else:
            logger.warning("dataset_source_empty", source=ds_name,
                           hint=f"no usable rows under {ds_name} — run: "
                                "python -m toffee xspark prepare --max-rows 4096")

    surviving_w = sum(DOMAIN_SOURCES[idx][2] for idx, (_, got) in enumerate(results) if got)
    samples: List[str] = []
    if surviving_w == 0:
        logger.error("all_dataset_sources_failed", hint="check network or dataset availability")
        return [f"Sample text for calibration {i}" for i in range(num_samples)]
    for idx, (ds_name, got) in enumerate(results):
        if not got:
            continue
        w = DOMAIN_SOURCES[idx][2]
        n = max(1, int(round(num_samples * w / surviving_w)))
        samples.extend(got[:n])
    if len(samples) > num_samples:
        samples = samples[:num_samples]
    while len(samples) < num_samples:
        samples.append(samples[len(samples) % max(len(samples), 1)])
    return samples
