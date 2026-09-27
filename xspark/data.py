"""
TOFFEE XSpark data preparation.

Handles dataset fetching via HF datasets-server API and local Parquet caching.
Supports both script-backed datasets (via isolated venv) and Parquet-native datasets.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from toffee.hardware import get_gpu_info

logger = structlog.get_logger()


HF_DATASETS_SERVER = "https://datasets-server.huggingface.co"
PARQUET_ROOT = Path("xspark_data/parquet")

# (hf_dataset_id, source_field(s), local_name, max_rows, served_by_datasets_server)
PREP_SOURCES = [
    ("el4/bartowski-imatrix-v5-semantic-parquet", "text", "bartowski_semantic", None, True),
    ("HuggingFaceH4/ultrachat_200k", "messages", "ultrachat", None, True),
    ("nvidia/Nemotron-Math-v2", ("problem", "expected_answer"), "nemotron_math", None, True),
    ("codeparrot/github-code-clean", "code", "github_code", None, True),
]

# Harvest reads from these LOCAL Parquet dirs. Tuple: (local_dir, stored_field, weight).
DOMAIN_SOURCES = [
    (str(PARQUET_ROOT / "bartowski_semantic"), "text", 0.25),
    (str(PARQUET_ROOT / "ultrachat"), "messages", 0.25),
    (str(PARQUET_ROOT / "nemotron_math"), "text", 0.25),
    (str(PARQUET_ROOT / "github_code"), "code", 0.25),
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


def _hfds_row_count(hf_id: str) -> Optional[int]:
    """Best-effort total row count for split=train via the /info endpoint."""
    try:
        info = _hfds_get("/info", {"dataset": hf_id})
        for ds_info in info.get("dataset_info", {}).values():
            for split_name, split_info in ds_info.get("splits", {}).items():
                if split_name == "train":
                    return int(split_info.get("num_examples", 0))
    except Exception as e:
        logger.warning("hfds_info_failed", dataset=hf_id, error=str(e))
    return None


def _hfds_row_field(row: Dict[str, Any], field) -> Any:
    """Extract one or more fields from a datasets-server `row` object."""
    fields = (field,) if isinstance(field, str) else tuple(field)

    def _get(src, key):
        if isinstance(src, dict) and key in src:
            return src[key]
        inner = src.get("row") if isinstance(src, dict) else None
        if isinstance(inner, dict) and key in inner:
            return inner[key]
        return None

    if len(fields) == 1:
        return _get(row, fields[0])
    parts = [_get(row, f) for f in fields]
    if any(p is None for p in parts):
        return None
    return "\n".join(str(p) for p in parts)


def _parquet_urls_for(hf_id: str) -> List[str]:
    """Return the list of train-split Parquet file URLs via the /parquet endpoint."""
    try:
        data = _hfds_get("/parquet", {"dataset": hf_id, "config": "default", "split": "train"})
        files = data.get("parquet_files", [])
        urls = [f["url"] for f in files if f.get("url")]
        if urls:
            return urls
    except Exception as e:
        logger.warning("hfds_parquet_failed", dataset=hf_id, error=str(e))
    return []


def _read_parquet_columns(url: str, fields) -> List[Any]:
    """Download one Parquet file and return the chosen column(s) as stored-field strings."""
    table = pq.read_table(url)
    field_list = [fields] if isinstance(fields, str) else list(fields)
    missing = [f for f in field_list if f not in table.column_names]
    if missing:
        raise ValueError(f"missing columns {missing} in {url}")
    if len(field_list) == 1:
        col = field_list[0]
        return [v for v in table.column(col).to_pylist() if v is not None]
    out = []
    cols = [table.column(f).to_pylist() for f in field_list]
    for vals in zip(*cols):
        if any(v is None for v in vals):
            continue
        out.append("\n".join(str(v) for v in vals))
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


def _write_shard(out_dir: Path, stored_field: str, rows: List[Dict[str, Any]], shard_idx: int):
    """Write a batch of rows as a Parquet shard."""
    table = pa.table({stored_field: [r[stored_field] for r in rows]})
    pq.write_table(table, out_dir / f"shard-{shard_idx:03d}.parquet")


def prepare_source(
    hf_id: str,
    field,
    local_name: str,
    max_rows: Optional[int],
    out_root: Path,
    resume: bool = True,
) -> int:
    """Fetch one HF dataset via datasets-server and write local Parquet shards.

    Returns the number of rows written.
    """
    out_dir = out_root / local_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stored_field = field if isinstance(field, str) else "text"

    existing = _count_existing_rows(out_dir, stored_field) if resume else 0
    if existing and max_rows is None:
        logger.info("prepare_resume", dataset=hf_id, existing_rows=existing)

    # Sampling: prefer the precise /rows pager
    if max_rows is not None:
        _prepare_via_rows(hf_id, field, local_name, out_dir, stored_field, max_rows, resume, existing)
        return _count_existing_rows(out_dir, stored_field)

    # Full fetch: whole-file Parquet download
    try:
        parquet_written = _fetch_via_parquet(hf_id, field, local_name, out_dir, stored_field, max_rows, resume)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            logger.error("hfds_unauthorized", dataset=hf_id,
                         hint="dataset is not served by datasets-server (gated/private); skip or replace")
            return _count_existing_rows(out_dir, stored_field)
        parquet_written = -1
    except Exception as e:
        logger.warning("hfds_parquet_path_failed", dataset=hf_id, error=str(e))
        parquet_written = -1

    if parquet_written == -1:
        _prepare_via_rows(hf_id, field, local_name, out_dir, stored_field, max_rows, resume, existing)

    return _count_existing_rows(out_dir, stored_field)


def _fetch_via_parquet(hf_id, field, local_name, out_dir, stored_field, max_rows, resume):
    """Fast path: download whole Parquet files concurrently."""
    import concurrent.futures as _cf

    urls = _parquet_urls_for(hf_id)
    if not urls:
        return -1

    fetched = 0
    existing_names = {p.name for p in out_dir.glob("*.parquet")} if resume else set()
    todo = []
    for u in urls:
        base = u.rsplit("/", 1)[-1]
        if base in existing_names:
            continue
        todo.append((u, base))
    if not todo:
        logger.info("prepare_resume_complete", dataset=hf_id, local_name=local_name)
        return _count_existing_rows(out_dir, stored_field)

    shard_idx = len(sorted(out_dir.glob("*.parquet")))

    def _worker(item):
        url, base = item
        try:
            values = _read_parquet_columns(url, field)
        except Exception as e:
            logger.error("hfds_parquet_read_failed", dataset=hf_id, url=url, error=str(e))
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
            _write_shard(out_dir, stored_field, [{stored_field: v} for v in values], shard_idx)
            shard_idx += 1
            fetched += len(values)
            logger.info("prepare_parquet_file", dataset=hf_id, file=base, rows=len(values))
    return fetched


def _prepare_via_rows(hf_id, field, local_name, out_dir, stored_field, max_rows, resume, existing):
    """Slow fallback: paginate the /rows endpoint."""
    total = _hfds_row_count(hf_id)
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
                "dataset": hf_id, "config": "default", "split": "train",
                "offset": offset, "length": length,
            })
        except urllib.error.HTTPError as e:
            if e.code == 401:
                logger.error("hfds_unauthorized", dataset=hf_id,
                             hint="dataset is not served by datasets-server (gated/private); skip or replace")
                break
            logger.error("hfds_rows_failed", dataset=hf_id, offset=offset, error=str(e))
            break
        except Exception as e:
            logger.error("hfds_rows_failed", dataset=hf_id, offset=offset, error=str(e))
            break

        batch = page.get("rows", [])
        if not batch:
            break
        for r in batch:
            val = _hfds_row_field(r, field)
            if val is None:
                continue
            rows.append({stored_field: val})
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
    logger.info("prepare_source_done", dataset=hf_id, local_name=local_name,
                rows=_count_existing_rows(out_dir, stored_field))


def prepare_all(out_root: Path, only: Optional[str] = None, max_rows: Optional[int] = None,
                resume: bool = True):
    """Prepare all SERVED sources via the datasets-server fast path."""
    jobs = [(hf_id, field, local_name) for hf_id, field, local_name, _, served in PREP_SOURCES
            if served and not (only and only != local_name)]

    def _run(job):
        hf_id, field, local_name = job
        prepare_source(hf_id, field, local_name, max_rows, out_root, resume=resume)

    if not jobs:
        logger.warning("prepare_all_no_served_sources", hint="use xspark_prepare_data.py for script-backed sources")
        return
    if len(jobs) <= 1:
        for job in jobs:
            _run(job)
        return
    import concurrent.futures as _cf
    with _cf.ThreadPoolExecutor(max_workers=min(len(jobs), 4)) as ex:
        list(ex.map(_run, jobs))


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


def build_dataset(num_samples: int, balance_domains: bool = True) -> List[str]:
    """Fetch diverse, domain-balanced calibration data."""
    if not balance_domains:
        samples = _stream_text(DOMAIN_SOURCES[0][0], DOMAIN_SOURCES[0][1], num_samples)
        if not samples:
            samples = [f"Sample text for calibration {i}" for i in range(num_samples)]
        return samples

    prep_by_dir = {str(PARQUET_ROOT / name): (hf_id, field, served)
                   for hf_id, field, name, _, served in PREP_SOURCES}

    total_w = sum(w for *_, w in DOMAIN_SOURCES)
    results = []
    for idx, (ds_name, field, w) in enumerate(DOMAIN_SOURCES):
        if not Path(ds_name).is_dir():
            prep = prep_by_dir.get(ds_name)
            if prep:
                hf_id, src_field, served = prep
                if served:
                    logger.info("auto_preparing_source", dir=ds_name)
                    prepare_source(hf_id, src_field, Path(ds_name).name, None, PARQUET_ROOT)
                else:
                    logger.error("source_not_prepared", dir=ds_name, hf_id=hf_id,
                                 hint="run: python xspark_prepare_data.py --only "
                                      + Path(ds_name).name + "  (in a datasets==2.21.0 venv)")
        n = int(round(num_samples * w / total_w))
        got = _stream_text(ds_name, field, n)
        results.append((ds_name, got))
        if got:
            logger.info("dataset_source_ok", source=ds_name, n=len(got))
        else:
            logger.warning("dataset_source_empty", source=ds_name,
                           hint="fetch failed (network/DNS) or dataset unavailable")

    surviving_w = sum(DOMAIN_SOURCES[idx][2] for idx, (_, got) in enumerate(results) if got)
    samples: List[str] = []
    if surviving_w == 0:
        logger.error("all_dataset_sources_failed", hint="check network or dataset availability")
        return [f"Sample text for calibration {i}" for i in range(num_samples)]
    for idx, (ds_name, got) in enumerate(results):
        if not got:
            continue
        w = DOMAIN_SOURCES[idx][2]
        n = int(round(num_samples * w / surviving_w))
        samples.extend(got[:n])
    if len(samples) > num_samples:
        samples = samples[:num_samples]
    while len(samples) < num_samples:
        samples.append(samples[len(samples) % max(len(samples), 1)])
    return samples
