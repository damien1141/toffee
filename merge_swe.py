#!/usr/bin/env python3
"""Merge agentic coding traces with Open-SWE-Traces and emit parquet.

Inputs
------
- Anchor: JSONL with keys {source, lang, tier, messages, tools}
- SWE:   parquet shards from nvidia/Open-SWE-Traces (deepseek_v4_flash/openhands)

Output
------
- train.parquet / val.parquet in the project's datasets/sft directory
"""

from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/solis/Documentos/Code/OPAL")
ANCHOR_PATH = PROJECT_ROOT / "toffee" / "datasets" / "sft" / "mixed_coding_traces2.jsonl"
SWE_DIR = PROJECT_ROOT / "toffee" / "datasets" / "open-swe-deepseek" / "data" / "openhands" / "deepseek_v4_flash" / "scale-swe"
OUT_DIR = PROJECT_ROOT / "toffee" / "datasets" / "sft"
TRAIN_OUT = OUT_DIR / "train.parquet"
VAL_OUT = OUT_DIR / "val.parquet"

# ---------------------------------------------------------------------------
# Mix settings
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
VAL_FRACTION = 0.05
SWE_TIER = "tier_swe_deepseek"
SWE_SOURCE = "open-swe"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_anchor(path: Path) -> list[dict]:
    """Load the existing JSONL anchor dataset."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_swe_shards(directory: Path) -> list[dict]:
    """Load all parquet shards from the SWE dataset."""
    rows = []
    shards = sorted(directory.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found in {directory}")
    for shard in shards:
        df = pd.read_parquet(shard)
        rows.extend(df.to_dict("records"))
    return rows


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------
def normalize_anchor(row: dict) -> dict:
    """Anchor rows are already in the target schema; just ensure tools is a string."""
    row = dict(row)
    tools = row.get("tools", "")
    if not isinstance(tools, str):
        tools = json.dumps(tools, ensure_ascii=False, default=str) if tools else ""
    row["tools"] = tools
    row["_split_source"] = "anchor"
    return row


def normalize_swe(row: dict) -> dict:
    """Map Open-SWE-Traces row to the anchor schema."""
    messages = row.get("messages", [])
    # Ensure messages is a list of dicts
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    messages = [dict(m) for m in messages]

    # Normalize each message: ensure tool_calls is a list (some may be strings)
    for msg in messages:
        tc = msg.get("tool_calls")
        if isinstance(tc, str):
            try:
                msg["tool_calls"] = json.loads(tc)
            except json.JSONDecodeError:
                msg["tool_calls"] = []
        elif tc is None:
            msg["tool_calls"] = []

    # tools: list of JSON strings -> single JSON string
    tools_raw = row.get("tools", [])
    if hasattr(tools_raw, "tolist"):
        tools_raw = tools_raw.tolist()
    if isinstance(tools_raw, list):
        tools = json.dumps(tools_raw, ensure_ascii=False, default=str)
    else:
        tools = str(tools_raw) if tools_raw else ""

    metadata = row.get("metadata", {})
    if hasattr(metadata, "tolist"):
        metadata = metadata.tolist()
    if hasattr(metadata, "to_dict"):
        metadata = metadata.to_dict()

    return {
        "source": SWE_SOURCE,
        "lang": row.get("language", "unknown"),
        "tier": SWE_TIER,
        "messages": messages,
        "tools": tools,
        "_split_source": "swe",
        "_swe_trajectory_id": row.get("trajectory_id", ""),
        "_swe_repo": row.get("repo", ""),
        "_swe_resolved": row.get("resolved", -1),
        "_swe_metadata": json.dumps(metadata, ensure_ascii=False, default=str) if metadata else "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    random.seed(RANDOM_SEED)

    print("Loading anchor dataset...")
    anchor_rows = [normalize_anchor(r) for r in load_anchor(ANCHOR_PATH)]
    print(f"  anchor rows: {len(anchor_rows)}")

    print("Loading SWE parquet shards...")
    swe_rows = [normalize_swe(r) for r in load_swe_shards(SWE_DIR)]
    print(f"  SWE rows:    {len(swe_rows)}")

    # Balance: downsample SWE to roughly match anchor size
    # This preserves the agentic prior while adding SWE reasoning diversity.
    target_swe = min(len(swe_rows), len(anchor_rows))
    if len(swe_rows) > target_swe:
        random.shuffle(swe_rows)
        swe_rows = swe_rows[:target_swe]
        print(f"  downsampled SWE to {len(swe_rows)}")

    combined = anchor_rows + swe_rows
    random.shuffle(combined)
    print(f"  combined:    {len(combined)}")

    # Train/val split
    val_size = max(1, int(len(combined) * VAL_FRACTION))
    val_rows = combined[:val_size]
    train_rows = combined[val_size:]
    print(f"  train: {len(train_rows)}, val: {len(val_rows)}")

    # Keep only columns that are safe for parquet / training
    # Drop internal bookkeeping fields before writing
    internal_cols = {"_split_source", "_swe_trajectory_id", "_swe_repo", "_swe_resolved", "_swe_metadata"}
    keep_cols = [c for c in combined[0].keys() if c not in internal_cols]

    def clean(rows: list[dict]) -> list[dict]:
        return [{k: row.get(k, "") for k in keep_cols} for row in rows]

    train_df = pd.DataFrame(clean(train_rows))
    val_df = pd.DataFrame(clean(val_rows))

    # Convert list/dict columns to JSON strings for parquet compatibility
    for col in ["messages", "tools"]:
        if col in train_df.columns:
            train_df[col] = train_df[col].apply(
                lambda v: json.dumps(v, ensure_ascii=False, default=str) if not isinstance(v, str) else v
            )
            val_df[col] = val_df[col].apply(
                lambda v: json.dumps(v, ensure_ascii=False, default=str) if not isinstance(v, str) else v
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(TRAIN_OUT, index=False, compression="snappy")
    val_df.to_parquet(VAL_OUT, index=False, compression="snappy")

    print(f"\nWrote:")
    print(f"  {TRAIN_OUT} ({train_df.shape[0]} rows)")
    print(f"  {VAL_OUT} ({val_df.shape[0]} rows)")
    print("\nDone.")


if __name__ == "__main__":
    main()
