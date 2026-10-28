"""
TOFFEE data preparation.

Handles dataset loading, formatting, and tokenization for Pass 1.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from toffee.config import DatasetConfig


class BatchRecord:
    """A tokenized batch saved to disk."""

    def __init__(self, input_ids, attention_mask, sample_id: int):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.sample_id = sample_id


def prepare_dataset(
    config: DatasetConfig,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    batch_size: int = 32,
) -> List[BatchRecord]:
    """Load dataset, tokenize, and save batches to disk.

    Returns a list of BatchRecord objects for in-memory use.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    try:
        path = Path(config.path)
        if path.suffix == ".parquet" or path.is_file():
            ds = load_dataset(
                "parquet",
                data_files=str(path),
                split=config.split,
                streaming=config.streaming,
            )
        else:
            ds = load_dataset(config.path, split=config.split, streaming=config.streaming)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {config.path}: {e}") from e

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    batches: List[BatchRecord] = []
    buffer: List[str] = []

    def _flush_buffer(buf: List[str], batch_idx: int) -> None:
        if not buf:
            return
        tokens = tokenizer(
            buf,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )
        for i in range(len(buf)):
            batches.append(BatchRecord(
                input_ids=tokens.input_ids[i],
                attention_mask=tokens.attention_mask[i],
                sample_id=batch_idx * batch_size + i,
            ))
        # Save to disk
        path = output_dir / f"batch_{batch_idx:04d}.pt"
        torch.save(
            {
                "input_ids": tokens.input_ids,
                "attention_mask": tokens.attention_mask,
            },
            path,
        )

    batch_idx = 0
    for item in ds:
        text = item.get(config.text_field, "")
        if not text or len(text.strip()) < 10:
            continue
        if config.format_template:
            text = config.format_template.format(text=text)
        buffer.append(text)
        if len(buffer) >= batch_size:
            _flush_buffer(buffer, batch_idx)
            batch_idx += 1
            buffer = []

    # Flush remainder
    _flush_buffer(buffer, batch_idx)

    return batches


def load_batches_from_disk(output_dir: Path, batch_size: int = 32) -> List[BatchRecord]:
    """Reconstruct BatchRecord list from saved .pt files."""
    import torch

    batches: List[BatchRecord] = []
    files = sorted(output_dir.glob("batch_*.pt"))
    for bf_idx, bf in enumerate(files):
        data = torch.load(bf, weights_only=True)
        for i in range(data["input_ids"].size(0)):
            batches.append(BatchRecord(
                input_ids=data["input_ids"][i],
                attention_mask=data["attention_mask"][i],
                sample_id=bf_idx * batch_size + i,
            ))
    return batches
