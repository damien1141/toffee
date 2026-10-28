"""
TOFFEE hardware management.

Mirrors ONYX's VRAM-aware approach: detect GPU memory, suggest optimal
training config, and enforce hard limits to keep consumer GPUs usable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

from toffee.config import LoRAConfig, TrainingConfig


@dataclass
class GPUInfo:
    """Detected GPU capabilities."""

    name: str
    total_memory_gb: float
    available_memory_gb: float
    compute_capability: tuple[int, int]
    supports_bf16: bool
    supports_fp16: bool


def get_gpu_info() -> Optional[GPUInfo]:
    """Detect primary GPU capabilities. Returns None if no CUDA device."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / (1024**3)
    # Reserve 1.5 GB overhead for PyTorch, drivers, and OS
    available = max(0.0, total - 1.5)
    cc = props.major, props.minor
    # bf16 supported on Ampere+ (sm_80+)
    supports_bf16 = cc[0] >= 8
    supports_fp16 = True
    return GPUInfo(
        name=props.name,
        total_memory_gb=total,
        available_memory_gb=available,
        compute_capability=cc,
        supports_bf16=supports_bf16,
        supports_fp16=supports_fp16,
    )


def suggest_lora_config(
    model_name: str,
    gpu: Optional[GPUInfo] = None,
    *,
    target_rank: int = 0,
) -> LoRAConfig:
    """Suggest a LoRA config sized for the available GPU.

    The heuristic is:
    - < 8 GB VRAM: rank 4, dropout 0.1, gradient checkpointing
    - 8–16 GB:     rank 8, dropout 0.05
    - 16–24 GB:    rank 16, dropout 0.05
    - > 24 GB:     rank 32, dropout 0.0
    """
    if gpu is None:
        gpu = get_gpu_info()

    if gpu is None:
        # CPU fallback: keep it tiny
        return LoRAConfig(r=4, alpha=8, dropout=0.1)

    mem = gpu.available_memory_gb
    if target_rank > 0:
        r = target_rank
    elif mem < 8:
        r = 4
    elif mem < 16:
        r = 8
    elif mem < 24:
        r = 16
    else:
        r = 32

    return LoRAConfig(
        r=r,
        alpha=r * 2,
        dropout=0.1 if mem < 8 else 0.05,
    )


def suggest_training_config(
    model_name: str,
    gpu: Optional[GPUInfo] = None,
    *,
    lora_r: int = 8,
) -> TrainingConfig:
    """Suggest a training config sized for the available GPU.

    Uses a simple memory model:
      per-device-batch × seq_len × hidden_size × 4 bytes ≈ activation memory
    and scales batch size / grad accum to fit within `available_memory_gb`.
    """
    if gpu is None:
        gpu = get_gpu_info()

    cfg = TrainingConfig()

    if gpu is None:
        cfg.per_device_train_batch_size = 1
        cfg.gradient_accumulation_steps = 4
        cfg.fp16 = False
        cfg.bf16 = False
        return cfg

    mem = gpu.available_memory_gb

    # Precision: prefer bf16 on Ampere+, else fp16
    cfg.bf16 = gpu.supports_bf16
    cfg.fp16 = not gpu.supports_bf16 and gpu.supports_fp16

    # Gradient checkpointing for anything under 24 GB
    cfg.gradient_checkpointing = mem < 24

    # Batch size heuristic for Qwen3.5-MoE-35B-A3B class models
    if mem < 8:
        cfg.per_device_train_batch_size = 1
        cfg.gradient_accumulation_steps = 16
    elif mem < 12:
        cfg.per_device_train_batch_size = 1
        cfg.gradient_accumulation_steps = 8
    elif mem < 16:
        cfg.per_device_train_batch_size = 2
        cfg.gradient_accumulation_steps = 4
    elif mem < 24:
        cfg.per_device_train_batch_size = 4
        cfg.gradient_accumulation_steps = 2
    else:
        cfg.per_device_train_batch_size = 8
        cfg.gradient_accumulation_steps = 1

    return cfg


def clamp_vram_limit(vram_limit_gb: float) -> float:
    """Clamp VRAM limit to a sensible range (mirrors ONYX)."""
    return max(4.0, min(vram_limit_gb, 80.0))


def bytes_to_gb(b: int) -> float:
    return b / (1024**3)
