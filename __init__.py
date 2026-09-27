"""
TOFFEE — Trainable Output-Free Fine-tuning Execution Environment.

A 3-pass, hardware-aware fine-tuning toolkit for Qwen3.5-MoE models:
  Pass 1 (Prepare): Format dataset, tokenize, suggest optimal training config.
  Pass 2 (Train):   LoRA/QLoRA fine-tuning with automatic VRAM-aware tuning.
  Pass 3 (Export):   Merge LoRA weights and export to GGUF or safetensors.

Architecture targets: Qwen3.5-MoE hybrid (linear_attention + full_attention + MoE).
"""

from __future__ import annotations

__version__ = "0.1.0"
