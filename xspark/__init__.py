"""
TOFFEE XSpark module.

Speculative decoding draft model training for Qwen3.5-MoE and other architectures.
Uses the DSpark/DSV4 head architecture with teacher forcing and 3-term loss.

Pipeline:
  prepare  → fetch datasets via HF datasets-server API
  harvest  → extract teacher activations and top-K logits
  train    → train XSpark draft head (backbone + Markov + confidence)
  export   → export trained head to GGUF for dflash/DSV4 runtime
"""

from __future__ import annotations

__version__ = "0.1.0"
