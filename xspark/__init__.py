"""
TOFFEE XSpark module v2.

Runtime-faithful speculative decoding draft training for ik_llama.cpp.

Two variants, both exported under the dense ``dflash`` GGUF architecture:

  dspark  — parallel backbone + Markov bigram head (DeepSpec-style), exported
            for ``--spec-type dspark`` (autoregressive corrected chain).
  dflash2 — parallel backbone + two-tap dynamic convolutions + candidate
            selector lattice (Inco's DFlash2), exported for the dflash2
            runtime (greedy/sampled lattice walk with rejection sampling).

Pipeline:
  prepare  → fetch datasets via HF datasets-server API
  harvest  → extract target-layer features + final post-norm hidden
  train    → train the draft head (anchor-block, exact full-vocab distillation)
  export   → export the head to an ik_llama.cpp-compatible GGUF
  eval     → offline acceptance evaluation (replays the runtime semantics)
"""

from __future__ import annotations

__version__ = "2.0.0"
