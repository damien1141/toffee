"""
@file toffee/amputate/canary/__init__.py
   The frozen PPL canary corpus + pre-registered verdict engine.

Implements SPEC.md (normative): 8 slots x 32,768 evaluated tokens,
frozen manifest (sha256), disjointness from the harvest calibration
data, log-space pooled PPL, noise-floor-adjusted decision rules with
exit codes, and offline selftests for every piece of machinery.

  prep   build + freeze the corpus (remote slots via `datasets`,
         local domain pack, `--selftest` offline)
  run    llama-perplexity A/B over the frozen corpus, verdict + triage
         (`--dry-run` prints commands, `--selftest` offline)
"""

from toffee.amputate.canary.slots import (
    CRITICAL_SLOTS,
    SLOT_TABLE,
    SLOTS_BY_NAME,
)

__all__ = ["SLOT_TABLE", "SLOTS_BY_NAME", "CRITICAL_SLOTS"]
