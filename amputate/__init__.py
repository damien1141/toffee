"""
@file toffee/amputate/__init__.py
   No-training MoEfication surgery: dense FFN -> sparse MoE ("amputation").

Turns a dense hybrid checkpoint (VeriLoop-E2 27B) into a Qwen3.5-MoE
layout checkpoint (32 routed experts + shared expert, top-4) by fitting
routers and expert assignments from activation statistics only -- no
gradient training, no quality farm, one laptop.

Pipeline (each stage is a CLI subcommand):

  report    scan the checkpoint shards, classify every tensor
            (backbone / vision / mtp.veriloop_* / mtp head), write the
            registry JSON that ``build`` consumes.
  harvest   stream the dense model layer-by-layer (same skeleton as
            ``toffee.quant.passes.pass2_harvest``) and collect, per
            layer: per-neuron importance, router sufficient statistics,
            and (sweep B) calibration + verification statistics against
            the exact top-k softmax semantics llama.cpp uses.
  build     slice the dense FFN into experts + shared expert, write the
            fitted router, bake the calibration scales, copy mixers /
            norms / embeddings / ``mtp.veriloop_*`` verbatim, and emit a
            ``qwen3_5_moe`` HF directory the existing TOFFEE quant
            pipeline can consume unchanged.
  verify    print the offline FFN-reconstruction canary (cosine vs the
            dense FFN, selection-mass recall, oracle ceiling) and the
            PPL canary commands + pre-registered kill criteria.
  canary    the frozen PPL canary corpus + verdict engine
            (``toffee/amputate/canary/SPEC.md``): ``canary-prep``
            builds + freezes 8 domain slots (sha256 manifest,
            disjoint from harvest calibration), ``canary-run`` runs
            llama-perplexity dense vs amputated and applies the
            pre-registered ACCEPT/CONDITIONAL/INVESTIGATE/KILL rules.
  plan      print the A12B preset math (active params, sparsity, bytes
            per token, the three quant dials).
  selftest  end-to-end synthetic run (tiny dense hybrid -> amputated
            MoE) with assertions; proves the machinery without the 54 GB
            download.

Run ``python -m toffee.amputate --help`` or ``toffee amputate --help``.
"""

from toffee.amputate.preset import AmputationPreset, A12B_PRESET

__all__ = ["AmputationPreset", "A12B_PRESET"]
