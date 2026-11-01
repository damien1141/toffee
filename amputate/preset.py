"""
@file toffee/amputate/preset.py
   The A12B amputation preset and its bandwidth arithmetic.

Every number here is the same math that was validated in the session
scripts (``a12b_math.py``): 64 layers kept, FFN 17408 -> 32 routed
experts x 544 neurons (a complete partition, every dense neuron lands in
exactly one routed expert) + a shared expert of 4096 neurons duplicated
from the global top-importance neurons, top-4 routing.

The three quant dials:

  1. SPEED   IQ2_XS everywhere         ~44-50 tps (2.0625 bpw active)
  2. QUALITY Q4_K_M on active paths    ~same tps as today, 2.3x precision
  3. SPLIT   RCO Q4 always-on / Q3 routed pool  ~26 tps, 12.9 GB file
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class AmputationPreset:
    """Surgery parameters for one dense -> MoE conversion.

    Attributes:
        name: short preset name.
        num_experts: routed experts per layer (E).
        experts_per_tok: top-k routing (K).
        shared_intermediate: shared-expert width in neurons.
        strategy: neuron -> expert assignment strategy
            (``balanced`` = round-robin over importance-sorted neurons,
            ``block`` = contiguous blocks).
        ridge_alpha: router ridge regularisation.
        router_logit_scale: target pooled std of the emitted router
            logits (temperature normalization; see ``surgery.fit_router``).
            The raw ridge fit predicts a SUM of I/E per-neuron log1p
            energies -- hundreds to thousands on a real FFN -- which
            saturates llama.cpp's softmax into a hard argmax and kills
            the top-k weights. 1.0 (v1.0.4) keeps the softmax in the
            near-linear regime where prob-space corr tracks logit-space
            corr: on the first real-model harvest (VeriLoop-E2, 64L/32E)
            the previous default 2.0 left the gated prob-space corr
            deeply attenuated (0.110 vs logit-corr 0.830, just under the
            0.12 gate) while selection (jaccard/mass) was invariant and
            a synthetic end-to-end sweep showed calibrated cos flat from
            2.0 down to 1.0. ``None``/0 disables (legacy behaviour,
            diagnostics only).
        calib_clamp: calibration scales are clamped to this range.
    """

    name: str = "a12b"
    num_experts: int = 32
    experts_per_tok: int = 4
    shared_intermediate: int = 4096
    strategy: str = "balanced"
    ridge_alpha: float = 1.0
    router_logit_scale: float | None = 1.0
    calib_clamp: float = 5.0

    def validate(self, intermediate_size: int) -> None:
        """Assert the preset divides the dense FFN exactly.

        A complete partition keeps every dense neuron reachable, so no
        capacity is silently thrown away: the amputated model can always
        express everything the dense FFN could, just not all at once.
        """
        if intermediate_size % self.num_experts != 0:
            raise ValueError(
                f"preset {self.name!r}: intermediate_size {intermediate_size} is not "
                f"divisible by num_experts {self.num_experts}"
            )
        if self.experts_per_tok > self.num_experts:
            raise ValueError(
                f"preset {self.name!r}: top-k {self.experts_per_tok} > "
                f"num_experts {self.num_experts}"
            )
        if not 0 < self.shared_intermediate <= intermediate_size:
            raise ValueError(
                f"preset {self.name!r}: shared_intermediate "
                f"{self.shared_intermediate} outside (0, {intermediate_size}]"
            )


A12B_PRESET = AmputationPreset()


# ---------------------------------------------------------------------------
# Bandwidth arithmetic (ported 1:1 from the validated a12b_math.py)
# ---------------------------------------------------------------------------

# Measured VeriLoop-E2 dense @ IQ2_XS numbers (RTX 4080 Mobile, 120 W).
DENSE_TOTAL_PARAMS = 27.5e9
DENSE_TPS = 22.0                     # conservative wallclock baseline
IQ2_XS_BPW = 2.0625
Q4_K_M_BPW = 4.7
Q3_K_BPW = 3.3

# VeriLoop-E2 text config (from the published config.json).
HIDDEN = 5120
LAYERS = 64
INTER = 17408
VOCAB = 248_320
MIXERS_PARAMS = 3.5e9               # always-on GDN + attention mid estimate


def dense_ffn_params(layers: int = LAYERS, hidden: int = HIDDEN,
                     inter: int = INTER) -> float:
    """Total parameters in the dense FFN blocks."""
    return layers * 3 * hidden * inter


def active_params(preset: AmputationPreset = A12B_PRESET,
                  layers: int = LAYERS, hidden: int = HIDDEN,
                  inter: int = INTER, mixers: float = MIXERS_PARAMS,
                  vocab: int = VOCAB) -> Dict[str, float]:
    """Per-token ACTIVE parameter census of the amputated model.

    Returns a dict with the components and the total. The routed pool
    contributes only the top-k experts; the shared expert, the mixers
    and the untied lm_head are always on.
    """
    e_inter = inter // preset.num_experts
    shared_p = layers * 3 * hidden * preset.shared_intermediate
    routed_active_p = layers * preset.experts_per_tok * 3 * hidden * e_inter
    router_p = layers * preset.num_experts * hidden
    lm_head_p = vocab * hidden
    active = shared_p + routed_active_p + mixers + lm_head_p
    return {
        "shared": shared_p,
        "routed_active": routed_active_p,
        "mixers": mixers,
        "lm_head": lm_head_p,
        "router": router_p,
        "active": active,
    }


def stored_params(preset: AmputationPreset = A12B_PRESET,
                  layers: int = LAYERS, hidden: int = HIDDEN,
                  inter: int = INTER, mixers: float = MIXERS_PARAMS,
                  vocab: int = VOCAB) -> Dict[str, float]:
    """Total STORED parameters of the amputated checkpoint.

    The routed pool keeps every dense neuron exactly once (complete
    partition) so it is the same size as the dense FFN; the shared
    expert is a duplicate of the top-importance neurons, so the file
    grows by the shared expert plus the routers.
    """
    ffn_pool = dense_ffn_params(layers, hidden, inter)
    ap = active_params(preset, layers, hidden, inter, mixers, vocab)
    stored = ffn_pool + ap["shared"] + ap["router"] + mixers + 2 * vocab * hidden
    return {
        "ffn_pool": ffn_pool,
        "shared": ap["shared"],
        "router": ap["router"],
        "mixers": mixers,
        "embed_and_head": 2 * vocab * hidden,
        "stored": stored,
    }


def summarize(preset: AmputationPreset = A12B_PRESET,
              eff_bw_gb_s: float | None = None) -> str:
    """Render the preset math + quant dials as a text report.

    Args:
        eff_bw_gb_s: effective decode bandwidth used for tps forecasts.
            Defaults to the conservative wallclock-implied value; the
            nsys per-kernel ceiling (~240 GB/s) is printed alongside.
    """
    ap = active_params(preset)
    sp = stored_params(preset)
    bytes_per_tok_now = DENSE_TOTAL_PARAMS * IQ2_XS_BPW / 8
    if eff_bw_gb_s is None:
        eff_bw_gb_s = bytes_per_tok_now * DENSE_TPS / 1e9  # ~156 GB/s

    ffn_sparsity = 1 - (ap["shared"] + ap["routed_active"]) / dense_ffn_params()
    lines = [
        f"preset: {preset.name}  (E={preset.num_experts}, top-{preset.experts_per_tok}, "
        f"shared={preset.shared_intermediate}, strategy={preset.strategy})",
        f"implied wallclock bandwidth from dense baseline: {eff_bw_gb_s:.0f} GB/s "
        f"(nsys per-kernel ceiling ~240 GB/s)",
        f"ACTIVE params:  {ap['active']/1e9:.2f} B  "
        f"(shared {ap['shared']/1e9:.2f} + routed {ap['routed_active']/1e9:.2f} "
        f"+ mixers {ap['mixers']/1e9:.1f} + lm_head {ap['lm_head']/1e9:.2f})",
        f"STORED params: {sp['stored']/1e9:.2f} B  (routed pool keeps every dense neuron "
        f"once; shared expert is the duplication cost)",
        f"FFN sparsity:  {ffn_sparsity*100:.0f}%",
        "",
        "--- dial 1: SPEED (IQ2_XS everywhere) ---",
    ]
    b1 = ap["active"] * IQ2_XS_BPW / 8
    lines.append(
        f"active bytes/token {b1/1e9:.2f} GB -> {eff_bw_gb_s*1e9/b1:.0f} tps naive; "
        f"MoE kernel overhead -> ~44-50 tps"
    )
    lines.append("")
    lines.append("--- dial 2: QUALITY (Q4_K_M on every active path) ---")
    b2 = ap["active"] * Q4_K_M_BPW / 8
    lines.append(
        f"active bytes/token {b2/1e9:.2f} GB -> {eff_bw_gb_s*1e9/b2:.0f} tps naive; "
        f"same speed as today but 4.7 bpw instead of 2.06"
    )
    lines.append("")
    lines.append("--- dial 3: SPLIT (RCO: Q4_K_M always-on+shared, Q3_K routed pool) ---")
    always_p = ap["mixers"] + ap["lm_head"] + ap["shared"]
    embed_p = VOCAB * HIDDEN
    pool_p = sp["ffn_pool"]
    b3 = always_p * Q4_K_M_BPW / 8 + ap["routed_active"] * Q3_K_BPW / 8
    size3 = (always_p * Q4_K_M_BPW / 8 + (pool_p + embed_p) * Q3_K_BPW / 8) / 1e9
    lines.append(
        f"per-token bytes {b3/1e9:.2f} GB -> {eff_bw_gb_s*1e9/b3:.0f} tps; "
        f"total file ~{size3:.1f} GB (routed pool + embeddings at Q3_K)"
    )
    lines.append("")
    lines.append("--- prefill ---")
    lines.append(
        f"FLOP ratio active/dense {ap['active']/DENSE_TOTAL_PARAMS:.2f} -> naive "
        f"{960* DENSE_TOTAL_PARAMS/ap['active']:.0f} t/s from today's 960; "
        f"MoE batch overhead -> ~1400-1800"
    )
    return "\n".join(lines)
