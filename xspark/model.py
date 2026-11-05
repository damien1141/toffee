"""
TOFFEE XSpark draft model — runtime-faithful reimplementation.

This module is a bit-exact PyTorch mirror of the *dense* ``dflash`` graph in
ik_llama.cpp (``src/graphs/build_dflash.cpp``), so that what we train is what
actually executes at inference time:

Context path (per decode round in ik_llama.cpp):
    fused   = RMSNorm(fc @ target_features, enc.output_norm)     # once
    K_ctx   = rope(per-head RMSNorm(wk @ fused))                  # per layer
    V_ctx   = wv @ fused                                          # per layer

Draft path (block = [seed, mask, mask, ...]):
    per layer:
        cur      = RMSNorm(h, attn_norm)
        [dflash2] dyn = attn_conv_proj @ cur;  cur = conv(cur, dyn, attn_conv_base, side=0)
        Q = rope(q_norm(wq @ cur));  K = rope(k_norm(wk @ cur));  V = wv @ cur
        attn = softmax(Q [K_ctx; K_blk]^T / sqrt(d)) [V_ctx; V_blk]   # block is bidirectional
        cur  = wo @ attn
        [dflash2] cur = conv(cur, dyn, attn_conv_base, side=1)
        h    = h + cur
        [dflash2] cur = RMSNorm(h, ffn_norm); dyn = ffn_conv_proj @ cur
                     cur = conv(cur, dyn, ffn_conv_base, side=0)
        [plain ] cur = RMSNorm(h, ffn_norm)
        h    = h + SwiGLU(cur)                                     # sequential residual
    logits = lm_head(RMSNorm(h, output_norm))                       # frozen target head

Heads:
    dspark : logits[i] += W2 @ W1[prev_token]  (rank-r bigram correction;
             teacher-forced in training, autoregressive greedy at inference)
    dflash2: selector over top-K candidates per position
             S_t(a, b) = U_t(b) + <A(a) * H(h_t), B(b)>
             (A = selector_predecessor, B = selector_successor, H = selector_hidden)

Position / mask geometry (verified against ``common/speculative-dflash-impl.h``
and mainline llama.cpp ``common/speculative.cpp``):
    * context feature rows live at their true positions 0..p-1
    * the seed token x_p is placed at its TRUE position p (the newest feature
      row is p-1 — features lag the committed tokens by exactly one)
    * mask rows sit at p+1 .. p+B-1
    * full (non-SWA) mask: every block row sees all context rows (pos < p)
      and all block rows (bidirectional within the block)
    * dspark  : row j logits predict x_{p+j+1}  (standard next-token rows;
                row 0 is the seed row and its prediction IS used)
    * dflash2 : row j (j>=1) logits predict x_{p+j}  (own position; the walk
                in llama.cpp reads lattice columns 1..B-1 with the anchor as
                the pos-1 predecessor, so the seed row's logits are unused)

The dflash2 two-tap dynamic convolution mirrors ``build_dflash2_conv`` exactly:
    dynamic = conv_proj @ x                       # [T, 2*K*G], layout (s, k, g)
    coeffs  = dynamic.view(T, 2, K, G)            # s = side (slowest), g fastest
    w[t,s,k,c] = coeffs[t,s,k,c//group_size] + base[s,k,c]
    out[t,c] = sum_k w[t,s,k,c] * x[t-k, c]       # x[-i] = 0 (the whole batch
                                                  # is one block: row 1 reads
                                                  # the seed row = last verified
                                                  # token, matching the blog)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm matching llama.cpp llm_build_norm(LLM_NORM_RMS)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def precompute_rope(head_dim: int, positions: torch.Tensor, base: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """positions: [P] int64 -> cos/sin [P, head_dim//2]."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device) / head_dim))
    freqs = torch.outer(positions.float(), inv_freq)  # [P, head_dim//2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: Optional[int] = None) -> torch.Tensor:
    """LLAMA_ROPE_TYPE_NORM: pairs (2i, 2i+1) of the first rope_dim channels.

    x: [..., head_dim]; cos/sin: broadcastable [..., rope_dim//2].
    """
    if cos is None:
        return x
    rd = rope_dim if rope_dim is not None else x.shape[-1]
    cos2 = torch.cat([cos, cos], dim=-1).to(x.dtype)
    sin2 = torch.cat([sin, sin], dim=-1).to(x.dtype)
    if rd < x.shape[-1]:
        x_rot, x_pass = x[..., :rd], x[..., rd:]
        return torch.cat([x_rot * cos2 + rotate_half(x_rot) * sin2, x_pass], dim=-1)
    return x * cos2 + rotate_half(x) * sin2


def dflash2_conv(x: torch.Tensor, dynamic: torch.Tensor, base: torch.Tensor, side: int) -> torch.Tensor:
    """K-tap dynamic depthwise convolution — mirrors build_dflash2_conv.

    x:       [T, n_embd]
    dynamic: [T, 2*K*G]  (conv_proj output; per-token layout: side slowest, then tap, then group)
    base:    [2, K, n_embd]  (PyTorch layout of ggml's [n_embd, K, 2])
    side:    0 = input conv, 1 = output conv
    """
    T, n_embd = x.shape
    K = base.shape[1]
    G = dynamic.shape[1] // (2 * K)
    group_size = n_embd // G
    coeffs = dynamic.view(T, 2, K, G)
    # broadcast per-group coefficients to channels and add the base kernel
    w = coeffs.repeat_interleave(group_size, dim=-1) + base[None]   # [T, 2, K, n_embd]
    w = w[:, side]                                                  # [T, K, n_embd]
    out = torch.zeros_like(x)
    for tap in range(K):
        if tap == 0:
            shifted = x
        else:
            shifted = torch.cat(
                [torch.zeros(tap, n_embd, dtype=x.dtype, device=x.device), x[:T - tap]], dim=0)
        out = out + w[:, tap, :] * shifted
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class XSparkConfig:
    """Configuration for the runtime-faithful draft model."""

    variant: str = "dspark"            # "dspark" | "dflash2"
    hidden_size: int = 2048            # == target hidden (shared IO tensors)
    vocab_size: int = 151936
    num_layers: int = 4
    num_heads: int = 16
    num_kv_heads: int = 16             # MHA by default; GQA supported
    head_dim: int = 128
    intermediate_size: int = 6144      # SwiGLU FFN width
    rope_base: float = 1000000.0
    rms_eps: float = 1e-6
    block_size: int = 8                # draft block rows: [seed] + (B-1) masks
    mask_token_id: int = 151669

    # target-feature fusion
    n_target_layers: int = 5
    target_layer_ids: Optional[List[int]] = None   # 0-based target layer ids

    # dspark head
    markov_rank: int = 256
    # dflash2 head
    selector_rank: int = 256
    selector_top_k: int = 16
    conv_kernel_size: int = 2
    conv_group_size: int = 16

    # training-only knobs
    use_confidence_head: bool = True   # dspark only (dead weight at runtime,
                                       # but the GGUF may carry it)

    def __post_init__(self):
        if self.variant not in ("dspark", "dflash2"):
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.variant == "dflash2":
            if self.hidden_size % self.conv_group_size != 0:
                raise ValueError("hidden_size must be divisible by conv_group_size")
            if self.conv_kernel_size <= 0 or self.conv_group_size <= 0:
                raise ValueError("conv_kernel_size/conv_group_size must be positive")
            if self.selector_rank <= 0 or self.selector_top_k <= 0:
                raise ValueError("selector_rank/selector_top_k must be positive")
            if self.hidden_size < self.selector_top_k * (self.selector_top_k + 1):
                raise ValueError(
                    f"dflash2 requires hidden_size >= top_k*(top_k+1) "
                    f"(got {self.hidden_size} < {self.selector_top_k * (self.selector_top_k + 1)})")
        if self.target_layer_ids is None:
            self.target_layer_ids = []

    @property
    def n_target_features(self) -> int:
        return self.hidden_size * self.n_target_layers

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "XSparkConfig":
        return cls(**json.loads(Path(path).read_text()))


def stream_embed_and_lm_head(model_path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Stream the target's embedding (and lm_head if untied) from safetensors.

    Returns (embed_weight [V, H], lm_head_weight [V, H] or None if tied/absent).
    """
    import json

    import safetensors.torch

    index_path = Path(model_path) / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        single = Path(model_path) / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"no safetensors found in {model_path}")
        w = safetensors.torch.load_file(single, device="cpu")
        weight_map = {k: "model.safetensors" for k in w}

    embed_key = lm_key = None
    for k in weight_map:
        if embed_key is None and ("embed_tokens" in k or k.endswith("tok_embeddings.weight")):
            embed_key = k
        if lm_key is None and ("lm_head" in k or k.endswith("output.weight")) and "final" not in k:
            lm_key = k

    if embed_key is None:
        raise RuntimeError("could not locate the target embedding tensor")

    wanted = {embed_key}
    if lm_key is not None:
        wanted.add(lm_key)
    files = {weight_map[k] for k in wanted}
    found = {}
    for f in sorted(files):
        w = safetensors.torch.load_file(Path(model_path) / f, device="cpu")
        for k in wanted:
            if k in w:
                found[k] = w[k]

    embed = found[embed_key].float()
    lm_head = found.get(lm_key)
    if lm_head is not None:
        lm_head = lm_head.float()
        if tuple(lm_head.shape) != tuple(embed.shape):
            logger.warning("lm_head_shape_mismatch", got=tuple(lm_head.shape),
                           expected=tuple(embed.shape))
            lm_head = None
    return embed, lm_head


def infer_config_from_target(target_config, variant: str = "dspark", **overrides) -> XSparkConfig:
    """Build an XSparkConfig from a HF target config (AutoConfig)."""
    text_config = getattr(target_config, "text_config", target_config)

    def _get(name, default):
        v = getattr(target_config, name, None)
        if v is None:
            v = getattr(text_config, name, None)
        return v if v is not None else default

    hidden = int(_get("hidden_size", 2048))
    vocab = int(_get("vocab_size", 151936))
    num_layers_tgt = int(_get("num_hidden_layers", 32))
    heads = int(_get("num_attention_heads", 16))
    head_dim = int(_get("head_dim", hidden // heads))
    rope_base = float(_get("rope_theta", 1000000.0))
    rms_eps = float(_get("rms_norm_eps", 1e-6))
    intermediate = int(_get("intermediate_size", hidden * 3))

    # evenly spread interior layers (0-based ids), DeepSpec-style
    n_tl = int(overrides.pop("n_target_layers", 5))
    target_layer_ids = overrides.pop(
        "target_layer_ids",
        [max(0, (num_layers_tgt * i) // (n_tl + 1) - 1) for i in range(1, n_tl + 1)],
    )

    cfg = XSparkConfig(
        variant=variant,
        hidden_size=hidden,
        vocab_size=vocab,
        num_layers=int(overrides.pop("num_layers", 4)),
        num_heads=heads,
        num_kv_heads=int(overrides.pop("num_kv_heads", heads)),
        head_dim=head_dim,
        intermediate_size=int(overrides.pop("intermediate_size", min(intermediate, hidden * 4))),
        rope_base=float(overrides.pop("rope_base", rope_base)),
        rms_eps=rms_eps,
        block_size=int(overrides.pop("block_size", 8)),
        mask_token_id=int(overrides.pop("mask_token_id", 151669)),
        n_target_layers=len(target_layer_ids),
        target_layer_ids=[int(i) for i in target_layer_ids],
        markov_rank=int(overrides.pop("markov_rank", 256)),
        selector_rank=int(overrides.pop("selector_rank", 256)),
        selector_top_k=int(overrides.pop("selector_top_k", 16)),
        conv_kernel_size=int(overrides.pop("conv_kernel_size", 2)),
        conv_group_size=int(overrides.pop("conv_group_size", 16)),
        use_confidence_head=bool(overrides.pop("use_confidence_head", variant == "dspark")),
    )
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class DraftAttention(nn.Module):
    """Dense attention projections matching build_dflash's per-layer structure."""

    def __init__(self, cfg: XSparkConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.num_heads
        self.n_kv = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.hidden_size, bias=False)
        # per-head RMSNorms (mandatory in the ik_llama.cpp graph contract)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_eps)

    def project(self, x: torch.Tensor):
        """x: [T, H] -> per-head Q/K/V with q/k norms applied (no rope).

        Matches: reshape_3d(w(x), head_dim, heads, T) -> RMSNorm -> (rope by caller)
        """
        T = x.shape[0]
        q = self.q_proj(x).view(T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(T, self.n_kv, self.head_dim)
        v = self.v_proj(x).view(T, self.n_kv, self.head_dim)
        return self.q_norm(q), self.k_norm(k), v


class DraftLayer(nn.Module):
    """One dense dflash block: attention + SwiGLU FFN with sequential residual,
    plus the four dflash2 dynamic-conv sites. The layer owns parameters and the
    FFN; the attention flow is driven by XSparkDraftModel.forward_blocks."""

    def __init__(self, cfg: XSparkConfig):
        super().__init__()
        self.cfg = cfg
        self.is_dflash2 = cfg.variant == "dflash2"
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.attn = DraftAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.ffn_gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.ffn_up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.ffn_down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        if self.is_dflash2:
            G = cfg.hidden_size // cfg.conv_group_size
            n_proj = 2 * cfg.conv_kernel_size * G
            self.attn_conv_proj = nn.Linear(cfg.hidden_size, n_proj, bias=False)
            self.register_parameter(
                "attn_conv_base", nn.Parameter(torch.zeros(2, cfg.conv_kernel_size, cfg.hidden_size)))
            self.ffn_conv_proj = nn.Linear(cfg.hidden_size, n_proj, bias=False)
            self.register_parameter(
                "ffn_conv_base", nn.Parameter(torch.zeros(2, cfg.conv_kernel_size, cfg.hidden_size)))

    def ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))


# ---------------------------------------------------------------------------
# The draft model
# ---------------------------------------------------------------------------

class XSparkDraftModel(nn.Module):
    """Runtime-faithful dspark / dflash2 draft model.

    Frozen (shared with target): embed_tokens, lm_head.
    Trainable: fc, hidden_norm, layers, output_norm, variant head.
    """

    def __init__(self, cfg: XSparkConfig, embed_weight: torch.Tensor,
                 lm_head_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size

        # frozen shared IO (streamed from the target model)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, H)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_weight.to(torch.float32))
        self.embed_tokens.weight.requires_grad = False
        self.lm_head = nn.Linear(H, cfg.vocab_size, bias=False)
        if lm_head_weight is not None and tuple(lm_head_weight.shape) == (cfg.vocab_size, H):
            with torch.no_grad():
                self.lm_head.weight.copy_(lm_head_weight.to(torch.float32))
            self.tied_lm_head = False
        else:
            self.lm_head.weight = self.embed_tokens.weight  # tied
            self.tied_lm_head = True
        self.lm_head.weight.requires_grad = False

        # target-feature fusion
        self.fc = nn.Linear(cfg.n_target_features, H, bias=False)
        self.hidden_norm = RMSNorm(H, cfg.rms_eps)   # exported as enc.output_norm

        # backbone
        self.layers = nn.ModuleList([DraftLayer(cfg) for _ in range(cfg.num_layers)])
        self.output_norm = RMSNorm(H, cfg.rms_eps)

        # heads
        if cfg.variant == "dspark":
            self.markov_w1 = nn.Embedding(cfg.vocab_size, cfg.markov_rank)
            self.markov_w2 = nn.Linear(cfg.markov_rank, cfg.vocab_size, bias=False)
            if cfg.use_confidence_head:
                self.conf_head = nn.Linear(H + cfg.markov_rank, 1, bias=True)
        else:
            self.selector_prev = nn.Embedding(cfg.vocab_size, cfg.selector_rank)      # A
            self.selector_next = nn.Embedding(cfg.vocab_size, cfg.selector_rank)      # B
            self.selector_hidden = nn.Linear(H, cfg.selector_rank, bias=False)        # H(.)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if not module.weight.requires_grad:
                    continue  # never touch the frozen shared IO
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, DraftLayer) and module.is_dflash2:
                nn.init.zeros_(module.attn_conv_base)
                nn.init.zeros_(module.ffn_conv_base)

    # ------------------------------------------------------------------
    # Core forward: anchor-block geometry
    # ------------------------------------------------------------------

    def forward_blocks(
        self,
        ctx_features: torch.Tensor,      # [S, L*H] fused target features (layer order)
        block_token_ids: torch.Tensor,   # [A, B] seed + masks (row 0 = seed x_p)
        anchor_pos: torch.Tensor,        # [A] p (true position of the seed token)
    ) -> torch.Tensor:
        """Returns the final pre-logits hidden states h: [A, B, H].

        Matches the runtime: one block per anchor; context K/V from fused
        features (positions 0..S-1, masked to pos < p); block rows at
        positions p..p+B-1; bidirectional within the block.
        """
        cfg = self.cfg
        S = ctx_features.shape[0]
        A, B = block_token_ids.shape
        device = ctx_features.device
        H = cfg.hidden_size

        # ---- context path (computed once, shared by all blocks) ----
        fused = self.hidden_norm(self.fc(ctx_features))               # [S, H]
        ctx_pos = torch.arange(S, device=device)
        cos_c, sin_c = precompute_rope(cfg.head_dim, ctx_pos, cfg.rope_base)
        cos_c, sin_c = cos_c[:, None, :], sin_c[:, None, :]           # [S, 1, hd/2]

        # ---- block path ----
        blk_pos = anchor_pos[:, None] + torch.arange(B, device=device)[None, :]   # [A, B]
        h = self.embed_tokens(block_token_ids)                                     # [A, B, H]

        allow_ctx = ctx_pos[None, None, None, :] < anchor_pos[:, None, None, None]   # [A, 1, 1, S]
        fmask = torch.zeros(A, 1, B, S + B, dtype=torch.float32, device=device)
        fmask[:, :, :, :S] = torch.where(allow_ctx, 0.0, float("-inf"))
        # block columns (S:) stay 0 -> fully visible (bidirectional)

        for layer in self.layers:
            # -- context K/V (from fused features, like build_dflash_kv_cache) --
            _, kk, vk = layer.attn.project(fused)                     # [S, kv, hd]
            kk = apply_rope(kk, cos_c, sin_c)
            # -- block Q/K/V --
            hb = layer.attn_norm(h.reshape(A * B, H))
            if cfg.variant == "dflash2":
                dyn = layer.attn_conv_proj(hb)
                hb = dflash2_conv(hb, dyn, layer.attn_conv_base, 0)
            else:
                dyn = None
            qb, kb, vb = layer.attn.project(hb)                       # [A*B, heads, hd]
            blk_pos_flat = blk_pos.reshape(A * B)
            cos_b, sin_b = precompute_rope(cfg.head_dim, blk_pos_flat, cfg.rope_base)
            cos_b, sin_b = cos_b[:, None, :], sin_b[:, None, :]       # [A*B, 1, hd/2]
            qb = apply_rope(qb, cos_b, sin_b)
            kb = apply_rope(kb, cos_b, sin_b)

            n_head, n_kv, hd = layer.attn.n_head, layer.attn.n_kv, layer.attn.head_dim
            # assemble K/V: [A, S+B, kv_heads, hd]
            K = torch.cat([kk[None].expand(A, S, n_kv, hd),
                           kb.reshape(A, B, n_kv, hd)], dim=1)
            V = torch.cat([vk[None].expand(A, S, n_kv, hd),
                           vb.reshape(A, B, n_kv, hd)], dim=1)
            Q = qb.reshape(A, B, n_head, hd)

            if n_head != n_kv:
                reps = n_head // n_kv
                K = K.repeat_interleave(reps, dim=2)
                V = V.repeat_interleave(reps, dim=2)

            # [A, n_head, B, S+B]
            attn = F.scaled_dot_product_attention(
                Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
                attn_mask=fmask.to(Q.dtype))
            attn = attn.transpose(1, 2).reshape(A * B, n_head * hd)
            attn_out = layer.attn.o_proj(attn)                        # [A*B, H]

            if cfg.variant == "dflash2":
                attn_out = dflash2_conv(attn_out, dyn, layer.attn_conv_base, 1)

            h_flat = h.reshape(A * B, H)
            h_flat = h_flat + attn_out
            # FFN sublayer
            cur = layer.ffn_norm(h_flat)
            if cfg.variant == "dflash2":
                dyn_f = layer.ffn_conv_proj(cur)
                cur = dflash2_conv(cur, dyn_f, layer.ffn_conv_base, 0)
            ffn_out = layer.ffn(cur)
            if cfg.variant == "dflash2":
                ffn_out = dflash2_conv(ffn_out, dyn_f, layer.ffn_conv_base, 1)
            h_flat = h_flat + ffn_out
            h = h_flat.reshape(A, B, H)

        return h

    # ------------------------------------------------------------------
    # Heads
    # ------------------------------------------------------------------

    def base_logits(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., H] -> logits [..., V] via the frozen target lm_head."""
        return self.lm_head(self.output_norm(h))

    def markov_bias(self, prev_ids: torch.Tensor) -> torch.Tensor:
        """prev_ids: [...] int -> [..., V] additive bigram correction."""
        return self.markov_w2(self.markov_w1(prev_ids))

    def selector_hidden_proj(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., H] -> [..., rank] (the H(.) gate, on the output-normed hidden)."""
        return self.selector_hidden(self.output_norm(h))

    def selector_scores(
        self,
        h: torch.Tensor,          # [A, B, H] final hidden
        logits: torch.Tensor,     # [A, B, V] base logits (fp32)
        anchor_ids: torch.Tensor, # [A] seed token ids
        top_k: Optional[int] = None,
    ):
        """Compute the runtime's lattice.

        Returns:
            candidates: [A, B, K]  top-k ids per row (sorted desc, like ggml_top_k)
            unary:      [A, B, K]  backbone logits at the candidates
            scores:     [A, B, K_succ, K_prev]
                        scores[:, j, i_succ, i_prev] = unary[j, i_succ]
                            + <B(cand_succ), A(cand_prev) * Hh(h_j)>
                        (element order matches the C++ lattice: the walk reads
                        row[succ + top_k * prev], i.e. dim -2 = successor,
                        dim -1 = predecessor.)
                        Column 1's predecessors are the anchor token (replicated
                        across all K columns, matching repeat_4d in C++); column
                        j>1's predecessors are row j-1's candidates.
        """
        cfg = self.cfg
        K = top_k or cfg.selector_top_k
        A, B, _ = h.shape
        cand_vals, candidates = torch.topk(logits.float(), K, dim=-1)   # [A, B, K]
        unary = cand_vals
        Hh = self.selector_hidden_proj(h).float()                       # [A, B, R]
        B_emb = self.selector_next(candidates).float()                  # [A, B, K, R]

        # predecessor ids per column j: anchor for j==1, candidates[j-1] for j>1
        prev_ids = candidates.new_empty(A, B, K)
        prev_ids[:, 0] = anchor_ids[:, None]        # unused column
        prev_ids[:, 1] = anchor_ids[:, None]        # anchor (replicated)
        if B > 2:
            prev_ids[:, 2:] = candidates[:, 1:B - 1]
        A_prev = self.selector_prev(prev_ids).float()                   # [A, B, K, R]

        scores = torch.zeros(A, B, K, K, dtype=torch.float32, device=h.device)
        for j in range(1, B):
            cond = A_prev[:, j] * Hh[:, j, None, :]                     # [A, K_prev, R]
            s = torch.einsum("asr,apr->asp", B_emb[:, j], cond)         # [A, K_succ, K_prev]
            s = s + unary[:, j, :, None]                                # + U(succ)
            scores[:, j] = s
        return candidates, unary, scores

    # ------------------------------------------------------------------
    # Inference helpers (mirror the C++ runtime exactly)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def draft_dspark(
        self,
        ctx_features: torch.Tensor,   # [S, L*H]
        seed_id: int,
        anchor_pos: int,
        block_size: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive markov rollout — mirrors build_dspark_logits.

        Returns (draft_tokens [B], corrected_logits [B, V]).
        """
        cfg = self.cfg
        B = block_size or cfg.block_size
        device = ctx_features.device
        ids = torch.full((1, B), cfg.mask_token_id, dtype=torch.long, device=device)
        ids[0, 0] = seed_id
        h = self.forward_blocks(ctx_features, ids, torch.tensor([anchor_pos], device=device))
        base = self.base_logits(h)[0].float()                           # [B, V]
        prev = int(seed_id)
        draft, corrected = [], []
        for i in range(B):
            row = base[i] + self.markov_bias(torch.tensor(prev, device=device))
            if temperature > 1e-5:
                tok = int(torch.multinomial(F.softmax(row / temperature, dim=-1), 1))
            else:
                tok = int(row.argmax())
            draft.append(tok)
            corrected.append(row)
            prev = tok
        return torch.tensor(draft, device=device), torch.stack(corrected)

    @torch.no_grad()
    def draft_dflash2(
        self,
        ctx_features: torch.Tensor,   # [S, L*H]
        seed_id: int,
        anchor_pos: int,
        block_size: Optional[int] = None,
        temperature: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ):
        """Greedy/sampled lattice walk — mirrors llama.cpp's decode readback.

        Returns (draft_tokens [B-1], candidates [B, K], scores [B, K_prev, K_succ]).
        """
        cfg = self.cfg
        B = block_size or cfg.block_size
        device = ctx_features.device
        ids = torch.full((1, B), cfg.mask_token_id, dtype=torch.long, device=device)
        ids[0, 0] = seed_id
        h = self.forward_blocks(ctx_features, ids, torch.tensor([anchor_pos], device=device))
        logits = self.base_logits(h)                                    # [1, B, V]
        candidates, _, scores = self.selector_scores(
            h, logits, torch.tensor([seed_id], device=device), top_k=cfg.selector_top_k)
        candidates = candidates[0]                                      # [B, K]
        scores = scores[0]                                              # [B, K_prev, K_succ]

        draft = []
        prev = 0
        for pos in range(1, B):
            row = scores[pos][:, prev]                                  # [K_succ]
            if temperature > 1e-5:
                idx = int(torch.multinomial(F.softmax(row / temperature, dim=-1), 1,
                                            generator=generator))
            else:
                idx = int(row.argmax())
            draft.append(int(candidates[pos, idx]))
            prev = idx
        return torch.tensor(draft, device=device), candidates, scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_head(self, path: Path):
        sd = {"config": self.cfg.to_dict()}
        sd.update({
            "fc": self.fc.state_dict(),
            "hidden_norm": self.hidden_norm.state_dict(),
            "layers": self.layers.state_dict(),
            "output_norm": self.output_norm.state_dict(),
        })
        if self.cfg.variant == "dspark":
            sd["markov_w1"] = self.markov_w1.state_dict()
            sd["markov_w2"] = self.markov_w2.state_dict()
            if self.cfg.use_confidence_head:
                sd["conf_head"] = self.conf_head.state_dict()
        else:
            sd["selector_prev"] = self.selector_prev.state_dict()
            sd["selector_next"] = self.selector_next.state_dict()
            sd["selector_hidden"] = self.selector_hidden.state_dict()
        torch.save(sd, path)
        logger.info("head_saved", path=str(path), variant=self.cfg.variant)

    @classmethod
    def load_head(cls, path: Path, embed_weight: torch.Tensor,
                  lm_head_weight: Optional[torch.Tensor] = None,
                  map_location: str = "cpu") -> "XSparkDraftModel":
        sd = torch.load(path, map_location=map_location, weights_only=False)
        cfg = XSparkConfig(**sd["config"])
        model = cls(cfg, embed_weight, lm_head_weight)
        model.fc.load_state_dict(sd["fc"])
        model.hidden_norm.load_state_dict(sd["hidden_norm"])
        model.layers.load_state_dict(sd["layers"])
        model.output_norm.load_state_dict(sd["output_norm"])
        if cfg.variant == "dspark":
            model.markov_w1.load_state_dict(sd["markov_w1"])
            model.markov_w2.load_state_dict(sd["markov_w2"])
            if cfg.use_confidence_head and "conf_head" in sd:
                model.conf_head.load_state_dict(sd["conf_head"])
        else:
            model.selector_prev.load_state_dict(sd["selector_prev"])
            model.selector_next.load_state_dict(sd["selector_next"])
            model.selector_hidden.load_state_dict(sd["selector_hidden"])
        return model
