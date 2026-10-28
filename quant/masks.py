"""
TOFFEE quantization mask helpers.

Ported from ONYX: build_4d_causal_mask, layer_wants_2d_mask, layer_needs_rope.
"""

from __future__ import annotations

from typing import Optional

import torch


def build_4d_causal_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """Convert [batch, seq_len] 0/1 mask → [batch, 1, q_len, k_len] additive mask.

    If `sliding_window` is given, keys beyond `sliding_window` positions to the
    left of each query are masked out.
    """
    bsz, tgt_len = attention_mask.shape
    min_dtype = torch.finfo(dtype).min

    mask = torch.full((tgt_len, tgt_len), min_dtype, device=device, dtype=dtype)
    mask_cond = torch.arange(mask.size(-1), device=device)
    causal = mask_cond < (mask_cond + 1).view(mask.size(-1), 1)
    mask.masked_fill_(causal, 0)
    if sliding_window is not None:
        window = mask_cond.view(-1, 1) - mask_cond.view(1, -1)
        mask.masked_fill_(window > sliding_window, min_dtype)

    mask = mask[None, None, :, :]
    mask = mask.expand(bsz, 1, tgt_len, tgt_len).contiguous()

    padding = (1 - attention_mask[:, None, None, :].to(dtype=dtype, device=device)) * min_dtype
    mask = mask + padding
    return mask


def layer_wants_2d_mask(layer_module) -> bool:
    """
    Hybrid models (e.g. Qwen3.5-MoE) mix `full_attention` blocks, which need a 4D
    additive causal mask, with linear-attention / SSM blocks, which need the raw
    2D [batch, seq_len] 0/1 mask.
    """
    block_type = getattr(layer_module, "block_type", None)
    if isinstance(block_type, str):
        if block_type == "full_attention":
            return False
        return True

    # Fall back to structural detection when block_type is absent.
    if hasattr(layer_module, "self_attn"):
        return False
    for attr in ("linear_attn", "mamba", "ssm", "mixer", "temporal_mixer"):
        if hasattr(layer_module, attr):
            return True
    return False


def layer_needs_rope(layer_module, layer_rope_theta, layer_idx) -> bool:
    """
    Whether this layer applies RoPE and must receive `position_embeddings`.

    - Hybrid block-typed models: only `full_attention` blocks rotate.
    - Muse Glimmer: `layer_rope_theta[layer_idx] == 0` marks a global NoPE layer.
    - Standard attention models (no block_type, no per-layer rope list): always True.
    """
    block_type = getattr(layer_module, "block_type", None)
    if isinstance(block_type, str):
        return block_type == "full_attention"
    if layer_rope_theta is not None and layer_idx < len(layer_rope_theta):
        return layer_rope_theta[layer_idx] != 0
    return True
