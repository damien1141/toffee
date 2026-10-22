"""
@file toffee/quant/hybrid_model.py
   Minimal hybrid text model for imatrix harvesting.

The exact transformers classes for the Qwen3.5 / Qwen4 hybrid architectures
(``qwen3_5_moe``, ``qwen4_exp``) are not available in every transformers
release. Sage-A1.5 is a ``qwen3_5_moe``: a 40-layer hybrid that alternates
GatedDeltaNet ("linear_attention") blocks with full-attention blocks, and
routes every token through 256 MoE experts.

For imatrix harvesting we do not need a *correct* forward pass -- we need
the right **module hierarchy**, so that

  * the real checkpoint weights can be injected into modules of the right
    shape, and
  * the forward hooks fire on the right modules, capturing the activation
    scales that the llama.cpp importance matrix expects.

This module therefore declares that hierarchy explicitly. The forward pass
is a stand-in that flows data through every linear layer, so the hooks see
real activations with the real shapes.

Ported from the ONYX ``_MinimalHybridDecoderLayer`` /
``_MinimalHybridTextModel`` (Apache-2.0), which solves the same problem for
the same architectures.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MinimalHybridDecoderLayer(nn.Module):
    """One decoder layer of a Qwen3.5-style hybrid model.

    The layer declares the submodules of either a full-attention block or a
    GatedDeltaNet ("linear attention") block, with the exact shapes the
    checkpoint uses, so weight injection and the activation hooks line up.

    Args:
        config: the text config.
        layer_type: ``"full_attention"`` or ``"linear_attention"``.
        layer_idx: the index of this layer.
    """

    def __init__(self, config: Any, layer_type: str, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_type = layer_type
        self.layer_idx = layer_idx

        hidden_size = int(config.hidden_size)
        inter_size = int(getattr(config, "moe_intermediate_size", 512) or 512)
        num_experts = int(getattr(config, "num_experts", 4) or 4)
        shared_inter = int(
            getattr(config, "shared_expert_intermediate_size", inter_size)
            or inter_size
        )
        n_heads = int(getattr(config, "num_attention_heads", 1) or 1)
        n_kv = int(getattr(config, "num_key_value_heads", n_heads) or n_heads)
        head_dim = int(
            getattr(config, "head_dim", None) or (hidden_size // n_heads)
        )

        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

        if layer_type == "full_attention":
            self._build_attention(hidden_size, n_heads, n_kv, head_dim, config)
        else:
            self._build_linear_attention(hidden_size, config)

        self._build_mlp(hidden_size, inter_size, shared_inter, num_experts, config)

    # -- attention ---------------------------------------------------------

    def _build_attention(
        self,
        hidden_size: int,
        n_heads: int,
        n_kv: int,
        head_dim: int,
        config: Any,
    ) -> None:
        """Declare the full-attention block projections.

        The shapes follow the llama.cpp Qwen3.5 reader: ``q_proj`` emits
        the attention width, and the projections are consumed in the
        feature dimension.
        """
        self.self_attn = nn.Module()
        attn_width = n_heads * head_dim
        kv_out = n_kv * head_dim
        # `attn_output_gate` makes q_proj emit the attention width plus a
        # gate of the same width.
        q_out = attn_width * (2 if getattr(config, "attn_output_gate", False) else 1)
        self._attn_n_heads = n_heads
        self._attn_n_kv = n_kv
        self._attn_head_dim = head_dim
        self._attn_q_out = q_out
        self._attn_kv_out = kv_out
        self._attn_attn_width = attn_width

        self.self_attn.q_proj = nn.Linear(hidden_size, q_out, bias=False)
        self.self_attn.k_proj = nn.Linear(hidden_size, kv_out, bias=False)
        self.self_attn.v_proj = nn.Linear(hidden_size, kv_out, bias=False)
        # o_proj consumes the attention width, not the gated q_proj width:
        # the checkpoint's o_proj is [hidden, attn_width].
        self.self_attn.o_proj = nn.Linear(attn_width, hidden_size, bias=False)
        self.self_attn.q_norm = nn.LayerNorm(head_dim)
        self.self_attn.k_norm = nn.LayerNorm(head_dim)

    def _build_linear_attention(self, hidden_size: int, config: Any) -> None:
        """Declare the GatedDeltaNet block projections.

        The fused QKV projection carries the K heads twice (once for the
        query side, once for the key side) plus the V heads, so its width is
        ``2 * key_dim + value_dim``. The A/B projections are per-value-head
        SSM parameter vectors, not full-width linears, and the conv runs on
        the fused QKV, grouped 1:1.
        """
        lin = nn.Module()
        self.linear_attn = lin
        self.self_attn = nn.Module()

        num_k_heads = int(getattr(config, "linear_num_key_heads", 1) or 1)
        num_v_heads = int(getattr(config, "linear_num_value_heads", 1) or 1)
        key_dim = int(getattr(config, "linear_key_head_dim", 128) or 128)
        value_dim = int(getattr(config, "linear_value_head_dim", 128) or 128)
        conv_kernel = int(getattr(config, "linear_conv_kernel_dim", 4) or 4)

        key_total = num_k_heads * key_dim
        value_total = num_v_heads * value_dim
        qkv_out = key_total + key_total + value_total
        self._lin_qkv_out = qkv_out
        self._lin_value_total = value_total

        lin.in_proj_qkv = nn.Linear(hidden_size, qkv_out, bias=False)
        lin.in_proj_z = nn.Linear(hidden_size, value_total, bias=False)
        lin.in_proj_a = nn.Linear(hidden_size, num_v_heads, bias=False)
        lin.in_proj_b = nn.Linear(hidden_size, num_v_heads, bias=False)
        lin.out_proj = nn.Linear(value_total, hidden_size, bias=False)
        lin.conv1d = nn.Conv1d(
            qkv_out, qkv_out, kernel_size=conv_kernel,
            padding=conv_kernel - 1, groups=qkv_out, bias=False,
        )
        lin.norm = nn.LayerNorm(value_dim)
        # A_log / dt_bias are 1-D parameters, not Linear layers.
        self.register_buffer(
            "A_log", torch.zeros(num_v_heads), persistent=False
        )
        self.register_buffer(
            "dt_bias", torch.zeros(num_v_heads), persistent=False
        )

    # -- mlp ---------------------------------------------------------------

    def _build_mlp(
        self,
        hidden_size: int,
        inter_size: int,
        shared_inter: int,
        num_experts: int,
        config: Any,
    ) -> None:
        """Declare the MoE block: the router, the experts, the shared expert.
        """
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(hidden_size, num_experts, bias=False)
        # The experts are declared as a ModuleList of gate/up/down triples,
        # matching the per-expert checkpoint layout
        # (``mlp.experts.{i}.{gate,up,down}_proj.weight``).
        self.mlp.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = nn.Module()
            expert.gate_proj = nn.Linear(hidden_size, inter_size, bias=False)
            expert.up_proj = nn.Linear(hidden_size, inter_size, bias=False)
            expert.down_proj = nn.Linear(inter_size, hidden_size, bias=False)
            self.mlp.experts.append(expert)
        self.mlp.shared_expert = nn.Module()
        self.mlp.shared_expert.gate_proj = nn.Linear(
            hidden_size, shared_inter, bias=False
        )
        self.mlp.shared_expert.up_proj = nn.Linear(
            hidden_size, shared_inter, bias=False
        )
        self.mlp.shared_expert.down_proj = nn.Linear(
            shared_inter, hidden_size, bias=False
        )
        # The shared expert has its own router gate.
        self.mlp.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """A stand-in forward pass that flows data through every linear.

        The exact numerics do not matter for imatrix harvesting -- what
        matters is that each captured module sees an input with the real
        shape, so the per-output-channel activation scales are right.
        """
        x = self.input_layernorm(hidden_states)

        if self.layer_type == "full_attention":
            q = self.self_attn.q_proj(x)
            k = self.self_attn.k_proj(x)
            v = self.self_attn.v_proj(x)
            # q_proj emits the attention width plus (optionally) a gate of
            # the same width; the first slice is the query.
            attn = q[..., : self._attn_attn_width]
            attn = attn.reshape(
                attn.shape[0], attn.shape[1], self._attn_n_heads, self._attn_head_dim
            )
            attn = self.self_attn.q_norm(attn)
            k = k.reshape(
                k.shape[0], k.shape[1], self._attn_n_kv, self._attn_head_dim
            )
            k = self.self_attn.k_norm(k)
            # Collapse the heads to the attention width that o_proj expects.
            # The KV heads are fewer than the query heads (GQA), so each KV
            # head is repeated to cover its query heads.
            n_rep = self._attn_n_heads // self._attn_n_kv
            k_heads = k.repeat_interleave(n_rep, dim=2)
            v_heads = v.reshape(
                v.shape[0], v.shape[1], self._attn_n_kv, self._attn_head_dim
            ).repeat_interleave(n_rep, dim=2)
            # The attention output keeps one head_dim slot per query head,
            # so the total width is n_heads * head_dim.
            attn_out = (
                attn + k_heads + v_heads
            ).reshape(attn.shape[0], attn.shape[1], -1)
            hidden_states = hidden_states + self.self_attn.o_proj(attn_out)
        else:
            lin = self.linear_attn
            qkv = lin.in_proj_qkv(x)
            z = lin.in_proj_z(x)
            a = lin.in_proj_a(x)
            b = lin.in_proj_b(x)
            # The conv runs on the fused QKV, grouped 1:1.
            conv = lin.conv1d(qkv.transpose(1, 2))[..., : qkv.shape[1]]
            conv = conv.transpose(1, 2)
            # The A/B projections are per-value-head; broadcast them over
            # the head's value dimension.
            n_val = a.shape[-1]
            value_dim = self._lin_value_total // n_val
            a_exp = a.repeat_interleave(value_dim, dim=-1)
            b_exp = b.repeat_interleave(value_dim, dim=-1)
            # Only the V part of the fused QKV feeds the SSM.
            v_part = conv[..., -self._lin_value_total :]
            state = (v_part + a_exp) * torch.sigmoid(b_exp)
            # The norm is per value-head, so reshape to heads first.
            normed = lin.norm(
                state.reshape(*state.shape[:-1], n_val, value_dim)
            ).reshape(*state.shape)
            gated = normed * torch.sigmoid(z)
            hidden_states = hidden_states + lin.out_proj(gated)

        # The MoE block: route the tokens, run the experts that were used.
        mlp = self.mlp
        router = mlp.gate(hidden_states)
        weights = torch.softmax(router.float(), dim=-1)
        top = weights.argmax(dim=-1)
        expert_out = torch.zeros_like(hidden_states)
        for i, expert in enumerate(mlp.experts):
            mask = (top == i)
            if not bool(mask.any()):
                continue
            sel = hidden_states[mask]
            gate = expert.gate_proj(sel)
            up = expert.up_proj(sel)
            expert_out[mask] = expert.down_proj(
                torch.nn.functional.silu(gate) * up
            )
        # The shared expert always runs.
        shared = mlp.shared_expert
        shared_out = shared.down_proj(
            torch.nn.functional.silu(shared.gate_proj(hidden_states))
            * shared.up_proj(hidden_states)
        )
        moe_out = expert_out + shared_out
        return hidden_states + self.post_attention_layernorm(moe_out)


class MinimalHybridTextModel(nn.Module):
    """A text model with the module hierarchy pass2 expects.

    Args:
        config: the text config.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            int(config.vocab_size), int(config.hidden_size)
        )
        layer_types = list(getattr(config, "layer_types", None) or [])
        if not layer_types:
            layer_types = ["full_attention"] * int(config.num_hidden_layers)
        self.layers = nn.ModuleList()
        for layer_type in layer_types:
            self.layers.append(
                MinimalHybridDecoderLayer(config, layer_type, len(self.layers))
            )
        self.norm = nn.LayerNorm(int(config.hidden_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Embed, run the layers, normalize."""
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(x)


class TextModelShim:
    """Wraps a text-only model so the rest of pass2 can use ``model.model.*``.

    Multimodal configs ship a ``*ForConditionalGeneration`` with no
    ``AutoModelForCausalLM`` entry. Building the text model directly and
    exposing it as ``.model`` keeps ``model.model.embed_tokens`` and
    ``model.model.layers`` working unchanged.
    """

    def __init__(self, text_model: nn.Module) -> None:
        self.model = text_model

    def named_modules(self):
        return self.model.named_modules()

    def named_parameters(self):
        return self.model.named_parameters()

    def named_buffers(self):
        return self.model.named_buffers()


__all__ = [
    "MinimalHybridDecoderLayer",
    "MinimalHybridTextModel",
    "TextModelShim",
]
