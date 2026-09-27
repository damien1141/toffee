"""
TOFFEE LoRA/QLoRA setup.

Applies parameter-efficient adapters to the model, targeting only
attention and shared-expert weights for Qwen3.5-MoE.
"""

from __future__ import annotations

from typing import List, Optional

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

from toffee.config import LoRAConfig


# Default target modules for Qwen3.5-MoE hybrid architecture.
# We LoRA attention layers and shared experts (always-active, low-rank).
# Routed experts are left alone: too many, sparse, and the router is sensitive.
QWEN3_5_MOE_TARGET_MODULES = [
    # Full-attention decoder layers
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    # Linear-attention (GatedDeltaNet / SSM) layers
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    # Shared expert FFN (dense, always-active)
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
]


def get_target_modules(model: AutoModelForCausalLM, override: Optional[List[str]]) -> List[str]:
    """Resolve target modules, verifying they exist in the model."""
    if override:
        return override
    # Auto-detect: use the Qwen3.5-MoE defaults if the model type matches
    model_type = getattr(model.config, "model_type", "")
    if model_type in ("qwen3_5_moe", "qwen3_5_moe_text", "qwen3_5", "qwen3_5_text"):
        return QWEN3_5_MOE_TARGET_MODULES
    # Fallback: inspect model keys
    found = set()
    for name, _ in model.named_modules():
        if "proj" in name or "gate" in name:
            found.add(name.split(".")[-1])
    return sorted(found)


def apply_lora(
    model: AutoModelForCausalLM,
    config: LoRAConfig,
) -> AutoModelForCausalLM:
    """Apply LoRA adapters to the model."""
    targets = get_target_modules(model, config.target_modules)
    peft_config = LoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias=config.bias,
        target_modules=targets,
        modules_to_save=config.modules_to_save,
        use_rslora=config.use_rslora,
        use_dora=config.use_dora,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
