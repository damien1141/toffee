"""
TOFFEE quantization configuration.

Port of ONYX's config loading with fallback for unregistered model types.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import AutoConfig


def load_config_with_fallback(model_name: str) -> Any:
    """Load model config, with a fallback for unregistered model types like qwen4_exp_text."""
    try:
        return AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except ValueError:
        pass

    # Fallback for model types not registered in transformers (e.g. qwen4_exp_text).
    # Load the raw config.json directly and construct the config object manually.
    cfg_path = Path(model_name) / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_name}")
    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)

    # qwen4_exp shares the Qwen3.5-MoE layer layout; try that first.
    try:
        from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig
        return Qwen3_5MoeTextConfig.from_dict(cfg_dict)
    except Exception:
        pass
    try:
        from transformers.models.qwen3_5 import Qwen3_5TextConfig
        return Qwen3_5TextConfig.from_dict(cfg_dict)
    except Exception:
        pass

    # Last resort: re-raise the original error.
    return AutoConfig.from_pretrained(model_name, trust_remote_code=True)


def get_num_hidden_layers(config: Any) -> int:
    """Extract num_hidden_layers from config, handling nested text_config."""
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None:
            num_hidden_layers = getattr(text_cfg, "num_hidden_layers", None)
    if num_hidden_layers is None:
        raise ValueError(
            f"Could not determine num_hidden_layers from config {type(config).__name__}. "
            f"Available attributes: {[a for a in dir(config) if not a.startswith('_')]}"
        )
    return int(num_hidden_layers)


def get_hidden_size(config: Any) -> int:
    """Extract hidden_size from config, handling nested text_config."""
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None:
            hidden_size = getattr(text_cfg, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Could not determine hidden_size from config")
    return int(hidden_size)


def get_intermediate_size(config: Any) -> int:
    """Extract intermediate_size from config, handling nested text_config."""
    intermediate_size = getattr(config, "intermediate_size", None)
    if intermediate_size is None:
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None:
            intermediate_size = getattr(text_cfg, "intermediate_size", None)
    if intermediate_size is None:
        # Fall back to MoE intermediate size
        intermediate_size = getattr(config, "moe_intermediate_size", None)
        if intermediate_size is None and hasattr(config, "text_config"):
            intermediate_size = getattr(config.text_config, "moe_intermediate_size", None)
    if intermediate_size is None:
        raise ValueError("Could not determine intermediate_size from config")
    return int(intermediate_size)


def get_num_experts(config: Any) -> int:
    """Extract num_experts from config, handling various key names."""
    num_experts = (
        getattr(config, "num_experts", None)
        or getattr(config, "n_routed_experts", None)
        or getattr(config, "num_local_experts", None)
    )
    if num_experts is None and hasattr(config, "text_config"):
        tc = config.text_config
        num_experts = (
            getattr(tc, "num_experts", None)
            or getattr(tc, "n_routed_experts", None)
            or getattr(tc, "num_local_experts", None)
        )
    if num_experts is None:
        num_experts = 128  # sensible default
    return int(num_experts)


def get_layer_types(config: Any) -> Optional[list[str]]:
    """Extract layer_types from config, handling nested text_config."""
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None and hasattr(config, "text_config"):
        layer_types = getattr(config.text_config, "layer_types", None)
    return layer_types
