"""
TOFFEE quantization configuration.

Port of ONYX's config loading with fallback for unregistered model types.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import AutoConfig

# ---------------------------------------------------------------------------
# Register qwen4_exp_text config for Whittle / Qwen3.8-Flash-Next models
# ---------------------------------------------------------------------------
try:
    from transformers import CONFIG_MAPPING
    from transformers.configuration_utils import PretrainedConfig

    class Qwen4ExpTextConfig(PretrainedConfig):
        model_type = "qwen4_exp_text"
        architectures = ["Qwen4ExpForCausalLM"]
        vocab_size = 248320
        hidden_size = 2048
        num_hidden_layers = 40
        num_attention_heads = 16
        num_key_value_heads = 2
        intermediate_size = 512
        num_experts = 180
        num_experts_per_tok = 8
        moe_intermediate_size = 512
        shared_expert_intermediate_size = 512
        max_position_embeddings = 262144
        rms_norm_eps = 1e-6
        hidden_act = "silu"
        initializer_range = 0.02
        use_cache = True
        tie_word_embeddings = False
        attention_bias = False
        attention_dropout = 0.0
        output_router_logits = False
        router_aux_loss_coef = 0.001
        layer_types = [
            "linear_attention", "linear_attention", "linear_attention", "full_attention"
        ] * 10
        head_dim = 256
        linear_num_key_heads = 16
        linear_num_value_heads = 32
        linear_key_head_dim = 128
        linear_value_head_dim = 128
        linear_conv_kernel_dim = 4
        mamba_ssm_dtype = "float32"
        partial_rotary_factor = 0.25
        attn_output_gate = True
        full_attention_interval = 4
        hc_count = 4
        hc_lowrank = 320
        indexer_n_heads = 4
        indexer_head_dim = 128
        indexer_budget = 262144
        indexer_compress_ratio = 4
        indexer_kv_heads = 1
        output_gate_type = "sigmoid"
        ple_layer_ids = [2]
        ngram_size = 3
        heads_per_ngram = 4
        ple_conv_kernel_size = 4
        split_ngram_parts = 5
        ple_embed_dim = 2048
        num_embeddings_per_layer_input = 256
        make_ngram_vocab_size_divisible_by = 128
        ngram_vocab_size_base = 4880000

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    if "qwen4_exp_text" not in CONFIG_MAPPING._extra_content:
        CONFIG_MAPPING.register("qwen4_exp_text", Qwen4ExpTextConfig)
        CONFIG_MAPPING.register("qwen4_exp", Qwen4ExpTextConfig)
except Exception:
    pass


def load_config_with_fallback(model_name: str) -> Any:
    """Load model config, with a fallback for unregistered model types like qwen4_exp_text."""
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except ValueError:
        # Fallback for model types not registered in transformers (e.g. qwen4_exp_text).
        # Load the raw config.json directly and construct the config object manually.
        cfg_path = Path(model_name) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_name}")
        with open(cfg_path, "r") as f:
            cfg_dict = json.load(f)

        # Use the locally-registered Qwen4ExpTextConfig for Whittle / qwen4_exp models.
        try:
            cfg = Qwen4ExpTextConfig.from_dict(cfg_dict)
        except Exception:
            # Last resort: re-raise the original error.
            return AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Ensure nested text_config is also a config object, not a raw dict.
    if isinstance(getattr(cfg, "text_config", None), dict):
        cfg.text_config = Qwen4ExpTextConfig.from_dict(cfg.text_config)
    return cfg


def _get_attr_from_instance(config: Any, name: str) -> Any:
    """Get an attribute from a config instance, checking __dict__ first to avoid class defaults."""
    if name in config.__dict__:
        return config.__dict__[name]
    return None


def get_num_hidden_layers(config: Any) -> int:
    """Extract num_hidden_layers from config, handling nested text_config."""
    num_hidden_layers = _get_attr_from_instance(config, "num_hidden_layers")
    if num_hidden_layers is None:
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None:
            num_hidden_layers = _get_attr_from_instance(text_cfg, "num_hidden_layers")
    if num_hidden_layers is None:
        raise ValueError(
            f"Could not determine num_hidden_layers from config {type(config).__name__}. "
            f"Available attributes: {[a for a in dir(config) if not a.startswith('_')]}"
        )
    return int(num_hidden_layers)


def get_hidden_size(config: Any) -> int:
    """Extract hidden_size from config, handling nested text_config."""
    hidden_size = _get_attr_from_instance(config, "hidden_size")
    if hidden_size is None:
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None:
            hidden_size = _get_attr_from_instance(text_cfg, "hidden_size")
    if hidden_size is None:
        raise ValueError("Could not determine hidden_size from config")
    return int(hidden_size)


def get_intermediate_size(config: Any) -> int:
    """Extract intermediate_size from config, handling nested text_config."""
    intermediate_size = _get_attr_from_instance(config, "intermediate_size")
    if intermediate_size is None:
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None:
            intermediate_size = _get_attr_from_instance(text_cfg, "intermediate_size")
    if intermediate_size is None:
        # Fall back to MoE intermediate size
        intermediate_size = _get_attr_from_instance(config, "moe_intermediate_size")
        if intermediate_size is None and hasattr(config, "text_config"):
            intermediate_size = _get_attr_from_instance(config.text_config, "moe_intermediate_size")
    if intermediate_size is None:
        raise ValueError("Could not determine intermediate_size from config")
    return int(intermediate_size)


def get_num_experts(config: Any) -> int:
    """Extract num_experts from config, handling various key names."""
    num_experts = (
        _get_attr_from_instance(config, "num_experts")
        or _get_attr_from_instance(config, "n_routed_experts")
        or _get_attr_from_instance(config, "num_local_experts")
    )
    if num_experts is None and hasattr(config, "text_config"):
        tc = config.text_config
        num_experts = (
            _get_attr_from_instance(tc, "num_experts")
            or _get_attr_from_instance(tc, "n_routed_experts")
            or _get_attr_from_instance(tc, "num_local_experts")
        )
    if num_experts is None:
        num_experts = 128  # sensible default
    return int(num_experts)


def get_layer_types(config: Any) -> Optional[list[str]]:
    """Extract layer_types from config, handling nested text_config."""
    layer_types = _get_attr_from_instance(config, "layer_types")
    if layer_types is None and hasattr(config, "text_config"):
        layer_types = _get_attr_from_instance(config.text_config, "layer_types")
    return layer_types
