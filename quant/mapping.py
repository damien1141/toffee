"""
TOFFEE quantization tensor mapping.

Ported from ONYX: infer_gguf_tensor_category, APEX tensor map generators.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional


def infer_gguf_tensor_category(hf_name: str, layer_idx: int) -> Optional[str]:
    """Dynamically infers the standard llama.cpp GGUF tensor name from an HF module name.

    Handles two hybrid schemas:
      * qwen3_5 (Qwen3.5 dense hybrid): GatedDeltaNet `linear_attn` + `self_attn`
        + dense `mlp`. llama.cpp names: attn_qkv/attn_q/attn_k/attn_v/attn_gate/
        attn_output/attn_q_norm/attn_k_norm, ffn_gate/ffn_up/ffn_down (DENSE, no
        _exps), ssm_a/ssm_alpha/ssm_beta/ssm_conv1d/ssm_dt/ssm_norm/ssm_out.
      * Qwen/Mistral-style MoE: ffn_*_exps / ffn_*_shexp.
    """
    local_name = hf_name.lower()

    # --- qwen3_5 GatedDeltaNet (linear_attention) block ---
    if "linear_attn" in local_name or "gateddeltanet" in local_name:
        if "in_proj_qkv" in local_name or "qkv" in local_name:
            return "attn_qkv"
        if "in_proj_z" in local_name:
            return "attn_gate"
        if "in_proj_a" in local_name:
            return "ssm_alpha"
        if "in_proj_b" in local_name:
            return "ssm_beta"
        if "out_proj" in local_name:
            return "ssm_out"
        if "conv1d" in local_name:
            return "ssm_conv1d"
        if "norm" in local_name:
            return "ssm_norm"

    # --- qwen3_5 self_attention (full_attention) block ---
    if "self_attn" in local_name:
        if "q_proj" in local_name or "qkv" in local_name:
            return "attn_q"
        if "k_proj" in local_name:
            return "attn_k"
        if "v_proj" in local_name:
            return "attn_v"
        if "o_proj" in local_name or "out_proj" in local_name:
            return "attn_output"
        if "q_norm" in local_name:
            return "attn_q_norm"
        if "k_norm" in local_name:
            return "attn_k_norm"

    # --- Routed Experts (MoE) ---
    if "expert" in local_name and "shared" not in local_name:
        if "w1" in local_name or "gate" in local_name:
            return "ffn_gate_exps"
        if "w3" in local_name or "up" in local_name:
            return "ffn_up_exps"
        if "w2" in local_name or "down" in local_name:
            return "ffn_down_exps"

    # --- Router (MoE) ---
    if "gate" in local_name and "ffn" not in local_name and "expert" not in local_name:
        return "ffn_gate_inp"

    # --- Shared Experts (MoE) ---
    if "shared" in local_name or "shexp" in local_name:
        if "w1" in local_name or "gate" in local_name:
            return "ffn_gate_shexp"
        if "w3" in local_name or "up" in local_name:
            return "ffn_up_shexp"
        if "w2" in local_name or "down" in local_name:
            return "ffn_down_shexp"

    # --- Dense MLP (no experts) ---
    if "mlp" in local_name or "feed_forward" in local_name or "ffn" in local_name:
        if "gate" in local_name or "w1" in local_name:
            return "ffn_gate"
        if "up" in local_name or "w3" in local_name:
            return "ffn_up"
        if "down" in local_name or "w2" in local_name:
            return "ffn_down"

    # --- Layer norms ---
    if "input_layernorm" in local_name or "attn_norm" in local_name:
        return "attn_norm"
    if "post_attention_layernorm" in local_name or "post_attention_norm" in local_name:
        return "post_attention_norm"

    # --- qwen4_exp hyper-connections (HC) ---
    if "hc_attn_norm" in local_name:
        return "hc_attn_norm"
    if "hc_attn_down" in local_name:
        return "hc_attn_down"
    if "hc_attn_up" in local_name:
        return "hc_attn_up"
    if "hc_attn_inject" in local_name:
        return "hc_attn_inject"
    if "hc_ffn_norm" in local_name:
        return "hc_ffn_norm"
    if "hc_ffn_down" in local_name:
        return "hc_ffn_down"
    if "hc_ffn_up" in local_name:
        return "hc_ffn_up"
    if "hc_ffn_inject" in local_name:
        return "hc_ffn_inject"

    # --- qwen4_exp PLE n-gram memory ---
    if "ple_key" in local_name and "norm" not in local_name:
        return "ple_key"
    if "ple_value" in local_name and "norm" not in local_name:
        return "ple_value"
    if "ple_norm_key" in local_name:
        return "ple_norm_key"
    if "ple_norm_query" in local_name:
        return "ple_norm_query"
    if "ple_norm_conv" in local_name:
        return "ple_norm_conv"
    if "ple_conv1d" in local_name:
        return "ple_conv1d"

    # --- qwen4_exp indexer ---
    if "indexer_q_proj" in local_name or "index_qk_proj" in local_name:
        return "indexer_q_proj"
    if "indexer_k_proj" in local_name:
        return "indexer_k_proj"
    if "indexer_q_norm" in local_name or "indexer_q_layernorm" in local_name:
        return "indexer_q_norm"
    if "indexer_k_norm" in local_name or "indexer_k_layernorm" in local_name:
        return "indexer_k_norm"

    # --- qwen4_exp output HC ---
    if "output_hc_norm" in local_name:
        return "output_hc_norm"
    if "output_hc_down" in local_name:
        return "output_hc_down"
    if "output_hc_up" in local_name:
        return "output_hc_up"

    # --- qwen4_exp per-layer token embedding (PLE table) ---
    if "per_layer_token_embd" in local_name:
        return "per_layer_token_embd"

    # --- Attention (generic) ---
    if "attn" in local_name or "attention" in local_name:
        if "qkv" in local_name:
            return "attn_qkv"
        if "q_proj" in local_name or "in_proj_q" in local_name:
            return "attn_q"
        if "k_proj" in local_name or "in_proj_k" in local_name:
            return "attn_k"
        if "v_proj" in local_name or "in_proj_v" in local_name:
            return "attn_v"
        if "o_proj" in local_name or "out_proj" in local_name:
            return "attn_output"

    # --- SSM / Mamba (Hybrid models) ---
    if "ssm" in local_name or "mamba" in local_name:
        if "in_proj" in local_name or "ssm_x" in local_name:
            return "ssm_in"
        if "conv1d" in local_name:
            return "ssm_conv1d"
        if "dt" in local_name:
            return "ssm_dt"
        if "out_proj" in local_name or "ssm_out" in local_name:
            return "ssm_out"

    return None


# Quant types that REQUIRE an importance-matrix entry in llama.cpp.
_IMATRIX_REQUIRED = frozenset({
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL",
    "Q2_K", "Q2_K_S", "Q3_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "TQ1_0", "TQ2_0",
})

_IMATRIX_FALLBACK = "Q6_K"


def _imatrix_covered_tensors(imatrix_path: Optional[Path]) -> Optional[set]:
    """Return the set of base tensor names present in the imatrix GGUF."""
    if imatrix_path is None or not imatrix_path.exists():
        return None
    try:
        import gguf as _gguf
    except ImportError:
        try:
            import sys
            sys.path.insert(0, "./llama.cpp")
            import gguf as _gguf
        except ImportError:
            return None
    try:
        reader = _gguf.GGUFReader(str(imatrix_path))
    except Exception as e:
        import logging
        logging.warning("pass3_imatrix_read_failed", error=str(e))
        return None
    covered = set()
    for t in reader.tensors:
        name = t.name
        if name.endswith(".in_sum2") or name.endswith(".counts"):
            covered.add(name[: -len(".in_sum2")])
        else:
            covered.add(name)
    return covered


def generate_dynamic_apex_tensor_map_from_gguf(
    f16_gguf_path: Path,
    output_path: Path,
    sensitivity_data: Dict[int, float],
    tier: str = "quality",
    imatrix_path: Optional[Path] = None,
) -> None:
    """Reads the actual GGUF file and generates a map dynamically based on real tensor names.

    When ``imatrix_path`` is supplied, any tensor assigned a very-low-bit quant
    that llama.cpp requires an importance matrix for is checked against the
    imatrix coverage. Tensors with no imatrix entry are bumped to ``Q6_K``.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info("pass3_map_start_gguf_read", gguf=str(f16_gguf_path))

    imatrix_covered = _imatrix_covered_tensors(imatrix_path)
    if imatrix_covered is not None:
        logger.info("pass3_imatrix_coverage", covered=len(imatrix_covered))

    sorted_layers = sorted(sensitivity_data.keys(), key=lambda i: sensitivity_data[i], reverse=True)
    edge_count = max(4, int(len(sorted_layers) * 0.25))
    near_edge_count = max(4, int(len(sorted_layers) * 0.25))

    edge_layers = set(sorted_layers[:edge_count])
    near_edge_layers = set(sorted_layers[edge_count : edge_count + near_edge_count])

    logger.info("pass3_dynamic_boundaries", edge=edge_layers, near_edge=near_edge_layers)

    try:
        import gguf
    except ImportError:
        try:
            import sys
            sys.path.insert(0, "./llama.cpp")
            import gguf
        except ImportError:
            raise RuntimeError("gguf python library not found. Please install it via 'pip install gguf'.")

    reader = gguf.GGUFReader(str(f16_gguf_path))
    tensor_names = [t.name for t in reader.tensors]

    with open(output_path, "w") as f:
        for name in tensor_names:
            if "norm" in name or "_norm" in name:
                f.write(f"{name}=Q8_0\n")
                continue

            match = re.match(r"blk\.(\d+)\.", name)
            if match:
                i = int(match.group(1))
                if i in edge_layers:
                    d = 0
                elif i in near_edge_layers:
                    d = 5
                else:
                    d = 10
            else:
                d = 0

            if tier == "quality":
                if d == 0: exp_q, shared_q, attn_q, attn_kv, ssm_q = "Q6_K", "Q8_0", "Q6_K", "Q6_K", "Q8_0"
                elif d == 5: exp_q, shared_q, attn_q, attn_kv, ssm_q = "Q5_K", "Q8_0", "Q6_K", "Q6_K", "Q8_0"
                else: exp_q, shared_q, attn_q, attn_kv, ssm_q = "IQ4_XS", "Q8_0", "Q6_K", "Q6_K", "Q8_0"
            elif tier == "balanced":
                if d == 0: exp_q, shared_q, attn_q, attn_kv, ssm_q = "Q6_K", "Q8_0", "Q6_K", "Q6_K", "Q6_K"
                else: exp_q, shared_q, attn_q, attn_kv, ssm_q = "Q5_K", "Q8_0", "Q5_K", "Q5_K", "Q6_K"
            elif tier == "compact":
                if d == 0: exp_q, shared_q, attn_q, attn_kv, ssm_q = "Q4_K", "Q6_K", "Q5_K", "Q5_K", "Q6_K"
                else: exp_q, shared_q, attn_q, attn_kv, ssm_q = "Q3_K", "Q6_K", "Q4_K", "Q4_K", "Q6_K"
            elif tier == "mini":
                if d == 0: exp_q, shared_q, attn_q, attn_kv, ssm_q = "Q3_K", "Q5_K", "Q4_K", "Q4_K", "Q5_K"
                elif d == 5: exp_q, shared_q, attn_q, attn_kv, ssm_q = "IQ2_S", "Q4_K", "Q3_K", "Q3_K", "Q4_K"
                else: exp_q, shared_q, attn_q, attn_kv, ssm_q = "IQ2_XXS", "Q4_K", "Q3_K", "Q3_K", "Q4_K"
            elif tier == "nano":
                if d == 0: exp_q, shared_q, attn_q, attn_kv, ssm_q = "Q3_K", "Q5_K", "Q4_K", "Q4_K", "Q5_K"
                elif d == 5: exp_q, shared_q, attn_q, attn_kv, ssm_q = "Q2_K", "Q4_K", "Q3_K", "Q3_K", "Q4_K"
                else: exp_q, shared_q, attn_q, attn_kv, ssm_q = "IQ1_M", "Q4_K", "Q3_K", "Q3_K", "Q4_K"
            else:
                raise ValueError(f"Unknown ONYX tier: {tier}")

            # --- Model-family aware dispatch ---
            if "ffn_gate_inp" in name:
                quant = "Q8_0"  # Router
            elif "exps" in name or "shexp" in name:
                is_moe = True
                if "ffn_gate_exps" in name or "ffn_up_exps" in name or "ffn_down_exps" in name:
                    quant = exp_q
                elif "ffn_gate_shexp" in name or "ffn_up_shexp" in name or "ffn_down_shexp" in name:
                    quant = shared_q
                elif "ssm_conv" in name or "ssm_dt" in name or "ssm_a" in name:
                    quant = "F32"
                elif "ssm_alpha" in name or "ssm_beta" in name:
                    quant = ssm_q
                elif "ssm_out" in name:
                    quant = attn_q
                elif "attn_qkv" in name:
                    quant = attn_kv
                elif "attn_q" in name or "attn_output" in name or "attn_gate" in name:
                    quant = attn_q
                elif "attn_k" in name or "attn_v" in name:
                    quant = attn_kv
                else:
                    quant = "Q8_0"
            else:
                # Dense model
                if "ffn_gate" in name or "ffn_up" in name or "ffn_down" in name:
                    quant = exp_q
                elif "ffn_gate_inp" in name:
                    quant = "Q8_0"
                elif "ssm_conv" in name or "ssm_dt" in name or "ssm_a" in name:
                    quant = "F32"
                elif "ssm_alpha" in name or "ssm_beta" in name:
                    quant = ssm_q
                elif "ssm_out" in name:
                    quant = attn_q
                elif "attn_qkv" in name:
                    quant = attn_kv
                elif "attn_q" in name or "attn_output" in name or "attn_gate" in name:
                    quant = attn_q
                elif "attn_k" in name or "attn_v" in name:
                    quant = attn_kv
                else:
                    quant = "Q8_0"

            # --- qwen4_exp hyper-connections (HC) ---
            if "hc_attn_norm" in name or "hc_ffn_norm" in name or "output_hc_norm" in name:
                quant = "Q8_0"
            elif "hc_attn_down" in name or "hc_attn_up" in name or "hc_attn_inject" in name:
                quant = attn_q
            elif "hc_ffn_down" in name or "hc_ffn_up" in name or "hc_ffn_inject" in name:
                quant = attn_q
            elif "output_hc_down" in name or "output_hc_up" in name:
                quant = attn_q

            # --- qwen4_exp PLE n-gram memory ---
            elif "ple_key" in name or "ple_value" in name:
                quant = attn_q
            elif "ple_conv1d" in name or "per_layer_token_embd" in name:
                quant = "Q8_0"

            # --- qwen4_exp indexer ---
            elif "indexer_q_proj" in name or "indexer_k_proj" in name:
                quant = attn_q

            if "output.weight" in name or "token_embd.weight" in name:
                quant = "Q8_0" if tier in ["quality", "balanced", "compact"] else "Q6_K"
            elif "_norm" in name or name.endswith(".norm") or "norm" in name.split(".")[-1]:
                quant = "Q8_0"

            # Guard: very-low-bit quants need an imatrix entry.
            if (
                imatrix_covered is not None
                and quant in _IMATRIX_REQUIRED
                and name not in imatrix_covered
            ):
                logger.warning(
                    "pass3_imatrix_fallback",
                    tensor=name,
                    from_quant=quant,
                    to_quant=_IMATRIX_FALLBACK,
                )
                quant = _IMATRIX_FALLBACK

            f.write(f"{name}={quant}\n")

    logger.info("pass3_map_complete_dynamic_gguf", path=str(output_path))
