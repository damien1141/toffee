"""
TOFFEE XSpark harvest module.

Streams a large model layer-by-layer to extract teacher activations and top-K logits.
Designed for 12GB VRAM using weight streaming and progressive layer loading.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import safetensors.torch
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from toffee.hardware import get_gpu_info
from toffee.quant.masks import build_4d_causal_mask
from toffee.state import save_state

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def safe_mkdir(path: Path) -> Path:
    """Create directory if it doesn't exist, return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def stream_specific_weights(model_path: str, target_keywords: List[str]) -> dict:
    """Reads only the requested tensors from safetensors files without loading the whole model."""
    index_path = Path(model_path) / "model.safetensors.index.json"
    with open(index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]

    files_to_load = set()
    for k, file in weight_map.items():
        if any(keyword in k for keyword in target_keywords):
            files_to_load.add(file)

    extracted = {}
    for f in files_to_load:
        w_cpu = safetensors.torch.load_file(Path(model_path) / f, device="cpu")
        for k, v in w_cpu.items():
            if any(keyword in k for keyword in target_keywords):
                extracted[k] = v
    return extracted


def inject_weights(module, weights_dict, prefix):
    """Injects weights from a dict into a module, handling fused experts."""
    module.to_empty(device="cuda")
    available_keys = list(weights_dict.keys())

    def _reconstruct_fused_expert(full_key, wdict, akeys):
        idx = full_key.rfind("experts.")
        if idx == -1:
            return None
        base = full_key[:idx] + "experts."
        pat = re.compile(re.escape(base) + r"(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
        experts = {}
        for k in akeys:
            m = pat.match(k)
            if m:
                experts.setdefault(int(m.group(1)), {})[m.group(2)] = wdict[k]
        if not experts:
            return None
        ordered = [experts[i] for i in sorted(experts.keys())]
        if full_key.endswith("gate_up_proj"):
            pieces = [torch.cat([e["gate_proj"], e["up_proj"]], dim=0) for e in ordered]
        else:
            pieces = [e["down_proj"] for e in ordered]
        return torch.stack(pieces, dim=0)

    def find_weight(full_key, name):
        if full_key in weights_dict:
            return weights_dict[full_key]
        for k in available_keys:
            if k.endswith(full_key) or k == name:
                return weights_dict[k]
        if "experts" in full_key and (full_key.endswith("gate_up_proj") or full_key.endswith("down_proj")):
            return _reconstruct_fused_expert(full_key, weights_dict, available_keys)
        return None

    for name, param in module.named_parameters(recurse=True):
        full_key = prefix + name
        tensor = find_weight(full_key, name)
        if tensor is not None:
            if tensor.dtype != param.dtype:
                tensor = tensor.to(param.dtype)
            if tensor.shape != param.shape and tensor.numel() == param.numel():
                tensor = tensor.reshape(param.shape)
            param.data = tensor.to("cuda")
        else:
            if "norm" in name or "ln" in name or "layernorm" in name:
                param.data.fill_(1.0)
            else:
                param.data.zero_()


def stream_and_harvest(
    model_name: str,
    batches: List[Dict[str, torch.Tensor]],
    output_dir: Path,
    target_layers: List[int],
    top_k: int,
    vram_limit_gb: float = 8.0,
) -> Path:
    """Main streaming loop to extract teacher data."""
    safe_mkdir(output_dir)

    gpu = get_gpu_info()
    if gpu:
        logger.info("harvest_gpu_detected", name=gpu.name, vram=gpu.total_memory_gb)
    else:
        logger.warning("harvest_no_gpu", msg="Harvest will be slow on CPU")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is None and getattr(config, "text_config", None) is not None:
        num_hidden_layers = config.text_config.num_hidden_layers

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # Fix RotaryEmbedding inv_freq
    for name, module in model.named_modules():
        if "RotaryEmbedding" in type(module).__name__:
            if hasattr(module, "inv_freq") and module.inv_freq is None:
                inv_freq = 1.0 / (module.base ** (torch.arange(0, module.dim, 2).float().to("cuda") / module.dim))
                module.register_buffer("inv_freq", inv_freq, persistent=False)

    index_path = Path(model_name) / "model.safetensors.index.json"
    with open(index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]

    # Load embeddings
    embed_files = set(v for k, v in weight_map.items() if "embed_tokens" in k)
    embed_weights = {}
    for f in embed_files:
        w_cpu = safetensors.torch.load_file(Path(model_name) / f, device="cpu")
        for k, v in w_cpu.items():
            if "embed_tokens" in k:
                embed_weights[k] = v
    inject_weights(model.model.embed_tokens, embed_weights, "model.embed_tokens.")

    # Find Rotary module
    rotary = None
    for name, module in model.named_modules():
        if "RotaryEmbedding" in type(module).__name__:
            rotary = module
            break

    # Process batches - get initial embeddings
    batch_hidden_states = []
    for batch in batches:
        input_ids = batch["input_ids"].to("cuda")
        attn_mask_2d = batch["attention_mask"].to("cuda")
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand_as(input_ids)

        with torch.no_grad():
            embeds = model.model.embed_tokens(input_ids)

            cos_sin = None
            if rotary is not None:
                cos, sin = rotary(embeds, position_ids)
                cos_sin = (cos.to(embeds.dtype).cpu(), sin.to(embeds.dtype).cpu())

            attn_mask_4d = build_4d_causal_mask(attn_mask_2d, embeds.dtype, embeds.device)

        batch_hidden_states.append({
            "hidden": embeds.cpu(),
            "attn_mask_2d": attn_mask_2d.cpu(),
            "position_ids": position_ids.cpu(),
            "cos_sin": cos_sin,
            "input_ids": batch["input_ids"]
        })
        del input_ids, attn_mask_2d, embeds, attn_mask_4d
        torch.cuda.empty_cache()

    # Free embeddings
    model.model.embed_tokens._apply(lambda t: torch.empty_like(t, device='meta'))
    torch.cuda.empty_cache()

    # Stream layers
    harvested_data = {i: [] for i in range(len(batches))}

    for layer_idx in range(num_hidden_layers):
        logger.info("harvest_layer_start", layer=layer_idx, total=num_hidden_layers)

        try:
            layer_module = getattr(model, "model").layers[layer_idx]
            prefix = f"model.layers.{layer_idx}."
        except AttributeError:
            layer_module = model.transformer.h[layer_idx]
            prefix = f"transformer.h.{layer_idx}."

        # Load weights for this layer
        files_to_load = set(v for k, v in weight_map.items() if k.startswith(prefix))
        layer_weights = {}
        for f in files_to_load:
            w_cpu = safetensors.torch.load_file(Path(model_name) / f, device="cpu")
            for k, v in w_cpu.items():
                if k.startswith(prefix):
                    layer_weights[k] = v
        inject_weights(layer_module, layer_weights, prefix)
        del layer_weights

        for i, batch_state in enumerate(batch_hidden_states):
            hidden = batch_state["hidden"].to("cuda")
            attn_mask_2d = batch_state["attn_mask_2d"].to("cuda")
            position_ids = batch_state["position_ids"].to("cuda")

            attn_mask_4d = build_4d_causal_mask(attn_mask_2d, hidden.dtype, hidden.device)
            kwargs = {"attention_mask": attn_mask_4d, "position_ids": position_ids}

            if batch_state.get("cos_sin") is not None:
                cos, sin = batch_state["cos_sin"]
                kwargs["position_embeddings"] = (cos.to("cuda"), sin.to("cuda"))

            with torch.no_grad():
                out = layer_module(hidden, **kwargs)
                new_hidden = out[0] if isinstance(out, tuple) else out

                if layer_idx in target_layers:
                    harvested_data[i].append(new_hidden.cpu())

                batch_state["hidden"] = new_hidden.cpu()

            del hidden, attn_mask_2d, position_ids, attn_mask_4d, new_hidden
            torch.cuda.empty_cache()

        # Free layer weights
        layer_module._apply(lambda t: torch.empty_like(t, device='meta'))
        torch.cuda.empty_cache()

        # Save state
        save_state(output_dir, type('S', (), {'current_pass': 1, 'last_completed_step': f'layer_{layer_idx}', 'model_name': model_name})())

    # Compute top-K logits at final layer
    logger.info("harvest_lm_head_start")
    lm_head_files = set(v for k, v in weight_map.items() if "lm_head" in k or "output" in k)
    lm_head_weights = {}
    for f in lm_head_files:
        w_cpu = safetensors.torch.load_file(Path(model_name) / f, device="cpu")
        for k, v in w_cpu.items():
            if "lm_head" in k or "output" in k:
                lm_head_weights[k] = v

    if not lm_head_weights:
        inject_weights(model.lm_head, embed_weights, "lm_head.")
    else:
        inject_weights(model.lm_head, lm_head_weights, "lm_head.")
    del embed_weights, lm_head_weights

    for i, batch_state in enumerate(batch_hidden_states):
        hidden = batch_state["hidden"].to("cuda")
        input_ids = batch_state["input_ids"]

        with torch.no_grad():
            shift_hidden = hidden[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            logits = model.lm_head(shift_hidden).float()

            topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)

            sample_path = output_dir / f"sample_{i:04d}.pt"
            torch.save({
                "ctx_features": harvested_data[i],
                "topk_vals": topk_vals.cpu(),
                "topk_idx": topk_idx.cpu(),
                "gt_tokens": shift_labels.cpu()
            }, sample_path)

            logger.info("harvest_sample_saved", sample=i, path=str(sample_path))

            del hidden, shift_hidden, shift_labels, logits, topk_vals, topk_idx
            torch.cuda.empty_cache()

    return output_dir
