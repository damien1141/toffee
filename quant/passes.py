"""
TOFFEE quantization passes.

Ported from ONYX: Pass 2 (imatrix harvest) and Pass 3 (GGUF quantization).
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from toffee.config import AppState
from toffee.hardware import get_gpu_info
from toffee.quant.config import (
    get_hidden_size,
    get_intermediate_size,
    get_layer_types,
    get_num_experts,
    get_num_hidden_layers,
    load_config_with_fallback,
)
from toffee.quant.imatrix import ActivationScaleCapturer, ImatrixEntry, ImatrixWriter
from toffee.quant.mapping import generate_dynamic_apex_tensor_map_from_gguf
from toffee.quant.masks import build_4d_causal_mask, layer_needs_rope, layer_wants_2d_mask
from toffee.state import load_state, save_state


class BatchRecord:
    """A single tokenized batch record for Pass 2."""

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, sample_id: int):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.sample_id = sample_id


def _enforce_vram_limit(vram_limit_gb: float) -> None:
    """Enforce VRAM limit at the allocator level."""
    if torch.cuda.is_available():
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        fraction = min(1.0, vram_limit_gb / total_mem_gb)
        try:
            torch.cuda.set_per_process_memory_fraction(fraction)
        except Exception:
            pass


def _inject_weights(module, weights_dict: Dict[str, torch.Tensor], prefix: str) -> Tuple[int, List[str]]:
    """Inject weights into a module, handling missing keys and dtype conversion."""
    assigned = 0
    missing = []
    available_keys = list(weights_dict.keys())

    def find_weight(full_key: str, name: str) -> Optional[torch.Tensor]:
        if full_key in weights_dict:
            return weights_dict[full_key]
        for k in available_keys:
            if k.endswith(full_key):
                return weights_dict[k]
            if k == name:
                return weights_dict[k]
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
            assigned += 1
        else:
            if "norm" in name or "ln" in name or "layernorm" in name:
                param.data.fill_(1.0)
            else:
                param.data.fill_(0.0)
            missing.append(full_key)

    for name, buf in module.named_buffers(recurse=True):
        full_key = prefix + name
        tensor = find_weight(full_key, name)
        if tensor is not None:
            if tensor.dtype != buf.dtype:
                tensor = tensor.to(buf.dtype)
            buf.data = tensor.to("cuda")
            assigned += 1
        else:
            if "norm" in name or "ln" in name or "layernorm" in name:
                buf.data.fill_(1.0)
            else:
                buf.data.fill_(0.0)

    return assigned, missing


def pass2_harvest(
    model_name: str,
    batches: List[BatchRecord],
    output_dir: Path,
    vram_limit_gb: float,
    samples: int = 256,
    last_completed_layer: int = -1,
    batch_size: int = 32,
) -> Path:
    """
    Stream the model weights layer-by-layer, running forward passes
    on disk-cached batches. Attach forward hooks to capture activation scales.
    """
    safe_mkdir(output_dir)
    imatrix_path = output_dir / "imatrix.dat"

    import logging
    logger = logging.getLogger(__name__)
    logger.info("pass2_start", model=model_name, vram_limit_gb=vram_limit_gb)

    _enforce_vram_limit(vram_limit_gb)

    batch_count = min(len(batches), math.ceil(samples / batch_size))
    batches_to_use = batches[:batch_count]

    config = load_config_with_fallback(model_name)
    num_hidden_layers = get_num_hidden_layers(config)

    # Build text-only model for imatrix harvesting
    _model_type = getattr(config, "model_type", None)

    class _TextModelShim:
        def __init__(self, text_model):
            self.model = text_model

        def named_modules(self):
            return self.model.named_modules()

        def named_parameters(self):
            return self.model.named_parameters()

        def named_buffers(self):
            return self.model.named_buffers()

    if _model_type in ("qwen3_5_moe", "qwen3_5_moe_text"):
        from transformers.models.qwen3_5_moe import (
            Qwen3_5MoeTextModel,
            Qwen3_5MoeTextConfig,
        )
        text_cfg = (
            config.text_config
            if _model_type == "qwen3_5_moe"
            else config
        )
        if not isinstance(text_cfg, Qwen3_5MoeTextConfig):
            text_cfg = Qwen3_5MoeTextConfig.from_dict(text_cfg.to_dict())

        with init_empty_weights():
            _text = Qwen3_5MoeTextModel(text_cfg)
        model = _TextModelShim(_text)
    elif _model_type in ("qwen3_5", "qwen3_5_text"):
        from transformers.models.qwen3_5 import (
            Qwen3_5TextModel,
            Qwen3_5TextConfig,
        )
        text_cfg = (
            config.text_config
            if _model_type == "qwen3_5"
            else config
        )
        if not isinstance(text_cfg, Qwen3_5TextConfig):
            text_cfg = Qwen3_5TextConfig.from_dict(text_cfg.to_dict())

        with init_empty_weights():
            _text = Qwen3_5TextModel(text_cfg)
        model = _TextModelShim(_text)
    else:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # Per-layer RoPE / sliding-window metadata
    layer_rope_theta = getattr(config, "layer_rope_theta", None)
    if layer_rope_theta is None:
        layer_rope_theta = getattr(
            getattr(config, "text_config", None), "layer_rope_theta", None
        )
    layer_types = get_layer_types(config)
    muse_layer_is_sliding = []
    if layer_types is not None:
        muse_layer_is_sliding = [
            (lt == "sliding_attention") for lt in layer_types
        ]

    index_path = Path(model_name) / "model.safetensors.index.json"
    with open(index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]

    # Load GGUF tensor sizes if available
    gguf_sizes: Dict[str, int] = {}
    _f16_gguf = output_dir / f"{Path(model_name).name}-f16.gguf"
    if _f16_gguf.exists():
        try:
            import gguf as _gguf
            _reader = _gguf.GGUFReader(str(_f16_gguf))
            for _t in _reader.tensors:
                gguf_sizes[_t.name] = int(_t.n_elements)
            logger.info("pass2_gguf_sizes_loaded", count=len(gguf_sizes))
        except Exception as e:
            logger.warning("pass2_gguf_sizes_failed", error=str(e))

    layer_idx = -1

    def inject_weights(module, weights_dict, prefix):
        module.to_empty(device="cuda")
        assigned, missing = _inject_weights(module, weights_dict, prefix)

        # Fix RotaryEmbedding if it exists
        for name, sub_module in module.named_modules():
            if "RotaryEmbedding" in type(sub_module).__name__:
                if hasattr(sub_module, "inv_freq"):
                    inv_freq = 1.0 / (sub_module.base ** (torch.arange(0, sub_module.dim, 2).float().to("cuda") / sub_module.dim))
                    sub_module.register_buffer("inv_freq", inv_freq, persistent=False)

    # Embeddings
    embed_keys = [k for k in weight_map if k.endswith("embed_tokens.weight")]
    embed_files = set(weight_map[k] for k in embed_keys)
    embed_weights = {}
    for f in embed_files:
        w_cpu = safetensors_load_file(Path(model_name) / f, device="cpu")
        for k, v in w_cpu.items():
            if k.endswith("embed_tokens.weight"):
                embed_weights[k] = v
        del w_cpu

    embed_module = None
    _candidate = getattr(model, "model", model)
    if hasattr(_candidate, "embed_tokens") and hasattr(_candidate.embed_tokens, "weight"):
        embed_module = _candidate.embed_tokens
    elif hasattr(_candidate, "embeddings") and hasattr(_candidate.embeddings, "weight"):
        embed_module = _candidate.embeddings
    else:
        for name, module in model.named_modules():
            if name == "embed_tokens" and hasattr(module, "weight"):
                embed_module = module
                break
    if embed_module is None:
        raise AttributeError("Could not locate an embedding module in the model for imatrix injection.")

    embed_prefix = ""
    if embed_weights:
        first_key = next(iter(embed_weights))
        embed_prefix = first_key[: first_key.rfind(".") + 1]
    inject_weights(embed_module, embed_weights, embed_prefix)
    del embed_weights

    rotary = None
    for name, module in model.named_modules():
        if "RotaryEmbedding" in type(module).__name__:
            rotary = module
            break

    batch_hidden_states = []
    for batch in batches_to_use:
        input_ids = batch.input_ids.unsqueeze(0).to("cuda")
        attention_mask_2d = batch.attention_mask.unsqueeze(0).to("cuda")
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand_as(input_ids)
        with torch.no_grad():
            embeds = embed_module(input_ids)
            cos_sin = None
            if rotary is not None:
                cos, sin = rotary(embeds, position_ids)
                cos_sin = (cos.to(embeds.dtype), sin.to(embeds.dtype))
            batch_hidden_states.append((embeds.cpu(), attention_mask_2d.cpu(), position_ids.cpu(), cos_sin))
        del input_ids, attention_mask_2d
        torch.cuda.empty_cache()

    embed_module._apply(lambda t: torch.empty_like(t, device='meta'))
    torch.cuda.empty_cache()

    writer = ImatrixWriter(imatrix_path)
    sensitivity_data = {}

    for layer_idx in range(num_hidden_layers):
        if layer_idx <= last_completed_layer:
            continue

        logger.info("pass2_layer_start", layer=layer_idx, total=num_hidden_layers)

        try:
            layer_module = getattr(model, "model").layers[layer_idx]
            prefix = f"model.layers.{layer_idx}."
        except AttributeError:
            try:
                layer_module = model.transformer.h[layer_idx]
                prefix = f"transformer.h.{layer_idx}."
            except AttributeError:
                continue

        layer_prefix = None
        for k in weight_map.keys():
            if f"layers.{layer_idx}." in k:
                layer_prefix = k.split(f"layers.{layer_idx}.")[0] + f"layers.{layer_idx}."
                break
            elif f"h.{layer_idx}." in k:
                layer_prefix = k.split(f"h.{layer_idx}.")[0] + f"h.{layer_idx}."
                break

        if layer_prefix:
            files_to_load = set(v for k, v in weight_map.items() if k.startswith(layer_prefix))
        else:
            files_to_load = set(v for k, v in weight_map.items() if k.startswith(prefix))

        layer_weights = {}
        for f in files_to_load:
            w_cpu = safetensors_load_file(Path(model_name) / f, device="cpu")
            for k, v in w_cpu.items():
                if layer_prefix and k.startswith(layer_prefix):
                    layer_weights[k] = v
                elif k.startswith(prefix):
                    layer_weights[k] = v
            del w_cpu

        inject_prefix = layer_prefix if layer_prefix else prefix
        inject_weights(layer_module, layer_weights, inject_prefix)
        del layer_weights

        # Router hook
        num_experts = get_num_experts(config)
        layer_router_probs = torch.zeros(num_experts)
        def router_hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                probs = torch.softmax(output.float(), dim=-1)
                layer_router_probs += probs.sum(dim=tuple(range(probs.dim()-1))).cpu()

        router_handle = None
        for name, module in layer_module.named_modules():
            if name == "mlp.gate" or name.endswith(".gate"):
                router_handle = module.register_forward_hook(router_hook)
                break

        # Standard hooks
        capturers = {}
        expert_capturers = {}
        found_individual_experts = False

        for name, module in layer_module.named_modules():
            if not hasattr(module, "weight"):
                continue
            gguf_suffix = _infer_gguf_tensor_category(name, layer_idx)
            if gguf_suffix:
                if "exps" in gguf_suffix:
                    match = re.search(r"(\d+)", name)
                    if match:
                        exp_id = int(match.group(1))
                        if (gguf_suffix, exp_id) not in expert_capturers:
                            expert_capturers[(gguf_suffix, exp_id)] = ActivationScaleCapturer(gguf_suffix)
                        expert_capturers[(gguf_suffix, exp_id)].register(module)
                        found_individual_experts = True
                else:
                    full_gguf_name = f"blk.{layer_idx}.{gguf_suffix}.weight"
                    if full_gguf_name not in capturers:
                        capturers[full_gguf_name] = ActivationScaleCapturer(full_gguf_name)
                    capturers[full_gguf_name].register(module)

        # Fused expert fallback
        if not found_individual_experts:
            expert_base_scale = None
            def expert_pre_hook(module, inputs):
                nonlocal expert_base_scale
                if not inputs: return
                x = inputs[0]
                if not isinstance(x, torch.Tensor) or x.dim() < 2: return
                x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
                reduce_dims = tuple(range(x.dim() - 1))
                sq = x.pow(2).sum(dim=reduce_dims).cpu()
                if expert_base_scale is None:
                    expert_base_scale = sq
                else:
                    expert_base_scale = expert_base_scale + sq

            expert_handle = None
            for name, module in layer_module.named_modules():
                if name == "mlp.experts" or name.endswith(".experts"):
                    expert_handle = module.register_forward_pre_hook(expert_pre_hook)
                    break
        else:
            expert_base_scale = None
            expert_handle = None

        logger.info("pass2_hooks_attached", layer=layer_idx, dense_count=len(capturers), expert_count=len(expert_capturers), has_expert_hook=(expert_handle is not None))

        for i, batch in enumerate(batches_to_use):
            hidden, attn_mask_2d, position_ids, cos_sin = batch_hidden_states[i]
            gpu_hidden = hidden.to("cuda")
            position_ids = position_ids.to("cuda")
            gpu_attn_mask_2d = attn_mask_2d.to("cuda")
            use_2d_mask = layer_wants_2d_mask(layer_module)
            if use_2d_mask:
                layer_attn_mask = gpu_attn_mask_2d
            else:
                win = None
                if muse_layer_is_sliding and layer_idx < len(muse_layer_is_sliding):
                    if muse_layer_is_sliding[layer_idx]:
                        win = getattr(config, "sliding_window", None)
                        if win is None:
                            win = getattr(getattr(config, "text_config", None), "sliding_window", None)
                layer_attn_mask = build_4d_causal_mask(
                    gpu_attn_mask_2d,
                    gpu_hidden.dtype,
                    gpu_hidden.device,
                    sliding_window=win,
                )
            kwargs = {
                "attention_mask": layer_attn_mask,
                "position_ids": position_ids,
                "position_embeddings": None,
            }
            if cos_sin is not None and layer_needs_rope(layer_module, layer_rope_theta, layer_idx):
                cos, sin = cos_sin
                kwargs["position_embeddings"] = (cos.to("cuda"), sin.to("cuda"))

            current_batch_size = gpu_hidden.shape[0]

            while True:
                try:
                    with torch.no_grad():
                        out = layer_module(gpu_hidden, **kwargs)
                        new_hidden = out[0] if isinstance(out, tuple) else out
                        batch_hidden_states[i] = (
                            new_hidden.cpu(),
                            gpu_attn_mask_2d.cpu(),
                            position_ids.cpu(),
                            cos_sin,
                        )
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    current_batch_size = current_batch_size // 2
                    if current_batch_size == 0:
                        raise RuntimeError("OOM: Cannot process even a single token for this layer.")
                    logger.warning("pass2_oom_fallback", layer=layer_idx, new_batch_size=current_batch_size)
                    gpu_hidden = gpu_hidden[:current_batch_size]
                    gpu_attn_mask_2d = gpu_attn_mask_2d[:current_batch_size]
                    position_ids = position_ids[:current_batch_size]
                    if use_2d_mask:
                        layer_attn_mask = gpu_attn_mask_2d
                    else:
                        win = None
                        if muse_layer_is_sliding and layer_idx < len(muse_layer_is_sliding):
                            if muse_layer_is_sliding[layer_idx]:
                                win = getattr(config, "sliding_window", None)
                                if win is None:
                                    win = getattr(getattr(config, "text_config", None), "sliding_window", None)
                        layer_attn_mask = build_4d_causal_mask(
                            gpu_attn_mask_2d,
                            gpu_hidden.dtype,
                            gpu_hidden.device,
                            sliding_window=win,
                        )
                    kwargs = {
                        "attention_mask": layer_attn_mask,
                        "position_ids": position_ids,
                        "position_embeddings": None,
                    }
                    if cos_sin is not None and layer_needs_rope(layer_module, layer_rope_theta, layer_idx):
                        cos, sin = cos_sin
                        kwargs["position_embeddings"] = (cos.to("cuda"), sin.to("cuda"))

            del gpu_hidden, layer_attn_mask, position_ids
            torch.cuda.empty_cache()

        # Unregister hooks
        for capturer in capturers.values():
            capturer.unregister()
        if router_handle: router_handle.remove()
        if expert_handle: expert_handle.remove()

        # Save dense capturer stats
        def _resize_scale(scale: List[float], out_feat: int) -> List[float]:
            if len(scale) == out_feat:
                return scale
            if not scale:
                return [100.0] * out_feat
            out = []
            for j in range(out_feat):
                out.append(scale[j % len(scale)])
            return out

        def _dense_numel(gguf_name: str) -> Optional[int]:
            suffix = gguf_name[len(f"blk.{layer_idx}."):] if gguf_name.startswith(f"blk.{layer_idx}.") else gguf_name
            pat = re.compile(re.escape(suffix) + r"\.weight$")
            for k, v in layer_weights.items():
                if pat.search(k):
                    return int(v.shape[1])
            if gguf_name in gguf_sizes:
                g = gguf_sizes[gguf_name]
                if isinstance(g, (list, tuple)) and len(g) == 2:
                    return int(g[0])
                return int(g)
            return None

        layer_sensitivity = 0.0
        for capturer in capturers.values():
            stats = capturer.get_statistics()
            if stats.ncall > 0:
                if len(stats.squared_sums) == 1 and stats.ncall == 0:
                    target = _dense_numel(stats.tensor_name)
                    if target:
                        stats = ImatrixEntry(
                            tensor_name=stats.tensor_name,
                            num_elements=target,
                            squared_sums=_resize_scale(stats.squared_sums, target),
                            ncall=0,
                        )
                writer.add_entry(stats)
                if stats.squared_sums:
                    sorted_sums = sorted(stats.squared_sums)
                    idx = int(len(sorted_sums) * 0.999)
                    layer_sensitivity += sorted_sums[idx]

        sensitivity_data[layer_idx] = layer_sensitivity

        # Normalize router probs
        if layer_router_probs.sum() > 0:
            layer_router_probs = layer_router_probs / layer_router_probs.sum()
        else:
            layer_router_probs = torch.ones(num_experts) / num_experts

        # Save expert imatrix
        def _expert_out_feat(suffix: str) -> int:
            _hidden = int(get_hidden_size(config))
            _inter = int(get_intermediate_size(config))
            if suffix == "ffn_down_exps":
                return _inter
            return _hidden

        from collections import defaultdict
        grouped_experts = defaultdict(dict)
        for (gguf_suffix, exp_id), capturer in expert_capturers.items():
            grouped_experts[gguf_suffix][exp_id] = capturer.get_statistics()

        if num_experts is None or num_experts < 1:
            num_experts = get_num_experts(config)

        for suffix in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]:
            out_feat = _expert_out_feat(suffix)
            exp_dict = grouped_experts.get(suffix, {})
            llama_name = f"blk.{layer_idx}.{suffix}.weight"
            all_sums: List[float] = []
            total_ncall = 0

            for i in range(num_experts):
                traffic_weight = layer_router_probs[i].item() if num_experts else 1.0
                if i in exp_dict and exp_dict[i].ncall > 0:
                    resized = _resize_scale(exp_dict[i].squared_sums, out_feat)
                    all_sums.extend(s * traffic_weight for s in resized)
                    total_ncall += exp_dict[i].ncall
                else:
                    all_sums.extend([100.0 * traffic_weight] * out_feat)

            writer.add_entry(ImatrixEntry(
                tensor_name=llama_name,
                num_elements=len(all_sums),
                squared_sums=all_sums,
                ncall=max(1, total_ncall // max(1, num_experts)),
            ))

        if num_experts > 0 and not found_individual_experts and expert_base_scale is None:
            logger.warning("pass2_expert_scale_missing", layer=layer_idx)

        layer_module._apply(lambda t: torch.empty_like(t, device='meta'))
        torch.cuda.empty_cache()

        state_path = output_dir / "state.json"
        state_started_at = (
            load_state(output_dir).started_at
            if load_state(output_dir)
            else datetime.now(timezone.utc).isoformat()
        )
        layer_state = AppState(
            current_pass=2,
            last_completed_step=f"layer_{layer_idx}",
            started_at=state_started_at,
            model_name=model_name,
        )
        tmp_state_path = output_dir / "state.json.tmp"
        with open(tmp_state_path, "w") as f:
            json.dump(layer_state.model_dump(), f, indent=2)
        os.replace(tmp_state_path, state_path)

        writer.write()
        logger.info("pass2_layer_complete", layer=layer_idx, sensitivity=layer_sensitivity)

    # Final imatrix fallbacks
    existing_entries = set(writer._names_on_disk)
    sens_path = output_dir / "sensitivity.json"
    with open(sens_path, "w") as f:
        json.dump(sensitivity_data, f, indent=2)
    logger.info("pass2_sensitivity_saved", path=str(sens_path), layers=len(sensitivity_data))

    _fallback_dim = get_hidden_size(config)
    for name in ["output.weight", "token_embd.weight"]:
        if name not in existing_entries:
            writer.add_entry(
                ImatrixEntry(name, _fallback_dim, [100.0] * _fallback_dim, ncall=0)
            )
            logger.info("imatrix_fallback_added", name=name, dim=_fallback_dim)
        else:
            logger.info("imatrix_fallback_present", name=name)

    if writer.entries:
        writer.write()

    writer.write()
    logger.info("pass2_complete", imatrix_path=str(imatrix_path))
    return imatrix_path


def pass3_quantize(
    model_path: Path,
    imatrix_path: Path,
    tensor_type_path: Path,
    output_path: Path,
    f16_gguf_path: Path,
    use_imatrix: bool = True,
) -> None:
    """
    Shell out to mainline llama-quantize.

    Passes --imatrix and --tensor-type-file when use_imatrix is True.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        "pass3_quantize_start",
        model=model_path,
        imatrix=imatrix_path,
        tensor_type=tensor_type_path,
        output=output_path,
        use_imatrix=use_imatrix,
    )

    if not f16_gguf_path.exists():
        convert_cmd = [
            sys.executable, "./llama.cpp/convert_hf_to_gguf.py",
            str(model_path),
            "--outtype", "f16",
            "--outfile", str(f16_gguf_path),
        ]
        logger.info("pass3_convert_start", cmd=" ".join(convert_cmd))
        convert_result = subprocess.run(
            convert_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if convert_result.returncode != 0:
            raise RuntimeError(
                f"convert_hf_to_gguf.py failed with exit code {convert_result.returncode}:\n"
                f"stdout: {convert_result.stdout}\n"
                f"stderr: {convert_result.stderr}"
            )

    cmd = [
        "./llama.cpp/build/bin/llama-quantize",
        "--tensor-type-file",
        str(tensor_type_path),
        str(f16_gguf_path),
        str(output_path),
        "Q4_K_M",
    ]
    if use_imatrix and imatrix_path.exists():
        cmd.insert(1, "--imatrix")
        cmd.insert(2, str(imatrix_path))
    elif use_imatrix:
        logger.warning(
            "pass3_no_imatrix_skip",
            msg="--imatrix requested but imatrix.dat missing; quantizing without it.",
        )

    logger.info("pass3_quantize_cmd", cmd=" ".join(cmd))
    process = subprocess.Popen(cmd)
    process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"llama-quantize failed with exit code {process.returncode}"
        )

    logger.info("pass3_quantize_complete", output=str(output_path))


def safe_mkdir(path: Path) -> Path:
    """Create directory if it doesn't exist, return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _infer_gguf_tensor_category(hf_name: str, layer_idx: int) -> Optional[str]:
    """Local wrapper around mapping.infer_gguf_tensor_category for internal use."""
    from toffee.quant.mapping import infer_gguf_tensor_category
    return infer_gguf_tensor_category(hf_name, layer_idx)
