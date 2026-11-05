"""
TOFFEE XSpark harvest module v2.

Streams a large target model layer-by-layer (12GB-VRAM friendly) and captures:

  1. ``ctx``           — the outputs of the configured target layers, fused in
                         layer order: [S, L*H] (bf16). These feed the draft's
                         context path (fc + enc.output_norm -> per-layer K/V),
                         exactly like the runtime's target-feature ring.
  2. ``final_hidden``  — the target's POST-FINAL-NORM last hidden state [S, H]
                         (bf16). Teacher logits are reconstructed EXACTLY at
                         train time as lm_head(final_hidden), which is both
                         smaller on disk and exact (DeepSpec's trick) — no more
                         lossy top-K approximations.
  3. ``input_ids``     — [S] int32.

Each sample is stored as ``sample_XXXX.pt`` plus a ``manifest.json`` describing
the geometry, so training and export can validate compatibility.

The final norm is applied during harvest (the previous implementation skipped
it, which silently biased every teacher distribution).
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import safetensors.torch
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from toffee.config import AppState
from toffee.hardware import get_gpu_info
from toffee.quant.masks import build_4d_causal_mask
from toffee.state import save_state

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Weight streaming helpers
# ---------------------------------------------------------------------------

def load_weight_map(model_path: str) -> Dict[str, str]:
    index_path = Path(model_path) / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    # single-file model
    single = Path(model_path) / "model.safetensors"
    if single.exists():
        w = safetensors.torch.load_file(single, device="cpu")
        return {k: "model.safetensors" for k in w.keys()}
    raise FileNotFoundError(f"No safetensors index or file found in {model_path}")


def stream_tensors(model_path: str, keys: List[str], weight_map: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """Reads only the requested tensors from safetensors files."""
    files = {weight_map[k] for k in keys if k in weight_map}
    wanted = set(keys)
    out = {}
    for f in sorted(files):
        w = safetensors.torch.load_file(Path(model_path) / f, device="cpu")
        for k, v in w.items():
            if k in wanted:
                out[k] = v
    return out


_FINAL_NORM_PATTERNS = [
    r"^model\.norm\.weight$",
    r"^model\.[^.]+\.norm\.weight$",   # glm-style inner norm
    r"^transformer\.ln_f\.weight$",
    r"^model\.final_layernorm\.weight$",
    r"^norm\.weight$",
]


def find_final_norm_key(weight_map: Dict[str, str]) -> Optional[str]:
    for pat in _FINAL_NORM_PATTERNS:
        for k in weight_map:
            if re.match(pat, k):
                return k
    return None


def find_embed_key(weight_map: Dict[str, str]) -> Optional[str]:
    for k in weight_map:
        if "embed_tokens" in k or k.endswith("tok_embeddings.weight") or k.endswith("word_embeddings.weight"):
            return k
    return None


def find_lm_head_key(weight_map: Dict[str, str]) -> Optional[str]:
    for k in weight_map:
        if "lm_head" in k or k.endswith("output.weight"):
            return k
    return None


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


# ---------------------------------------------------------------------------
# Main harvest
# ---------------------------------------------------------------------------

def stream_and_harvest(
    model_name: str,
    batches: List[Dict[str, torch.Tensor]],
    output_dir: Path,
    target_layers: List[int],
) -> Path:
    """Stream the target layer-by-layer and capture ctx features + final hidden.

    Args:
        model_name: HF model path (local dir with safetensors).
        batches: list of {"input_ids": [1, S], "attention_mask": [1, S]}.
        output_dir: destination for sample_XXXX.pt + manifest.json.
        target_layers: 0-based target layer ids whose outputs are captured.
    """
    safe_mkdir(output_dir)

    gpu = get_gpu_info()
    if gpu:
        logger.info("harvest_gpu_detected", gpu_name=gpu.name, vram_gb=gpu.total_memory_gb)
    else:
        logger.warning("harvest_no_gpu", reason="Harvest will be slow on CPU")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is None and getattr(config, "text_config", None) is not None:
        num_hidden_layers = config.text_config.num_hidden_layers

    bad = [l for l in target_layers if not (0 <= l < num_hidden_layers)]
    if bad:
        raise ValueError(
            f"target layers {bad} out of range for a {num_hidden_layers}-layer model "
            f"(use 0-based ids; avoid the final layer — its raw output is unnormalized)")

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # Fix RotaryEmbedding inv_freq
    for name, module in model.named_modules():
        if "RotaryEmbedding" in type(module).__name__:
            if hasattr(module, "inv_freq") and module.inv_freq is None:
                inv_freq = 1.0 / (module.base ** (torch.arange(0, module.dim, 2).float().to("cuda") / module.dim))
                module.register_buffer("inv_freq", inv_freq, persistent=False)

    weight_map = load_weight_map(model_name)

    # --- embeddings ---
    embed_key = find_embed_key(weight_map)
    if embed_key is None:
        raise RuntimeError("Could not locate the embedding tensor in safetensors")
    embed_weights = stream_tensors(model_name, [embed_key], weight_map)
    inject_weights(model.model.embed_tokens, embed_weights, "model.embed_tokens.")

    # --- final norm (applied to the last layer output before we store it) ---
    final_norm_key = find_final_norm_key(weight_map)
    final_norm_weight = None
    if final_norm_key:
        final_norm_weight = stream_tensors(model_name, [final_norm_key], weight_map)[final_norm_key].float().cuda()
        logger.info("final_norm_loaded", key=final_norm_key, shape=tuple(final_norm_weight.shape))
    else:
        logger.warning("final_norm_not_found", hint="teacher logits will be computed on the raw last-layer output")

    # --- rotary module ---
    rotary = None
    for name, module in model.named_modules():
        if "RotaryEmbedding" in type(module).__name__:
            rotary = module
            break

    # --- run embeddings for all batches ---
    batch_states = []
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
            # keep the 2d mask; the 4d mask is rebuilt per layer on demand

        batch_states.append({
            "hidden": embeds.cpu(),
            "attn_mask_2d": attn_mask_2d.cpu(),
            "position_ids": position_ids.cpu(),
            "cos_sin": cos_sin,
            "input_ids": batch["input_ids"].cpu(),
        })
        del input_ids, attn_mask_2d, embeds
        torch.cuda.empty_cache()

    # free embeddings
    model.model.embed_tokens._apply(lambda t: torch.empty_like(t, device="meta"))
    torch.cuda.empty_cache()

    # --- stream layers ---
    target_set = set(target_layers)
    harvested: List[List[torch.Tensor]] = [[] for _ in range(len(batches))]

    for layer_idx in range(num_hidden_layers):
        t0 = time.time()
        try:
            layer_module = getattr(model, "model").layers[layer_idx]
            prefix = f"model.layers.{layer_idx}."
        except AttributeError:
            layer_module = model.transformer.h[layer_idx]
            prefix = f"transformer.h.{layer_idx}."

        files_to_load = {v for k, v in weight_map.items() if k.startswith(prefix)}
        layer_weights = {}
        for f in files_to_load:
            w_cpu = safetensors.torch.load_file(Path(model_name) / f, device="cpu")
            for k, v in w_cpu.items():
                if k.startswith(prefix):
                    layer_weights[k] = v
        inject_weights(layer_module, layer_weights, prefix)
        del layer_weights

        for i, batch_state in enumerate(batch_states):
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
                if layer_idx in target_set:
                    harvested[i].append(new_hidden.to(torch.bfloat16).cpu())
                batch_state["hidden"] = new_hidden.cpu()

            del hidden, attn_mask_2d, position_ids, attn_mask_4d, new_hidden
            torch.cuda.empty_cache()

        layer_module._apply(lambda t: torch.empty_like(t, device="meta"))
        torch.cuda.empty_cache()
        logger.info("harvest_layer_done", layer=layer_idx,
                    seconds=round(time.time() - t0, 1))
        save_state(output_dir, AppState(
            current_pass=1,
            last_completed_step=f"layer_{layer_idx}",
            started_at=datetime.now(timezone.utc).isoformat(),
            model_name=model_name,
        ))

    # --- assemble samples: ctx features + final post-norm hidden ---
    logger.info("harvest_assemble_start")
    rms_eps = float(getattr(
        config, "rms_norm_eps",
        getattr(getattr(config, "text_config", config), "rms_norm_eps", 1e-6)))
    for i, batch_state in enumerate(batch_states):
        input_ids = batch_state["input_ids"]                      # [1, S]
        hidden = batch_state["hidden"].to("cuda")                 # [1, S, H] raw last layer
        with torch.no_grad():
            if final_norm_weight is not None:
                hidden = hidden.float()
                hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + rms_eps)
                hidden = hidden * final_norm_weight.view(1, 1, -1)
        final_hidden = hidden.to(torch.bfloat16).cpu().squeeze(0)  # [S, H]

        ctx = torch.cat([feat.squeeze(0) for feat in harvested[i]], dim=-1)  # [S, L*H] bf16
        sample_path = output_dir / f"sample_{i:04d}.pt"
        torch.save({
            "input_ids": input_ids.squeeze(0).to(torch.int32),
            "ctx": ctx,
            "final_hidden": final_hidden,
        }, sample_path)
        logger.info("harvest_sample_saved", sample=i, path=str(sample_path),
                    seq_len=int(input_ids.shape[1]))

        del hidden, final_hidden, ctx
        torch.cuda.empty_cache()

    # --- manifest ---
    hidden_size = getattr(config, "hidden_size",
                          getattr(getattr(config, "text_config", config), "hidden_size", None))
    manifest = {
        "model_name": model_name,
        "target_layer_ids": sorted(target_layers),
        "n_target_layers": len(target_layers),
        "hidden_size": int(hidden_size),
        "n_samples": len(batches),
        "format": "v2",
        "note": "ctx = [S, L*H] layer outputs in target_layer_ids order; "
                "final_hidden = post-final-norm; teacher logits = lm_head(final_hidden)",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("harvest_complete", output_dir=str(output_dir), **{
        "layers": sorted(target_layers), "samples": len(batches)})
    return output_dir
