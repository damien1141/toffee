"""
TOFFEE XSpark harvest module v2.

Streams a large target model layer-by-layer and captures:

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

Memory model (v2.1 — bounded on all three axes):

  * GPU   — one decoder layer (+ its activations) at a time. ``--vram-limit``
            caps the CUDA allocator via set_per_process_memory_fraction and,
            when an fp32 layer would not fit under the limit, weights are
            auto-downcast to bf16 (with a warning).
  * RAM   — one *chunk* of samples at a time. The samples are split so that
            (live hidden + harvested target features + a transient copy)
            stay under ``--ram-budget`` (default: 45% of system RAM, capped
            at 8 GB). Each chunk is swept through all layers and written to
            disk immediately, so peak system RAM is independent of
            ``--samples``. Weights are re-read from safetensors per chunk via
            mmap (page cache makes repeat sweeps cheap).
  * DISK  — ~ samples x S x H x (L+1) x 2 bytes of sample files; the plan is
            logged and checked against free disk space before starting.

The final norm is applied during harvest (the previous implementation skipped
it, which silently biased every teacher distribution).
"""

from __future__ import annotations

import gc
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import safetensors.torch
import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from toffee.config import AppState
from toffee.hardware import get_gpu_info
from toffee.quant.masks import build_4d_causal_mask
from toffee.state import save_state

try:
    # transformers >= 5: the model-agnostic mask factories hybrid models use
    # (Qwen3.5, Qwen3-Next, ...) — mirrors what the model's own forward passes
    # to each decoder layer.
    from transformers.masking_utils import (  # type: ignore
        create_causal_mask, create_recurrent_attention_mask)
    _HAS_MASKING_UTILS = True
except ImportError:  # pragma: no cover — transformers < 5
    _HAS_MASKING_UTILS = False

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "f16": torch.float16,
    "fp32": torch.float32,
    "f32": torch.float32,
    "auto": None,
}


# ---------------------------------------------------------------------------
# System resources
# ---------------------------------------------------------------------------

def _system_ram_gb() -> Optional[float]:
    """Total system RAM in GB, or None if it cannot be determined."""
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    try:
        if hasattr(__import__("os"), "sysconf"):
            import os
            pages = os.sysconf("SC_PHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            return pages * size / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        pass
    return None


def _auto_ram_budget_gb() -> float:
    """Default system-RAM budget for one chunk: 45% of RAM, clamped to [2, 8] GB."""
    ram = _system_ram_gb()
    if ram is None:
        return 4.0
    return max(2.0, min(8.0, ram * 0.45))


def _enforce_vram_limit(vram_limit_gb: Optional[float]) -> None:
    """Cap the CUDA allocator at vram_limit_gb (mirrors the quant module)."""
    if not vram_limit_gb or not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if vram_limit_gb >= total:
        return
    try:
        torch.cuda.set_per_process_memory_fraction(vram_limit_gb / total)
        logger.info("harvest_vram_limit_enforced",
                    limit_gb=round(vram_limit_gb, 2), gpu_total_gb=round(total, 2))
    except Exception as e:  # pragma: no cover — best effort
        logger.warning("harvest_vram_limit_failed", error=str(e))


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
        with safe_open(single, framework="pt") as sf:
            return {k: "model.safetensors" for k in sf.keys()}
    raise FileNotFoundError(f"No safetensors index or file found in {model_path}")


def stream_tensors(model_path: str, keys: List[str], weight_map: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """Read only the requested tensors from safetensors files.

    Uses ``safe_open`` (mmap) per file so a shard is never fully materialised
    in RAM — only the requested tensors are copied out. Repeat reads hit the
    OS page cache, which is what makes the per-chunk layer sweeps cheap.
    """
    wanted = set(keys)
    files = sorted({weight_map[k] for k in keys if k in weight_map})
    out: Dict[str, torch.Tensor] = {}
    for f in files:
        with safe_open(Path(model_path) / f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if k in wanted:
                    out[k] = sf.get_tensor(k)
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


def _text_config(config):
    """The text-backbone config (composite VLM-style configs carry it in
    .text_config — e.g. Qwen3.5's `model.language_model.*` layout)."""
    return getattr(config, "text_config", None) or config


def _resolve_text_backbone(model):
    """Return (module_with_layers, module_prefix) for the decoder stack.

    Handles the standard `model.layers` text layout, composite multimodal
    wrappers (`model.language_model.layers`) and legacy gpt-style
    `transformer.h`.
    """
    m = getattr(model, "model", None)
    if m is not None and hasattr(m, "layers"):
        return m, "model."
    lm = getattr(m, "language_model", None) if m is not None else None
    if lm is not None and hasattr(lm, "layers"):
        return lm, "model.language_model."
    t = getattr(model, "transformer", None)
    if t is not None and hasattr(t, "h"):
        return t, "transformer."
    raise AttributeError("Could not locate the decoder layers in the target model")


def _detect_ckpt_stem(weight_map: Dict[str, str]) -> str:
    """Checkpoint-side key stem of the text backbone.

    Standard HF layout: ``model.layers.N.*``. Composite multimodal
    checkpoints (Qwen3.5 ForConditionalGeneration-style) store the text stack
    under ``model.language_model.layers.N.*`` **even when the loaded class
    exposes its module tree as ``model.layers.N.*```` — key collection and
    injection must use the checkpoint-side stem, not the module-side one.
    """
    for stem, back in (("model.language_model.layers.", "layers."),
                       ("model.layers.", "layers."),
                       ("transformer.h.", "h.")):
        if any(k.startswith(stem) for k in weight_map):
            return stem[: -len(back)]
    raise RuntimeError("Could not locate decoder-layer weights in the checkpoint")


def _build_hybrid_layer_masks(text_cfg, hidden, attn_mask_2d, position_ids):
    """Per-layer-type masks for hybrid (linear+full attention) targets.

    Mirrors e.g. Qwen3_5TextModel.forward exactly:
      * full_attention  layers -> create_causal_mask (4D, or None when the
        attention backend can skip it)
      * linear_attention layers -> create_recurrent_attention_mask (the 2D
        padding mask, or None when un-padded — GatedDeltaNet zeroes padded
        states with it and would crash on a 4D mask)
    """
    if _HAS_MASKING_UTILS:
        full = create_causal_mask(
            config=text_cfg, inputs_embeds=hidden, attention_mask=attn_mask_2d,
            past_key_values=None, position_ids=position_ids)
        linear = create_recurrent_attention_mask(
            config=text_cfg, inputs_embeds=hidden, attention_mask=attn_mask_2d)
    else:
        full = build_4d_causal_mask(attn_mask_2d, hidden.dtype, hidden.device)
        linear = attn_mask_2d if bool((attn_mask_2d == 0).any()) else None
    return full, linear


def inject_weights(module, weights_dict, prefix, device, dtype=None):
    """Injects weights from a dict into a module, handling fused experts.

    ``dtype``: None keeps each checkpoint tensor's own dtype (no fp32 upcast —
    the skeleton's declared dtype is ignored on purpose); a torch dtype force-
    casts every injected tensor. Missing keys are filled (ones for norms,
    zeros elsewhere) so a partial mismatch still runs. ``to_empty`` first
    materialises the skeleton off meta — torch refuses ``param.data =`` on a
    meta parameter ("incompatible tensor type").
    """
    module.to_empty(device=device)
    available_keys = list(weights_dict.keys())
    missing = []

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
            if dtype is not None and tensor.dtype != dtype:
                tensor = tensor.to(dtype)
            if tensor.shape != param.shape and tensor.numel() == param.numel():
                tensor = tensor.reshape(param.shape)
            param.data = tensor.to(device)
        else:
            fill = 1.0 if ("norm" in name or "ln" in name or "layernorm" in name) else 0.0
            param.data = torch.full(param.shape, fill, dtype=dtype or torch.float32,
                                    device=device)
            missing.append(name)
    if missing:
        logger.warning("inject_weights_missing", prefix=prefix, n_missing=len(missing),
                       names=missing[:6],
                       hint="filled with zeros/ones — checkpoint/module key layout mismatch?")


def _free_module(module) -> None:
    """Return every parameter of a module to meta (frees its device memory)."""
    module._apply(lambda t: torch.empty_like(t, device="meta"))


def _resolve_vram_dtype(run_dtype: torch.dtype, layer_bytes: int, ckpt_itemsize: int,
                        embed_elems: int, act_elems: int, vram_limit_gb: float):
    """Pick the compute dtype under which one streamed layer fits the GPU budget.

    Returns (dtype, downcast_from: Optional[torch.dtype]). Raises RuntimeError
    when even bf16 cannot fit — better to fail fast with numbers than to OOM
    an hour into the sweep.
    """
    limit = vram_limit_gb * (1024 ** 3)

    def need_for(dt: torch.dtype) -> int:
        layer = layer_bytes * (dt.itemsize / ckpt_itemsize)
        act = act_elems * dt.itemsize
        embed = embed_elems * dt.itemsize
        # max(weights phase, layer phase) + activations + fixed overhead
        return int(max(layer, embed) + act + 0.6 * (1024 ** 3))

    need = need_for(run_dtype)
    if need <= limit:
        return run_dtype, None
    if run_dtype != torch.bfloat16:
        bf16_need = need_for(torch.bfloat16)
        if bf16_need <= limit:
            return torch.bfloat16, run_dtype
        need = bf16_need
    raise RuntimeError(
        f"a single decoder layer needs ~{need / (1024 ** 3):.1f} GB but "
        f"--vram-limit is {vram_limit_gb} GB. Raise --vram-limit (the GPU "
        "budget for one streamed layer) or use a smaller/sharded model.")


# ---------------------------------------------------------------------------
# Main harvest
# ---------------------------------------------------------------------------

def _plan_chunks(batches: List[Dict[str, torch.Tensor]], hidden_size: int,
                 itemsize: int, n_target: int, budget_bytes: int):
    """Split batches into chunks whose resident RAM stays under budget.

    Per chunk we keep: the live hidden states (1 copy), a transient copy while
    a layer's output replaces the previous hidden, and n_target harvested
    feature tensors (bf16). Padding is included (that is what is stored).
    """
    def est(batch):
        rows, seq = batch["input_ids"].shape
        per = hidden_size * (itemsize * 2 + 2 * n_target)
        return rows * seq * per

    chunks: List[List[Dict[str, torch.Tensor]]] = []
    cur: List[Dict[str, torch.Tensor]] = []
    cur_bytes = 0
    for b in batches:
        b_bytes = est(b)
        if cur and cur_bytes + b_bytes > budget_bytes:
            chunks.append(cur)
            cur, cur_bytes = [], 0
        cur.append(b)
        cur_bytes += b_bytes
    if cur:
        chunks.append(cur)
    return chunks


def stream_and_harvest(
    model_name: str,
    batches: List[Dict[str, torch.Tensor]],
    output_dir: Path,
    target_layers: List[int],
    vram_limit_gb: Optional[float] = None,
    ram_budget_gb: Optional[float] = None,
    dtype: str = "auto",
) -> Path:
    """Stream the target layer-by-layer and capture ctx features + final hidden.

    Args:
        model_name: HF model path (local dir with safetensors).
        batches: list of {"input_ids": [B, S], "attention_mask": [B, S]}.
        output_dir: destination for sample_XXXX.pt + manifest.json.
        target_layers: 0-based target layer ids whose outputs are captured.
        vram_limit_gb: GPU budget in GB (allocator cap + layer-fit check;
            fp32 layers auto-downcast to bf16 when needed).
        ram_budget_gb: system-RAM budget per sample chunk in GB (default:
            45% of installed RAM, capped at 8 GB).
        dtype: "auto" (checkpoint dtype, recommended), "bf16", "fp16", "fp32".
    """
    if not batches:
        raise ValueError("harvest received no batches")
    safe_mkdir(output_dir)

    gpu = get_gpu_info()
    if gpu:
        logger.info("harvest_gpu_detected", gpu_name=gpu.name, vram_gb=round(gpu.total_memory_gb, 2))
    else:
        logger.warning("harvest_no_gpu", reason="Harvest will be slow on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _enforce_vram_limit(vram_limit_gb)

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    text_cfg = _text_config(config)
    num_hidden_layers = getattr(text_cfg, "num_hidden_layers", None)
    hidden_size = int(getattr(text_cfg, "hidden_size", 0) or 0)
    # Hybrid targets (Qwen3.5 / Qwen3-Next) interleave linear-attention and
    # full-attention layers; each type consumes a different mask format.
    layer_types = list(getattr(text_cfg, "layer_types", None) or [])

    bad = [l for l in target_layers if not (0 <= l < num_hidden_layers)]
    if bad:
        raise ValueError(
            f"target layers {bad} out of range for a {num_hidden_layers}-layer model "
            f"(use 0-based ids; avoid the final layer — its raw output is unnormalized)")

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    backbone, base_prefix = _resolve_text_backbone(model)

    # Fix RotaryEmbedding inv_freq (meta/None under init_empty_weights)
    for name, module in model.named_modules():
        if "RotaryEmbedding" in type(module).__name__:
            if hasattr(module, "inv_freq") and (
                    module.inv_freq is None or getattr(module.inv_freq, "is_meta", False)):
                inv_freq = 1.0 / (module.base ** (torch.arange(0, module.dim, 2).float().to(device) / module.dim))
                module.register_buffer("inv_freq", inv_freq, persistent=False)

    weight_map = load_weight_map(model_name)
    # Checkpoint-side stem for the text backbone keys — differs from the
    # module-side prefix for composite layouts (see _detect_ckpt_stem).
    ckpt_stem = _detect_ckpt_stem(weight_map)
    logger.info("text_backbone_resolved", module_prefix=base_prefix,
                ckpt_stem=ckpt_stem, hybrid_layer_types=bool(layer_types))

    # --- decide the compute dtype + check the layer fits the VRAM budget ---
    forced_dtype = _DTYPE_MAP.get((dtype or "auto").lower())
    if dtype and (dtype or "auto").lower() not in _DTYPE_MAP:
        raise ValueError(f"unknown dtype {dtype!r} (use auto/bf16/fp16/fp32)")

    layer0_prefix = f"{ckpt_stem}layers.0."
    layer0_weights = stream_tensors(model_name,
                                    [k for k in weight_map if k.startswith(layer0_prefix)],
                                    weight_map)
    if not layer0_weights:
        raise RuntimeError("layer 0 has no weights in the checkpoint — wrong stem?")
    ckpt_dtype = next(iter(layer0_weights.values())).dtype
    run_dtype = forced_dtype if forced_dtype is not None else ckpt_dtype

    layer_bytes = sum(t.numel() * t.element_size() for t in layer0_weights.values())
    max_rows = max(b["input_ids"].shape[0] for b in batches)
    max_seq = max(b["input_ids"].shape[1] for b in batches)
    vocab = int(getattr(text_cfg, "vocab_size", 0) or 0)
    act_elems = max_rows * max_seq * hidden_size * 12
    embed_elems = vocab * hidden_size
    del layer0_weights
    gc.collect()

    if vram_limit_gb and device.type == "cuda":
        run_dtype, downcast_from = _resolve_vram_dtype(
            run_dtype, layer_bytes, ckpt_dtype.itemsize,
            embed_elems, act_elems, vram_limit_gb)
        if downcast_from is not None:
            logger.warning("harvest_dtype_downcast", frm=str(downcast_from), to="bfloat16",
                           reason="layer does not fit under --vram-limit",
                           limit_gb=vram_limit_gb)
    layer_bytes_run = layer_bytes * (run_dtype.itemsize / ckpt_dtype.itemsize)

    # --- chunk plan (bounds system RAM regardless of --samples) ---
    if ram_budget_gb is None:
        ram_budget_gb = _auto_ram_budget_gb()
    ram_budget_bytes = int(ram_budget_gb * (1024 ** 3) * 0.92)
    chunks = _plan_chunks(batches, hidden_size, run_dtype.itemsize,
                          len(target_layers), ram_budget_bytes)

    total_rows = sum(b["input_ids"].shape[0] for b in batches)
    est_disk = sum(int(b["attention_mask"].sum()) for b in batches) \
        * hidden_size * (len(target_layers) + 1) * 2
    try:
        free_disk = shutil.disk_usage(str(output_dir)).free
    except OSError:
        free_disk = None
    logger.info("harvest_plan",
                samples=total_rows, n_batches=len(batches), n_chunks=len(chunks),
                chunk_rows=[sum(b["input_ids"].shape[0] for b in c) for c in chunks],
                ram_budget_gb=round(ram_budget_gb, 2), compute_dtype=str(run_dtype),
                vram_limit_gb=vram_limit_gb,
                layer_gb=round(layer_bytes_run / (1024 ** 3), 2),
                est_disk_gb=round(est_disk / (1024 ** 3), 1))
    if free_disk is not None and est_disk > free_disk * 0.9:
        logger.warning("harvest_disk_tight",
                       est_gb=round(est_disk / (1024 ** 3), 1),
                       free_gb=round(free_disk / (1024 ** 3), 1),
                       hint="reduce --samples or free disk space")

    # --- final norm (applied to the last layer output before we store it) ---
    # Prefer calling the target's OWN norm module — norm families differ
    # (Qwen3.5 uses zero-init Gemma-style `x * (1 + w)`; Llama/Qwen2 use
    # `x * w`; some archs use LayerNorm) and the module encodes the exact math.
    final_norm_key = find_final_norm_key(weight_map)
    norm_module, final_norm_weight = None, None
    if final_norm_key:
        norm_attr = final_norm_key.split(".")[-2]
        norm_module = getattr(backbone, norm_attr, None) or getattr(backbone, "norm", None)
        norm_prefix = final_norm_key[: final_norm_key.rfind(".") + 1]
        if norm_module is not None:
            norm_keys = [k for k in weight_map if k.startswith(norm_prefix)]
            inject_weights(norm_module, stream_tensors(model_name, norm_keys, weight_map),
                           norm_prefix, device, dtype=run_dtype)
            logger.info("final_norm_loaded", key=final_norm_key,
                        shape=tuple(norm_module.weight.shape), via="module")
        else:
            final_norm_weight = stream_tensors(
                model_name, [final_norm_key], weight_map)[final_norm_key].float().to(device)
            logger.warning("final_norm_module_missing", key=final_norm_key,
                           hint="manual classic-RMS fallback — wrong for (1+w)-style norms")
    else:
        logger.warning("final_norm_not_found", hint="teacher logits will be computed on the raw last-layer output")

    # --- rotary module ---
    rotary = None
    for name, module in model.named_modules():
        if "RotaryEmbedding" in type(module).__name__:
            rotary = module
            break

    embed_key = find_embed_key(weight_map)
    if embed_key is None:
        raise RuntimeError("Could not locate the embedding tensor in safetensors")
    embed_prefix = embed_key[: embed_key.rfind(".") + 1]

    # --- sweep chunk-by-chunk: RAM stays bounded no matter the sample count ---
    target_set = set(target_layers)
    sample_idx = 0
    manifest = {
        "model_name": model_name,
        "target_layer_ids": sorted(target_layers),
        "n_target_layers": len(target_layers),
        "hidden_size": hidden_size,
        "n_samples": 0,
        "format": "v2",
        "compute_dtype": str(run_dtype).replace("torch.", ""),
        "vram_limit_gb": vram_limit_gb,
        "ram_budget_gb": round(ram_budget_gb, 2),
        "note": "ctx = [S, L*H] layer outputs in target_layer_ids order; "
                "final_hidden = post-final-norm; teacher logits = lm_head(final_hidden)",
    }

    for ci, chunk in enumerate(chunks):
        t_chunk = time.time()

        # -- embeddings for this chunk (embed weights live on GPU only briefly) --
        inject_weights(backbone.embed_tokens,
                       stream_tensors(model_name, [embed_key], weight_map),
                       embed_prefix, device, dtype=run_dtype)
        batch_states = []
        for batch in chunk:
            input_ids = batch["input_ids"].to(device)
            attn_mask_2d = batch["attention_mask"].to(device)
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(input_ids)

            with torch.no_grad():
                embeds = backbone.embed_tokens(input_ids)
                cos_sin = None
                if rotary is not None:
                    pos_rope = position_ids
                    if getattr(rotary, "mrope_section", None) is not None:
                        # 3-axis mrope (Qwen3.5 / Qwen2.5-VL): text-only inputs use
                        # the same arange on every axis — and the rotary expects a
                        # [3, B, S] position tensor (2D only works for B == 1).
                        pos_rope = torch.arange(seq_len, device=device).view(1, 1, -1) \
                            .expand(3, input_ids.shape[0], -1)
                    cos, sin = rotary(embeds, pos_rope)
                    cos_sin = (cos.to(embeds.dtype).cpu(), sin.to(embeds.dtype).cpu())

            batch_states.append({
                "hidden": embeds.cpu(),
                "attn_mask_2d": attn_mask_2d.cpu(),
                "position_ids": position_ids.cpu(),
                "cos_sin": cos_sin,
                "input_ids": batch["input_ids"].cpu(),
            })
            del input_ids, attn_mask_2d, embeds
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _free_module(backbone.embed_tokens)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # -- stream every layer over this chunk --
        harvested: List[List[torch.Tensor]] = [[] for _ in range(len(chunk))]
        for layer_idx in range(num_hidden_layers):
            t0 = time.time()
            layer_module = backbone.layers[layer_idx]
            prefix = f"{ckpt_stem}layers.{layer_idx}."
            layer_weights = stream_tensors(
                model_name, [k for k in weight_map if k.startswith(prefix)], weight_map)
            inject_weights(layer_module, layer_weights, prefix, device, dtype=run_dtype)
            del layer_weights

            for i, batch_state in enumerate(batch_states):
                hidden = batch_state["hidden"].to(device)
                attn_mask_2d = batch_state["attn_mask_2d"].to(device)
                position_ids = batch_state["position_ids"].to(device)

                if layer_types:
                    # hybrid model: each layer type gets its own mask format,
                    # exactly like the model's own forward does
                    full_mask, linear_mask = _build_hybrid_layer_masks(
                        text_cfg, hidden, attn_mask_2d, position_ids)
                    if layer_idx < len(layer_types) and layer_types[layer_idx] == "linear_attention":
                        layer_mask = linear_mask
                    else:
                        layer_mask = full_mask
                else:
                    layer_mask = build_4d_causal_mask(attn_mask_2d, hidden.dtype, hidden.device)

                kwargs = {"attention_mask": layer_mask, "position_ids": position_ids}
                if batch_state.get("cos_sin") is not None:
                    cos, sin = batch_state["cos_sin"]
                    kwargs["position_embeddings"] = (cos.to(device), sin.to(device))

                with torch.no_grad():
                    out = layer_module(hidden, **kwargs)
                    new_hidden = out[0] if isinstance(out, tuple) else out
                    if layer_idx in target_set:
                        harvested[i].append(new_hidden.to(torch.bfloat16).cpu())
                    batch_state["hidden"] = new_hidden.cpu()

                del hidden, attn_mask_2d, position_ids, layer_mask, new_hidden

            _free_module(layer_module)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info("harvest_layer_done", chunk=ci + 1, of=len(chunks),
                        layer=layer_idx, seconds=round(time.time() - t0, 1))

        # -- assemble + persist this chunk's samples immediately --
        rms_eps = float(getattr(text_cfg, "rms_norm_eps", 1e-6))
        for i, batch_state in enumerate(batch_states):
            n_rows = batch_state["input_ids"].shape[0]
            for b in range(n_rows):
                # Crop each row to its true (unpadded) length
                valid = int(batch_state["attn_mask_2d"][b].sum()) or batch_state["input_ids"].shape[1]
                input_ids = batch_state["input_ids"][b:b + 1, :valid]        # [1, S]
                hidden = batch_state["hidden"][b:b + 1, :valid].to(device)   # [1, S, H] raw last layer
                with torch.no_grad():
                    if norm_module is not None:
                        hidden = norm_module(hidden)
                    elif final_norm_weight is not None:
                        hidden = hidden.float()
                        hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + rms_eps)
                        hidden = hidden * final_norm_weight.view(1, 1, -1)
                final_hidden = hidden.to(torch.bfloat16).cpu().squeeze(0)    # [S, H]

                ctx = torch.cat([feat[b, :valid] for feat in harvested[i]], dim=-1)  # [S, L*H] bf16
                sample_path = output_dir / f"sample_{sample_idx:04d}.pt"
                torch.save({
                    "input_ids": input_ids.squeeze(0).to(torch.int32),
                    "ctx": ctx,
                    "final_hidden": final_hidden,
                }, sample_path)
                sample_idx += 1

                del hidden, final_hidden, ctx
        logger.info("harvest_chunk_done", chunk=ci + 1, of=len(chunks),
                    samples_saved=sample_idx, seconds=round(time.time() - t_chunk, 1))

        # refresh manifest + state so partial runs stay usable
        manifest["n_samples"] = sample_idx
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        save_state(output_dir, AppState(
            current_pass=1,
            last_completed_step=f"chunk_{ci + 1}_of_{len(chunks)}",
            started_at=datetime.now(timezone.utc).isoformat(),
            model_name=model_name,
        ))

        del batch_states, harvested
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("harvest_complete", output_dir=str(output_dir), **{
        "layers": sorted(target_layers), "samples": sample_idx,
        "chunks": len(chunks)})
    return output_dir
