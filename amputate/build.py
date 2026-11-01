"""
@file toffee/amputate/build.py
   Emit the amputated checkpoint: dense FFN -> experts + shared + router.

Reads the dense safetensors shards one file at a time, transforms each
tensor exactly once, and writes a standard ``qwen3_5_moe`` HF directory
-- the same tensor layout Sage-A1.5 uses, which the existing TOFFEE pipeline
(``toffee quant run-all``, convert_hf_to_gguf.py, llama-quantize) already
consumes unchanged.  The emitted ``config.json`` mirrors the Qwen3.5-MoE
family WRAPPER layout (``architectures: [Qwen3_5MoeForConditionalGeneration]``
+ nested ``text_config``), because convert_hf_to_gguf.py resolves by
``architectures[0]`` and rejects invented class names (v1.0.5 field fix:
the amputated run died with "Model Qwen3_5MoeForCausalLM is not supported").
``--config-only`` re-emits that config without re-slicing the ~55 GB of
shards when only the metadata layout changed.

Tensor surgery per layer:

  dense mlp.gate_proj [I, H]  ->  mlp.experts.{e}.gate_proj [I/E, H]
  dense mlp.up_proj   [I, H]  ->  mlp.experts.{e}.up_proj   [I/E, H]
  dense mlp.down_proj [H, I]  ->  mlp.experts.{e}.down_proj [H, I/E]
                                     (columns scaled by the calibration
                                      factor c_e -- zero runtime cost)
  shared expert (new)           mlp.shared_expert.{gate,up,down}_proj
                                 from the top-importance neurons;
                                 down scaled by 2*s (the emitted
                                 shared_expert_gate is all-zero, so
                                 llama.cpp's sigmoid gives a constant
                                 0.5: 0.5 * 2s = s)
  router (new)                  mlp.gate.weight [E, H] from the ridge fit

Mixers, norms, embeddings, lm_head and the ``mtp.veriloop_*`` memory
tensors are copied verbatim; the vision tower and the MTP head are
dropped (registry policy).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file, save_file

from toffee.amputate import registry as reg_mod
from toffee.amputate import surgery
from toffee.amputate.preset import AmputationPreset

logger = logging.getLogger(__name__)

def _log(logger, level, event, **kw):
    """structlog-style event logging on a stdlib logger."""
    if kw:
        rendered = " ".join(f"{k}={v}" for k, v in kw.items())
        getattr(logger, level)(f"{event} {rendered}")
    else:
        getattr(logger, level)(event)

SHARD_MAX_BYTES = 4 * 1024**3  # 4 GiB of tensor data per output shard


class _ShardWriter:
    """Buffer tensors and flush safetensors shards of <= ~4 GiB."""

    def __init__(self, out_dir: Path, max_bytes: int = SHARD_MAX_BYTES):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.buffer: Dict[str, torch.Tensor] = {}
        self.buffer_bytes = 0
        self.weight_map: Dict[str, str] = {}
        self.shard_names: List[str] = []
        self.total_bytes = 0

    def _est_bytes(self, t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    def add(self, name: str, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        nbytes = self._est_bytes(tensor)
        if self.buffer and self.buffer_bytes + nbytes > self.max_bytes:
            self.flush()
        self.buffer[name] = tensor
        self.buffer_bytes += nbytes

    def flush(self) -> None:
        if not self.buffer:
            return
        n = len(self.shard_names) + 1
        # total shard count is patched in finalize(); use a wide placeholder
        shard_name = f"model-{n:05d}-of-00000.safetensors"
        n_buf = len(self.buffer)
        save_file(self.buffer, str(self.out_dir / shard_name))
        for k in self.buffer:
            self.weight_map[k] = shard_name
        self.shard_names.append(shard_name)
        self.total_bytes += self.buffer_bytes
        self.buffer = {}
        self.buffer_bytes = 0
        _log(logger, "info", "amputate_shard_flushed", shard=shard_name, tensors=n_buf)
    def finalize(self) -> Path:
        self.flush()
        total = max(len(self.shard_names), 1)
        rename = {}
        for old in self.shard_names:
            n = int(old.split("-")[1])
            new = f"model-{n:05d}-of-{total:05d}.safetensors"
            if new != old:
                (self.out_dir / old).rename(self.out_dir / new)
            rename[old] = new
        final_map = {k: rename[shard] for k, shard in self.weight_map.items()}
        index = {
            "metadata": {"total_size": self.total_bytes},
            "weight_map": {k: final_map[k] for k in sorted(final_map)},
        }
        with open(self.out_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
        return self.out_dir / "model.safetensors.index.json"


# convert_hf_to_gguf.py resolves a model by ``architectures[0]`` and only
# accepts registered family class names, so the output must carry the real
# Qwen3.5-MoE wrapper class -- NOT an invented "Qwen3_5MoeForCausalLM"
# (2026-10-31 field report: 'Model Qwen3_5MoeForCausalLM is not supported').
GGUF_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"

# Top-level wrapper keys mirrored verbatim from the source config when
# present: vocab-level token ids / bookkeeping that stay valid for the
# text-only checkpoint.
_MIRRORED_TOP_LEVEL = (
    "image_token_id", "video_token_id",
    "vision_start_token_id", "vision_end_token_id",
    "tie_word_embeddings", "transformers_version",
)


def _output_config(source_config: dict, preset: AmputationPreset,
                   inter: int, stats_meta: dict,
                   mirror_vision_config: bool = True) -> dict:
    """Build the amputated model's config.json.

    The output mirrors the Qwen3.5-MoE family WRAPPER layout that
    convert_hf_to_gguf.py registers: ``architectures:
    [Qwen3_5MoeForConditionalGeneration]`` with the text backbone NESTED
    under ``text_config`` (``model_type: qwen3_5_moe_text``), exactly like
    the real family members.  MoE fields are set from the preset; vision
    and MTP keys are dropped from the text config.  Provenance is
    embedded for traceability.

    The checkpoint is text-only -- the vision TOWER either lives in the
    tensor set or it doesn't, and it doesn't.  ``mirror_vision_config``
    controls whether the source ``vision_config`` block is still copied
    into the wrapper: the converter reads it, but a loader may dislike
    CLIP metadata with no vision weights behind it; ``--no-vision-config``
    drops it.
    """
    text = dict(source_config.get("text_config", source_config))
    text = {k: v for k, v in text.items()
            if not k.startswith(("vision", "mtp", "vllm", "veriloop"))}
    text.update({
        "model_type": "qwen3_5_moe_text",
        "num_experts": preset.num_experts,
        "num_experts_per_tok": preset.experts_per_tok,
        "moe_intermediate_size": inter // preset.num_experts,
        "shared_expert_intermediate_size": preset.shared_intermediate,
        "intermediate_size": inter // preset.num_experts,
        "output_router_logits": False,
        "amputate_provenance": {
            "source_model": stats_meta.get("model"),
            "preset": preset.name,
            "dense_intermediate_size": inter,
            "num_layers": stats_meta.get("num_layers"),
            "hidden": stats_meta.get("hidden"),
            "note": "dense FFN amputated into a sparse MoE by toffee; "
                    "vision tower and MTP head dropped; text-only weights",
        },
    })
    out = {k: source_config[k] for k in _MIRRORED_TOP_LEVEL
           if k in source_config}
    out.update({
        "architectures": [GGUF_ARCHITECTURE],
        "model_type": "qwen3_5_moe",
        "text_config": text,
    })
    if mirror_vision_config and "vision_config" in source_config:
        out["vision_config"] = source_config["vision_config"]
    return out


def _write_metadata(model_dir: Path, output_dir: Path, reg,
                    preset: AmputationPreset, inter: int, stats_meta: dict,
                    mirror_vision_config: bool) -> None:
    """Emit everything that is not a weight shard: config.json, aux files
    (tokenizer etc.), generation_config.  Shared by the full build and
    ``--config-only`` so the two paths can never drift."""
    with open(model_dir / "config.json") as f:
        src_cfg = json.load(f)
    out_cfg = _output_config(src_cfg, preset, inter, stats_meta,
                             mirror_vision_config=mirror_vision_config)
    with open(output_dir / "config.json", "w") as f:
        json.dump(out_cfg, f, indent=2)
    for fname in reg.aux_files:
        shutil.copy2(model_dir / fname, output_dir / fname)
    gen_cfg = model_dir / "generation_config.json"
    if gen_cfg.exists():
        shutil.copy2(gen_cfg, output_dir / "generation_config.json")


def build(
    model_dir: str | Path,
    stats_path: str | Path,
    output_dir: str | Path,
    preset: AmputationPreset,
    registry_json: Optional[str | Path] = None,
    drop_mtp: bool = False,
    keep_vision: bool = False,
    drop_unknown: bool = False,
    config_only: bool = False,
    mirror_vision_config: bool = True,
) -> Path:
    """Run the surgery; returns the output directory.

    Args:
        model_dir: the dense checkpoint directory.
        stats_path: ``surgery_stats.pt`` from harvest.
        output_dir: where the amputated HF directory is written.
        preset: must match the harvest preset (checked).
        registry_json: optional registry JSON from ``report``;
            re-scanned otherwise.
        drop_mtp: also drop the ``mtp.veriloop_*`` tensors.
        keep_vision: keep the vision tower (NOT recommended: the output
            config is text-only and convert would not know what to do
            with it).
        drop_unknown: drop unknown tensors instead of keeping them.
        config_only: refresh config.json + aux files WITHOUT re-slicing
            the shards.  Use when the tensor set is already correct and
            only the metadata layout changed (e.g. the v1.0.5
            converter-key fix).  Requires a previous full build at
            ``output_dir``.
        mirror_vision_config: copy the source ``vision_config`` block
            into the output wrapper.  The amputated model ships no vision
            weights either way; the flag only controls the metadata.
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    hp = stats["preset"]
    if (hp["num_experts"] != preset.num_experts
            or hp["experts_per_tok"] != preset.experts_per_tok
            or hp["shared_intermediate"] != preset.shared_intermediate):
        raise ValueError(
            "preset mismatch between harvest and build; re-run harvest with "
            "the preset you intend to build"
        )
    layer_stats = stats["layers"]
    num_layers = int(stats["meta"]["num_layers"])
    inter = int(stats["meta"]["inter"])
    hidden = int(stats["meta"]["hidden"])
    E = preset.num_experts

    reg = (reg_mod.load_registry(registry_json) if registry_json
           else reg_mod.scan_checkpoint(model_dir))
    if sorted(int(k) for k in layer_stats.keys()) != list(range(num_layers)):
        raise ValueError("stats file does not cover every layer; re-run harvest")

    if config_only:
        if not (output_dir / "model.safetensors.index.json").exists():
            raise ValueError(
                f"--config-only: no built model found at {output_dir}; "
                "run the full build first"
            )
        _write_metadata(model_dir, output_dir, reg, preset, inter,
                        stats["meta"], mirror_vision_config)
        _log(logger, "info", "amputate_config_refreshed",
             output=str(output_dir), architecture=GGUF_ARCHITECTURE)
        return output_dir

    # Pre-compute per-layer slicing orders (deterministic: neuron id order).
    slices: Dict[int, dict] = {}
    for lstr, ls in layer_stats.items():
        l = int(lstr)
        partition = ls["partition"].long()
        idx_lists = surgery.expert_index_lists(partition)
        slices[l] = {
            "idx": idx_lists,
            "shared": ls["shared_idx"].long(),
            "router_w": ls["router_w"],          # [E, H]
            "c": ls["calib_c"].double(),         # [E]
            "s": float(ls["calib_s"]),
        }

    # Unknown tensors: refuse unless told otherwise.
    unknown = reg.by_category("unknown")
    if unknown and not drop_unknown:
        names = [t.name for t in unknown][:10]
        raise RuntimeError(
            f"unknown tensors in checkpoint ({len(unknown)}): {names} ... "
            "inspect amputate_registry.json, then pass --drop-unknown or "
            "extend toffee.amputate.registry.classify"
        )

    writer = _ShardWriter(output_dir)
    copied, dropped, sliced = 0, 0, 0

    for shard_name in sorted({t.shard for t in reg.tensors}):
        shard_tensors = load_file(model_dir / shard_name, device="cpu")
        try:
            for key in list(shard_tensors.keys()):
                rec = next(t for t in reg.tensors
                           if t.name == key and t.shard == shard_name)
                w = shard_tensors.pop(key)
                cat = rec.category
                if cat == "vision" and not keep_vision:
                    dropped += 1
                    continue
                if cat == "mtp_head":
                    dropped += 1
                    continue
                if cat == "mtp_veriloop" and drop_mtp:
                    dropped += 1
                    continue
                if cat == "ffn":
                    l, part = rec.layer, rec.ffn_part
                    sl = slices[l]
                    prefix = key[: key.rfind("mlp.") ]
                    if part in ("gate", "up"):
                        for e in range(E):
                            writer.add(
                                f"{prefix}mlp.experts.{e}.{part}_proj.weight",
                                w[sl["idx"][e]],
                            )
                        writer.add(f"{prefix}mlp.shared_expert.{part}_proj.weight",
                                   w[sl["shared"]])
                    else:  # down [H, I]
                        for e in range(E):
                            writer.add(
                                f"{prefix}mlp.experts.{e}.down_proj.weight",
                                w[:, sl["idx"][e]] * sl["c"][e].to(w.dtype),
                            )
                        # shared down scaled by 2*s (zero gate -> sigmoid 0.5)
                        writer.add(
                            f"{prefix}mlp.shared_expert.down_proj.weight",
                            w[:, sl["shared"]] * (2.0 * sl["s"]),
                        )
                    sliced += 1
                    continue
                # backbone / mtp_veriloop / kept-unknown: verbatim
                writer.add(key, w)
                copied += 1
        finally:
            del shard_tensors

    # router + shared_expert_gate per layer
    for l in range(num_layers):
        sl = slices[l]
        prefix = next(
            (t.name[: t.name.rfind("mlp.")] for t in reg.tensors
             if t.category == "ffn" and t.layer == l),
            f"model.layers.{l}.",
        )
        router = sl["router_w"]
        writer.add(f"{prefix}mlp.gate.weight",
                   router.to(torch.bfloat16 if router.dtype == torch.float32
                             else router.dtype))
        writer.add(f"{prefix}mlp.shared_expert_gate.weight",
                   torch.zeros(1, hidden, dtype=torch.bfloat16))

    writer.finalize()

    # config.json (family wrapper layout) + aux files + generation_config
    _write_metadata(model_dir, output_dir, reg, preset, inter, stats["meta"],
                    mirror_vision_config)

    _log(logger, "info", "amputate_build_complete", output=str(output_dir), copied=copied, sliced_ffn=sliced, dropped=dropped, shards=len(writer.shard_names))
    return output_dir
