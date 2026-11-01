"""
@file toffee/amputate/registry.py
   Checkpoint shard scan and tensor classification.

Answers the question that blocked the surgery: what exactly lives in
the VeriLoop-E2 shards? In particular the 34 ``mtp.veriloop_*``
tensors ("veriloop_conditional_memory", semantic-gate routing,
``vllm_ignored_mtp_namespace``) that squat in the MTP namespace of
ordinary model shards.

Classification and default policy:

  backbone        ``model.layers.*``, ``model.embed_tokens.*``,
                  ``model.norm.*``, ``lm_head.*``          -> keep
  ffn             ``model.layers.N.mlp.{gate,up,down}_proj`` -> keep,
                  sliced into experts by ``build``
  vision          ``model.visual.*`` / vision towers / projectors  -> drop
                  (text-only amputation)
  mtp_veriloop    ``mtp.veriloop*`` conditional-memory tensors -> keep,
                  copied verbatim (semantics unverified on GPU; the
                  Qwen3.5 loader skips ``mtp.`` so they are inert for
                  llama.cpp, and dropping an unexplained memory bank
                  from a *reasoning* model is not a decision to make
                  with a zip gun)
  mtp_head        other ``mtp.*`` tensors (the real MTP head)    -> drop
                  (they were already skipped at GGUF load time: the
                  332 MiB VRAM overrun)
  unknown         anything else                                  -> keep
                  with a warning (build refuses to proceed on unknown
                  tensors unless ``--drop-unknown`` is passed)

``report`` writes ``amputate_registry.json``; ``build`` reads it so the
copy policy is decided once, inspectably, before any 54 GB of IO.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from safetensors import safe_open

VISION_PREFIXES = (
    "model.visual.",
    "model.vision_tower.",
    "model.vision_model.",
    "model.vision_encoder.",
    "model.multimodal_project.",
    "model.multi_modal_project.",
    "visual.",
    "vision_tower.",
    "vision_model.",
    "vision_encoder.",
    "multimodal_project.",
    "multi_modal_project.",
)

MTP_PREFIXES = ("mtp.", "model.mtp.")

FFN_RE = re.compile(
    r"^(?P<prefix>.*?layers\.(?P<layer>\d+)\.)mlp\.(?P<part>gate|up|down)_proj\.weight$"
)

AUX_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "preprocessor_config.json",
)


@dataclass
class TensorRecord:
    """One classified tensor."""

    name: str
    shard: str
    shape: List[int]
    dtype: str
    nbytes: int
    category: str          # backbone | ffn | vision | mtp_veriloop | mtp_head | unknown
    layer: Optional[int] = None
    ffn_part: Optional[str] = None
    policy: str = "keep"   # keep | slice | drop


@dataclass
class ShardRegistry:
    """The classification of an entire checkpoint directory."""

    model_dir: str
    tensors: List[TensorRecord] = field(default_factory=list)
    aux_files: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # -- queries -----------------------------------------------------------

    def by_category(self, category: str) -> List[TensorRecord]:
        return [t for t in self.tensors if t.category == category]

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for t in self.tensors:
            out[t.category] = out.get(t.category, 0) + 1
        return out

    def bytes_by_category(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for t in self.tensors:
            out[t.category] = out.get(t.category, 0) + t.nbytes
        return out

    def ffn_layers(self) -> List[int]:
        return sorted({t.layer for t in self.tensors if t.category == "ffn"})

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    # -- rendering ---------------------------------------------------------

    def render(self) -> str:
        counts = self.counts()
        nbytes = self.bytes_by_category()
        lines = [
            f"registry: {self.model_dir}",
            f"tensors: {len(self.tensors)}  shards: "
            f"{len({t.shard for t in self.tensors})}  aux files: {len(self.aux_files)}",
            "",
            f"{'category':<14}{'count':>7}{'MiB':>12}  policy",
            "-" * 52,
        ]
        policy = {
            "backbone": "keep verbatim",
            "ffn": "slice into experts + shared",
            "vision": "drop (text-only amputation)",
            "mtp_veriloop": "keep verbatim (inert for llama.cpp)",
            "mtp_head": "drop (was skipped at GGUF load anyway)",
            "unknown": "keep + warn",
        }
        for cat in ("backbone", "ffn", "vision", "mtp_veriloop", "mtp_head", "unknown"):
            if cat in counts:
                lines.append(
                    f"{cat:<14}{counts[cat]:>7}{nbytes[cat]/2**20:>12.1f}  {policy[cat]}"
                )
        if self.by_category("mtp_veriloop"):
            lines.append("")
            lines.append("mtp.veriloop_* tensors (preserved):")
            for t in self.by_category("mtp_veriloop"):
                lines.append(f"  {t.name}  {t.shape} {t.dtype}")
        if self.by_category("unknown"):
            lines.append("")
            lines.append("UNKNOWN tensors (inspect before build):")
            for t in self.by_category("unknown"):
                lines.append(f"  {t.name}  {t.shape} {t.dtype}")
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)


def classify(name: str) -> tuple:
    """Classify one checkpoint tensor name.

    Returns ``(category, layer, ffn_part)``.
    """
    m = FFN_RE.match(name)
    if m:
        return "ffn", int(m.group("layer")), m.group("part")
    if name.startswith(MTP_PREFIXES):
        if "veriloop" in name:
            return "mtp_veriloop", None, None
        return "mtp_head", None, None
    if name.startswith(VISION_PREFIXES):
        return "vision", None, None
    if ".visual." in name or ".vision_tower." in name or ".vision_model." in name:
        return "vision", None, None
    return "backbone", None, None


def scan_checkpoint(model_dir: str | Path) -> ShardRegistry:
    """Scan a HF checkpoint directory without loading tensor data.

    ``safe_open`` + ``get_slice`` read only the safetensors headers, so
    this works on a 54 GB checkpoint in seconds and ~0 RAM.
    """
    model_dir = Path(model_dir)
    reg = ShardRegistry(model_dir=str(model_dir))

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            idx = json.load(f)
        shard_names = sorted(set(idx["weight_map"].values()))
        reg.notes.append(
            f"index: {len(idx['weight_map'])} tensors across {len(shard_names)} shards"
        )
    else:
        shard_names = sorted(p.name for p in model_dir.glob("*.safetensors"))
        if not shard_names:
            raise FileNotFoundError(
                f"no *.safetensors found in {model_dir}; point --model at the "
                "unpacked checkpoint directory"
            )
        reg.notes.append("no index json found; scanned loose shards")

    for shard in shard_names:
        with safe_open(model_dir / shard, framework="pt") as f:
            for key in f.keys():
                sl = f.get_slice(key)
                shape = [int(d) for d in sl.get_shape()]
                dtype = str(sl.get_dtype())
                nbytes = 1
                for d in shape:
                    nbytes *= d
                # bf16/f16 = 2 bytes; f32 = 4. get_dtype returns e.g. 'BF16'.
                nbytes *= 2 if dtype.upper() in ("BF16", "F16", "I16") else 4
                category, layer, ffn_part = classify(key)
                policy = {
                    "ffn": "slice",
                    "vision": "drop",
                    "mtp_head": "drop",
                }.get(category, "keep")
                reg.tensors.append(
                    TensorRecord(
                        name=key,
                        shard=shard,
                        shape=shape,
                        dtype=dtype,
                        nbytes=nbytes,
                        category=category,
                        layer=layer,
                        ffn_part=ffn_part,
                        policy=policy,
                    )
                )

    reg.aux_files = [p.name for p in sorted(model_dir.iterdir())
                     if p.is_file() and p.name in AUX_FILES]
    return reg


def save_registry(reg: ShardRegistry, path: str | Path) -> Path:
    """Write the registry JSON next to the surgery outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(reg.to_json())
    return path


def load_registry(path: str | Path) -> ShardRegistry:
    """Read back a registry JSON."""
    with open(path) as f:
        d = json.load(f)
    reg = ShardRegistry(model_dir=d["model_dir"])
    reg.tensors = [TensorRecord(**t) for t in d["tensors"]]
    reg.aux_files = list(d.get("aux_files", []))
    reg.notes = list(d.get("notes", []))
    return reg
