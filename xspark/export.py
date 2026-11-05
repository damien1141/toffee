"""
TOFFEE XSpark export module v2 — writes ik_llama.cpp-compatible GGUF draft models.

Two variants, both under ``general.architecture = "dflash"`` (the variant is
auto-detected by ik_llama.cpp from the tensor signature / metadata):

* **dspark**  — dense backbone + Markov head. ik_llama.cpp detects the dense
  signature via ``blk.0.attn_q.weight``; ``--spec-type dspark`` runs
  ``build_dspark_logits`` (autoregressive bigram-corrected chain).
* **dflash2** — dense backbone + two-tap dynamic convolutions + candidate
  selector. Detected via ``dflash.selector_rank > 0``; the runtime builds the
  candidate lattice and walks it (greedy or temperature-sampled with
  rejection sampling).

Tensor layouts (ggml ne <- numpy shape is reversed):

    fc.weight                    ggml [n_target_features, n_embd]  <- (n_embd, n_feat)
    enc.output_norm.weight       ggml [n_embd]
    output_norm.weight           ggml [n_embd]
    blk.N.attn_q.weight          ggml [n_embd, n_head*head_dim]    <- (n_head*hd, n_embd)
    blk.N.attn_k.weight          ggml [n_embd, n_kv*head_dim]      <- (n_kv*hd, n_embd)
    blk.N.attn_v.weight          ggml [n_embd, n_kv*head_dim]      <- (n_kv*hd, n_embd)
    blk.N.attn_output.weight     ggml [n_head*hd, n_embd]          <- (n_embd, n_head*hd)
    blk.N.ffn_gate/up.weight     ggml [n_embd, n_ff]               <- (n_ff, n_embd)
    blk.N.ffn_down.weight        ggml [n_ff, n_embd]               <- (n_embd, n_ff)
    markov_w1.weight             ggml [rank, vocab]                <- (vocab, rank)
    markov_w2.weight             ggml [rank, vocab]                <- (vocab, rank)
    conf_proj.weight             ggml [n_embd+rank, 1]             <- (1, n_embd+rank)
    selector_predecessor.weight  ggml [rank, vocab]                <- (vocab, rank)
    selector_successor.weight    ggml [rank, vocab]                <- (vocab, rank)
    selector_hidden.weight       ggml [n_embd, rank]               <- (rank, n_embd)
    blk.N.attn_conv_base         ggml [n_embd, K, 2]               <- (2, K, n_embd)
                                  ** no ".weight" suffix **
    blk.N.attn_conv_proj.weight  ggml [n_embd, 2*K*G]              <- (2*K*G, n_embd)

IO sharing: by default the draft omits ``token_embd``/``output`` —
ik_llama.cpp aliases them from the target model at spec-init (the preferred
mode; dflash2 requires it). ``--self-contained`` embeds frozen copies instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from toffee.xspark.model import XSparkConfig, XSparkDraftModel, stream_embed_and_lm_head

try:
    import gguf
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install gguf") from e

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer plumbing
# ---------------------------------------------------------------------------

def copy_tokenizer_from_gguf(target_gguf_path: str, writer: "gguf.GGUFWriter"):
    """Copy tokenizer.* metadata from the target GGUF into the draft GGUF."""
    reader = gguf.GGUFReader(target_gguf_path)
    for key, field in reader.fields.items():
        if not key.startswith("tokenizer.") and not key.startswith("general.gguf"):
            continue
        if key in ("general.gguf.architecture",):
            continue
        main_type = field.types[0]
        if main_type == gguf.GGUFValueType.ARRAY:
            sub_type = field.types[-1]
            contents = [v for v in field.contents()]
            if not contents:
                continue
            # pygguf returns the flat contents; re-add as array
            try:
                writer.add_key_value(key, contents, main_type, sub_type)
            except Exception:
                writer.add_array(key, contents)
        else:
            try:
                writer.add_key_value(key, field.contents(), main_type)
            except Exception:
                pass
    logger.info("tokenizer_copied_from_gguf", source=target_gguf_path)


def find_mask_token_id(
    target_model: Optional[str],
    target_gguf: Optional[str],
    explicit: Optional[int] = None,
) -> int:
    """Resolve the mask token id: explicit flag > HF tokenizer > target GGUF."""
    if explicit is not None:
        return int(explicit)
    if target_model:
        tok_json = Path(target_model) / "tokenizer.json"
        if tok_json.exists():
            data = json.loads(tok_json.read_text())
            for a in data.get("added_tokens", []):
                if a.get("content") in ("<|mask|>", "<mask>", "[MASK]"):
                    return int(a["id"])
        tok_cfg = Path(target_model) / "tokenizer_config.json"
        if tok_cfg.exists():
            cfg = json.loads(tok_cfg.read_text())
            mid = cfg.get("mask_token_id") or cfg.get("mask_token")
            if isinstance(mid, int):
                return mid
            if isinstance(mid, dict):
                return int(mid.get("id", -1))
    if target_gguf and Path(target_gguf).exists():
        reader = gguf.GGUFReader(target_gguf)
        for key, field in reader.fields.items():
            if key == "tokenizer.ggml.mask_token_id":
                return int(field.contents())
    raise ValueError(
        "Could not resolve the mask token id — pass --mask-token-id explicitly")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _np(t: torch.Tensor, dtype: torch.dtype = torch.float32) -> np.ndarray:
    return t.detach().to(dtype).contiguous().numpy()


def export_to_gguf(
    head_path: str,
    target_model: str,
    output_path: Optional[str] = None,
    target_gguf: Optional[str] = None,
    mask_token_id: Optional[int] = None,
    self_contained: bool = False,
    dtype: str = "f32",
) -> Path:
    """Export a trained head to an ik_llama.cpp-compatible GGUF."""
    torch_dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[dtype]

    cfg = XSparkConfig(**torch.load(head_path, map_location="cpu", weights_only=False)["config"])

    # ---- resolve the mask token early ----
    mask_id = find_mask_token_id(target_model, target_gguf, mask_token_id)
    logger.info("mask_token_resolved", mask_token_id=mask_id)

    # ---- load the trained head (needs the frozen IO for self-contained mode) ----
    embed_w, lm_head_w = stream_embed_and_lm_head(target_model)
    model = XSparkDraftModel.load_head(head_path, embed_w, lm_head_w)
    model.eval()
    sd = torch.load(head_path, map_location="cpu", weights_only=False)

    if output_path is None:
        output_path = str(Path(head_path).parent /
                          ("DSpark-Draft.gguf" if cfg.variant == "dspark" else "DFlash2-Draft.gguf"))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig
    tgt_config = AutoConfig.from_pretrained(target_model, trust_remote_code=True)
    tgt_hidden = int(getattr(tgt_config, "hidden_size",
                             getattr(getattr(tgt_config, "text_config", tgt_config), "hidden_size")))
    if tgt_hidden != cfg.hidden_size:
        raise ValueError(
            f"head hidden_size {cfg.hidden_size} != target hidden_size {tgt_hidden}; "
            "the dense dflash arch requires matching dims for IO sharing")

    writer = gguf.GGUFWriter(str(out), "dflash")
    n_embd = cfg.hidden_size
    hd = cfg.head_dim

    # ---- metadata ----
    writer.add_uint32("dflash.block_size", cfg.block_size)
    writer.add_uint32("tokenizer.ggml.mask_token_id", mask_id)
    # 1-based layer ids (the loader normalizes to 0-based)
    one_based = [i + 1 for i in (cfg.target_layer_ids or [])]
    if not one_based:
        raise ValueError("target_layer_ids missing from the head config")
    writer.add_array("dflash.target_layers", one_based)
    writer.add_uint32("dflash.target_hidden_size", tgt_hidden)

    writer.add_uint32("embedding_length", n_embd)
    writer.add_uint32("block_count", cfg.num_layers)
    writer.add_uint32("feed_forward_length", cfg.intermediate_size)
    writer.add_uint32("attention.head_count", cfg.num_heads)
    writer.add_uint32("attention.head_count_kv", cfg.num_kv_heads)
    writer.add_uint32("attention.key_length", hd)
    writer.add_uint32("attention.value_length", hd)
    writer.add_float32("attention.layer_norm_rms_epsilon", cfg.rms_eps)
    writer.add_bool("attention.causal", False)   # blocks are bidirectional
    writer.add_float32("rope.freq_base", cfg.rope_base)
    writer.add_uint32("rope.dimension_count", hd)

    if cfg.variant == "dflash2":
        writer.add_uint32("dflash.conv_kernel_size", cfg.conv_kernel_size)
        writer.add_uint32("dflash.conv_group_size", cfg.conv_group_size)
        writer.add_uint32("dflash.selector_rank", cfg.selector_rank)
        writer.add_uint32("dflash.selector_top_k", cfg.selector_top_k)
        if cfg.hidden_size < cfg.selector_top_k * (cfg.selector_top_k + 1):
            raise ValueError("ik_llama.cpp requires n_embd >= top_k*(top_k+1)")

    # ---- tokenizer ----
    if target_gguf and Path(target_gguf).exists():
        copy_tokenizer_from_gguf(target_gguf, writer)
    else:
        logger.warning("no_target_gguf", hint="tokenizer metadata will be minimal")

    # ---- shared / self-contained IO ----
    if self_contained:
        writer.add_tensor("token_embd.weight", _np(embed_w, torch_dtype))
        if lm_head_w is not None:
            writer.add_tensor("output.weight", _np(lm_head_w, torch_dtype))
        logger.info("io_self_contained")
    else:
        logger.info("io_shared_with_target", note="ik_llama.cpp aliases token_embd/output at spec init")

    # ---- global tensors ----
    writer.add_tensor("fc.weight", _np(model.fc.weight, torch_dtype))
    writer.add_tensor("enc.output_norm.weight", _np(model.hidden_norm.weight, torch_dtype))
    writer.add_tensor("output_norm.weight", _np(model.output_norm.weight, torch_dtype))

    # ---- per-layer tensors ----
    for i, layer in enumerate(model.layers):
        writer.add_tensor(f"blk.{i}.attn_norm.weight", _np(layer.attn_norm.weight, torch_dtype))
        writer.add_tensor(f"blk.{i}.attn_q.weight", _np(layer.attn.q_proj.weight, torch_dtype))
        writer.add_tensor(f"blk.{i}.attn_q_norm.weight", _np(layer.attn.q_norm.weight, torch_dtype))
        writer.add_tensor(f"blk.{i}.attn_k.weight", _np(layer.attn.k_proj.weight, torch_dtype))
        writer.add_tensor(f"blk.{i}.attn_k_norm.weight", _np(layer.attn.k_norm.weight, torch_dtype))
        writer.add_tensor(f"blk.{i}.attn_v.weight", _np(layer.attn.v_proj.weight, torch_dtype))
        writer.add_tensor(f"blk.{i}.attn_output.weight", _np(layer.attn.o_proj.weight, torch_dtype))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", _np(layer.ffn_norm.weight, torch_dtype))
        writer.add_tensor(f"blk.{i}.ffn_gate.weight", _np(layer.ffn_gate.weight, torch_dtype))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", _np(layer.ffn_down.weight, torch_dtype))
        writer.add_tensor(f"blk.{i}.ffn_up.weight", _np(layer.ffn_up.weight, torch_dtype))
        if cfg.variant == "dflash2":
            # NB: conv_base carries NO ".weight" suffix in ik_llama.cpp
            writer.add_tensor(f"blk.{i}.attn_conv_base", _np(layer.attn_conv_base, torch_dtype))
            writer.add_tensor(f"blk.{i}.attn_conv_proj.weight", _np(layer.attn_conv_proj.weight, torch_dtype))
            writer.add_tensor(f"blk.{i}.ffn_conv_base", _np(layer.ffn_conv_base, torch_dtype))
            writer.add_tensor(f"blk.{i}.ffn_conv_proj.weight", _np(layer.ffn_conv_proj.weight, torch_dtype))

    # ---- variant head tensors ----
    if cfg.variant == "dspark":
        writer.add_tensor("markov_w1.weight", _np(model.markov_w1.weight, torch_dtype))
        writer.add_tensor("markov_w2.weight", _np(model.markov_w2.weight, torch_dtype))
        if cfg.use_confidence_head:
            writer.add_tensor("conf_proj.weight", _np(model.conf_head.weight, torch_dtype))
            writer.add_tensor("conf_proj.bias", _np(model.conf_head.bias, torch_dtype))
    else:
        writer.add_tensor("selector_predecessor.weight", _np(model.selector_prev.weight, torch_dtype))
        writer.add_tensor("selector_successor.weight", _np(model.selector_next.weight, torch_dtype))
        writer.add_tensor("selector_hidden.weight", _np(model.selector_hidden.weight, torch_dtype))

    # ---- write ----
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = out.stat().st_size / 1e6
    logger.info("gguf_export_complete", path=str(out), size_mb=round(size_mb, 1),
                variant=cfg.variant, dtype=dtype)
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="XSpark GGUF Exporter v2")
    parser.add_argument("--head", type=str, required=True, help="Trained head (.pt)")
    parser.add_argument("--target-model", type=str, required=True, help="HF target model dir")
    parser.add_argument("--target-gguf", type=str, default=None,
                        help="Target GGUF (tokenizer copy; optional)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--self-contained", action="store_true",
                        help="Embed token_embd/output instead of sharing with the target")
    parser.add_argument("--dtype", type=str, default="f32", choices=["f32", "f16", "bf16"])
    args = parser.parse_args()
    export_to_gguf(
        head_path=args.head,
        target_model=args.target_model,
        output_path=args.output,
        target_gguf=args.target_gguf,
        mask_token_id=args.mask_token_id,
        self_contained=args.self_contained,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
