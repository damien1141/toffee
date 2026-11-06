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

Storage dtypes (``--dtype``): 1-D tensors (norms, biases) and the per-layer
conv bases are ALWAYS F32 — ik_llama.cpp's fused_rms_norm / conv-add kernels
assert F32 storage for their weights, matching the llama.cpp convention of
keeping 1-D tensors F32 even in MOSTLY_F16/BF16 GGUFs. 2-D matmul weights
follow the requested dtype. BF16 is downgraded to F16 for get_rows-consumed
tensors (selector_*, markov_w1, token_embd) because the CUDA get_rows kernel
has no BF16 path.

Block quantization (``--quant q8_0`` / ``--quant q6_k``): 2-D matmul weights
are written directly as GGUF Q8_0 (8.5 bpw, ~53% of F16) or Q6_K (6.5625
bpw, ~41% of F16) blocks — reference-faithful numpy ports of ik_llama.cpp's
``ggml-quants.c`` (see :mod:`toffee.xspark.quantize`). Kernel constraints
applied per tensor: CUDA get_rows has no k-quant path, so selector_* /
token_embd cap at Q8_0 under ``--quant q6_k``; ``markov_w1`` (dspark) is
never quantized (get_rows_ext same-type dst assert); 1-D tensors and conv
bases stay F32; a row length not divisible by the block size (32 / 256)
keeps float storage (rows are never padded — the runtime builds graph
shapes from hparams). Quantization always reads the fp32 weights, never an
F16-rounded copy. Run ``xspark eval --quant ...`` first to measure the
acceptance impact before committing to a quantized export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from toffee.xspark.model import XSparkConfig, XSparkDraftModel, stream_embed_and_lm_head
from toffee.xspark.quantize import (
    GGML_TYPE_IDS,
    normalize_quant,
    plan_quant,
    quantize_tensor,
)

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

def copy_tokenizer_from_gguf(
    target_gguf_path: str, writer: "gguf.GGUFWriter"
) -> set:
    """Copy tokenizer.* metadata from the target GGUF into the draft GGUF.

    Returns the set of keys actually copied. The caller must verify that the
    critical keys (tokenizer.ggml.model / tokens) made it through — ik_llama.cpp's
    llm_load_vocab unconditionally throws "cannot find tokenizer vocab in model
    file" without them, so a draft GGUF lacking the tokenizer can never load.
    """
    reader = gguf.GGUFReader(target_gguf_path)
    copied: set = set()
    # mask_token_id is resolved separately (explicit flag / trained head default
    # take precedence over the target GGUF's value) — do not overwrite it here.
    skip = {"general.gguf.architecture", "tokenizer.ggml.mask_token_id"}
    for key, field in reader.fields.items():
        if not key.startswith("tokenizer.") and not key.startswith("general.gguf"):
            continue
        if key in skip:
            continue
        main_type = field.types[0]
        if main_type == gguf.GGUFValueType.ARRAY:
            sub_type = field.types[-1]
            contents = [v for v in field.contents()]
            if not contents:
                continue
            # pygguf returns the flat contents; re-add as array with the
            # original element type preserved (older gguf versions without the
            # 4-arg add_key_value fall back to add_array's type inference)
            try:
                writer.add_key_value(key, contents, main_type, sub_type)
            except Exception:
                writer.add_array(key, contents)
            copied.add(key)
        else:
            try:
                writer.add_key_value(key, field.contents(), main_type)
                copied.add(key)
            except Exception as e:
                logger.warning("tokenizer_key_copy_failed", key=key, error=str(e))
    logger.info("tokenizer_copied_from_gguf", source=target_gguf_path,
                n_keys=len(copied))
    return copied


def _gguf_token_count(target_gguf_path: str) -> int:
    """Number of entries in the target GGUF's tokenizer.ggml.tokens array."""
    reader = gguf.GGUFReader(target_gguf_path)
    field = reader.fields.get("tokenizer.ggml.tokens")
    if field is None:
        return -1
    if (field.types[0] == gguf.GGUFValueType.ARRAY
            and field.types[-1] == gguf.GGUFValueType.STRING):
        # ReaderField.data holds one part-index per string
        return len(field.data)
    try:
        return len(list(field.contents()))
    except Exception:
        return -1


def find_mask_token_id(
    target_model: Optional[str],
    target_gguf: Optional[str],
    explicit: Optional[int] = None,
    head_default: Optional[int] = None,
) -> int:
    """Resolve the mask token id.

    Precedence: explicit flag > the head's TRAINED mask id (heads are trained
    with a specific mask token; the runtime must feed exactly that id) > HF
    tokenizer > target GGUF.
    """
    if explicit is not None:
        return int(explicit)
    if head_default is not None and head_default >= 0:
        return int(head_default)
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

def _gguf_bf16_type():
    """The GGUF BF16 tensor type enum, across gguf-py versions."""
    qtype = getattr(gguf, "GGMLQuantizationType", None) or gguf.GGMLType
    bf16 = getattr(qtype, "BF16", None)
    if bf16 is None:
        raise RuntimeError(
            "installed 'gguf' package has no BF16 tensor type — upgrade it "
            "(pip install -U gguf) or export with --dtype f16")
    return bf16


def _qtype_enum(qtype: str):
    """GGMLQuantizationType member for a block-quant type, value-checked.

    The GGUF tensor header carries the ggml_type enum VALUE, so the installed
    gguf package must agree with ik_llama.cpp's ggml.h (Q8_0 = 8, Q6_K = 14).
    A renumbered package fails loudly here instead of silently writing
    tensors that decode as the wrong type.
    """
    qcls = getattr(gguf, "GGMLQuantizationType", None) or gguf.GGMLType
    member = getattr(qcls, qtype, None)
    if member is None or int(member) != GGML_TYPE_IDS[qtype]:
        found = None if member is None else int(member)
        raise RuntimeError(
            f"installed 'gguf' package reports {qtype}={found!r} but "
            f"ik_llama.cpp's ggml_type enum is {qtype}={GGML_TYPE_IDS[qtype]} "
            "— pin a matching gguf version (gguf==0.19.0 is verified)")
    return member


def _np(t: torch.Tensor, dtype: torch.dtype = torch.float32) -> np.ndarray:
    """Detach, cast, and hand off to numpy.

    numpy has no bfloat16, so bf16 tensors are bitcast to uint16 (same 2-byte
    element size, shape preserved); the GGUF tensor type is tagged BF16 via
    raw_dtype at the add_tensor call site — the same approach llama.cpp's
    convert_hf_to_gguf.py uses.
    """
    t = t.detach().to(dtype).contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy()
    return t.numpy()


# tensors consumed via ggml_get_rows: the CUDA kernel (getrows.cu) supports
# F16/F32/quantized sources only — BF16 aborts — so BF16 exports downgrade
# these to F16 (bf16 -> f16 is exact within f16's normal range)
_GET_ROWS_TENSORS = ("selector_predecessor", "selector_successor",
                     "markov_w1", "token_embd")


def _storage_dtype(name: str, t: torch.Tensor,
                   requested: torch.dtype) -> torch.dtype:
    """Per-tensor GGUF storage dtype honoring ik_llama.cpp kernel constraints.

    Verified against ik_llama.cpp's ggml sources:
      * 1-D tensors (norm weights, biases) feed ggml_fused_rms_norm / the
        two-tap conv adds, whose CPU and CUDA kernels assert F32 storage —
        always F32 (matches the llama.cpp convention of keeping 1-D tensors
        F32 even in MOSTLY_F16/BF16 GGUFs).
      * blk.N.attn_conv_base / ffn_conv_base are added elementwise to F32
        dynamic coefficients; the CPU add dispatcher would silently read
        BF16 bits as F16. Always F32 (they are tiny).
      * get_rows consumers (selector_*, markov_w1, token_embd) must not be
        BF16 — CUDA getrows aborts on it; BF16 requests downgrade to F16.
      * everything else is pure matmul: F32/F16/BF16 are all supported.
    """
    if t.dim() <= 1 or name.endswith(("attn_conv_base", "ffn_conv_base")):
        return torch.float32
    if name.startswith(_GET_ROWS_TENSORS) and requested == torch.bfloat16:
        return torch.float16
    return requested


def export_to_gguf(
    head_path: str,
    target_model: str,
    output_path: Optional[str] = None,
    target_gguf: Optional[str] = None,
    mask_token_id: Optional[int] = None,
    self_contained: bool = False,
    dtype: str = "f32",
    quant: Optional[str] = None,
) -> Path:
    """Export a trained head to an ik_llama.cpp-compatible GGUF."""
    torch_dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[dtype]
    quant = normalize_quant(quant)   # None | "Q8_0" | "Q6_K"

    cfg = XSparkConfig(**torch.load(head_path, map_location="cpu", weights_only=False)["config"])

    # ---- resolve the mask token early ----
    # the head's trained mask id wins unless --mask-token-id overrides it:
    # the runtime must feed the draft exactly the token it was trained with
    mask_id = find_mask_token_id(target_model, target_gguf, mask_token_id,
                                 head_default=cfg.mask_token_id)
    if not (0 <= mask_id < cfg.vocab_size):
        raise ValueError(
            f"resolved mask_token_id {mask_id} is outside the target vocab "
            f"[0, {cfg.vocab_size}) — retrain the head or pass --mask-token-id")
    logger.info("mask_token_resolved", mask_token_id=mask_id,
                head_default=cfg.mask_token_id)

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

    # ik_llama.cpp formats every standard LLM_KV as "%s.<key>" with the arch
    # name resolved from general.architecture (llama-arch.cpp), and
    # llm_load_hparams reads these as REQUIRED keys (llama-hparams.cpp):
    #   dflash.block_count, dflash.context_length, dflash.embedding_length,
    #   dflash.attention.layer_norm_rms_epsilon (arch branch), tokenizer.ggml.mask_token_id
    # The "optional" ones (head counts, key/value length, rope base) MUST still
    # be written — if the loader cannot see them it silently falls back to
    # n_embd/n_head-derived defaults (e.g. head_dim = 5120/24 = 213 instead of
    # 256, rope base 10000 instead of 1e7) and tensor shape checks explode.
    text_cfg = getattr(tgt_config, "text_config", None) or tgt_config
    n_ctx_train = int(
        getattr(tgt_config, "max_position_embeddings", None)
        or getattr(text_cfg, "max_position_embeddings", None)
        or 0)
    if n_ctx_train <= 0:
        n_ctx_train = 8192
        logger.warning("context_length_fallback", value=n_ctx_train,
                       hint="target config has no max_position_embeddings")

    writer = gguf.GGUFWriter(str(out), "dflash")
    n_embd = cfg.hidden_size
    hd = cfg.head_dim

    quant_records: list[tuple] = []   # (name, QuantPlan, QuantStats)

    def _add(name: str, t: torch.Tensor) -> None:
        if quant is not None:
            plan = plan_quant(name, t, quant)
            if plan.qtype is not None:
                # quantize from the fp32 weights (never an F16-rounded copy)
                packed, st = quantize_tensor(_np(t, torch.float32), plan.qtype,
                                             name=name)
                # (n_rows, row_bytes) uint8 + raw_dtype: pygguf converts the
                # byte shape back to the logical ne when writing the header
                writer.add_tensor(name, packed, raw_dtype=_qtype_enum(plan.qtype))
                quant_records.append((name, plan, st))
                return
            logger.info("quant_skipped", tensor=name, reason=plan.reason)
        storage = _storage_dtype(name, t, torch_dtype)
        if storage == torch.float16 and torch_dtype == torch.bfloat16:
            logger.info("dtype_downgrade", tensor=name, requested="bf16",
                        storage="f16",
                        reason="ik_llama.cpp CUDA get_rows has no BF16 path")
        raw_dtype = _gguf_bf16_type() if storage == torch.bfloat16 else None
        writer.add_tensor(name, _np(t, storage), raw_dtype=raw_dtype)

    if torch_dtype != torch.float32:
        logger.info("gguf_dtype_policy",
                    note="1-D tensors and conv bases stay F32 (ik_llama.cpp "
                         "fused_rms_norm / conv-add kernels require F32 "
                         "storage); matmul weights use the requested dtype")
    if quant is not None:
        logger.info("gguf_quant_policy", quant=quant,
                    note="2-D matmul weights quantized at export; get_rows "
                         "tensors (selector_*/token_embd) cap at Q8_0 "
                         "(CUDA get_rows has no k-quant path); markov_w1 is "
                         "never quantized; 1-D/conv-base tensors stay F32")

    # ---- metadata ----
    writer.add_uint32("dflash.block_size", cfg.block_size)
    writer.add_uint32("tokenizer.ggml.mask_token_id", mask_id)
    # 1-based layer ids (the loader normalizes to 0-based)
    one_based = [i + 1 for i in (cfg.target_layer_ids or [])]
    if not one_based:
        raise ValueError("target_layer_ids missing from the head config")
    writer.add_array("dflash.target_layers", one_based)
    writer.add_uint32("dflash.target_hidden_size", tgt_hidden)

    writer.add_uint32("dflash.embedding_length", n_embd)
    writer.add_uint32("dflash.block_count", cfg.num_layers)
    writer.add_uint32("dflash.vocab_size", cfg.vocab_size)
    writer.add_uint32("dflash.context_length", n_ctx_train)
    writer.add_uint32("dflash.feed_forward_length", cfg.intermediate_size)
    writer.add_uint32("dflash.attention.head_count", cfg.num_heads)
    writer.add_uint32("dflash.attention.head_count_kv", cfg.num_kv_heads)
    writer.add_uint32("dflash.attention.key_length", hd)
    writer.add_uint32("dflash.attention.value_length", hd)
    writer.add_float32("dflash.attention.layer_norm_rms_epsilon", cfg.rms_eps)
    writer.add_bool("dflash.attention.causal", False)   # blocks are bidirectional
    writer.add_float32("dflash.rope.freq_base", cfg.rope_base)
    writer.add_uint32("dflash.rope.dimension_count", hd)

    if cfg.variant == "dflash2":
        writer.add_uint32("dflash.conv_kernel_size", cfg.conv_kernel_size)
        writer.add_uint32("dflash.conv_group_size", cfg.conv_group_size)
        writer.add_uint32("dflash.selector_rank", cfg.selector_rank)
        writer.add_uint32("dflash.selector_top_k", cfg.selector_top_k)
        if cfg.hidden_size < cfg.selector_top_k * (cfg.selector_top_k + 1):
            raise ValueError("ik_llama.cpp requires n_embd >= top_k*(top_k+1)")

    # ---- tokenizer (MANDATORY) ----
    # llm_load_vocab throws "cannot find tokenizer vocab in model file" when
    # tokenizer.ggml.tokens is absent (llama-vocab.cpp), and the draft model
    # goes through the exact same llama_model_load path as any target — a
    # tokenizer-less draft GGUF can never load in ik_llama.cpp.
    if not target_gguf:
        raise ValueError(
            "--target-gguf is required: the draft GGUF must embed the target's "
            "tokenizer metadata or ik_llama.cpp refuses to load it")
    if not Path(target_gguf).exists():
        raise FileNotFoundError(
            f"--target-gguf not found: {Path(target_gguf).resolve()} — pass the "
            "path of the target model's .gguf so the tokenizer can be copied "
            "into the draft")
    copied_keys = copy_tokenizer_from_gguf(target_gguf, writer)
    missing = {"tokenizer.ggml.model", "tokenizer.ggml.tokens"} - copied_keys
    if missing:
        raise RuntimeError(
            f"target GGUF {target_gguf} lacks required tokenizer keys: "
            f"{sorted(missing)} — cannot build a loadable draft GGUF from it")
    n_tokens = _gguf_token_count(target_gguf)
    if n_tokens != cfg.vocab_size:
        raise ValueError(
            f"vocab mismatch: the head was trained with vocab_size="
            f"{cfg.vocab_size} but the target GGUF tokenizer has {n_tokens} "
            "tokens — export from the head trained against this exact target")

    # ---- shared / self-contained IO ----
    if self_contained:
        _add("token_embd.weight", embed_w)
        if lm_head_w is not None:
            _add("output.weight", lm_head_w)
        logger.info("io_self_contained")
    else:
        logger.info("io_shared_with_target", note="ik_llama.cpp aliases token_embd/output at spec init")

    # ---- global tensors ----
    _add("fc.weight", model.fc.weight)
    _add("enc.output_norm.weight", model.hidden_norm.weight)
    _add("output_norm.weight", model.output_norm.weight)

    # ---- per-layer tensors ----
    for i, layer in enumerate(model.layers):
        _add(f"blk.{i}.attn_norm.weight", layer.attn_norm.weight)
        _add(f"blk.{i}.attn_q.weight", layer.attn.q_proj.weight)
        _add(f"blk.{i}.attn_q_norm.weight", layer.attn.q_norm.weight)
        _add(f"blk.{i}.attn_k.weight", layer.attn.k_proj.weight)
        _add(f"blk.{i}.attn_k_norm.weight", layer.attn.k_norm.weight)
        _add(f"blk.{i}.attn_v.weight", layer.attn.v_proj.weight)
        _add(f"blk.{i}.attn_output.weight", layer.attn.o_proj.weight)
        _add(f"blk.{i}.ffn_norm.weight", layer.ffn_norm.weight)
        _add(f"blk.{i}.ffn_gate.weight", layer.ffn_gate.weight)
        _add(f"blk.{i}.ffn_down.weight", layer.ffn_down.weight)
        _add(f"blk.{i}.ffn_up.weight", layer.ffn_up.weight)
        if cfg.variant == "dflash2":
            # NB: conv_base carries NO ".weight" suffix in ik_llama.cpp
            _add(f"blk.{i}.attn_conv_base", layer.attn_conv_base)
            _add(f"blk.{i}.attn_conv_proj.weight", layer.attn_conv_proj.weight)
            _add(f"blk.{i}.ffn_conv_base", layer.ffn_conv_base)
            _add(f"blk.{i}.ffn_conv_proj.weight", layer.ffn_conv_proj.weight)

    # ---- variant head tensors ----
    if cfg.variant == "dspark":
        _add("markov_w1.weight", model.markov_w1.weight)
        _add("markov_w2.weight", model.markov_w2.weight)
        if cfg.use_confidence_head:
            _add("conf_proj.weight", model.conf_head.weight)
            _add("conf_proj.bias", model.conf_head.bias)
    else:
        _add("selector_predecessor.weight", model.selector_prev.weight)
        _add("selector_successor.weight", model.selector_next.weight)
        _add("selector_hidden.weight", model.selector_hidden.weight)

    # ---- write ----
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = out.stat().st_size / 1e6
    logger.info("gguf_export_complete", path=str(out), size_mb=round(size_mb, 1),
                variant=cfg.variant, dtype=dtype, quant=quant or "none")

    if quant_records:
        total_f16 = sum(st.f16_bytes for _, _, st in quant_records)
        total_q = sum(st.quant_bytes for _, _, st in quant_records)
        worst_name, _, worst_st = max(quant_records, key=lambda r: r[2].rel_rmse)
        for name, plan, st in quant_records:
            logger.info(
                "gguf_quant_tensor", tensor=name, qtype=st.qtype,
                shape=list(st.shape),
                f16_kib=round(st.f16_bytes / 1024, 1),
                quant_kib=round(st.quant_bytes / 1024, 1),
                ratio=round(st.ratio, 3),
                rel_rmse_pct=round(st.rel_rmse * 100, 4),
                note=plan.reason)
        logger.info(
            "gguf_quant_summary", quant=quant, tensors=len(quant_records),
            f16_mib=round(total_f16 / 2**20, 1),
            quant_mib=round(total_q / 2**20, 1),
            ratio=round(total_q / max(total_f16, 1), 3),
            worst_rel_rmse_pct=round(worst_st.rel_rmse * 100, 4),
            worst_tensor=worst_name)
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
    parser.add_argument("--dtype", type=str, default="f32", choices=["f32", "f16", "bf16"],
                        help="storage dtype for 2-D matmul weights; 1-D tensors "
                             "and conv bases always stay F32 (ik_llama.cpp "
                             "kernel requirement)")
    parser.add_argument("--quant", type=str, default="none",
                        choices=["none", "q8", "q8_0", "q6", "q6_k"],
                        help="block-quantize 2-D matmul weights at export: "
                             "q8_0 ~53%% of F16 size (near-lossless), q6_k "
                             "~41%%; get_rows tensors (selector_*, "
                             "token_embd) cap at Q8_0 because ik_llama.cpp's "
                             "CUDA get_rows kernel has no k-quant path; "
                             "markov_w1 (dspark) is never quantized")
    args = parser.parse_args()
    export_to_gguf(
        head_path=args.head,
        target_model=args.target_model,
        output_path=args.output,
        target_gguf=args.target_gguf,
        mask_token_id=args.mask_token_id,
        self_contained=args.self_contained,
        dtype=args.dtype,
        quant=args.quant,
    )


if __name__ == "__main__":
    main()
