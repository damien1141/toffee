"""
Ternary quantization pipeline: f16 GGUF -> ternary GGUF.

Creates a NEW GGUF (not an in-place patch) with:
  - All source metadata copied (architecture, hparams, tokenizer).
  - prism.hadamard.* metadata for the runtime's activation-side FWHT.
  - Matrix-weight tensors encoded as PTQ1_0 or PQ2_0 ternary blocks.
  - Non-matrix tensors (norms, embeddings where applicable) kept as-is.

The output is loadable by the PrismML llama.cpp fork:
  https://github.com/PrismML-Eng/llama.cpp
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from toffee.quant.ternary.codec import (
    encode_ptq1_0_row,
    encode_pq2_0_row,
    encode_ptq1_0_batch,
    encode_pq2_0_batch,
    bits_per_weight,
    block_size_bytes,
    block_elems,
)
from toffee.quant.ternary.hadamard import rotate_weights, random_signs

logger = logging.getLogger(__name__)

# Tensor name pattern for weights that should be rotated + ternarized.
# Matches the PrismML convention (see conversion/base.py _HADAMARD_KINDS).
_ROTATE_PATTERN = re.compile(
    r"output\.weight|"
    r"blk\.\d+\.("
    r"attn_q|attn_k|attn_v|attn_qkv|attn_output"
    r"|ffn_gate|ffn_up|ffn_down"
    r"|ffn_gate_exps|ffn_up_exps|ffn_down_exps"
    r"|ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp"
    r"|ssm_out"
    r")\.weight"
)

# Regex to extract the blk index from a tensor name (e.g. "blk.40.attn_q.weight" -> 40).
_BLK_INDEX_PATTERN = re.compile(r"^blk\.(\d+)\.")

# Default n_layer_nextn value if the KV is absent (most models have no MTP).
_DEFAULT_N_LAYER_NEXTN = 0


def _read_n_layer_nextn(reader, arch: str) -> int:
    """Read {arch}.nextn_predict_layers from the source GGUF's KV metadata.

    Returns 0 if the key is absent (model has no MTP layers). The value
    tells us how many "extra" decoder blocks past the main n_layer are
    MTP / next-token-prediction blocks. The PrismML runtime skips these
    by default (load_mtp=false), so their tensors must NOT be in
    prism.hadamard.weight_names.
    """
    key = f"{arch}.nextn_predict_layers"
    field = None
    try:
        field = reader.fields.get(key)
    except Exception:
        return _DEFAULT_N_LAYER_NEXTN
    if field is None or not field.parts:
        return _DEFAULT_N_LAYER_NEXTN
    try:
        return int(field.parts[-1].flatten()[-1])
    except Exception:
        return _DEFAULT_N_LAYER_NEXTN


def _read_n_layer(reader, arch: str) -> int:
    """Read {arch}.block_count (= n_layer, the main trunk layer count)."""
    key = f"{arch}.block_count"
    field = None
    try:
        field = reader.fields.get(key)
    except Exception:
        return 0
    if field is None or not field.parts:
        return 0
    try:
        return int(field.parts[-1].flatten()[-1])
    except Exception:
        return 0


def _make_mtp_filter(n_layer: int, n_layer_nextn: int):
    """Return a function name -> bool that returns True for MTP tensor names.

    MTP tensors are blk.{N}.* where N >= n_layer. We require n_layer > 0
    (so we know where the trunk ends). The n_layer_nextn value is only
    used to decide whether to log a notice -- some GGUFs contain MTP
    tensors but didn't get the nextn_predict_layers KV written during
    conversion, so we don't strictly require it.
    """
    if n_layer <= 0:
        return lambda name: False

    def _is_mtp(name: str) -> bool:
        m = _BLK_INDEX_PATTERN.match(name)
        if not m:
            return False
        idx = int(m.group(1))
        return idx >= n_layer

    return _is_mtp


@dataclass
class TernaryConfig:
    """Configuration for the ternary quantization pipeline.

    Attributes:
        format: "PTQ1_0" (1.75 bpw) or "PQ2_0" (2.125 bpw).
        hadamard_block_size: the rotation block size (power of 2).
            Bonsai uses 1024. Set to 0 to skip rotation.
        hadamard_seed: RNG seed for the +/-1 sign pattern.
        sign_mode: "identity" (no signs) or "explicit" (random +/-1).
        device: torch device for the rotation ("cpu" or "cuda").
            "cuda" uses torch.bmm on the GPU for ~60x speed-up on
            large weight matrices.
    """

    format: str = "PTQ1_0"
    hadamard_block_size: int = 1024
    hadamard_seed: int = 0
    sign_mode: str = "explicit"
    device: str = "cpu"


@dataclass
class TernaryReport:
    tensors_quantized: int = 0
    tensors_skipped: int = 0
    tensors_rotated: int = 0
    total_params: int = 0
    output_bytes: int = 0
    effective_bpw: float = 0.0
    per_tensor: List[Tuple[str, str, int, int]] = field(default_factory=list)


def _gguf_module():
    """Import the gguf package (pip or bundled in llama.cpp)."""
    try:
        import gguf
        return gguf
    except ImportError:
        import sys
        for candidate in (
            Path.cwd() / "llama.cpp" / "gguf-py",
            Path(__file__).resolve().parents[3] / "llama.cpp" / "gguf-py",
        ):
            if candidate.is_dir():
                sys.path.insert(0, str(candidate))
                import gguf
                return gguf
        raise


def _get_ggml_type(gguf, type_name: str):
    """Look up a GGMLQuantizationType by name, or fall back to raw int.

    The PrismML fork registers PQ2_0 as type 142 and PTQ1_0 as 143.
    The upstream gguf library doesn't know about these, so we pass
    the raw integer to GGUFWriter.add_tensor(raw_dtype=142) which
    the gguf library accepts directly.
    """
    # Raw type IDs from the PrismML fork's ggml.h.
    RAW_TYPE_IDS = {
        "PQ2_0": 142,
        "PTQ1_0": 143,
        "TQ1_0": 34,
        "TQ2_0": 35,
    }
    qt = getattr(gguf, "GGMLQuantizationType", None)
    if qt is not None:
        val = getattr(qt, type_name.upper(), None)
        if val is not None:
            return val
    # Fall back to the raw integer ID.
    return RAW_TYPE_IDS.get(type_name.upper())


def _copy_kv_metadata(gguf, reader, writer, format_name: str = "PQ2_0"):
    """Copy all KV metadata from the reader to the writer.

    Iterates reader.fields and re-adds each key with the appropriate
    writer method.
    """
    # The gguf library stores fields as a dict of {key: ReaderField}.
    # Each ReaderField has .parts (list of np.ndarray) and .types (list
    # of GGUFValueType). We need to reconstruct the value and re-add it.
    for key, field in reader.fields.items():
        try:
            # Skip prism.hadamard.* -- we write our own.
            if key.startswith("prism.hadamard."):
                continue
            # Skip ALL general.* keys -- the GGUFWriter constructor sets
            # general.architecture, and we set general.file_type below.
            # The KV-copy code mishandles scalar fields (writes them as
            # arrays), which causes "wrong type arr but expected type str/u32"
            # at load time. Skipping general.* avoids this for the keys
            # the runtime validates strictly.
            if key.startswith("general."):
                continue
            _copy_one_kv(gguf, writer, key, field)
        except Exception as exc:
            logger.debug("ternary_kv_copy_skip: %s: %s", key, str(exc))

    # Set general.file_type to the PrismML ftype for the ternary format.
    # GGML_FTYPE_MOSTLY_PQ2_0 = 128, GGML_FTYPE_MOSTLY_PTQ1_0 = 129
    # (from the PrismML fork's ggml.h).
    FTYPE_MAP = {"PQ2_0": 128, "PTQ1_0": 129}
    ftype = FTYPE_MAP.get(format_name.upper(), 0)
    if ftype:
        writer.add_uint32("general.file_type", ftype)


def _copy_one_kv(gguf, writer, key: str, field):
    """Copy one KV pair from the reader field to the writer.

    Uses the GGUF value type system to distinguish scalars from arrays:
    - GGUFValueType.ARRAY (9) = array field; the element type is in
      field.types[1], and the values are in the subsequent parts.
    - Anything else = scalar field; the value is in the last part of
      field.parts.
    """
    parts = field.parts
    types = field.types
    if not parts or not types:
        return

    vt = getattr(gguf, "GGUFValueType", None)
    if vt is None:
        return

    val_type = int(types[0])
    val_type_name = vt(val_type).name if hasattr(vt, "__members__") else str(val_type)

    # ARRAY type = 9 in the GGUF spec.
    if val_type == 9:
        # Array field. The element type is types[1].
        if len(types) < 2:
            return
        elem_type = int(types[1])
        elem_name = vt(elem_type).name if hasattr(vt, "__members__") else str(elem_type)
        try:
            if elem_name == "STRING":
                # String arrays: the parts alternate between uint64
                # (string length prefix) and uint8 (string data). The
                # key name bytes are also uint8 (the first one). Filter
                # for uint8 parts, skip the first (key name), decode the
                # rest as strings. Include empty strings (valid tokens).
                strings = []
                uint8_parts = [p for p in parts if p.dtype == np.uint8]
                for p in uint8_parts[1:]:  # skip first = key name
                    s = bytes(p.tobytes()).decode('utf-8', errors='replace')
                    strings.append(s)
                if strings:
                    writer.add_array(key, strings)
            else:
                # Numeric arrays: the values are spread across multiple
                # parts, one value per part. The first few parts are
                # metadata (key name, type indicators, array length) with
                # different dtypes (uint8, uint32, uint64). Filter by the
                # expected element dtype to get only the value parts.
                DTYPE_MAP = {
                    "INT32": np.int32, "UINT32": np.uint32,
                    "FLOAT32": np.float32, "FLOAT64": np.float64,
                    "INT64": np.int64, "UINT64": np.uint64,
                    "INT8": np.int8, "UINT8": np.uint8, "BOOL": np.bool_,
                }
                expected = DTYPE_MAP.get(elem_name)
                if expected is not None:
                    vals = []
                    for p in parts:
                        if p.dtype == np.dtype(expected):
                            vals.extend(p.flatten().tolist())
                    if vals:
                        writer.add_array(key, vals)
        except Exception:
            pass
        return

    # Scalar field. The value is in the LAST part (parts[-1]).
    # The earlier parts contain the key name length, key name, and type
    # indicator -- not the actual value.
    try:
        last = parts[-1]
        if val_type_name == "STRING":
            s = bytes(last.tobytes()).decode('utf-8', errors='replace')
            if s:
                writer.add_string(key, s)
        elif val_type_name in ("UINT32",):
            writer.add_uint32(key, int(last.flatten()[-1]))
        elif val_type_name in ("INT32",):
            writer.add_int32(key, int(last.flatten()[-1]))
        elif val_type_name in ("UINT64",):
            writer.add_uint64(key, int(last.flatten()[-1]))
        elif val_type_name in ("INT64",):
            writer.add_int64(key, int(last.flatten()[-1]))
        elif val_type_name in ("FLOAT32",):
            writer.add_float32(key, float(last.flatten()[-1]))
        elif val_type_name in ("FLOAT64",):
            writer.add_float64(key, float(last.flatten()[-1]))
        elif val_type_name in ("BOOL",):
            writer.add_bool(key, bool(last.flatten()[-1]))
    except Exception:
        pass


def _write_prism_metadata(writer, config: TernaryConfig, rotated_names: List[str],
                          sign_values: Optional[List[int]] = None,
                          sign_widths: Optional[List[int]] = None):
    """Write the prism.hadamard.* metadata to the GGUF.

    This tells the PrismML runtime:
    - Which tensors were rotated (so it applies the FWHT to activations).
    - The rotation block size and sign pattern.
    """
    bs = config.hadamard_block_size
    writer.add_uint32("prism.hadamard.version", 1)
    writer.add_uint32("prism.hadamard.block_size", bs)
    writer.add_string("prism.hadamard.transform",
                       "normalized-sylvester-walsh-hadamard")
    writer.add_string("prism.hadamard.axis", "input-last-dimension")
    writer.add_string("prism.hadamard.sign_mode", config.sign_mode)
    writer.add_array("prism.hadamard.weight_names", rotated_names)
    if config.sign_mode == "explicit" and sign_values is not None:
        writer.add_array("prism.hadamard.sign_widths", sign_widths or [])
        writer.add_array("prism.hadamard.sign_values", sign_values)
    logger.info(
        "ternary_hadamard_metadata_written: block=%d, sign_mode=%s, %d rotated",
        bs, config.sign_mode, len(rotated_names),
    )


def _encode_row(w: np.ndarray, fmt: str) -> bytes:
    if fmt == "PTQ1_0":
        return encode_ptq1_0_row(w)
    elif fmt == "PQ2_0":
        return encode_pq2_0_row(w)
    else:
        raise ValueError(f"Unknown ternary format: {fmt}")


def _encode_q8_0_batch(weights: np.ndarray) -> bytes:
    """Quantize a 2D float32 matrix to Q8_0 blocks via the gguf library.

    Used for MTP (next-token-prediction) tensors that we don't want to
    ternarize: Q8_0 keeps them at ~8.5 bpw (vs 2.125 for PQ2_0) for
    better speculative-decoding quality, and keeps them out of the
    prism.hadamard.weight_names list (so the runtime doesn't try to
    apply FWHT to activations going into them when load_mtp=false).

    Returns the packed Q8_0 bytes (uint8) for the entire matrix.
    """
    gguf = _gguf_module()
    try:
        from gguf import quants as gguf_quants
        GGMLQuantizationType = gguf.GGMLQuantizationType
    except ImportError:
        raise RuntimeError(
            "MTP Q8_0 quantization requires the gguf library's quants "
            "module (gguf.quants.quantize). Make sure the PrismML llama.cpp "
            "fork's gguf-py is on the Python path."
        )
    # The gguf quantize() expects a flat or 2D float32 array. We pass
    # the 2D (n_rows, n_per_row) array; it returns uint8 bytes.
    out = gguf_quants.quantize(weights, GGMLQuantizationType.Q8_0)
    return bytes(np.frombuffer(out, dtype=np.uint8).tobytes())


def _encode_batch(w: np.ndarray, fmt: str) -> bytes:
    """Vectorized batch encoder for a full (n_rows, n_per_row) matrix."""
    if fmt == "PTQ1_0":
        return encode_ptq1_0_batch(w)
    elif fmt == "PQ2_0":
        return encode_pq2_0_batch(w)
    else:
        raise ValueError(f"Unknown ternary format: {fmt}")


def quantize_tensor_to_ternary(
    weights: np.ndarray,
    config: TernaryConfig,
    tensor_name: str = "",
    is_mtp_fn: Optional[Any] = None,
) -> Tuple[bytes, bool]:
    """Quantize one weight tensor to ternary blocks.

    Returns (encoded_bytes, was_rotated).

    The optional ``is_mtp_fn`` is a predicate name -> bool that returns
    True for MTP / next-token-prediction tensor names. MTP tensors are
    *not* rotated (the runtime skips them by default with load_mtp=false
    so they would never get the FWHT-on-activations side; rotating them
    would silently corrupt their output for the load_mtp=true path too
    unless we also list them in weight_names, which then triggers
    "weight not found" when load_mtp=false).
    """
    n_rows, n_per_row = weights.shape
    be = block_elems(config.format)
    if n_per_row % be != 0:
        raise ValueError(
            f"Tensor {tensor_name}: n_per_row={n_per_row} not divisible by "
            f"{config.format} block size {be}"
        )

    is_mtp = bool(is_mtp_fn(tensor_name)) if is_mtp_fn is not None else False
    do_rotate = (
        config.hadamard_block_size > 0
        and n_per_row % config.hadamard_block_size == 0
        and bool(_ROTATE_PATTERN.fullmatch(tensor_name))
        and not is_mtp
    )
    if do_rotate:
        if config.device and config.device != "cpu":
            # GPU path: torch.bmm on the GPU.
            from toffee.quant.ternary.hadamard import rotate_weights_torch
            weights = rotate_weights_torch(
                weights,
                block_size=config.hadamard_block_size,
                seed=config.hadamard_seed,
                device=config.device,
            )
        else:
            # CPU path: numpy einsum.
            from toffee.quant.ternary.hadamard import rotate_weights
            weights = rotate_weights(
                weights,
                block_size=config.hadamard_block_size,
                seed=config.hadamard_seed,
            )

    # Vectorized batch encode (no per-row Python loop).
    encoded = _encode_batch(weights, config.format)
    return encoded, do_rotate


def quantize_gguf_to_ternary(
    f16_gguf_path: Path,
    output_path: Path,
    config: Optional[TernaryConfig] = None,
    tensor_types: Optional[Dict[str, str]] = None,
    max_tensors: Optional[int] = None,
) -> TernaryReport:
    """Convert an f16 GGUF to a ternary GGUF.

    Creates a new GGUF with:
    - All source metadata copied (architecture, hparams, tokenizer).
    - prism.hadamard.* metadata for the runtime's FWHT.
    - Matrix weights encoded as PTQ1_0/PQ2_0 ternary blocks.
    - Non-matrix tensors copied as-is (F16).

    The output is loadable by the PrismML llama.cpp fork.
    """
    gguf = _gguf_module()
    cfg = config or TernaryConfig()
    report = TernaryReport()

    # Read the source GGUF.
    reader = gguf.GGUFReader(str(f16_gguf_path))

    # Register the ternary type sizes in the gguf library's quant-size
    # dict, if not already present. The upstream gguf library doesn't know
    # about PQ2_0 (type 142) or PTQ1_0 (type 143); without this, the
    # GGUFWriter's quant_shape_from_byte_shape raises KeyError.
    try:
        from gguf.quants import GGML_QUANT_SIZES
        GGML_QUANT_SIZES.setdefault(142, (128, 34))  # PQ2_0: 128 elems, 34 bytes/block
        GGML_QUANT_SIZES.setdefault(143, (128, 28))  # PTQ1_0: 128 elems, 28 bytes/block
    except (ImportError, AttributeError):
        # Older gguf versions may not have this module.
        pass

    # Determine the arch name from the source metadata.
    arch = "llama"
    try:
        arch_field = reader.fields.get("general.architecture")
        if arch_field and arch_field.parts:
            # The GGUF reader stores string fields with multiple parts:
            # the last part is the actual string bytes. Try parts[-1]
            # first (the string), then fall back to scanning all parts.
            for part in reversed(arch_field.parts):
                raw = bytes(part.tobytes())
                s = raw.decode('utf-8', errors='ignore').strip('\x00')
                # A valid arch name is ASCII, no control chars, non-empty.
                if s and all(c.isprintable() and c.isascii() for c in s):
                    arch = s
                    break
    except Exception:
        pass

    # Fallback: if arch extraction failed, look at the non-general
    # key prefixes (e.g., "qwen35moe.block_count" -> "qwen35moe").
    if not arch or arch == "llama":
        for key in reader.fields:
            if key.startswith(("general.", "tokenizer.", "prism.", "GGUF.")):
                continue
            prefix = key.split(".")[0]
            if prefix and prefix != "llama":
                arch = prefix
                break

    # Create the output GGUF writer.
    writer = gguf.GGUFWriter(str(output_path), arch)

    # Build the MTP tensor filter. MTP (next-token-prediction) layers
    # are extra decoder blocks past the main trunk (e.g. Sage-A1.5 has
    # n_layer=40 trunk blocks + n_layer_nextn=1 MTP block at blk.40).
    # The PrismML runtime skips MTP tensors by default (load_mtp=false),
    # so we must NOT add them to prism.hadamard.weight_names -- otherwise
    # the runtime's lookup of `blk.40.attn_q.weight` returns nullptr and
    # the model fails to load with "prism.hadamard weight not found".
    #
    # IMPORTANT: {arch}.block_count as written by the PrismML converter
    # (conversion/qwen.py:_QwenMtpMixin) is the TOTAL layer count
    # (trunk + MTP), not the trunk count. For Sage-A1.5 this is
    # 40 (trunk) + 1 (MTP) = 41. To get the trunk count we subtract
    # n_layer_nextn (= {arch}.nextn_predict_layers).
    n_layer_total    = _read_n_layer(reader, arch)
    n_layer_nextn    = _read_n_layer_nextn(reader, arch)
    n_layer_trunk    = max(0, n_layer_total - n_layer_nextn)
    is_mtp_fn        = _make_mtp_filter(n_layer_trunk, n_layer_nextn)
    # Diagnostic: count how many tensors in reader.tensors look like MTP
    # by name (blk.{N}.* with N >= n_layer_trunk). If this count is > 0
    # but n_layer_nextn == 0, it means the source GGUF has MTP tensors
    # without the nextn_predict_layers KV -- we'll still filter them
    # out by their blk index.
    mtp_tensor_count = 0
    if n_layer_trunk > 0:
        for t in reader.tensors:
            if is_mtp_fn(t.name):
                mtp_tensor_count += 1
    logger.warning(
        "ternary_mtp_detection: arch=%s, block_count_total=%d, "
        "n_layer_nextn=%d, n_layer_trunk=%d, mtp_tensors_found=%d "
        "(filter_active=%s)",
        arch, n_layer_total, n_layer_nextn, n_layer_trunk,
        mtp_tensor_count,
        "yes" if n_layer_trunk > 0 else "no",
    )
    if mtp_tensor_count > 0:
        logger.warning(
            "ternary_mtp_detected: %d MTP tensors (blk.%d+) will be excluded "
            "from Hadamard rotation and from prism.hadamard.weight_names",
            mtp_tensor_count, n_layer_trunk,
        )
    elif n_layer_trunk == 0:
        logger.warning(
            "ternary_mtp_filter_inactive: %s.block_count KV not found or "
            "unreadable; MTP tensors (if any) will NOT be excluded from "
            "prism.hadamard.weight_names -- this will cause 'prism.hadamard "
            "weight not found' errors at load time if the source GGUF has "
            "any MTP/nextn-prediction tensors past the main trunk",
            arch,
        )

    # Copy all KV metadata from the source.
    _copy_kv_metadata(gguf, reader, writer, format_name=cfg.format)

    # Determine the format for each tensor.
    overrides = tensor_types or {}

    # -----------------------------------------------------------------
    # Compute the EFFECTIVE Hadamard block size.
    # -----------------------------------------------------------------
    # The PrismML runtime enforces BOTH:
    #   (a) every non-MTP regex-matching tensor must be in
    #       prism.hadamard.weight_names, AND
    #   (b) every tensor in weight_names must have n_per_row divisible
    #       by prism.hadamard.block_size.
    # So the block_size must divide *every* non-MTP regex-matching
    # tensor's n_per_row. We shrink the user's requested block down
    # (by factors of 2) until it divides all of them. For Sage-A1.5
    # this typically means 1024 -> 512 (because GQA V/K projections
    # and shared-expert tensors have n_per_row=512).
    requested_bs = cfg.hadamard_block_size
    if requested_bs > 0:
        effective_bs = requested_bs
        for t in reader.tensors:
            if not _ROTATE_PATTERN.fullmatch(t.name):
                continue
            if is_mtp_fn(t.name):
                continue
            n_per_row_t = int(t.shape[0]) if len(t.shape) >= 1 else int(t.n_elements)
            if n_per_row_t == 0:
                continue
            # Shrink effective_bs (power-of-2) until it divides n_per_row_t.
            while effective_bs > 1 and n_per_row_t % effective_bs != 0:
                effective_bs //= 2
        if effective_bs != requested_bs:
            logger.warning(
                "ternary_hadamard_block_size_adjusted: requested=%d, effective=%d "
                "(shrunk to divide every regex-matching tensor's n_per_row)",
                requested_bs, effective_bs,
            )
        # Replace cfg with an adjusted copy so every downstream call
        # uses the effective block size.
        cfg = TernaryConfig(
            format=cfg.format,
            hadamard_block_size=effective_bs,
            hadamard_seed=cfg.hadamard_seed,
            sign_mode=cfg.sign_mode,
            device=cfg.device,
        )

    # Collect rotated tensor names and sign data for the metadata.
    rotated_names: List[str] = []
    sign_values: Optional[List[int]] = None
    sign_widths: Optional[List[int]] = None

    if cfg.hadamard_block_size > 0 and cfg.sign_mode == "explicit":
        # Generate the sign pattern. For each unique n_per_row (= ne[0],
        # the innermost dim) of regex-matching NON-MTP tensors, generate
        # a sign vector. NOTE: t.shape[0] is ne[0] (innermost), NOT
        # t.shape[-1] (which is the outermost / row count). Using
        # shape[-1] here was a bug that put row counts in sign_widths.
        # MTP tensors are excluded -- they're not rotated, so they
        # don't consume a sign vector.
        unique_widths = set()
        for t in reader.tensors:
            if not bool(_ROTATE_PATTERN.fullmatch(t.name)):
                continue
            if is_mtp_fn(t.name):
                continue
            n = int(t.shape[0]) if len(t.shape) >= 1 else 0
            if n > 0 and n % cfg.hadamard_block_size == 0:
                unique_widths.add(n)
        sign_widths = sorted(unique_widths)
        sign_values = []
        rng = np.random.default_rng(cfg.hadamard_seed)
        for w in sign_widths:
            signs = rng.choice([-1, 1], size=w).tolist()
            sign_values.extend(int(s) for s in signs)

    # Process each tensor.
    ternary_type = _get_ggml_type(gguf, cfg.format)
    if ternary_type is None:
        logger.warning(
            "ternary_type_not_found: %s not in this gguf build; using fallback",
            cfg.format,
        )

    tensors = list(reader.tensors)
    # Cache: tensor_name -> (encoded_bytes, write_type) for ternary tensors,
    # or None for non-quantized tensors (use source bytes).
    tensor_data_cache = {}
    for t in tensors:
        name = t.name
        if max_tensors is not None and report.tensors_quantized >= max_tensors:
            # Not processed — will be copied from source.
            tensor_data_cache[name] = None
            continue

        type_name = str(t.tensor_type.name).replace(
            "GGMLQuantizationType.", ""
        ).upper()
        n_elements = int(t.n_elements)
        # GGML convention: ne[0] (= shape[0]) is the innermost dimension.
        n_per_row = int(t.shape[0]) if len(t.shape) >= 1 else n_elements
        n_rows = n_elements // n_per_row if n_per_row > 0 else 1

        # Only quantize float tensors (F16, BF16, F32).
        if type_name not in ("F16", "BF16", "F32"):
            tensor_data_cache[name] = None
            continue

        # Read the weights as float32 (used by both ternary and Q8_0 paths).
        if type_name == "F16":
            weights = np.frombuffer(
                t.data.tobytes(), dtype=np.float16
            ).astype(np.float32).reshape(n_rows, n_per_row)
        elif type_name == "BF16":
            raw = np.frombuffer(t.data.tobytes(), dtype=np.uint16)
            f32_bits = raw.astype(np.uint32) << 16
            weights = f32_bits.view(np.float32).reshape(n_rows, n_per_row)
        else:
            weights = np.frombuffer(
                t.data.tobytes(), dtype=np.float32
            ).reshape(n_rows, n_per_row)

        # ----- MTP tensor path: quantize as Q8_0 (no Hadamard rotation) -----
        # MTP (next-token-prediction) tensors are skipped by the runtime
        # when load_mtp=false, so they must NOT appear in
        # prism.hadamard.weight_names. We also keep them at Q8_0 (8.5 bpw)
        # instead of PQ2_0 (2.125 bpw) for better speculative-decoding
        # quality -- MTP is a small extra block so the size cost is
        # negligible, and the higher precision keeps the draft tokens
        # useful.
        if is_mtp_fn(name):
            Q8_0_BLOCK = 32  # ggml Q8_0 block size
            if n_per_row % Q8_0_BLOCK != 0:
                # Too small to quantize as Q8_0 (e.g. an MTP-specific
                # SSM tensor with n_per_row=4). Keep as F16 source bytes.
                tensor_data_cache[name] = None
                report.tensors_skipped += 1
                continue
            try:
                encoded = _encode_q8_0_batch(weights)
            except Exception as exc:
                logger.warning(
                    "ternary_mtp_q8_0_failed: %s: %s", name, str(exc),
                )
                tensor_data_cache[name] = None
                report.tensors_skipped += 1
                continue
            # GGML type ID for Q8_0 is 8 (ggml.h GGML_TYPE_Q8_0).
            Q8_0_TYPE = 8
            tensor_data_cache[name] = (encoded, Q8_0_TYPE)
            report.tensors_quantized += 1
            report.total_params += n_elements
            report.output_bytes += len(encoded)
            report.per_tensor.append((name, "Q8_0", n_elements, len(encoded)))
            continue

        # ----- Non-MTP tensor path: ternary quantization with Hadamard -----
        fmt = overrides.get(name, cfg.format).upper()
        be = block_elems(fmt)
        # Skip tensors whose n_per_row is not divisible by the ternary
        # block size (e.g., SSM state tensors with n_per_row=4 or 32).
        if n_per_row % be != 0:
            tensor_data_cache[name] = None
            report.tensors_skipped += 1
            continue

        if fmt not in ("PTQ1_0", "PQ2_0"):
            tensor_data_cache[name] = None
            continue

        # Quantize. Pass the MTP filter so MTP tensors are not rotated
        # (they'd otherwise be added to weight_names and trigger
        # "weight not found" at load time when load_mtp=false). MTP
        # tensors never reach this branch -- they took the Q8_0 path
        # above -- so the is_mtp check inside quantize_tensor_to_ternary
        # is now a safety net.
        try:
            encoded, was_rotated = quantize_tensor_to_ternary(
                weights, cfg, tensor_name=name, is_mtp_fn=is_mtp_fn,
            )
        except Exception as exc:
            logger.warning(
                "ternary_quantize_failed: %s: %s", name, str(exc),
            )
            tensor_data_cache[name] = None
            report.tensors_skipped += 1
            continue

        # Cache the encoded bytes + the write type.
        write_type = ternary_type or 142  # fallback to PQ2_0's raw ID
        tensor_data_cache[name] = (encoded, write_type)

        report.tensors_quantized += 1
        if was_rotated:
            report.tensors_rotated += 1
            rotated_names.append(name)
        report.total_params += n_elements
        report.output_bytes += len(encoded)
        report.per_tensor.append((name, fmt, n_elements, len(encoded)))

    # Also add any regex-matching NON-MTP tensor that wasn't picked up
    # by the first pass (e.g. kept as F32 because n_per_row didn't
    # divide 128, or skipped due to max_tensors limit). The PrismML
    # runtime requires every regex-matching tensor it *loads* to be in
    # prism.hadamard.weight_names, AND every tensor in weight_names to
    # have n_per_row divisible by block_size. The effective_bs
    # computation above guarantees the latter for every non-MTP
    # regex-matching tensor, so we can safely include them all here.
    # MTP tensors are excluded because the runtime skips them by
    # default (load_mtp=false) and would emit "weight not found".
    unrotated_regex_tensors = []
    for t in reader.tensors:
        name = t.name
        if name in rotated_names:
            continue
        if is_mtp_fn(name):
            continue
        if _ROTATE_PATTERN.fullmatch(name):
            rotated_names.append(name)
            unrotated_regex_tensors.append(name)

    if unrotated_regex_tensors:
        logger.warning(
            "ternary_unrotated_regex_tensors: %d regex-matching tensors were "
            "added to weight_names but NOT actually rotated at quantize time "
            "(their weights are stored unrotated, so runtime FWHT on "
            "activations will produce wrong output for these tensors): %s",
            len(unrotated_regex_tensors),
            ", ".join(unrotated_regex_tensors[:10]) + (
                ", ..." if len(unrotated_regex_tensors) > 10 else ""
            ),
        )

    if rotated_names:
        _write_prism_metadata(
            writer, cfg, rotated_names, sign_values, sign_widths,
        )
        logger.warning(
            "ternary_rotated_tensor_list: %d tensors (of %d total), "
            "effective_block_size=%d, mtp_excluded=%s",
            len(rotated_names), len(tensors), cfg.hadamard_block_size,
            "yes" if n_layer_nextn > 0 or mtp_tensor_count > 0 else "no",
        )

    # Register ALL tensors (in source order) with the writer.
    for t in reader.tensors:
        name = t.name
        cached = tensor_data_cache.get(name)
        # Original ggml shape: ne[0] (innermost), ne[1], ne[2], ...
        orig_shape = [int(d) for d in t.shape]
        n_elements = int(t.n_elements)

        if cached is not None:
            # Ternary tensor.
            encoded_bytes, write_type = cached
            arr = np.frombuffer(encoded_bytes, dtype=np.uint8)
            # Build the byte-space numpy shape:
            # - The ggml shape is [ne0, ne1, ne2, ...] (ne0=innermost).
            # - quant_shape_from_byte_shape only scales the LAST numpy
            #   dim (= ne0 after the gguf lib reverses to ggml order).
            # - So byte_shape[-1] = ne0 * type_size / block_size, and
            #   the other dims stay the same.
            # - In numpy (C-order), the shape is reversed from ggml:
            #   [..ne2, ne1, ne0_bytes].
            block_sz, type_sz = GGML_QUANT_SIZES.get(
                int(write_type) if isinstance(write_type, int) else 142,
                (128, 34),  # default to PQ2_0
            )
            byte_shape_np = list(reversed(orig_shape))
            if byte_shape_np:
                byte_shape_np[-1] = byte_shape_np[-1] * type_sz // block_sz
            try:
                arr = arr.reshape(byte_shape_np)
            except ValueError:
                # Fallback: 2D [n_rows, bytes_per_row].
                ne0 = orig_shape[0] if orig_shape else n_elements
                n_rows = n_elements // ne0 if ne0 > 0 else 1
                bpr = len(encoded_bytes) // n_rows if n_rows > 0 else len(encoded_bytes)
                arr = arr.reshape(n_rows, bpr) if n_rows > 1 else arr
            writer.add_tensor(name, arr, raw_dtype=write_type)
        else:
            # Non-ternary tensor: copy from source with the original type.
            raw = np.frombuffer(t.data.tobytes(), dtype=np.uint8)
            # Same approach: preserve original dimensionality.
            src_type_id = int(t.tensor_type) if hasattr(t.tensor_type, '__int__') else 0
            block_sz, type_sz = GGML_QUANT_SIZES.get(src_type_id, (1, 1))
            byte_shape_np = list(reversed(orig_shape))
            if byte_shape_np and type_sz > 0 and block_sz > 0:
                byte_shape_np[-1] = byte_shape_np[-1] * type_sz // block_sz
            try:
                raw = raw.reshape(byte_shape_np)
            except ValueError:
                ne0 = orig_shape[0] if orig_shape else n_elements
                n_rows = n_elements // ne0 if ne0 > 0 else 1
                bpr = raw.size // n_rows if n_rows > 0 else raw.size
                raw = raw.reshape(n_rows, bpr) if n_rows > 1 else raw
            src_type = t.tensor_type
            writer.add_tensor(name, raw, raw_dtype=src_type)

    # Finalize the GGUF.
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    # Write the actual tensor data in the same order as registered.
    for t in reader.tensors:
        name = t.name
        cached = tensor_data_cache.get(name)
        if cached is not None:
            encoded_bytes, _ = cached
            writer.write_tensor_data(
                np.frombuffer(encoded_bytes, dtype=np.uint8)
            )
        else:
            raw = np.frombuffer(t.data.tobytes(), dtype=np.uint8)
            writer.write_tensor_data(raw)

    writer.close()

    report.effective_bpw = (
        report.output_bytes * 8.0 / max(report.total_params, 1)
    )
    logger.info(
        "ternary_quantize_complete: %d quantized, %d skipped, %d rotated, bpw=%.4f",
        report.tensors_quantized, report.tensors_skipped,
        report.tensors_rotated, report.effective_bpw,
    )
    return report


__all__ = [
    "TernaryConfig",
    "TernaryReport",
    "quantize_tensor_to_ternary",
    "quantize_gguf_to_ternary",
]
