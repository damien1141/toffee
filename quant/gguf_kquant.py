"""
@file toffee/quant/gguf_kquant.py
   Exact codec for llama.cpp's scalar quant block layouts.

GSQ refines a quantized tensor *in format*: it changes the integer codes and
the per-group scales, then re-emits the tensor in the very same llama.cpp
block layout so the result loads in llama.cpp unchanged. That requires
reading and writing the block layouts exactly as llama.cpp defines them.

Design contract
---------------
The codec is built so that correctness is *checkable* rather than assumed:

  1. Every decoder reproduces llama.cpp's ``to_float`` output exactly
     (verified by the test-suite against the real ggml shared library).
  2. Every encoder is the exact mirror of its decoder -- it writes the same
     field at the same bit position the decoder reads it from. Therefore
     ``encode(decode(bytes)) == bytes`` for any llama.cpp-produced bytes, and
     anything we emit decodes identically under llama.cpp.

Supported types
---------------
The scalar types TOFFEE's APEX tiers can emit::

    Q8_0   32 elements   fp16 scale + 32 int8 codes
    Q2_K  256 elements   16 groups of 16, 4 codes, fp16 super-scale/min
    Q3_K  256 elements   16 groups of 16, 8 codes, signed fp16 super-scale
    Q4_K  256 elements    8 groups of 32, 16 codes, fp16 super-scale/min
    Q5_K  256 elements    8 groups of 32, 32 codes, fp16 super-scale/min
    Q6_K  256 elements   16 groups of 16, 64 codes, signed fp16 super-scale

Value conventions
-----------------
Every type realises a per-sub-group affine map

    value = (code - bias) * scale - min

``scale`` is the *effective* per-group scale (the block's fp16 super-scale
already folded in) and ``min`` the effective per-group minimum. Types whose
sub-scale field is signed (Q3_K, Q6_K) fold the sign into ``scale``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Type descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeSpec:
    """Structural description of one llama.cpp quant block layout.

    Attributes:
        block_elems: elements per block.
        n_groups: sub-groups per block (the granularity of the sub-scale).
        code_min: smallest representable code.
        code_max: largest representable code.
        bias: code offset in the value map: value = (code - bias) * scale.
        scale_ratio_max: largest |effective_scale / d| representable, where
            ``d`` is the block's fp16 super-scale.
        scale_offset: offset applied to the stored sub-scale before use
            (Q3_K stores ``ratio + 32``).
        sc_raw_min: smallest storable sub-scale.
        sc_raw_max: largest storable sub-scale.
        scale_bias: value subtracted from the raw sub-scale to get the
            effective scale (Q3_K stores ``scale + 32``).
        has_min: whether the type carries a per-group minimum.
        min_ratio_max: largest |effective_min / dmin| representable.
        min_raw_max: largest storable sub-min.
    """

    block_elems: int
    n_groups: int
    code_min: int
    code_max: int
    bias: int
    scale_ratio_max: int = 0
    scale_offset: int = 0
    sc_raw_min: int = 0
    sc_raw_max: int = 0
    scale_bias: int = 0
    has_min: bool = False
    min_ratio_max: int = 0
    min_raw_max: int = 0
    #: Optional non-uniform codebook. When set, the value map is
    #: ``codebook[code] * scale`` instead of ``(code - bias) * scale``.
    #: IQ4_XS is the only type with one.
    codebook: Optional[Tuple[float, ...]] = None

    @property
    def group_size(self) -> int:
        return self.block_elems // self.n_groups

    @property
    def n_levels(self) -> int:
        return self.code_max - self.code_min + 1


#: llama.cpp's ``kvalues_iq4nl`` codebook (16 entries).
IQ4NL_CODEBOOK = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)

TYPE_SPECS: Dict[str, TypeSpec] = {
    "Q8_0": TypeSpec(32, 1, -128, 127, 0),
    "Q2_K": TypeSpec(256, 16, 0, 3, 0, 15, 0, 0, 15, 0, True, 15, 15),
    "Q3_K": TypeSpec(256, 16, -4, 3, 0, 31, 32, 0, 63, 32),
    "Q4_K": TypeSpec(256, 8, 0, 15, 0, 63, 0, 0, 63, 0, True, 63, 63),
    "Q5_K": TypeSpec(256, 8, 0, 31, 0, 63, 0, 0, 63, 0, True, 63, 63),
    "Q6_K": TypeSpec(256, 16, 0, 63, 32, 127, 0, -128, 127, 0),
    # IQ4_XS: 8 groups of 32, 4-bit codes indexing a non-uniform codebook.
    # The sub-scale is 6 bits and *signed* about 32 (``d * (ls - 32)``), so
    # the scale_bias is 32 and the raw range is 0..63.
    "IQ4_XS": TypeSpec(
        256, 8, 0, 15, 0, 31, 32, 0, 63, 32,
        codebook=tuple(float(v) for v in IQ4NL_CODEBOOK),
    ),
}

#: Bytes per block for each supported type.
BLOCK_BYTES: Dict[str, int] = {
    "Q8_0": 2 + 32,
    "Q2_K": 4 + 16 + 64,
    "Q3_K": 2 + 12 + 32 + 64,
    "Q4_K": 4 + 12 + 128,
    "Q5_K": 4 + 12 + 32 + 128,
    "Q6_K": 2 + 16 + 192,
    "IQ4_XS": 2 + 2 + 4 + 128,
}

#: Elements per block for each supported type.
BLOCK_ELEMS: Dict[str, int] = {k: v.block_elems for k, v in TYPE_SPECS.items()}

#: Number of sub-groups per block.
GROUPS_PER_BLOCK: Dict[str, int] = {k: v.n_groups for k, v in TYPE_SPECS.items()}

#: Elements per sub-group.
GROUP_SIZE: Dict[str, int] = {k: v.group_size for k, v in TYPE_SPECS.items()}

#: Code-space size per type (number of representable levels).
CODE_LEVELS: Dict[str, int] = {k: v.n_levels for k, v in TYPE_SPECS.items()}


class UnsupportedQuantType(ValueError):
    """Raised when a quant type has no implemented block codec."""


def fp32_to_fp16_bytes(value: float) -> bytes:
    """Encode a float32 as llama.cpp's ``ggml_half`` (IEEE fp16, LE)."""
    return struct.pack("<e", value)


def fp16_bytes_to_fp32(raw: bytes) -> float:
    """Decode llama.cpp's ``ggml_half`` bytes into a Python float."""
    return float(struct.unpack("<e", raw)[0])


def nearest_int(fval: float) -> int:
    """llama.cpp's ``nearest_int``: round to nearest, ties away from zero."""
    fval = float(fval)
    if not np.isfinite(fval):
        raise ValueError(f"nearest_int got a non-finite value: {fval}")
    return int(np.floor(fval + 0.5)) if fval >= 0 else int(np.ceil(fval - 0.5))


def get_scale_min_k4(j: int, q: np.ndarray) -> Tuple[int, int]:
    """Unpack the 6-bit scale and min of sub-group ``j`` from ``q`` (12 bytes).

    Mirrors ``get_scale_min_k4`` in ggml-quants.c.
    """
    q = np.asarray(q, dtype=np.uint8)
    if j < 4:
        return int(q[j] & 63), int(q[j + 4] & 63)
    d = int((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4))
    m = int((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4))
    return d, m


def pack_scale_min_k4(scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    """Pack 8 six-bit scales and 8 six-bit mins into llama.cpp's 12 bytes.

    Exact inverse of :func:`get_scale_min_k4`.
    """
    scales = np.asarray(scales, dtype=np.int64)
    mins = np.asarray(mins, dtype=np.int64)
    out = np.zeros(12, dtype=np.uint8)
    for j in range(8):
        ls = int(scales[j]) & 63
        lm = int(mins[j]) & 63
        if j < 4:
            out[j] |= ls
            out[j + 4] |= lm
        else:
            out[j + 4] = (out[j + 4] & 0xF0) | (ls & 0x0F) | ((lm & 0x0F) << 4)
            out[j - 4] |= ((ls >> 4) << 6)
            out[j - 0] |= ((lm >> 4) << 6)
    return out


# ---------------------------------------------------------------------------
# Raw block state
# ---------------------------------------------------------------------------

@dataclass
class BlockRaw:
    """llama.cpp's raw scale fields for one block.

    Attributes:
        d: the fp16 super-scale (or the single scale for Q8_0).
        dmin: the fp16 super-minimum (asymmetric K-quants only).
        sc_raw: raw integer sub-scales, one per sub-group.
        m_raw: raw integer sub-mins, one per sub-group (asymmetric K-quants).
    """

    d: float = 0.0
    dmin: float = 0.0
    sc_raw: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    m_raw: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def effective_scales(raw: BlockRaw, spec: TypeSpec) -> np.ndarray:
    """Per-sub-group effective scale implied by a block's raw fields."""
    if spec.n_groups == 1:
        return np.full(spec.n_groups, raw.d, dtype=np.float32)
    return (raw.d * (raw.sc_raw - spec.scale_bias)).astype(np.float32)


def effective_mins(raw: BlockRaw, spec: TypeSpec) -> np.ndarray:
    """Per-sub-group effective minimum implied by a block's raw fields."""
    if not spec.has_min:
        return np.zeros(spec.n_groups, dtype=np.float32)
    if spec.n_groups == 1:
        return np.full(spec.n_groups, raw.dmin, dtype=np.float32)
    return (raw.dmin * raw.m_raw).astype(np.float32)


# ---------------------------------------------------------------------------
# Decoders
#
# Element-index -> (byte, shift) conventions, derived from llama.cpp:
#   * Q2_K / Q3_K: byte = (e // 128) * 32 + (e % 32),
#                   shift = 2 * ((e % 128) // 32),
#                   group = (e // 128) * 8 + (e % 128) // 16
#   * Q4_K / Q5_K: byte = (e // 64) * 32 + (e % 32),
#                   low nibble when (e % 64) < 32 else high nibble,
#                   group = e // 32
#   * Q6_K:        ql byte = (e // 128) * 64 + (e % 32) + (32 if the 32-block
#                   index is odd else 0), high nibble when the 32-block index
#                   is >= 2, qh byte = (e // 128) * 32 + (e % 32),
#                   qh bit pair = 2 * ((e % 128) // 32),
#                   group = (e // 128) * 8 + (e % 128) // 16
# ---------------------------------------------------------------------------

def _decode_q8_0(buf: bytes, spec: TypeSpec) -> Tuple[np.ndarray, BlockRaw]:
    d = fp16_bytes_to_fp32(buf[0:2])
    qs = np.frombuffer(buf, dtype=np.int8, offset=2, count=32).astype(np.int32)
    return qs, BlockRaw(d=d)


def _decode_q2_k(buf: bytes, spec: TypeSpec) -> Tuple[np.ndarray, BlockRaw]:
    # struct block_q2_K { uint8_t scales[16]; uint8_t qs[64]; half d; half dmin; }
    sc_raw = np.frombuffer(buf, dtype=np.uint8, offset=0, count=16)
    packed = np.frombuffer(buf, dtype=np.uint8, offset=16, count=64)
    d = fp16_bytes_to_fp32(buf[80:82])
    dmin = fp16_bytes_to_fp32(buf[82:84])
    codes = np.empty(256, dtype=np.int32)
    for e in range(256):
        byte = int(packed[(e // 128) * 32 + (e % 32)])
        codes[e] = (byte >> (2 * ((e % 128) // 32))) & 0x03
    return codes, BlockRaw(
        d=d,
        dmin=dmin,
        sc_raw=(sc_raw & 0x0F).astype(np.int64),
        m_raw=((sc_raw >> 4) & 0x0F).astype(np.int64),
    )


def _decode_q3_k(buf: bytes, spec: TypeSpec) -> Tuple[np.ndarray, BlockRaw]:
    # struct block_q3_K { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; half d; }
    hmask = np.frombuffer(buf, dtype=np.uint8, offset=0, count=32)
    packed = np.frombuffer(buf, dtype=np.uint8, offset=32, count=64)
    sc_raw = np.frombuffer(buf, dtype=np.uint8, offset=96, count=12)
    d = fp16_bytes_to_fp32(buf[108:110])

    # Unpack the 6-bit scales exactly as llama.cpp's dequantize_row_q3_K does.
    # llama.cpp memcpy's 12 bytes into a 16-byte `aux` and then rewrites all
    # four 32-bit words, so the transform must be reproduced in 32-bit words.
    aux = np.zeros(16, dtype=np.uint8)
    aux[0:12] = sc_raw
    w = aux.view("<u4")
    tmp = w[2]
    kmask1 = np.uint32(0x03030303)
    kmask2 = np.uint32(0x0F0F0F0F)
    w0, w1 = w[0], w[1]
    w[2] = ((w0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    w[3] = ((w1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    w[0] = (w0 & kmask2) | (((tmp >> 0) & kmask1) << 4)
    w[1] = (w1 & kmask2) | (((tmp >> 2) & kmask1) << 4)
    # aux now holds 16 bytes = 16 int8 scales, each biased by 32.
    scales_raw = aux.view(np.int8)[:16].astype(np.int64)

    codes = np.empty(256, dtype=np.int32)
    for e in range(256):
        byte = int(packed[(e // 128) * 32 + (e % 32)])
        code = (byte >> (2 * ((e % 128) // 32))) & 0x03
        # The hmask supplies the high bit: element e -> byte (e % 32),
        # bit (e // 32). A *clear* bit means the code is negative.
        if not (hmask[e % 32] & (1 << (e // 32))):
            code -= 4
        codes[e] = code
    return codes, BlockRaw(d=d, sc_raw=scales_raw)


def _decode_q4_k(buf: bytes, spec: TypeSpec) -> Tuple[np.ndarray, BlockRaw]:
    # struct block_q4_K { half d; half dmin; uint8_t scales[12]; uint8_t qs[128]; }
    d = fp16_bytes_to_fp32(buf[0:2])
    dmin = fp16_bytes_to_fp32(buf[2:4])
    sc_raw = np.frombuffer(buf, dtype=np.uint8, offset=4, count=12)
    nib = np.frombuffer(buf, dtype=np.uint8, offset=16, count=128)
    codes = np.empty(256, dtype=np.int32)
    for e in range(256):
        byte = int(nib[(e // 64) * 32 + (e % 32)])
        codes[e] = (byte & 0x0F) if (e % 64) < 32 else (byte >> 4)
    sc = np.empty(8, dtype=np.int64)
    mn = np.empty(8, dtype=np.int64)
    for j in range(8):
        s, m = get_scale_min_k4(j, sc_raw)
        sc[j] = s
        mn[j] = m
    return codes, BlockRaw(d=d, dmin=dmin, sc_raw=sc, m_raw=mn)


def _decode_q5_k(buf: bytes, spec: TypeSpec) -> Tuple[np.ndarray, BlockRaw]:
    d = fp16_bytes_to_fp32(buf[0:2])
    dmin = fp16_bytes_to_fp32(buf[2:4])
    sc_raw = np.frombuffer(buf, dtype=np.uint8, offset=4, count=12)
    qh = np.frombuffer(buf, dtype=np.uint8, offset=16, count=32)
    nib = np.frombuffer(buf, dtype=np.uint8, offset=48, count=128)
    codes = np.empty(256, dtype=np.int32)
    for e in range(256):
        byte = int(nib[(e // 64) * 32 + (e % 32)])
        code = (byte & 0x0F) if (e % 64) < 32 else (byte >> 4)
        # The 5th bit lives in qh: byte (e % 32), bit
        # 2 * (e // 64) + (e % 64) // 32.
        bit = 2 * (e // 64) + ((e % 64) // 32)
        if qh[e % 32] & (1 << bit):
            code += 16
        codes[e] = code
    sc = np.empty(8, dtype=np.int64)
    mn = np.empty(8, dtype=np.int64)
    for j in range(8):
        s, m = get_scale_min_k4(j, sc_raw)
        sc[j] = s
        mn[j] = m
    return codes, BlockRaw(d=d, dmin=dmin, sc_raw=sc, m_raw=mn)


def _decode_q6_k(buf: bytes, spec: TypeSpec) -> Tuple[np.ndarray, BlockRaw]:
    # struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; }
    ql = np.frombuffer(buf, dtype=np.uint8, offset=0, count=128)
    qh = np.frombuffer(buf, dtype=np.uint8, offset=128, count=64)
    sc_raw = np.frombuffer(buf, dtype=np.int8, offset=192, count=16).astype(np.int64)
    d = fp16_bytes_to_fp32(buf[208:210])
    codes = np.empty(256, dtype=np.int32)
    for e in range(256):
        blk = (e % 128) // 32          # 0..3
        l = e % 32
        ql_byte = (e // 128) * 64 + l + (32 if blk % 2 == 1 else 0)
        nibble = (ql[ql_byte] >> 4) if blk >= 2 else (ql[ql_byte] & 0x0F)
        qh_byte = (e // 128) * 32 + l
        high = (qh[qh_byte] >> (2 * blk)) & 0x03
        codes[e] = nibble | (high << 4)
    return codes, BlockRaw(d=d, sc_raw=sc_raw)


# ---------------------------------------------------------------------------
# Encoders -- exact mirrors of the decoders above.
# ---------------------------------------------------------------------------

def _encode_q8_0(codes: np.ndarray, raw: BlockRaw, spec: TypeSpec) -> bytes:
    return fp32_to_fp16_bytes(raw.d) + codes.astype(np.int8).tobytes()


def _encode_q2_k(codes: np.ndarray, raw: BlockRaw, spec: TypeSpec) -> bytes:
    # struct block_q2_K { uint8_t scales[16]; uint8_t qs[64]; half d; half dmin; }
    packed_scales = ((raw.sc_raw.astype(np.uint8) & 0x0F)
                      | ((raw.m_raw.astype(np.uint8) & 0x0F) << 4))
    packed = np.zeros(64, dtype=np.uint8)
    for e in range(256):
        packed[(e // 128) * 32 + (e % 32)] |= (
            int(codes[e]) & 0x03) << (2 * ((e % 128) // 32))
    return (packed_scales.tobytes() + packed.tobytes()
            + fp32_to_fp16_bytes(raw.d) + fp32_to_fp16_bytes(raw.dmin))


def _encode_q3_k(codes: np.ndarray, raw: BlockRaw, spec: TypeSpec) -> bytes:
    # struct block_q3_K { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; half d; }
    # Pack the 16 six-bit sub-scales exactly the way llama.cpp does.
    s = raw.sc_raw.astype(np.int64) & 0x3F
    if s.size != 16:
        raise ValueError("Q3_K needs 16 sub-scales")
    o = np.zeros(12, dtype=np.uint8)
    for j in range(16):
        v = int(s[j])
        if j < 8:
            o[j] |= v & 0x0F
        else:
            o[j - 8] |= (v & 0x0F) << 4
        o[8 + (j % 4)] |= ((v >> 4) & 0x03) << (2 * (j // 4))

    hmask = np.zeros(32, dtype=np.uint8)
    packed = np.zeros(64, dtype=np.uint8)
    for e in range(256):
        code = int(codes[e])
        if code < 0:
            code += 4
        else:
            hmask[e % 32] |= np.uint8(1 << (e // 32))
        packed[(e // 128) * 32 + (e % 32)] |= (code & 0x03) << (2 * ((e % 128) // 32))
    return (hmask.tobytes() + packed.tobytes() + o.tobytes()
            + fp32_to_fp16_bytes(raw.d))


def _encode_q4_k(codes: np.ndarray, raw: BlockRaw, spec: TypeSpec) -> bytes:
    packed_scales = pack_scale_min_k4(raw.sc_raw, raw.m_raw)
    nib = np.zeros(128, dtype=np.uint8)
    for e in range(256):
        idx = (e // 64) * 32 + (e % 32)
        if (e % 64) < 32:
            nib[idx] |= int(codes[e]) & 0x0F
        else:
            nib[idx] |= (int(codes[e]) & 0x0F) << 4
    return (fp32_to_fp16_bytes(raw.d) + fp32_to_fp16_bytes(raw.dmin)
            + packed_scales.tobytes() + nib.tobytes())


def _encode_q5_k(codes: np.ndarray, raw: BlockRaw, spec: TypeSpec) -> bytes:
    packed_scales = pack_scale_min_k4(raw.sc_raw, raw.m_raw)
    qh = np.zeros(32, dtype=np.uint8)
    nib = np.zeros(128, dtype=np.uint8)
    for e in range(256):
        idx = (e // 64) * 32 + (e % 32)
        code = int(codes[e])
        if (e % 64) < 32:
            nib[idx] |= code & 0x0F
        else:
            nib[idx] |= (code & 0x0F) << 4
        if code > 15:
            bit = 2 * (e // 64) + ((e % 64) // 32)
            qh[e % 32] |= np.uint8(1 << bit)
    return (fp32_to_fp16_bytes(raw.d) + fp32_to_fp16_bytes(raw.dmin)
            + packed_scales.tobytes() + qh.tobytes() + nib.tobytes())


def _encode_q6_k(codes: np.ndarray, raw: BlockRaw, spec: TypeSpec) -> bytes:
    ql = np.zeros(128, dtype=np.uint8)
    qh = np.zeros(64, dtype=np.uint8)
    for e in range(256):
        blk = (e % 128) // 32
        l = e % 32
        ql_idx = (e // 128) * 64 + l + (32 if blk % 2 == 1 else 0)
        qh_idx = (e // 128) * 32 + l
        code = int(codes[e]) & 0x3F
        if blk >= 2:
            ql[ql_idx] |= (code & 0x0F) << 4
        else:
            ql[ql_idx] |= code & 0x0F
        qh[qh_idx] |= ((code >> 4) & 0x03) << (2 * blk)
    return (ql.tobytes() + qh.tobytes() + raw.sc_raw.astype(np.int8).tobytes()
            + fp32_to_fp16_bytes(raw.d))


# ---------------------------------------------------------------------------
# IQ4_XS
#
# IQ4_XS is not a K-quant: it has its own block layout, a 6-bit sub-scale
# and a 4-bit codebook that is *not* uniform. The block is
#
#     ggml_half d;
#     uint16_t scales_h;    // the high 2 bits of each of the 8 sub-scales
#     uint8_t  scales_l[4];  // the low 4 bits of each of the 8 sub-scales
#     uint8_t  qs[128];      // 256 nibbles
#
# so the sub-scale of group g is
#
#     ls = (scales_l[g // 2] >> 4 * (g % 2) & 0xf) | ((scales_h >> 2 * g) & 3) << 4
#
# and the effective scale is ``d * (ls - 32)``. The codes index
# ``kvalues_iq4nl``, a fixed non-uniform 4-bit codebook.
#
# The value map is therefore value = codebook[code] * d * (ls - 32), i.e. a
# pure scale with no per-group offset -- so the search has exactly one
# knob per group: the 6-bit sub-scale.
# ---------------------------------------------------------------------------

#: The 8 sub-scales of an IQ4_XS block, unpacked from ``scales_h``/``scales_l``.
def _unpack_iq4_xs_scales(scales_h: int, scales_l: bytes) -> np.ndarray:
    """Unpack the 8 six-bit sub-scales of one IQ4_XS block.

    Mirrors the expression in ``dequantize_row_iq4_xs``:
    ``(scales_l[ib/2] >> 4*(ib%2) & 0xf) | ((scales_h >> 2*ib) & 3) << 4``.

    Args:
        scales_h: the block's 16-bit high-scale field.
        scales_l: the block's 4-byte low-scale field.

    Returns:
        An int64 array of 8 sub-scales.
    """
    out = np.zeros(8, dtype=np.int64)
    for ib in range(8):
        lo = (scales_l[ib // 2] >> (4 * (ib % 2))) & 0xF
        hi = (scales_h >> (2 * ib)) & 0x3
        out[ib] = lo | (hi << 4)
    return out


def _pack_iq4_xs_scales(scales: np.ndarray) -> Tuple[int, bytes]:
    """Pack 8 six-bit sub-scales into ``(scales_h, scales_l)``.

    The exact inverse of :func:`_unpack_iq4_xs_scales`.

    Args:
        scales: 8 sub-scales, each in 0..63.

    Returns:
        ``(scales_h, scales_l)``.
    """
    scales_h = 0
    scales_l = bytearray(4)
    for ib in range(8):
        s = int(scales[ib]) & 0x3F
        scales_l[ib // 2] |= (s & 0xF) << (4 * (ib % 2))
        scales_h |= (s >> 4) << (2 * ib)
    return scales_h, bytes(scales_l)


def _decode_iq4_xs(chunk: bytes, spec: TypeSpec):
    """Decode one IQ4_XS block.

    Args:
        chunk: the 146-byte block.
        spec: the type descriptor.

    Returns:
        (codes, raw).
    """
    d = fp16_bytes_to_fp32(chunk[0:2])
    scales_h = struct.unpack_from("<H", chunk, 2)[0]
    scales_l = chunk[4:8]
    qs = np.frombuffer(chunk[8:8 + 128], dtype=np.uint8)

    sc_raw = _unpack_iq4_xs_scales(scales_h, scales_l)

    # 8 groups of 32; each group holds 16 bytes, low nibble first.
    codes = np.empty(256, dtype=np.int32)
    for g in range(8):
        base = g * 16
        nib = qs[base:base + 16]
        grp = np.empty(32, dtype=np.int32)
        grp[0:16] = nib & 0x0F
        grp[16:32] = nib >> 4
        codes[g * 32:(g + 1) * 32] = grp

    raw = BlockRaw(
        d=d,
        dmin=0.0,
        sc_raw=sc_raw,
        m_raw=np.zeros(0, dtype=np.int64),
    )
    return codes, raw


def _encode_iq4_xs(codes: np.ndarray, raw: BlockRaw, spec: TypeSpec) -> bytes:
    """Encode one IQ4_XS block.

    Args:
        codes: 256 integer codes.
        raw: the block's raw scale fields.
        spec: the type descriptor.

    Returns:
        The 146-byte block.
    """
    scales_h, scales_l = _pack_iq4_xs_scales(raw.sc_raw)
    qs = bytearray(128)
    for g in range(8):
        base = g * 16
        grp = codes[g * 32:(g + 1) * 32]
        lo = grp[0:16].astype(np.uint8) & 0x0F
        hi = grp[16:32].astype(np.uint8) & 0x0F
        for i in range(16):
            qs[base + i] = int(lo[i]) | (int(hi[i]) << 4)
    return (
        fp32_to_fp16_bytes(raw.d)
        + struct.pack("<H", scales_h)
        + scales_l
        + bytes(qs)
    )


#: type name -> (decode, encode) pair.
_CODECS: Dict[str, Tuple] = {
    "Q8_0": (_decode_q8_0, _encode_q8_0),
    "Q2_K": (_decode_q2_k, _encode_q2_k),
    "Q3_K": (_decode_q3_k, _encode_q3_k),
    "Q4_K": (_decode_q4_k, _encode_q4_k),
    "Q5_K": (_decode_q5_k, _encode_q5_k),
    "Q6_K": (_decode_q6_k, _encode_q6_k),
    "IQ4_XS": (_decode_iq4_xs, _encode_iq4_xs),
}


def is_supported(type_name: str) -> bool:
    """True when a block codec exists for this quant type."""
    return type_name.upper() in _CODECS


def spec_for(type_name: str) -> TypeSpec:
    """Structural descriptor for a quant type."""
    key = type_name.upper()
    if key not in TYPE_SPECS:
        raise UnsupportedQuantType(
            f"No block codec for {type_name!r}; supported: {sorted(_CODECS)}"
        )
    return TYPE_SPECS[key]


def decode_blocks(data: bytes, type_name: str):
    """Decode a buffer of quant blocks.

    Args:
        data: raw block bytes.
        type_name: llama.cpp quant type name.

    Returns:
        (codes, scales, mins, raw) where ``codes`` has shape
        (n_blocks, block_elems), ``scales``/``mins`` have shape
        (n_blocks, n_groups) and hold the *effective* per-group values.
    """
    key = type_name.upper()
    spec = spec_for(key)
    decode = _CODECS[key][0]
    block_bytes = BLOCK_BYTES[key]
    if len(data) % block_bytes != 0:
        raise ValueError(
            f"Buffer of {len(data)} bytes is not a multiple of the {key} block "
            f"size {block_bytes}"
        )
    n_blocks = len(data) // block_bytes
    all_codes = np.empty((n_blocks, spec.block_elems), dtype=np.int32)
    all_scales = np.empty((n_blocks, spec.n_groups), dtype=np.float32)
    all_mins = np.empty((n_blocks, spec.n_groups), dtype=np.float32)
    raws: List[BlockRaw] = []
    for b in range(n_blocks):
        chunk = data[b * block_bytes:(b + 1) * block_bytes]
        codes, raw = decode(chunk, spec)
        all_codes[b] = codes
        all_scales[b] = effective_scales(raw, spec)
        all_mins[b] = effective_mins(raw, spec)
        raws.append(raw)
    return all_codes, all_scales, all_mins, raws


def encode_blocks(
    codes: np.ndarray,
    type_name: str,
    raw: Optional[List[BlockRaw]] = None,
) -> bytes:
    """Encode quant blocks into llama.cpp's binary layout.

    Args:
        codes: (n_blocks, block_elems) integer codes.
        type_name: llama.cpp quant type name.
        raw: per-block raw scale fields, as returned by :func:`decode_blocks`.
            Required for byte-exact re-encoding.

    Returns:
        The encoded bytes.
    """
    key = type_name.upper()
    spec = spec_for(key)
    encode = _CODECS[key][1]
    codes = np.asarray(codes, dtype=np.int32)
    n_blocks = codes.shape[0]
    if raw is None or len(raw) != n_blocks:
        raise ValueError(
            "encode_blocks needs the per-block raw scale fields from "
            "decode_blocks to reproduce the layout exactly"
        )
    out = bytearray()
    for b in range(n_blocks):
        out += encode(codes[b], raw[b], spec)
    return bytes(out)


def dequantize_blocks(
    codes: np.ndarray,
    scales: np.ndarray,
    mins: np.ndarray,
    type_name: str,
) -> np.ndarray:
    """Reconstruct float values from decoded blocks.

    Args:
        codes: (n_blocks, block_elems) integer codes.
        scales: (n_blocks, n_groups) effective scales.
        mins: (n_blocks, n_groups) effective minimums.
        type_name: llama.cpp quant type name.

    Returns:
        (n_blocks, block_elems) float array.
    """
    spec = spec_for(type_name)
    group_size = spec.group_size
    codes = np.asarray(codes, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    mins = np.asarray(mins, dtype=np.float32)
    per_group_scale = scales.repeat(group_size, axis=1)
    per_group_min = mins.repeat(group_size, axis=1)
    if spec.codebook is not None:
        # A non-uniform codebook: the value is codebook[code] * scale.
        book = np.asarray(spec.codebook, dtype=np.float32)
        return book[codes.astype(np.int32)] * per_group_scale - per_group_min
    return (codes - spec.bias) * per_group_scale - per_group_min


def fit_blocks(
    values: np.ndarray,
    type_name: str,
    imatrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit values onto a type's integer grid, per sub-group.

    This is the reference grid fit that GSQ then refines: it picks, per
    sub-group, the scale (and minimum) that best represents the values on the
    type's code range, and returns the resulting codes.

    Args:
        values: (n_blocks, block_elems) float array.
        type_name: llama.cpp quant type name.
        imatrix: optional per-column importance weights.

    Returns:
        (codes, scales, mins) with effective scales and minimums.
    """
    spec = spec_for(type_name)
    values = np.asarray(values, dtype=np.float32)
    n_blocks, block_elems = values.shape
    group_size = spec.group_size
    lo_code, hi_code = spec.code_min, spec.code_max

    if imatrix is not None:
        w = np.broadcast_to(
            np.asarray(imatrix, dtype=np.float32).reshape(1, -1), values.shape
        ).astype(np.float32)
    else:
        w = np.ones_like(values)

    codes = np.zeros((n_blocks, block_elems), dtype=np.int32)
    scales = np.zeros((n_blocks, spec.n_groups), dtype=np.float32)
    mins = np.zeros((n_blocks, spec.n_groups), dtype=np.float32)

    for b in range(n_blocks):
        for g in range(spec.n_groups):
            lo = g * group_size
            hi = lo + group_size
            v = values[b, lo:hi].astype(np.float64)
            wt = w[b, lo:hi].astype(np.float64)
            if wt.sum() <= 0:
                wt = np.ones_like(wt)

            # Weighted mean is the optimal grid centre for a squared error.
            centre = float((v * wt).sum() / wt.sum())
            span = float(v.max() - v.min())
            if span <= 0:
                span = max(abs(float(v.max())), 1e-8)
            base = span / max(hi_code - lo_code, 1)

            best = None
            for mult in (0.85, 0.92, 0.96, 1.0, 1.04, 1.08, 1.15):
                s = base * mult
                c = np.clip(np.rint((v - centre) / s), lo_code - spec.bias, hi_code - spec.bias)
                recon = (c + spec.bias) * s - (centre + spec.bias) * s
                total = float((recon ** 2 * wt).sum())
                if best is None or total < best[0]:
                    best = (total, s, c)
            _, s, c = best
            codes[b, lo:hi] = c.astype(np.int32)
            scales[b, g] = s
            mins[b, g] = centre
    return codes, scales, mins


__all__ = [
    "BLOCK_BYTES",
    "BLOCK_ELEMS",
    "BlockRaw",
    "CODE_LEVELS",
    "GROUP_SIZE",
    "GROUPS_PER_BLOCK",
    "TYPE_SPECS",
    "TypeSpec",
    "UnsupportedQuantType",
    "decode_blocks",
    "dequantize_blocks",
    "effective_mins",
    "effective_scales",
    "encode_blocks",
    "fit_blocks",
    "get_scale_min_k4",
    "is_supported",
    "nearest_int",
    "pack_scale_min_k4",
    "spec_for",
]
