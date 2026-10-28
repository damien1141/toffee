"""
Ternary weight codecs for the PrismML llama.cpp fork.

Implements the PTQ1_0 and PQ2_0 block formats exactly as defined in
``ggml/src/ggml-common.h`` and ``ggml/src/ggml-quants.c`` of
https://github.com/PrismML-Eng/llama.cpp.

Both formats are per-group-128 ternary: each weight is mapped to
{-1, 0, +1} via round-to-nearest against the group's absmax, then packed.

- **PTQ1_0** (1.75 bpw): trit-based packing. 5 trits per byte in the
  ``qs`` array (24 bytes -> 120 values) + 4 trits per byte in the
  ``qh`` array (2 bytes -> 8 values) + 1 fp16 scale. Total: 28 bytes
  per 128 weights = 1.75 bpw.

- **PQ2_0** (2.125 bpw): 2-bit packing. 4 values per byte in the ``qs``
  array (32 bytes -> 128 values) + 1 fp16 scale. Total: 34 bytes per
  128 weights = 2.125 bpw. The 4th 2-bit value (3) is unused for ternary
  inputs but the codec is identical to Q2_0.

The trit packing uses a ceiling-division trick: the base-3 value
[0, 242] is mapped to a byte [0, 255] via ``byte = (val * 256 + 242) / 243``
(integer arithmetic). The inverse uses ``trit_n = (byte * 3^(n+1)) >> 8``.
This is a 1-byte-in, 5-trits-out mapping that fits exactly in uint8.

The C reference is the source of truth; this Python implementation is
verified to produce byte-identical output via ``test_codec.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Block size (group size) for both PTQ1_0 and PQ2_0.
QK_PTQ1_0 = 128
QK_PQ2_0 = 128

# Trit packing constants (matches the C ``pow3`` table).
_POW3 = np.array([1, 3, 9, 27, 81, 243], dtype=np.uint16)


@dataclass(frozen=True)
class TernaryBlockRaw:
    """Raw fields of one ternary block (PTQ1_0 or PQ2_0).

    Attributes:
        d: the fp16 scale (absmax of the group).
        qs: the packed codes (24 bytes for PTQ1_0, 32 bytes for PQ2_0).
        qh: the high-byte packed codes (2 bytes for PTQ1_0, 0 for PQ2_0).
    """
    d: float
    qs: np.ndarray  # uint8
    qh: np.ndarray  # uint8


# ---------------------------------------------------------------------------
# PQ2_0: 2-bit packing, 4 values per byte.
# ---------------------------------------------------------------------------

def _quantize_pq2_0_group(w: np.ndarray) -> Tuple[np.ndarray, float]:
    """Quantize one group of 128 weights to PQ2_0 codes.

    Mirrors ``quantize_row_pq2_0_ref`` in ggml-quants.c.

    The value mapping is:
        q = round(w / absmax) + 1   # in {0, 1, 2, 3}
        w_recon = (q - 1) * absmax  # in {-absmax, 0, +absmax, ...}

    For ternary inputs the 4th value (3) is never produced (it would
    require |w| > absmax, which is impossible).

    Returns:
        (codes, scale) where codes is a (128,) uint8 array in {0, 1, 2}
        and scale is the absmax.
    """
    amax = float(np.abs(w).max())
    d = amax
    id_ = 1.0 / d if d > 0.0 else 0.0
    q = np.round(w * id_).astype(np.int32) + 1
    q = np.clip(q, 0, 3).astype(np.uint8)
    return q, d


def _pack_pq2_0(codes: np.ndarray) -> np.ndarray:
    """Pack 128 PQ2_0 codes (uint8 in {0,1,2,3}) into 32 bytes.

    4 codes per byte, low bits first.
    """
    # codes: (128,) uint8 -> reshape (32, 4) -> pack 4 per byte.
    c = codes.astype(np.uint8).reshape(-1, 4)
    packed = (
        c[:, 0]
        | (c[:, 1] << 2)
        | (c[:, 2] << 4)
        | (c[:, 3] << 6)
    ).astype(np.uint8)
    return packed


def _unpack_pq2_0(qs: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_pack_pq2_0`.

    Args:
        qs: (32,) uint8 array of packed codes.

    Returns:
        (128,) uint8 array of codes in {0, 1, 2, 3}.
    """
    qs = qs.astype(np.uint8)
    out = np.empty(qs.shape[0] * 4, dtype=np.uint8)
    out[0::4] = qs & 0b00000011
    out[1::4] = (qs >> 2) & 0b00000011
    out[2::4] = (qs >> 4) & 0b00000011
    out[3::4] = (qs >> 6) & 0b00000011
    return out


def encode_pq2_0_row(w: np.ndarray) -> bytes:
    """Encode one row of weights to PQ2_0 block bytes.

    Args:
        w: (N,) float array. N must be a multiple of 128.

    Returns:
        bytes of length ``len(w) // 128 * 34`` (one 34-byte block per
        128 weights).
    """
    N = w.shape[0]
    assert N % QK_PQ2_0 == 0, f"N={N} not a multiple of {QK_PQ2_0}"
    n_blocks = N // QK_PQ2_0
    out = bytearray()
    for b in range(n_blocks):
        chunk = w[b * QK_PQ2_0 : (b + 1) * QK_PQ2_0]
        codes, d = _quantize_pq2_0_group(chunk)
        packed = _pack_pq2_0(codes)
        # Block layout: d (2 bytes fp16) + qs (32 bytes).
        out += _fp16_to_bytes(d)
        out += packed.tobytes()
    return bytes(out)


def encode_pq2_0_batch(w: np.ndarray) -> bytes:
    """Encode a full (n_rows, n_per_row) weight matrix to PQ2_0 blocks.

    Vectorized over rows and blocks -- no Python per-row loop.

    Args:
        w: (n_rows, n_per_row) float array. n_per_row must be a multiple
            of 128.

    Returns:
        bytes of length ``n_rows * n_per_row // 128 * 34``.
    """
    n_rows, n_per_row = w.shape
    assert n_per_row % QK_PQ2_0 == 0
    n_blocks_per_row = n_per_row // QK_PQ2_0
    # Reshape to (n_rows, n_blocks_per_row, QK_PQ2_0) for per-group ops.
    w_grp = w.astype(np.float64).reshape(n_rows, n_blocks_per_row, QK_PQ2_0)
    # Per-group absmax (the scale).
    amax = np.abs(w_grp).max(axis=2)  # (n_rows, n_blocks_per_row)
    # Quantize: q = round(w / amax) + 1, clip to [0, 3].
    safe_amax = np.where(amax > 0, amax, 1.0)
    q = np.round(w_grp / safe_amax[:, :, None]).astype(np.int32) + 1
    q = np.clip(q, 0, 3).astype(np.uint8)  # (n_rows, n_blocks_per_row, QK_PQ2_0)
    # Pack 4 codes per byte: (n_rows, n_blocks_per_row, QK_PQ2_0/4) -> (n_rows, n_blocks_per_row, 32).
    q_flat = q.reshape(n_rows, n_blocks_per_row, QK_PQ2_0 // 4, 4)
    packed = (
        q_flat[:, :, :, 0]
        | (q_flat[:, :, :, 1] << 2)
        | (q_flat[:, :, :, 2] << 4)
        | (q_flat[:, :, :, 3] << 6)
    ).astype(np.uint8)  # (n_rows, n_blocks_per_row, 32)
    # Scales as fp16 bytes.
    scales_fp16 = amax.astype(np.float16).astype(np.float32)  # round-trip through fp16
    # Build the output: for each (row, block): [scale (2 bytes)] + [packed (32 bytes)].
    # Interleave: (n_rows, n_blocks_per_row, 34) -> flatten to bytes.
    out = np.empty((n_rows, n_blocks_per_row, 34), dtype=np.uint8)
    scale_bytes = amax.astype(np.float16).view(np.uint8).reshape(n_rows, n_blocks_per_row, 2)
    out[:, :, :2] = scale_bytes
    out[:, :, 2:] = packed
    return out.tobytes()


def encode_ptq1_0_batch(w: np.ndarray) -> bytes:
    """Encode a full (n_rows, n_per_row) weight matrix to PTQ1_0 blocks.

    Vectorized over rows and blocks.

    Args:
        w: (n_rows, n_per_row) float array. n_per_row must be a multiple
            of 128.

    Returns:
        bytes of length ``n_rows * n_per_row // 128 * 28``.
    """
    n_rows, n_per_row = w.shape
    assert n_per_row % QK_PTQ1_0 == 0
    n_blocks_per_row = n_per_row // QK_PTQ1_0
    # Reshape to (n_rows, n_blocks_per_row, QK_PTQ1_0).
    w_grp = w.astype(np.float64).reshape(n_rows, n_blocks_per_row, QK_PTQ1_0)
    # Per-group absmax.
    amax = np.abs(w_grp).max(axis=2)  # (n_rows, n_blocks_per_row)
    safe_amax = np.where(amax > 0, amax, 1.0)
    # Quantize: q = round(w / amax) + 1, clip to [0, 2] (ternary).
    q = np.round(w_grp / safe_amax[:, :, None]).astype(np.int32) + 1
    q = np.clip(q, 0, 2).astype(np.uint8)  # (n_rows, n_blocks_per_row, 128)
    # Reshape to (n_rows, n_blocks_per_row, 24+2, ...) for packing.
    # qs: 24 bytes of 5 trits per byte (120 values).
    # qh: 2 bytes of 4 trits per byte (8 values).
    # Total: 28 bytes per block.
    out = np.empty((n_rows, n_blocks_per_row, 28), dtype=np.uint8)
    # Pack the 24 qs bytes (5 trits each) and 2 qh bytes (4 trits each).
    # This is the slow part -- do it per-block but vectorized over rows.
    for b in range(n_blocks_per_row):
        codes = q[:, b, :]  # (n_rows, 128)
        # qs: 24 bytes, 5 trits per byte.
        for i in range(24):
            t5 = codes[:, i*5:(i+1)*5]  # (n_rows, 5)
            base3 = t5[:, 0] * 81 + t5[:, 1] * 27 + t5[:, 2] * 9 + t5[:, 3] * 3 + t5[:, 4]
            byte = (base3.astype(np.int64) * 256 + 242) // 243
            out[:, b, i] = (byte & 0xFF).astype(np.uint8)
        # qh: 2 bytes, 4 trits per byte.
        for i in range(2):
            t4 = codes[:, 120 + i*4:120 + (i+1)*4]  # (n_rows, 4)
            q_val = t4[:, 0] * 27 + t4[:, 1] * 9 + t4[:, 2] * 3 + t4[:, 3]
            q_val = q_val * 3  # extra shift
            byte = (q_val.astype(np.int64) * 256 + 242) // 243
            out[:, b, 24 + i] = (byte & 0xFF).astype(np.uint8)
        # Scale (fp16, 2 bytes) at offset 26.
        scale_bytes = amax[:, b].astype(np.float16).view(np.uint8).reshape(n_rows, 2)
        out[:, b, 26:28] = scale_bytes
    return out.tobytes()


def decode_pq2_0_row(data: bytes, n_blocks: int) -> np.ndarray:
    """Decode PQ2_0 blocks back to float32 weights.

    Args:
        data: bytes of length ``n_blocks * 34``.
        n_blocks: number of 128-weight blocks.

    Returns:
        (n_blocks * 128,) float32 array.
    """
    block_size = 34
    assert len(data) == n_blocks * block_size, (
        f"decode_pq2_0_row: expected {n_blocks * block_size} bytes, got {len(data)}"
    )
    out = np.empty(n_blocks * QK_PQ2_0, dtype=np.float32)
    for b in range(n_blocks):
        d = _fp16_from_bytes(data[b * block_size : b * block_size + 2])
        qs = np.frombuffer(
            data, dtype=np.uint8, count=QK_PQ2_0 // 4,
            offset=b * block_size + 2,
        ).copy()
        codes = _unpack_pq2_0(qs)
        # w_recon = (q - 1) * d, with q in {0, 1, 2, 3}.
        out[b * QK_PQ2_0 : (b + 1) * QK_PQ2_0] = (
            codes.astype(np.float32) - 1.0
        ) * d
    return out


# ---------------------------------------------------------------------------
# PTQ1_0: trit packing, 5 trits per byte in qs + 4 trits per byte in qh.
# ---------------------------------------------------------------------------

def _quantize_ptq1_0_group(w: np.ndarray) -> Tuple[np.ndarray, float]:
    """Quantize one group of 128 weights to PTQ1_0 codes.

    Same value mapping as PQ2_0 (round-to-nearest against absmax), but
    the codes are restricted to {0, 1, 2} (ternary).

    Returns:
        (codes, scale) where codes is a (128,) uint8 array in {0, 1, 2}
        and scale is the absmax.
    """
    amax = float(np.abs(w).max())
    d = amax
    id_ = 1.0 / d if d > 0.0 else 0.0
    q = np.round(w * id_).astype(np.int32) + 1
    # PTQ1_0 is strictly ternary: clip to {0, 1, 2}.
    q = np.clip(q, 0, 2).astype(np.uint8)
    return q, d


def _pack_5_trits(trits: np.ndarray) -> int:
    """Pack 5 ternary values (each in {0,1,2}) into one byte.

    Matches the C reference's convention: ``trits[0]`` is the MSB
    (coefficient 3^4 = 81), ``trits[4]`` is the LSB (coefficient 1).
    This mirrors the C loop ``q = 0; for n in 5: q = q*3 + x[n]``
    which produces ``q = x[0]*81 + x[1]*27 + x[2]*9 + x[3]*3 + x[4]``.

    The ceiling-division mapping is:
        base3 = x[0]*81 + x[1]*27 + x[2]*9 + x[3]*3 + x[4]  # in [0, 242]
        byte = (base3 * 256 + 242) // 243  # in [0, 255]
    """
    # C convention: trits[0] is MSB (coefficient 81).
    base3 = (
        int(trits[0]) * 81
        + int(trits[1]) * 27
        + int(trits[2]) * 9
        + int(trits[3]) * 3
        + int(trits[4])
    )
    byte = (base3 * 256 + 242) // 243
    return byte & 0xFF


def _unpack_5_trits(byte: int) -> np.ndarray:
    """Inverse of :func:`_pack_5_trits`.

    Extracts 5 ternary values from one byte using the C reference's
    formula: ``trit_n = (byte * 3^(n+1)) >> 8``, mod 3.
    """
    byte = int(byte) & 0xFF
    trits = np.empty(5, dtype=np.uint8)
    for n in range(5):
        # q = byte * 3^n (as uint8, wraps around).
        q = (byte * int(_POW3[n])) & 0xFF
        # xi = (q * 3) >> 8 (as uint16, then shift).
        xi = ((q * 3) & 0xFFFF) >> 8
        trits[n] = xi
    return trits


def _pack_4_trits(trits: np.ndarray) -> int:
    """Pack 4 ternary values into one byte (the qh layout).

    Mirrors the C reference's qh packing, which packs 4 trits per byte
    with an extra ``q *= 3`` shift before the same ceiling-division.
    The extra shift means the 4-trit value is encoded as if it were
    a 5-trit value with the LSB zero, allowing the same unpack code.
    """
    # trits[0] is the first input; the C code packs them as
    # q = trits[0], then q = q*3 + trits[i], then q *= 3 (the extra shift).
    q = 0
    for i in range(4):
        q = q * 3 + int(trits[i])
    q *= 3  # extra shift: makes the 4-trit value occupy trits 1-4 of a 5-trit slot.
    byte = (q * 256 + 242) // 243
    return byte & 0xFF


def _unpack_4_trits(byte: int) -> np.ndarray:
    """Inverse of :func:`_pack_4_trits`.

    Extracts 4 ternary values from one byte. The unpack formula is the
    same as for 5 trits, but only the top 4 trits are valid (the LSB is
    always zero by construction).
    """
    byte = int(byte) & 0xFF
    trits = np.empty(4, dtype=np.uint8)
    for n in range(4):
        q = (byte * int(_POW3[n])) & 0xFF
        xi = ((q * 3) & 0xFFFF) >> 8
        trits[n] = xi
    return trits


def encode_ptq1_0_row(w: np.ndarray) -> bytes:
    """Encode one row of weights to PTQ1_0 block bytes.

    Block layout (28 bytes per 128 weights):
        - qs: 24 bytes, 5 trits per byte (120 values)
        - qh: 2 bytes, 4 trits per byte (8 values)
        - d: 2 bytes, fp16 scale

    The 128 weights are split into 24 groups of 5 (for qs) + 2 groups
    of 4 (for qh), in order.

    Args:
        w: (N,) float array. N must be a multiple of 128.

    Returns:
        bytes of length ``len(w) // 128 * 28``.
    """
    N = w.shape[0]
    assert N % QK_PTQ1_0 == 0, f"N={N} not a multiple of {QK_PTQ1_0}"
    n_blocks = N // QK_PTQ1_0
    out = bytearray()
    for b in range(n_blocks):
        chunk = w[b * QK_PTQ1_0 : (b + 1) * QK_PTQ1_0]
        codes, d = _quantize_ptq1_0_group(chunk)
        # qs: 24 bytes of 5 trits each = 120 values.
        qs_bytes = bytearray(24)
        for i in range(24):
            trits = codes[i * 5 : (i + 1) * 5]
            qs_bytes[i] = _pack_5_trits(trits)
        # qh: 2 bytes of 4 trits each = 8 values.
        qh_bytes = bytearray(2)
        for i in range(2):
            trits = codes[120 + i * 4 : 120 + (i + 1) * 4]
            qh_bytes[i] = _pack_4_trits(trits)
        # Block layout: qs (24) + qh (2) + d (2).
        out += bytes(qs_bytes)
        out += bytes(qh_bytes)
        out += _fp16_to_bytes(d)
    return bytes(out)


def decode_ptq1_0_row(data: bytes, n_blocks: int) -> np.ndarray:
    """Decode PTQ1_0 blocks back to float32 weights.

    Args:
        data: bytes of length ``n_blocks * 28``.
        n_blocks: number of 128-weight blocks.

    Returns:
        (n_blocks * 128,) float32 array.
    """
    block_size = 28
    assert len(data) == n_blocks * block_size, (
        f"decode_ptq1_0_row: expected {n_blocks * block_size} bytes, got {len(data)}"
    )
    out = np.empty(n_blocks * QK_PTQ1_0, dtype=np.float32)
    for b in range(n_blocks):
        base = b * block_size
        qs = np.frombuffer(data, dtype=np.uint8, count=24, offset=base).copy()
        qh = np.frombuffer(data, dtype=np.uint8, count=2, offset=base + 24).copy()
        d = _fp16_from_bytes(data[base + 26 : base + 28])
        # Unpack 24 bytes of 5 trits = 120 values.
        codes = np.empty(128, dtype=np.uint8)
        for i in range(24):
            codes[i * 5 : (i + 1) * 5] = _unpack_5_trits(qs[i])
        # Unpack 2 bytes of 4 trits = 8 values.
        for i in range(2):
            codes[120 + i * 4 : 120 + (i + 1) * 4] = _unpack_4_trits(qh[i])
        # w_recon = (q - 1) * d.
        out[b * QK_PTQ1_0 : (b + 1) * QK_PTQ1_0] = (
            codes.astype(np.float32) - 1.0
        ) * d
    return out


# ---------------------------------------------------------------------------
# FP16 helpers (little-endian, matches ggml_half).
# ---------------------------------------------------------------------------

def _fp16_to_bytes(f: float) -> bytes:
    """Convert a float to 2 little-endian fp16 bytes."""
    return np.array([f], dtype=np.float16).tobytes()


def _fp16_from_bytes(b: bytes) -> float:
    """Convert 2 little-endian fp16 bytes to a float."""
    return float(np.frombuffer(b, dtype=np.float16, count=1)[0])


# ---------------------------------------------------------------------------
# Block-level helpers (for the TOFFEE codec integration).
# ---------------------------------------------------------------------------

def block_size_bytes(format_name: str) -> int:
    """The byte size of one ternary block (128 weights)."""
    if format_name == "PTQ1_0":
        return 28
    elif format_name == "PQ2_0":
        return 34
    elif format_name == "TQ1_0":
        return 54  # 2 + 4 + 48 bytes per 256 weights.
    elif format_name == "TQ2_0":
        return 66  # 2 + 64 bytes per 256 weights.
    else:
        raise ValueError(f"Unknown ternary format: {format_name}")


def block_elems(format_name: str) -> int:
    """The number of weights in one ternary block."""
    if format_name in ("PTQ1_0", "PQ2_0"):
        return 128
    elif format_name in ("TQ1_0", "TQ2_0"):
        return 256
    else:
        raise ValueError(f"Unknown ternary format: {format_name}")


def bits_per_weight(format_name: str) -> float:
    """Effective bits per weight (block_bytes * 8 / block_elems)."""
    bs = block_size_bytes(format_name)
    be = block_elems(format_name)
    return bs * 8.0 / be


__all__ = [
    "TernaryBlockRaw",
    "QK_PTQ1_0",
    "QK_PQ2_0",
    "encode_pq2_0_row",
    "decode_pq2_0_row",
    "encode_ptq1_0_row",
    "decode_ptq1_0_row",
    "block_size_bytes",
    "block_elems",
    "bits_per_weight",
]
