"""
TOFFEE XSpark export-time GGUF quantization (Q8_0 / Q6_K).

Writes draft-head weights directly in llama.cpp/ik_llama.cpp block-quantized
formats so the exported GGUF is far smaller than any float storage:

    Q8_0    8.5    bits/weight  (34 B per 32 elems)    ~53% of F16
    Q6_K    6.5625 bits/weight  (210 B per 256 elems)  ~41% of F16

Both encoders are numpy ports of ik_llama.cpp's reference quantizers
(``ggml/src/ggml-quants.c``):

* ``quantize_row_q8_0_ref`` — ported bit-exactly (amax/127 scale, roundf),
  matching the gguf package's official ``Q8_0.quantize_blocks``.
* ``quantize_row_q6_K_ref`` — ported including ``make_qx_quants(16, 32, ...,
  rmse_type=1, NULL)``: the x^2-weighted least-squares group scale, the
  optimal-scale solve, and the +/-0.1*is grid sweep.  The *final greedy
  single-coordinate polish* of ``make_qx_quants`` (a <= 1008-iteration
  steepest descent per 16-element group) is NOT ported: it cannot be
  vectorized cheaply and buys < 1% RMSE.  Note the reference itself
  re-derives the final per-element codes from ``x/(d*sc)`` after the scales
  are fixed, so the polish's code tweaks are largely overwritten anyway —
  only the per-group scales keep its contribution.

Numeric deviations from the C reference (both safe):

* Weighted sums accumulate in float64 (C uses float32) — strictly more
  precise, and deterministic between the vectorized path and the scalar
  reference port used by the test-suite.
* ``nearest_int`` is the literal bit-trick port (float32 add of 12582912.0),
  so knife-edge inputs round exactly like ggml.

Any block we emit is a *valid* Q8_0/Q6_K block: ggml / ik_llama.cpp
dequantizers read it identically to llama-quantize output.

ik_llama.cpp kernel constraints baked into :func:`plan_quant` (verified in
the fork's sources, not upstream folklore):

* CUDA ``get_rows`` (``ggml-cuda/getrows.cu``) supports F16/F32/Q4_0/Q4_1/
  Q5_0/Q5_1/Q8_0 and hard-ABORTS on k-quants (``// TODO: k-quants``), so
  get_rows-consumed tensors (``selector_*``, ``markov_w1``, ``token_embd``)
  cap at Q8_0 even under ``--quant q6_k``.
* ``ggml_get_rows_ext(..., same_type=true)`` (dspark ``markov_w1``) makes
  the dst the SOURCE's type, and CUDA get_rows asserts ``dst == F32`` —
  so ``markov_w1`` is never quantized at all.
* 1-D tensors (norms/biases) and the per-layer conv bases feed
  fused_rms_norm / conv-add kernels that assert F32 storage — never
  quantized (they are tiny anyway).
* A quantized tensor's row length (ggml ``ne[0]`` == torch ``shape[-1]``)
  must be a multiple of the block size (32 / 256).  Rows are NOT
  zero-padded here because the runtime graph is built from hparams-derived
  shapes — a padded tensor would break the build.  Tensors that fail the
  divisibility check keep float storage (logged).

This module is deliberately torch-free (numpy only) so the pure-quantizer
test-suite does not need a torch install.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Block geometry (ggml-common.h)
# ---------------------------------------------------------------------------

QK8_0 = 32            # elements per Q8_0 block
QS8_0 = 34            # bytes  per Q8_0 block (fp16 d + 32 x int8)
QK_K = 256            # elements per K-quant super-block
QS6_K = 210           # bytes  per Q6_K block (ql[128] qh[64] scales[16] d[2])
GROUP_MAX_EPS = 1e-15 # ggml-common.h

BLOCK = {"Q8_0": QK8_0, "Q6_K": QK_K}
BLOCK_BYTES = {"Q8_0": QS8_0, "Q6_K": QS6_K}

#: ik_llama.cpp ggml_type enum values (ggml/include/ggml.h) — the uint32 the
#: GGUF tensor header must carry.  Asserted against the installed gguf
#: package at write time so a renumbered package fails loudly, not silently.
GGML_TYPE_IDS = {"Q8_0": 8, "Q6_K": 14, "F16": 1, "F32": 0, "BF16": 30}

_QUANT_ALIASES = {
    None: None, "": None, "none": None, "off": None, "f16": None, "float": None,
    "q8": "Q8_0", "q8_0": "Q8_0",
    "q6": "Q6_K", "q6_k": "Q6_K",
}


def normalize_quant(name: Optional[str]) -> Optional[str]:
    """Map a CLI quant flag (with aliases) to a canonical type name or None."""
    key = (name or "").strip().lower()
    if key not in _QUANT_ALIASES:
        raise ValueError(
            f"unknown quant type {name!r} — expected one of "
            f"{sorted(k for k in _QUANT_ALIASES if k)}")
    return _QUANT_ALIASES[key]


# ---------------------------------------------------------------------------
# ggml helpers
# ---------------------------------------------------------------------------

def nearest_int(f: np.ndarray) -> np.ndarray:
    """Vectorized ggml ``nearest_int`` (ggml-quants.c): the 12582912.0 magic add.

    Rounds ties to even, exactly like the C bit trick.  Valid for finite
    ``|f| <= 4194303`` (the C asserts the same); callers stay far below that.
    """
    f = np.asarray(f, dtype=np.float32)
    val = f + np.float32(12582912.0)
    bits = val.view(np.int32)
    return (bits & np.int32(0x007FFFFF)) - np.int32(0x00400000)


def _roundf(f: np.ndarray) -> np.ndarray:
    """C ``roundf`` (half away from zero), same formula as gguf's np_roundf."""
    a = np.abs(f)
    floored = np.floor(a)
    return np.sign(f) * (floored + np.floor(2.0 * (a - floored)))


# ---------------------------------------------------------------------------
# Q8_0 (bit-exact port of quantize_row_q8_0_ref, fudge = 1.0)
# ---------------------------------------------------------------------------

def _quant_q8_0_chunk(x: np.ndarray) -> np.ndarray:
    """x: (nb, 32) float32 -> (nb, 34) uint8 packed blocks."""
    amax = np.abs(x).max(axis=1, keepdims=True)                # (nb, 1)
    d = amax / np.float32(127.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        id_ = np.where(d == 0, np.float32(0.0), np.float32(1.0) / d)
    qs = _roundf(x * id_)
    out = np.empty((x.shape[0], QS8_0), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = qs.astype(np.int8).view(np.uint8)
    return out


def _dequant_q8_0_chunk(b: np.ndarray) -> np.ndarray:
    """b: (nb, 34) uint8 -> (nb, 32) float32 (ggml dequantize_row_q8_0)."""
    d = b[:, :2].copy().view(np.float16).astype(np.float32)     # (nb, 1)
    q = b[:, 2:].copy().view(np.int8).astype(np.float32)        # (nb, 32)
    return q * d


# ---------------------------------------------------------------------------
# Q6_K (port of quantize_row_q6_K_ref + make_qx_quants(rmse_type=1),
#       greedy polish omitted — see module docstring; fudge = 1.0)
# ---------------------------------------------------------------------------

def _quant_q6_k_chunk(x: np.ndarray) -> np.ndarray:
    """x: (nb, 256) float32 -> (nb, 210) uint8 packed blocks."""
    nb = x.shape[0]
    g = x.reshape(nb, 16, 16)                                   # 16 groups of 16
    g64 = g.astype(np.float64)

    # ---- per-group amax / signed max (make_qx_quants prologue) ----
    ag = np.abs(g)
    ix = ag.argmax(axis=2, keepdims=True)
    amax = np.take_along_axis(ag, ix, 2).squeeze(2)             # (nb, 16)
    maxv = np.take_along_axis(g, ix, 2).squeeze(2)              # signed
    nz = amax >= GROUP_MAX_EPS
    safe = np.where(nz, maxv.astype(np.float64), 1.0)

    # ---- make_qx_quants(16, 32, x, L, rmse_type=1, NULL) ----
    # first pass: iscale = -32/max, l = clip(nearest_int(iscale*x), -32, 31)
    iscale0 = np.where(nz, -32.0 / safe, 0.0)                   # (nb, 16) f64
    L = np.clip(nearest_int((iscale0[..., None] * g64).astype(np.float32)),
                -32, 31)                                        # (nb, 16, 16)
    w = g64 * g64                                               # rmse_type=1 weights
    sumlx = (w * g64 * L).sum(axis=2)                           # f64 accumulators
    suml2 = (w * L * L).sum(axis=2)
    scale = np.where(suml2 > 0, sumlx / np.where(suml2 > 0, suml2, 1.0), 0.0)
    best = scale * sumlx

    # grid sweep: is in [-9, 9], isc = -(32 + 0.1*is)/max and (31 + 0.1*is)/max
    for is_ in range(-9, 10):
        f = 0.1 * is_
        for base, sign in ((32.0, -1.0), (31.0, 1.0)):
            isc = np.where(nz, sign * (base + f) / safe, 0.0)   # (nb, 16)
            lc = np.clip(nearest_int((isc[..., None] * g64).astype(np.float32)),
                         -32, 31)
            slx = (w * g64 * lc).sum(axis=2)
            sl2 = (w * lc * lc).sum(axis=2)
            win = (sl2 > 0) & (slx * slx > best * sl2)
            if not win.any():
                continue
            L = np.where(win[..., None], lc, L)
            sumlx = np.where(win, slx, sumlx)
            suml2 = np.where(win, sl2, suml2)
            scale = np.where(win, slx / np.where(sl2 > 0, sl2, 1.0), scale)
            best = np.where(win, scale * sumlx, best)

    # all-(~)zero groups: C sets the biased codes L[i] = 0 -> unbiased -32
    L = np.where(nz[..., None], L, -32)
    scale = np.where(nz, scale, 0.0)

    # ---- super-block scale quantization ----
    ascale = np.abs(scale)
    msb = ascale.argmax(axis=1, keepdims=True)                  # first max, like C's strict >
    max_abs = np.take_along_axis(ascale, msb, 1)                # (nb, 1)
    max_scale = np.take_along_axis(scale, msb, 1)               # signed (nb, 1)
    zero_blk_col = max_abs < GROUP_MAX_EPS                      # (nb, 1)
    zero_blk = zero_blk_col[..., 0]                             # (nb,)

    msafe = np.where(zero_blk_col, -128.0, max_scale)           # dummy: keeps 1/iscale finite
    iscale_sb = -128.0 / msafe                                  # (nb, 1)
    d16 = (1.0 / iscale_sb).astype(np.float16)                  # fudge = 1.0
    if not np.isfinite(d16.astype(np.float32)).all():
        raise ValueError(
            "Q6_K super-block scale overflows fp16 — weights have |max| > "
            "~8.4M; refusing to emit inf blocks")
    sc = np.minimum(127, nearest_int((iscale_sb * scale).astype(np.float32)))
    sc = sc.astype(np.int8)                                     # (nb, 16), [-128, 127]

    # ---- final codes: l = clip(nearest_int(x / (fp16(d) * sc)), -32, 31) ----
    d_eff = d16.astype(np.float64) * sc.astype(np.float64)      # (nb, 16)
    ok = (d_eff != 0)[..., None]                                # (nb, 16, 1)
    l_final = np.clip(nearest_int((g64 / np.where(ok, d_eff[..., None], 1.0)).astype(np.float32)),
                      -32, 31)
    Lc = np.where(ok, l_final + 32, L + 32).astype(np.int64)    # biased codes [0, 63]

    # ---- pack (mirrors the C loop bit-for-bit) ----
    q = Lc.reshape(nb, 2, 4, 32)                                # [blk, half, quarter, l]
    A, B, C, D = q[:, :, 0, :], q[:, :, 1, :], q[:, :, 2, :], q[:, :, 3, :]
    ql = np.empty((nb, 2, 64), dtype=np.uint8)
    ql[:, :, :32] = ((A & 0xF) | ((C & 0xF) << 4)).astype(np.uint8)
    ql[:, :, 32:] = ((B & 0xF) | ((D & 0xF) << 4)).astype(np.uint8)
    qh = ((A >> 4) | ((B >> 4) << 2) | ((C >> 4) << 4) | ((D >> 4) << 6)).astype(np.uint8)
    out = np.empty((nb, QS6_K), dtype=np.uint8)
    out[:, :128] = ql.reshape(nb, 128)
    out[:, 128:192] = qh.reshape(nb, 64)
    out[:, 192:208] = sc.view(np.uint8)
    out[:, 208:210] = d16.view(np.uint8).reshape(nb, 2)
    out[zero_blk] = 0                                           # memset(&y[i], 0, ...)
    return out


def _dequant_q6_k_chunk(b: np.ndarray) -> np.ndarray:
    """b: (nb, 210) uint8 -> (nb, 256) float32 (ggml dequantize_row_q6_K)."""
    nb = b.shape[0]
    ql = b[:, :128].reshape(nb, 2, 2, 32)                       # [blk, half, lo/hi row, l]
    qh = b[:, 128:192].reshape(nb, 2, 32)
    sc = b[:, 192:208].copy().view(np.int8).astype(np.float32)  # (nb, 16)
    d = b[:, 208:210].copy().view(np.float16).astype(np.float32)  # (nb, 1)

    lo, hi = ql[:, :, 0, :], ql[:, :, 1, :]
    q0 = (((lo & 0x0F) | ((qh & 0x03) << 4)).astype(np.int8) - 32).astype(np.float32)
    q1 = (((hi & 0x0F) | (((qh >> 2) & 0x03) << 4)).astype(np.int8) - 32).astype(np.float32)
    q2 = (((lo >> 4) | (((qh >> 4) & 0x03) << 4)).astype(np.int8) - 32).astype(np.float32)
    q3 = (((hi >> 4) | (((qh >> 6) & 0x03) << 4)).astype(np.int8) - 32).astype(np.float32)
    q = np.stack([q0, q1, q2, q3], axis=2).reshape(nb, 256)     # (nb, 256)

    eff = (d * sc).astype(np.float32)                           # (nb, 16)
    return q * np.repeat(eff, 16, axis=1)


_QUANT_CHUNK = {"Q8_0": _quant_q8_0_chunk, "Q6_K": _quant_q6_k_chunk}
_DEQUANT_CHUNK = {"Q8_0": _dequant_q8_0_chunk, "Q6_K": _dequant_q6_k_chunk}


# ---------------------------------------------------------------------------
# Tensor drivers (row-chunked to bound peak memory)
# ---------------------------------------------------------------------------

@dataclass
class QuantStats:
    """Per-tensor quantization record for the export report."""
    name: str
    qtype: str
    shape: Tuple[int, ...]
    n_elements: int
    f16_bytes: int
    quant_bytes: int
    rmse: float
    rel_rmse: float          # rmse / rms(x)
    max_abs_err: float

    @property
    def ratio(self) -> float:
        return self.quant_bytes / max(self.f16_bytes, 1)


class QuantPlan(NamedTuple):
    qtype: Optional[str]     # None -> keep float storage
    reason: str


#: tensors consumed via ggml_get_rows (CUDA kernel supports Q8_0 but aborts
#: on k-quants); must stay in sync with export.py's _GET_ROWS_TENSORS
GET_ROWS_TENSORS = ("selector_predecessor", "selector_successor",
                    "markov_w1", "token_embd")


def plan_quant(name: str, tensor, quant: Optional[str]) -> QuantPlan:
    """Decide the effective GGUF quant type for one head tensor.

    ``tensor`` is duck-typed (torch Tensor or numpy ndarray — anything with
    ``.shape`` and ``.dim()``/``.ndim``).  Returns the effective qtype (or
    None = keep float storage) plus the reason, so the exporter can log an
    honest per-tensor report.
    """
    quant = normalize_quant(quant)
    if quant is None:
        return QuantPlan(None, "no_quant")
    ndim = tensor.dim() if hasattr(tensor, "dim") else tensor.ndim
    if ndim != 2:
        # 1-D (norms, biases) and 3-D conv bases: ik_llama.cpp's
        # fused_rms_norm / conv-add kernels assert F32 storage
        return QuantPlan(None, "not_2d_f32_kernel")
    if name.startswith("markov_w1"):
        # ggml_get_rows_ext(same_type=true) -> dst takes the source type and
        # CUDA get_rows asserts dst == F32
        return QuantPlan(None, "markov_w1_cuda_same_type_get_rows")
    eff = quant
    reason = "ok"
    if name.startswith(GET_ROWS_TENSORS) and quant == "Q6_K":
        # CUDA get_rows aborts on k-quants ("TODO: k-quants")
        eff, reason = "Q8_0", "q6_k_capped_q8_0_cuda_get_rows"
    block = BLOCK[eff]
    ne0 = tensor.shape[-1]
    if ne0 % block != 0:
        # rows cannot be padded: the runtime graph shapes come from hparams
        return QuantPlan(None, f"row_len_{ne0}_not_divisible_by_{block}")
    return QuantPlan(eff, reason)


def _check_finite(w: np.ndarray, name: str) -> None:
    if not np.isfinite(w).all():
        raise ValueError(f"tensor {name!r} contains non-finite weights — "
                         "refusing to quantize garbage")


def quantize_tensor(w: np.ndarray, qtype: str, name: str = "<tensor>",
                    chunk_elems: int = 48_000_000) -> Tuple[np.ndarray, QuantStats]:
    """Quantize a 2-D float weight into GGUF block format.

    Args:
        w: (n_rows, ne0) float array; ``ne0`` must be divisible by the
           block size (32 for Q8_0, 256 for Q6_K).
        qtype: "Q8_0" or "Q6_K".
        name: tensor name for error messages / stats.
        chunk_elems: row-chunk target size (memory bound).

    Returns:
        (packed, stats) where ``packed`` is a (n_rows, row_bytes) uint8
        array — pass it to ``GGUFWriter.add_tensor(..., raw_dtype=...)``
        and pygguf converts the byte shape back to the logical ne.
    """
    qtype = normalize_quant(qtype) or ""
    if qtype not in _QUANT_CHUNK:
        raise ValueError(f"unsupported quant type {qtype!r}")
    w = np.ascontiguousarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"{name}: expected 2-D, got shape {w.shape}")
    _check_finite(w, name)
    blk, bsz = BLOCK[qtype], BLOCK_BYTES[qtype]
    n_rows, ne0 = w.shape
    if ne0 % blk:
        raise ValueError(f"{name}: row length {ne0} not divisible by {blk}")
    blocks_per_row = ne0 // blk
    row_bytes = blocks_per_row * bsz

    packed = np.empty((n_rows, row_bytes), dtype=np.uint8)
    quant_fn, dequant_fn = _QUANT_CHUNK[qtype], _DEQUANT_CHUNK[qtype]
    rows_per_chunk = max(1, min(n_rows, chunk_elems // max(ne0, 1)))
    sse = 0.0
    max_err = 0.0
    for r0 in range(0, n_rows, rows_per_chunk):
        chunk = w[r0:r0 + rows_per_chunk]
        qb = quant_fn(chunk.reshape(-1, blk))
        packed[r0:r0 + chunk.shape[0]] = qb.reshape(chunk.shape[0], row_bytes)
        err = dequant_fn(qb).reshape(chunk.shape).astype(np.float64) \
            - chunk.astype(np.float64)
        sse += float((err * err).sum())
        max_err = max(max_err, float(np.abs(err).max()))

    rms = float(np.sqrt((w.astype(np.float64) ** 2).sum() / max(w.size, 1)))
    rmse = float(np.sqrt(sse / max(w.size, 1)))
    stats = QuantStats(
        name=name, qtype=qtype, shape=tuple(w.shape), n_elements=int(w.size),
        f16_bytes=int(w.size) * 2, quant_bytes=int(packed.nbytes),
        rmse=rmse, rel_rmse=rmse / max(rms, 1e-30), max_abs_err=max_err,
    )
    return packed, stats


def fake_quantize_tensor(w: np.ndarray, qtype: str, name: str = "<tensor>",
                         chunk_elems: int = 48_000_000) -> np.ndarray:
    """Quantize + dequantize a 2-D weight (round-trip), float32 out.

    Used by ``xspark eval --quant`` to measure acceptance against exactly
    the weights a quantized GGUF would reconstruct.
    """
    qtype = normalize_quant(qtype) or ""
    if qtype not in _QUANT_CHUNK:
        raise ValueError(f"unsupported quant type {qtype!r}")
    w = np.ascontiguousarray(w, dtype=np.float32)
    _check_finite(w, name)
    blk = BLOCK[qtype]
    n_rows, ne0 = w.shape
    if ne0 % blk:
        raise ValueError(f"{name}: row length {ne0} not divisible by {blk}")
    out = np.empty_like(w)
    quant_fn, dequant_fn = _QUANT_CHUNK[qtype], _DEQUANT_CHUNK[qtype]
    rows_per_chunk = max(1, min(n_rows, chunk_elems // max(ne0, 1)))
    for r0 in range(0, n_rows, rows_per_chunk):
        chunk = w[r0:r0 + rows_per_chunk]
        qb = quant_fn(chunk.reshape(-1, blk))
        out[r0:r0 + chunk.shape[0]] = dequant_fn(qb).reshape(chunk.shape)
    return out


# ---------------------------------------------------------------------------
# In-model fake quantization (eval pre-flight)
# ---------------------------------------------------------------------------

#: live-module parameter name -> GGUF tensor name (mirrors export.py's _add)
_PARAM_TO_GGUF = {
    "fc.weight": "fc.weight",
    "hidden_norm.weight": "enc.output_norm.weight",
    "output_norm.weight": "output_norm.weight",
    "selector_prev.weight": "selector_predecessor.weight",
    "selector_next.weight": "selector_successor.weight",
    "selector_hidden.weight": "selector_hidden.weight",
    "markov_w1.weight": "markov_w1.weight",
    "markov_w2.weight": "markov_w2.weight",
    "conf_head.weight": "conf_proj.weight",
}
_LAYER_PARAM_TO_GGUF = {
    "attn_norm.weight": "attn_norm.weight",
    "attn.q_proj.weight": "attn_q.weight",
    "attn.q_norm.weight": "attn_q_norm.weight",
    "attn.k_proj.weight": "attn_k.weight",
    "attn.k_norm.weight": "attn_k_norm.weight",
    "attn.v_proj.weight": "attn_v.weight",
    "attn.o_proj.weight": "attn_output.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "ffn_gate.weight": "ffn_gate.weight",
    "ffn_down.weight": "ffn_down.weight",
    "ffn_up.weight": "ffn_up.weight",
    "attn_conv_base": "attn_conv_base",
    "attn_conv_proj.weight": "attn_conv_proj.weight",
    "ffn_conv_base": "ffn_conv_base",
    "ffn_conv_proj.weight": "ffn_conv_proj.weight",
}


def apply_fake_quant(model, quant: Optional[str]) -> List[QuantStats]:
    """Fake-quantize a loaded XSparkDraftModel in place (CPU, float32 math).

    Applies the same per-tensor policy as the exporter (:func:`plan_quant`)
    so ``xspark eval --quant q8_0`` measures acceptance against exactly the
    weights a quantized GGUF reconstructs.  Run before ``.to(device)``.
    Returns the per-tensor stats for logging.
    """
    quant = normalize_quant(quant)
    if quant is None:
        return []
    stats: List[QuantStats] = []
    for pname, param in model.named_parameters():
        if pname.startswith("layers."):
            parts = pname.split(".", 2)                     # layers, i, rest
            suffix = _LAYER_PARAM_TO_GGUF.get(parts[2])
            gguf_name = f"blk.{parts[1]}.{suffix}" if suffix else None
        else:
            gguf_name = _PARAM_TO_GGUF.get(pname)
        if gguf_name is None:
            continue
        plan = plan_quant(gguf_name, param, quant)
        if plan.qtype is None:
            continue
        w = param.detach().cpu().float().numpy()
        deq = fake_quantize_tensor(w, plan.qtype, gguf_name)
        st = QuantStats(
            name=gguf_name, qtype=plan.qtype, shape=tuple(w.shape),
            n_elements=int(w.size), f16_bytes=int(w.size) * 2,
            quant_bytes=int(w.size) * BLOCK_BYTES[plan.qtype] // BLOCK[plan.qtype],
            rmse=float(np.sqrt(((deq - w) ** 2).sum() / max(w.size, 1))),
            rel_rmse=0.0, max_abs_err=float(np.abs(deq - w).max()),
        )
        st.rel_rmse = st.rmse / max(float(np.sqrt((w ** 2).sum() / max(w.size, 1))), 1e-30)
        param.data.copy_(torch_from_numpy(deq).to(param.dtype))
        stats.append(st)
    return stats


def torch_from_numpy(arr: np.ndarray):
    """Import torch lazily and convert (keeps this module torch-free)."""
    import torch
    return torch.from_numpy(arr)


__all__ = [
    "BLOCK", "BLOCK_BYTES", "GGML_TYPE_IDS", "GET_ROWS_TENSORS",
    "QuantPlan", "QuantStats",
    "apply_fake_quant", "fake_quantize_tensor", "nearest_int",
    "normalize_quant", "plan_quant", "quantize_tensor",
]
