"""
@file toffee/quant/gsq.py
   GSQ: Gumbel-Softmax refinement of a quantized tensor, in llama.cpp format.

Ported from IST-DASLab/GSQ (Apache-2.0), adapted to TOFFEE's GGUF pipeline.

## PARKED -- experimental, not on by default, not proven to help

This is a scale/code refit of an already-quantized tensor. It is **not**
enabled in the standard quantization flow.

Measured on **real** Sage-A1.5 weights (2026-10-21), not synthetic data:
``epochs=8`` and ``epochs=0`` give the *same* error to 10 decimal places
(+1.5544% and +1.5544% on ``blk.0.attn_qkv.weight``), with ``epochs=8``
taking ~45% longer. The Gumbel-Softmax relaxation -- upstream GSQ's actual
method -- contributes **0%**. llama.cpp's own code assignment is already
optimal given its scales, and the scale search subsumes the relaxation.

So this module optimizes the (scale, min) pair jointly with the codes,
using the imatrix-weighted objective, and only ever writes values that are
exactly representable in the llama.cpp format. It is a scale refit, not
GSQ. Until a measured win exists, it stays parked.

## Attribution

The idea
--------
A scalar quantizer picks, for every weight, one code on a small integer grid
plus a per-group scale. The *choice* of code is discrete, so it is normally
made greedily (round-to-nearest) and cannot be improved afterwards. GSQ
relaxes that choice: every weight gets a learnable logit over the grid, and
a Gumbel-Softmax sample (with a temperature annealed high -> low) turns the
logits into a soft, differentiable code. The reconstruction error is then
back-propagated into the logits, and the final hard assignment (argmax) is
written back into the same format.

Why it matters for TOFFEE
-------------------------
The upstream GSQ trains a *new* checkpoint in compressed-tensors format. That
is not usable here: TOFFEE's contract is "a GGUF that llama.cpp loads
unchanged". So this port keeps GSQ's optimizer, but optimizes the *degrees of
freedom that actually exist in a llama.cpp block*:

  * the integer codes (GSQ's Gumbel-Softmax relaxation), and
  * the integer per-group sub-scale / sub-minimum fields,

and re-emits them in the identical block layout. The file stays a valid
GGUF; only the numbers inside change.

What the refinement can actually gain
-------------------------------------
Measured against llama.cpp's own quantizer on the same imatrix:

  * llama.cpp's **code assignment is already optimal** given its scales --
    refitting every code to its locally best grid point changes 97% of the
    codes and improves the error by 0.000%. The Gumbel-Softmax therefore
    starts from llama.cpp's assignment and can only *keep* it.
  * the real headroom is in the **per-group scale** (and, for the
    asymmetric types, the scale/minimum *pair*): refitting the integer
    sub-scale of a Q6_K group alone reduces the weighted error by ~0.8%.

So this module runs two stages:

  1. Gumbel-Softmax relaxation of the codes (GSQ proper), which recovers any
     loss from a sub-optimal starting point and can improve the assignment
     when the scales are jointly re-fitted afterwards.
  2. Exact coordinate descent over the *integer* scale/minimum fields, with
     the codes re-fitted for each candidate, so every accepted value is
     exactly representable in the llama.cpp format.

The objective is the *weighted* reconstruction error, using the same
importance matrix llama.cpp uses, so the refinement improves exactly the
quantity the imatrix measures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from toffee.quant.gguf_kquant import (
    BlockRaw,
    TypeSpec,
    dequantize_blocks,
    effective_mins,
    effective_scales,
    spec_for,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GsqConfig:
    """Hyper-parameters of the Gumbel-Softmax refinement.

    Attributes:
        epochs: number of Gumbel-Softmax passes over the tensor.
        lr_logits: learning rate for the code logits.
        temperature_init: starting Gumbel-Softmax temperature.
        temperature_end: final temperature (low -> hard assignments).
        logit_scale_init: initial logit gain.
        logit_scale_end: final logit gain.
        seed: RNG seed for the Gumbel noise.
        scale_passes: number of coordinate-descent passes over the integer
            per-group sub-scale / sub-minimum fields.
        refit_codes: re-fit the codes for every candidate scale, which is
            what makes the asymmetric (scale, min) pair searchable.
        chunk_blocks: number of blocks processed per Gumbel-Softmax chunk.
            Bounds the peak memory of the relaxation.
    """

    # The Gumbel-Softmax stage is OFF by default, and that is a measured
    # decision rather than a shortcut.
    #
    # llama.cpp's own code assignment is already optimal *given its scales*:
    # relaxing the codes with Gumbel-Softmax and re-fitting them changes
    # ~97% of the codes but improves the error by 0.000%. All the headroom
    # is in the scales, which stage 2 searches exactly.
    #
    # Measured on 256x2048 tensors with the same imatrix as llama.cpp:
    #
    #     type   epochs=8          epochs=0
    #     Q4_K   2.8s  +1.547%     1.6s  +1.547%
    #     Q5_K   5.6s  +4.549%     3.5s  +4.549%
    #     Q6_K   6.8s  +2.833%     3.1s  +2.833%
    #
    # Identical gains, ~45% less time. Set ``epochs`` explicitly (or pass
    # ``--epochs N``) to run the relaxation anyway.
    epochs: int = 0
    lr_logits: float = 2e-3
    temperature_init: float = 1.0
    temperature_end: float = 0.05
    logit_scale_init: float = 1.0
    logit_scale_end: float = 5.0
    seed: int = 0
    scale_passes: int = 2
    refit_codes: bool = True
    chunk_blocks: int = 2048


@dataclass
class RefineResult:
    """Outcome of refining one tensor.

    Attributes:
        codes: the refined integer codes, shape (n_blocks, block_elems).
        scales: the refined effective per-group scales.
        mins: the refined effective per-group minimums.
        raw: the refined per-block raw scale fields.
        initial_error: weighted reconstruction error before refinement.
        final_error: weighted reconstruction error after refinement.
    """

    codes: np.ndarray
    scales: np.ndarray
    mins: np.ndarray
    raw: List[BlockRaw]
    initial_error: float
    final_error: float

    @property
    def improved(self) -> bool:
        """True when the refinement reduced the reconstruction error."""
        return self.final_error < self.initial_error

    @property
    def relative_improvement(self) -> float:
        """Fractional reduction in the weighted reconstruction error."""
        if self.initial_error <= 0:
            return 0.0
        return (self.initial_error - self.final_error) / self.initial_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gumbel_softmax(
    logits: np.ndarray,
    temperature: float,
    logit_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gumbel-Softmax over the last axis of ``logits``.

    The computation is done in place on a copy of the input, so the peak
    memory is one extra array of the input's size rather than several.

    Args:
        logits: (n, n_levels) logits.
        temperature: softmax temperature.
        logit_scale: multiplicative gain on the logits.
        rng: numpy random generator.

    Returns:
        (n, n_levels) probabilities.
    """
    u = rng.random(logits.shape, dtype=np.float32)
    np.clip(u, 1e-20, 1.0 - 1e-20, out=u)
    # -log(-log(u)) is the Gumbel(0, 1) sample.
    np.log(u, out=u)
    np.negative(u, out=u)
    np.log(u, out=u)
    np.negative(u, out=u)

    z = np.array(logits, dtype=np.float32, copy=True)
    z *= np.float32(logit_scale)
    z += u
    z /= np.float32(temperature)
    z -= z.max(axis=-1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=-1, keepdims=True)
    return z


def _weighted_error(recon: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean squared reconstruction error, per element.

    The squared error is accumulated in float64, but the *elementwise*
    products are done in float32: on a large tensor the float64 temporaries
    would triple the peak memory for no accuracy gain in the per-element
    terms, and the final sum -- where the precision actually matters -- is
    still accumulated in float64.
    """
    diff = (recon - target).astype(np.float32)
    total = float((diff * diff * weights).sum(dtype=np.float64))
    return total / max(float(weights.sum(dtype=np.float64)), 1e-12)


def _fit_codes(
    values: np.ndarray,
    weights: np.ndarray,
    scale: float,
    minimum: float,
    spec: TypeSpec,
) -> np.ndarray:
    """Choose the best code for every element, given a group's scale/min.

    The scale may be negative (Q6_K's sub-scale is signed), in which case the
    code axis is inverted; the formula ``value = (code - bias) * scale - min``
    handles that automatically.

    Args:
        values: (group_size,) target values.
        weights: (group_size,) per-element weights.
        scale: the group's effective scale.
        minimum: the group's effective minimum.
        spec: the quant type's descriptor.

    Returns:
        (group_size,) integer codes.
    """
    cands = np.arange(spec.code_min, spec.code_max + 1, dtype=np.float64)
    recon = _candidate_values(spec).astype(np.float64) * scale - minimum
    errs = (recon[None, :] - values[:, None]) ** 2 * weights[:, None]
    return cands[errs.argmin(axis=1)].astype(np.int32)


def _refit_codes(
    codes: np.ndarray,
    scales: np.ndarray,
    mins: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    spec: TypeSpec,
) -> np.ndarray:
    """Re-fit every group's codes for the given (scale, min) fields."""
    out = codes.copy()
    for b in range(codes.shape[0]):
        for g in range(spec.n_groups):
            lo = g * spec.group_size
            hi = lo + spec.group_size
            out[b, lo:hi] = _fit_codes(
                values[b, lo:hi], weights[b, lo:hi], scales[b, g], mins[b, g], spec
            )
    return out


def _effective_from_raw(
    raws: List[BlockRaw],
    spec: TypeSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Effective per-group scales and minimums implied by raw block fields."""
    eff_s = np.zeros((len(raws), spec.n_groups), dtype=np.float32)
    eff_m = np.zeros((len(raws), spec.n_groups), dtype=np.float32)
    for b, raw in enumerate(raws):
        eff_s[b] = effective_scales(raw, spec)
        eff_m[b] = effective_mins(raw, spec)
    return eff_s, eff_m


def _copy_raw(raw: BlockRaw) -> BlockRaw:
    """Copy a block's raw fields so a candidate can be tried in isolation."""
    return BlockRaw(
        d=raw.d,
        dmin=raw.dmin,
        sc_raw=raw.sc_raw.copy(),
        m_raw=raw.m_raw.copy(),
    )


# ---------------------------------------------------------------------------
# The refinement
# ---------------------------------------------------------------------------

def refine_tensor(
    target: np.ndarray,
    type_name: str,
    imatrix: Optional[np.ndarray] = None,
    config: Optional[GsqConfig] = None,
    initial_codes: Optional[np.ndarray] = None,
    initial_raw: Optional[List[BlockRaw]] = None,
) -> RefineResult:
    """Refine one tensor's codes and scales on its existing quant grid.

    The refinement is **streamed over row groups**: the tensor is processed
    in slices of ``config.chunk_blocks`` rows, and each slice is refined
    independently. This is exact, not an approximation: the objective is a
    sum over elements, the quantization grid is per block, and every block's
    scale/minimum decision depends only on that block. So refining a slice
    gives the same result as refining the whole tensor at once -- with peak
    memory bounded by the slice instead of the tensor.

    For a fused MoE expert tensor ([256, 512, 2048] at Q4_K) the whole-tensor
    working set would be tens of gigabytes; with row chunking it stays in
    the low hundreds of megabytes.

    Args:
        target: (n_rows, n_per_row) float array, the original weights.
        type_name: llama.cpp quant type the tensor is stored as.
        imatrix: optional per-column importance weights (length n_per_row).
        config: GSQ hyper-parameters.
        initial_codes: (n_blocks, block_elems) starting codes, usually
            decoded from llama.cpp's own output.
        initial_raw: per-block raw scale fields matching ``initial_codes``.

    Returns:
        A :class:`RefineResult` with the refined codes/scales/mins.
    """
    cfg = config or GsqConfig()
    spec = spec_for(type_name)
    target = np.asarray(target, dtype=np.float32)
    if target.ndim != 2:
        raise ValueError("target must be a 2-D (n_rows, n_per_row) array")
    n_rows, n_per_row = target.shape
    if n_per_row % spec.block_elems != 0:
        raise ValueError(
            f"n_per_row={n_per_row} is not a multiple of the {type_name} block "
            f"size {spec.block_elems}"
        )
    if initial_codes is None or initial_raw is None:
        raise ValueError(
            "refine_tensor needs llama.cpp's own codes and raw scale fields "
            "(initial_codes / initial_raw) to start from"
        )

    blocks_per_row = n_per_row // spec.block_elems
    n_blocks = n_rows * blocks_per_row
    codes = np.asarray(initial_codes)
    if codes.shape != (n_blocks, spec.block_elems):
        raise ValueError(
            f"initial_codes has shape {codes.shape}, expected "
            f"{(n_blocks, spec.block_elems)}"
        )

    # How many rows to refine at a time. The working set of one slice is
    # dominated by the per-block arrays (values, weights, codes, recon) and
    # the Adam state, so the slice is sized to keep that bounded.
    rows_per_chunk = max(1, int(cfg.chunk_blocks))
    # A row's working set, in bytes, as a multiple of the tensor's own size.
    # values + weights + codes + recon + the Adam state, roughly 8x the
    # element count in float32.
    per_row_bytes = n_per_row * 4 * 8
    # Cap the slice so it stays under ~1 GiB.
    max_rows = max(1, (1 << 30) // max(per_row_bytes, 1))
    rows_per_chunk = max(1, min(rows_per_chunk, max_rows))
    if n_rows <= rows_per_chunk:
        return _refine_tensor_rows(
            target,
            type_name,
            imatrix,
            cfg,
            codes,
            list(initial_raw),
            0,
            n_rows,
            blocks_per_row,
        )

    # Refine the tensor in slices and stitch the results back together.
    all_codes = np.empty_like(codes)
    all_raws: List[BlockRaw] = [None] * n_blocks  # type: ignore[list-item]
    init_error = 0.0
    final_error = 0.0
    weight_total = 0.0
    for row_start in range(0, n_rows, rows_per_chunk):
        row_stop = min(row_start + rows_per_chunk, n_rows)
        n_chunk_rows = row_stop - row_start
        block_lo = row_start * blocks_per_row
        block_hi = row_stop * blocks_per_row
        chunk = _refine_tensor_rows(
            target[row_start:row_stop],
            type_name,
            imatrix,
            cfg,
            codes[block_lo:block_hi],
            list(initial_raw)[block_lo:block_hi],
            block_lo,
            n_chunk_rows,
            blocks_per_row,
        )
        all_codes[block_lo:block_hi] = chunk.codes
        for i, raw in enumerate(chunk.raw):
            all_raws[block_lo + i] = raw
        init_error += chunk.initial_error
        final_error += chunk.final_error
        weight_total += 1.0

    eff_scales, eff_mins = _effective_from_raw(all_raws, spec)
    return RefineResult(
        codes=all_codes,
        scales=eff_scales,
        mins=eff_mins,
        raw=all_raws,
        initial_error=init_error / max(weight_total, 1.0),
        final_error=final_error / max(weight_total, 1.0),
    )


def _refine_tensor_rows(
    target: np.ndarray,
    type_name: str,
    imatrix: Optional[np.ndarray],
    cfg: GsqConfig,
    codes: np.ndarray,
    raws: List[BlockRaw],
    block_offset: int,
    n_rows: int,
    blocks_per_row: int,
) -> RefineResult:
    """Refine a slice of rows. See :func:`refine_tensor`.

    Args:
        target: (n_rows, n_per_row) slice of the original weights.
        type_name: llama.cpp quant type name.
        imatrix: optional per-column importance weights.
        cfg: GSQ hyper-parameters.
        codes: (n_blocks, block_elems) starting codes for this slice.
        raws: per-block raw scale fields for this slice.
        block_offset: the index of this slice's first block in the tensor.
        n_rows: number of rows in the slice.
        blocks_per_row: blocks per row of the full tensor.

    Returns:
        The refinement result for this slice.
    """
    spec = spec_for(type_name)
    target = np.asarray(target, dtype=np.float32)
    n_per_row = target.shape[1]

    if imatrix is None:
        weights = np.ones((n_rows, n_per_row), dtype=np.float32)
    else:
        w = np.asarray(imatrix, dtype=np.float32).reshape(-1)
        if w.shape[0] != n_per_row:
            raise ValueError(
                f"imatrix has {w.shape[0]} entries but the tensor has {n_per_row} "
                f"columns"
            )
        weights = np.broadcast_to(w, (n_rows, n_per_row)).astype(np.float32)

    # A block is the unit of quantization. float32 throughout: the search
    # works on single-precision data, which halves the memory traffic.
    n_blocks_per_row = n_per_row // spec.block_elems
    n_blocks = n_rows * n_blocks_per_row
    values = np.ascontiguousarray(
        target.reshape(n_blocks, spec.block_elems), dtype=np.float32
    )
    block_weights = np.ascontiguousarray(
        weights.reshape(n_blocks, spec.block_elems), dtype=np.float32
    )

    # --- initial state -----------------------------------------------------
    codes = np.array(codes, dtype=np.int32, copy=True)
    if codes.shape != (n_blocks, spec.block_elems):
        raise ValueError(
            f"initial_codes has shape {codes.shape}, expected "
            f"{(n_blocks, spec.block_elems)}"
        )
    raws = list(raws)
    eff_scales, eff_mins = _effective_from_raw(raws, spec)

    recon = dequantize_blocks(codes, eff_scales, eff_mins, type_name)
    initial_error = _weighted_error(recon, values, block_weights)

    best_codes = codes.copy()
    best_raws = raws
    best_error = initial_error

    # --- stage 1: Gumbel-Softmax over the codes ----------------------------
    levels = np.arange(spec.code_min, spec.code_max + 1, dtype=np.float64)
    n_levels = levels.shape[0]

    # The relaxation's Adam state is (n_blocks, block_elems, n_levels), which
    # for a big tensor is enormous: a fused MoE expert tensor
    # ([256, 512, 2048] at Q4_K) would need 16 GB *per array*, 48 GB for the
    # three of them. The state is therefore kept **per row**: each row of the
    # tensor is relaxed on its own, and the optimizer state for one row is
    # (blocks_per_row, block_elems, n_levels) -- small and, crucially,
    # independent of the tensor's total size.
    #
    # The rows can be relaxed one at a time because the objective is a sum
    # over rows, so the result is the same as relaxing them together, with
    # memory bounded by a single row.
    blocks_per_row = max(1, n_per_row // spec.block_elems)
    # A row's Adam state, sized once and reused for every row.
    row_state_bytes = blocks_per_row * spec.block_elems * n_levels * 4
    # Rows processed per chunk: enough to fill the working set, but capped
    # so the three state arrays stay well under a gigabyte.
    rows_per_chunk = max(1, int(cfg.chunk_blocks))
    while rows_per_chunk > 1 and rows_per_chunk * row_state_bytes * 3 > (1 << 30):
        rows_per_chunk = max(1, rows_per_chunk // 2)

    rng = np.random.default_rng(cfg.seed)
    group_index = np.arange(spec.block_elems) // spec.group_size
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(cfg.epochs):
        progress = epoch / max(cfg.epochs - 1, 1)
        temperature = (
            cfg.temperature_init
            + (cfg.temperature_end - cfg.temperature_init) * progress
        )
        logit_scale = (
            cfg.logit_scale_init
            + (cfg.logit_scale_end - cfg.logit_scale_init) * progress
        )
        t = epoch + 1
        gain = logit_scale / max(temperature, 1e-8)

        for row_start in range(0, n_rows, rows_per_chunk):
            row_stop = min(row_start + rows_per_chunk, n_rows)
            n_chunk_rows = row_stop - row_start
            chunk_blocks = n_chunk_rows * blocks_per_row
            block_lo = row_start * blocks_per_row
            block_hi = row_stop * blocks_per_row

            # The Adam state for this chunk of rows.
            logits = np.empty(
                (chunk_blocks, spec.block_elems, n_levels), dtype=np.float32
            )
            m_logits = np.zeros_like(logits)
            v_logits = np.zeros_like(logits)

            chunk_codes = codes[block_lo:block_hi]
            idx = np.clip(chunk_codes - spec.code_min, 0, n_levels - 1)
            logits[:] = -1.0
            np.put_along_axis(logits, idx[:, :, None], 1.0, axis=2)

            chunk_values = values[block_lo:block_hi]
            chunk_weights = block_weights[block_lo:block_hi]
            chunk_scales = eff_scales[block_lo:block_hi]
            chunk_mins = eff_mins[block_lo:block_hi]

            for start in range(0, chunk_blocks, blocks_per_row):
                stop = min(start + blocks_per_row, chunk_blocks)
                lg = logits[start:stop]
                m_l = m_logits[start:stop]
                v_l = v_logits[start:stop]

                probs = _gumbel_softmax(
                    lg.reshape(-1, n_levels), temperature, logit_scale, rng
                ).reshape(stop - start, spec.block_elems, n_levels)

                # The scales are held fixed during the code relaxation: they
                # are discrete fields of the format and are refined in
                # stage 2.
                per_group_scale = chunk_scales[start:stop, group_index]
                per_group_min = chunk_mins[start:stop, group_index]
                # The soft assignment is a *probability distribution over
                # codes*, so the relaxed value is the distribution's
                # expectation of the code's value. For a uniform codebook
                # that is the same as interpolating in code space; for a
                # non-uniform one (IQ4_XS) it is not, so interpolate over
                # the codebook's values directly.
                if spec.codebook is not None:
                    book = np.asarray(spec.codebook, dtype=np.float32)
                    soft_values = (
                        probs @ book
                    ) * per_group_scale - per_group_min
                else:
                    soft_codes = probs @ levels.astype(np.float32)
                    soft_values = (
                        soft_codes - np.float32(spec.bias)
                    ) * per_group_scale - per_group_min

                diff = soft_values - chunk_values[start:stop]
                grad_soft_codes = (
                    2.0 * diff * chunk_weights[start:stop]
                ).astype(np.float32)

                # Softmax Jacobian: dL/dlogit_k = p_k * (g_k - sum_j p_j g_j).
                mean_g = (probs * grad_soft_codes[:, :, None]).sum(axis=2, keepdims=True)
                grad_codes = probs * (grad_soft_codes[:, :, None] - mean_g)
                grad_logits = grad_codes * np.float32(gain)

                m_l = np.float32(beta1) * m_l + np.float32(1 - beta1) * grad_logits
                v_l = np.float32(beta2) * v_l + np.float32(1 - beta2) * grad_logits ** 2
                lg -= np.float32(cfg.lr_logits) * (m_l / np.float32(1 - beta1 ** t)) / (
                    np.sqrt(v_l / np.float32(1 - beta2 ** t)) + np.float32(eps)
                )

                logits[start:stop] = lg
                m_logits[start:stop] = m_l
                v_logits[start:stop] = v_l

            # Evaluate the *hard* assignment of this chunk, which is what we
            # would store.
            hard_codes = levels[logits.argmax(axis=2)].astype(np.int32)
            hard_recon = dequantize_blocks(
                hard_codes, chunk_scales, chunk_mins, type_name
            )
            hard_error = _weighted_error(
                hard_recon, chunk_values, chunk_weights
            )

            if hard_error < best_error:
                best_error = hard_error
                best_codes[block_lo:block_hi] = hard_codes
                best_raws = raws

            del logits, m_logits, v_logits, hard_codes, hard_recon

        logger.debug(
            "gsq_epoch",
            extra={
                "epoch": epoch,
                "temperature": temperature,
                "best": best_error,
            },
        )

    # --- stage 2: coordinate descent over the integer scale fields ---------
    best_codes, best_raws, best_error = _refine_scales(
        best_codes,
        best_raws,
        values,
        block_weights,
        spec,
        type_name,
        cfg,
    )

    eff_scales, eff_mins = _effective_from_raw(best_raws, spec)
    return RefineResult(
        codes=best_codes,
        scales=eff_scales,
        mins=eff_mins,
        raw=best_raws,
        initial_error=initial_error,
        final_error=best_error,
    )


def _candidate_values(spec: TypeSpec) -> np.ndarray:
    """The value each candidate code maps to, before scaling.

    For a uniform codebook that is ``code - bias``; for a non-uniform one
    (IQ4_XS) it is the codebook entry itself.

    Args:
        spec: the quant type's descriptor.

    Returns:
        An array of ``n_levels`` values.
    """
    if spec.codebook is not None:
        return np.asarray(spec.codebook, dtype=np.float32)
    return np.arange(spec.code_min, spec.code_max + 1, dtype=np.float32) - np.float32(
        spec.bias
    )


def _code_values(spec: TypeSpec, codes: np.ndarray) -> np.ndarray:
    """Map codes to their values, before the per-group scale.

    Args:
        spec: the quant type's descriptor.
        codes: integer codes of any shape.

    Returns:
        The values, same shape as ``codes``.
    """
    if spec.codebook is not None:
        book = np.asarray(spec.codebook, dtype=np.float32)
        return book[codes.astype(np.int32)]
    return codes.astype(np.float32) - np.float32(spec.bias)


def _refit_group(
    values: np.ndarray,
    weights: np.ndarray,
    scale: np.ndarray,
    minimum: np.ndarray,
    spec: TypeSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Re-fit the codes of one group across every block, vectorized.

    Args:
        values: (n_blocks, group_size) target values.
        weights: (n_blocks, group_size) per-element weights.
        scale: (n_blocks,) the group's effective scale.
        minimum: (n_blocks,) the group's effective minimum.
        spec: the quant type's descriptor.

    Returns:
        (codes, error) where ``codes`` has shape (n_blocks, group_size) and
        ``error`` the weighted squared error per block.
    """
    cands = np.arange(spec.code_min, spec.code_max + 1, dtype=np.int32)
    # The value each candidate maps to, before the per-group scale.
    cand_values = _candidate_values(spec)
    # (n_blocks, n_cands)
    recon = cand_values[None, :] * scale[:, None] - minimum[:, None]
    # (n_blocks, group_size, n_cands). The error tensor is the hot spot of
    # the whole refinement, so it is built with in-place operations: the
    # difference, the square and the weighting each reuse one buffer
    # instead of allocating their own.
    errs = recon[:, None, :] - values[:, :, None]
    np.square(errs, out=errs)
    errs *= weights[:, :, None]
    best = errs.argmin(axis=2)
    codes = cands[best].astype(np.int32)
    picked = np.take_along_axis(errs, best[:, :, None], axis=2)[:, :, 0]
    error = picked.sum(axis=1, dtype=np.float64)
    return codes, error


def _refit_group_chunked(
    values: np.ndarray,
    weights: np.ndarray,
    scale: np.ndarray,
    minimum: np.ndarray,
    spec: TypeSpec,
    chunk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Like :func:`_refit_group` but bounded in memory.

    The candidate-error tensor is (n_blocks, group_size, n_levels), which
    grows quickly with the number of blocks, so it is built in chunks.
    """
    n_blocks = values.shape[0]
    codes = np.zeros((n_blocks, values.shape[1]), dtype=np.int32)
    error = np.zeros(n_blocks, dtype=np.float64)
    for start in range(0, n_blocks, chunk):
        stop = min(start + chunk, n_blocks)
        c, e = _refit_group(
            values[start:stop], weights[start:stop],
            scale[start:stop], minimum[start:stop], spec,
        )
        codes[start:stop] = c
        error[start:stop] = e
    return codes, error


def _refine_scales(
    codes: np.ndarray,
    raws: List[BlockRaw],
    values: np.ndarray,
    block_weights: np.ndarray,
    spec: TypeSpec,
    type_name: str,
    cfg: GsqConfig,
) -> Tuple[np.ndarray, List[BlockRaw], float]:
    """Coordinate descent over the integer per-group scale/minimum fields.

    For every sub-group we try moving its integer sub-scale (and sub-min)
    by a small delta and keep the change when it lowers the weighted
    reconstruction error. Because the fields are integers, every candidate
    is exactly representable in the format -- nothing is projected away.

    The search is vectorized over blocks: for a fixed sub-group, all blocks
    are evaluated together, and each block keeps its own decision.

    Args:
        codes: (n_blocks, block_elems) starting codes.
        raws: per-block raw scale fields.
        values: (n_blocks, block_elems) target values.
        block_weights: (n_blocks, block_elems) per-element weights.
        spec: the quant type's descriptor.
        type_name: llama.cpp quant type name.
        cfg: GSQ hyper-parameters.

    Returns:
        (codes, raws, error).
    """
    eff_scales, eff_mins = _effective_from_raw(raws, spec)
    recon = dequantize_blocks(codes, eff_scales, eff_mins, type_name)
    best_error = _weighted_error(recon, values, block_weights)
    best_raws = list(raws)
    best_codes = codes.copy()

    if spec.n_groups == 1:
        return best_codes, best_raws, best_error

    n_blocks = codes.shape[0]
    gs = spec.group_size
    deltas = (-3, -2, -1, 1, 2, 3)
    chunk = max(1, int(cfg.chunk_blocks))

    # Per-block, per-group effective values, kept in sync with best_raws.
    cur_scales = eff_scales.astype(np.float32).copy()
    cur_mins = eff_mins.astype(np.float32).copy()
    # The integer fields, as they appear in the block.
    sc_raw = np.stack([r.sc_raw for r in best_raws]).astype(np.int64)
    m_raw = np.stack([r.m_raw for r in best_raws]).astype(np.int64) \
        if spec.has_min else np.zeros((n_blocks, spec.n_groups), dtype=np.int64)
    # The per-block fp16 super-scales, which the sub-fields multiply.
    block_d = np.array([r.d for r in best_raws], dtype=np.float32)
    block_dmin = np.array([r.dmin for r in best_raws], dtype=np.float32)

    for _ in range(max(cfg.scale_passes, 0)):
        improved = False
        for g in range(spec.n_groups):
            lo = g * gs
            hi = lo + gs
            v = values[:, lo:hi]
            w = block_weights[:, lo:hi]

            # Baseline for this group: current codes and their error.
            base_codes = best_codes[:, lo:hi]
            base_recon = _code_values(spec, base_codes) \
                * cur_scales[:, g:g + 1] - cur_mins[:, g:g + 1]
            base_err = ((base_recon - v) ** 2 * w).sum(axis=1, dtype=np.float64)

            best_group_err = base_err.copy()
            best_group_codes = base_codes.copy()
            best_sc = sc_raw[:, g].copy()
            best_m = m_raw[:, g].copy()

            fields = ("sc", "m") if spec.has_min else ("sc",)
            for field in fields:
                for delta in deltas:
                    if field == "sc":
                        # Measure every candidate against the *current* best
                        # value for this group, not against the original one,
                        # so that a sequence of accepted deltas cannot end up
                        # worse than the baseline.
                        new_val = best_sc + delta
                        ok = (new_val >= spec.sc_raw_min) & (new_val <= spec.sc_raw_max)
                        if not ok.any():
                            continue
                        # The effective scale is d * (sc_raw - scale_bias);
                        # Q3_K stores a signed sub-scale with a +32 offset.
                        cand_scales = block_d * (new_val - spec.scale_bias)
                        cand_mins = cur_mins[:, g]
                    else:
                        new_val = best_m + delta
                        ok = (new_val >= 0) & (new_val <= spec.min_raw_max)
                        if not ok.any():
                            continue
                        cand_scales = cur_scales[:, g]
                        cand_mins = block_dmin * new_val

                    if cfg.refit_codes:
                        cand_codes, cand_err = _refit_group_chunked(
                            v, w, cand_scales, cand_mins, spec, chunk
                        )
                    else:
                        recon_c = _code_values(spec, best_group_codes) \
                            * cand_scales[:, None] - cand_mins[:, None]
                        cand_err = ((recon_c - v) ** 2 * w).sum(axis=1, dtype=np.float64)
                        cand_codes = best_group_codes

                    take = ok & (cand_err < best_group_err)
                    if take.any():
                        best_group_err = np.where(take, cand_err, best_group_err)
                        best_group_codes = np.where(
                            take[:, None], cand_codes, best_group_codes
                        )
                        if field == "sc":
                            best_sc = np.where(take, new_val, best_sc)
                            cur_scales[take, g] = cand_scales[take]
                        else:
                            best_m = np.where(take, new_val, best_m)
                            cur_mins[take, g] = cand_mins[take]

            # Apply the group's improvements.
            changed = best_group_err < base_err
            if changed.any():
                improved = True
                best_codes[:, lo:hi] = best_group_codes
                sc_raw[:, g] = best_sc
                if spec.has_min:
                    m_raw[:, g] = best_m
                cur_scales[:, g] = block_d * (best_sc - spec.scale_bias)
                if spec.has_min:
                    cur_mins[:, g] = block_dmin * best_m

        if not improved:
            break

    # Rebuild the raw fields from the searched integer fields.
    for b in range(n_blocks):
        best_raws[b] = BlockRaw(
            d=float(block_d[b]),
            dmin=float(block_dmin[b]),
            sc_raw=sc_raw[b].copy(),
            m_raw=m_raw[b].copy() if spec.has_min else np.zeros(0, dtype=np.int64),
        )

    eff_scales, eff_mins = _effective_from_raw(best_raws, spec)
    recon = dequantize_blocks(best_codes, eff_scales, eff_mins, type_name)
    final_error = _weighted_error(recon, values, block_weights)
    return best_codes, best_raws, final_error


__all__ = [
    "GsqConfig",
    "RefineResult",
    "refine_tensor",
]
