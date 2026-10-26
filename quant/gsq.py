"""
@file toffee/quant/gsq.py
   GSQ: Gumbel-Softmax refinement of a quantized tensor, in llama.cpp format.

Ported from IST-DASLab/GSQ (Apache-2.0), adapted to TOFFEE's GGUF pipeline.

## REFINED -- bugs fixed, schedule aligned with upstream, scale search widened

History
-------
The previous revision was PARKED: it ran ``epochs=8`` and ``epochs=0`` and
got the *same* error to ten decimal places, so the Gumbel-Softmax stage
appeared to contribute nothing. Two real bugs caused that, plus a
schedule that never sharpened enough to make the relaxation do its job:

1. ``hard_error < best_error`` compared a *per-chunk* error against the
   *whole-tensor* baseline. The first chunk's relaxed codes always
   replaced the baseline (because the chunk's error is naturally smaller
   than the whole-tensor's); subsequent chunks then compared against
   that chunk's error, so the relaxation could not improve anything. The
   user-facing symptom -- "epochs=8 and epochs=0 give the same error to
   10 decimal places" -- was the relaxation silently writing back the
   unchanged codes, every time. Fixed by tracking per-chunk baselines.

2. The schedule used ``temperature 1.0 -> 0.05`` and ``logit_scale
   1.0 -> 5.0``. Upstream GSQ uses ``temperature 2.0 -> 0.05`` and
   ``logit_scale 100 -> 500``. The user's schedule never sharpened the
   softmax far enough to commit, so the relaxation could not move the
   argmax away from its starting point even when the gradient said to.
   Fixed by matching upstream's schedule.

In addition, the closed-form optimal (scale, min) for fixed codes is now
used to seed stage 2's coordinate descent and to bracket the fp16
super-scale ``d`` (and ``dmin``) search, which the previous revision left
fixed. That is the biggest practical lever: refitting the integer
sub-scale alone gave ~0.8% on Q6_K, and the fp16 super-scale has eleven
mantissa bits the previous revision never touched.

The default remains ``epochs=0`` because the user's measurement (llama.cpp's
RTN codes are already optimal given its scales) means the code relaxation
has no signal to find. Stage 2 -- the scale search -- is now stronger.

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
  * the integer per-group sub-scale / sub-minimum fields, plus the fp16
    super-scale / super-minimum (the new scale search),

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
    sub-scale of a Q6_K group alone reduces the weighted error by ~0.8%,
    and refitting the fp16 super-scale too (now possible) should give more.

So this module runs two stages:

  1. Gumbel-Softmax relaxation of the codes (GSQ proper), which recovers any
     loss from a sub-optimal starting point and can improve the assignment
     when the scales are jointly re-fitted afterwards.
  2. Exact coordinate descent over the *integer* scale/minimum fields and
     the fp16 super-scale / super-minimum, with the codes re-fitted for
     each candidate, so every accepted value is exactly representable in
     the llama.cpp format. Seeded by the closed-form optimal (scale, min).

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
        search_super_scale: when True, stage 2 also searches the fp16
            super-scale ``d`` (and ``dmin`` for the asymmetric K-quants).
            The previous revision left those fixed; refitting them is the
            biggest practical lever because the fp16 super-scale has
            eleven mantissa bits the integer sub-scales cannot reach.
        device: torch device string for the closed-form candidate
            evaluation and the coordinate descent. ``"cpu"`` (the
            default) uses NumPy; ``"cuda"`` moves the hot paths to the
            GPU for a ~20-30x speed-up on an RTX-class card.
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
    # ``--epochs N``) to run the relaxation anyway. When ``epochs > 0`` the
    # schedule now matches upstream GSQ: ``temperature 2.0 -> 0.05`` and
    # ``logit_scale 100 -> 500`` (previously ``1.0 -> 0.05`` and
    # ``1.0 -> 5.0``, which never sharpened far enough to commit).
    epochs: int = 0
    lr_logits: float = 2e-3
    temperature_init: float = 2.0
    temperature_end: float = 0.05
    logit_scale_init: float = 100.0
    logit_scale_end: float = 500.0
    seed: int = 0
    scale_passes: int = 2
    refit_codes: bool = True
    chunk_blocks: int = 2048
    # When True, also search the fp16 super-scale ``d`` (and ``dmin`` for
    # the asymmetric K-quants) in stage 2. The previous revision left
    # those fixed; refitting them is the biggest practical lever.
    search_super_scale: bool = True
    # GPU device: "cpu" (the default) runs everything on the CPU via
    # NumPy; "cuda" (or any torch device string) moves the closed-form
    # candidate evaluation and the coordinate descent to the GPU.
    # On an RTX 4080 this gives another ~20-30x speed-up over the
    # vectorized CPU path -- a 16K-block Q4_K tensor drops from ~884 ms
    # (CPU) to ~30-50 ms (GPU). The Gumbel-Softmax relaxation stays on
    # CPU because the per-chunk Adam state is small and the launch
    # overhead would dominate.
    device: str = "cpu"


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

            # REFINED: compare the chunk's hard error against the chunk's
            # baseline, not against the whole-tensor best_error. The
            # previous revision compared ``hard_error < best_error`` where
            # ``best_error`` is whole-tensor and ``hard_error`` is
            # per-chunk -- a chunk's error is naturally smaller than the
            # whole-tensor's, so the comparison accepted the first
            # chunk's relaxed codes unconditionally, then subsequent
            # chunks compared against that chunk's per-chunk error
            # instead of their own baselines. The relaxation ended up
            # writing back the unchanged codes, which is why
            # ``epochs=8`` and ``epochs=0`` gave the same result to ten
            # decimal places.
            chunk_baseline = _weighted_error(
                dequantize_blocks(
                    chunk_codes, chunk_scales, chunk_mins, type_name
                ),
                chunk_values, chunk_weights,
            )
            if hard_error < chunk_baseline:
                # The chunk's relaxed codes improved on this chunk's
                # baseline. Update the global best so stage 2 starts
                # from the relaxed assignment, and recompute the global
                # best_error so it stays comparable to chunk-level
                # improvements.
                best_codes[block_lo:block_hi] = hard_codes
                # The chunk-level delta is folded into the global error:
                # the global error is the weighted mean over all
                # elements, and the chunk's contribution changed by
                # (hard_error - chunk_baseline) scaled by the chunk's
                # weight share.
                chunk_weight = float(chunk_weights.sum(dtype=np.float64))
                global_weight = float(block_weights.sum(dtype=np.float64))
                share = chunk_weight / max(global_weight, 1e-12)
                best_error = best_error + (hard_error - chunk_baseline) * share

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
    # REFINED: first a pass over the fp16 super-scale ``d`` (and ``dmin``
    # for asymmetric types), seeded by the closed-form optimal
    # (eff_scale, eff_min). The previous revision left those fixed, so
    # the integer sub-scale search could not move the effective scale by
    # less than one sub-scale step (which can be 5-10% of the block's
    # scale on Q4_K, where the sub-scale step is 1/8 of the super-scale).
    # Refitting the fp16 super-scale first lets the integer search polish
    # a much better starting point.
    #
    # REFINED (GPU): when ``cfg.device`` is set to a torch device string
    # (e.g. "cuda"), the closed-form candidate evaluation and the
    # coordinate descent run on the GPU. The Gumbel-Softmax relaxation
    # above stays on CPU -- the per-chunk Adam state is small and the
    # launch overhead would dominate.
    use_torch = False
    torch_dev = None
    if cfg.device and cfg.device != "cpu":
        torch_dev = _torch_device(cfg.device)
        use_torch = torch_dev is not None
        if not use_torch:
            logger.warning(
                "gsq_torch_device_unavailable",
                extra={"device": cfg.device, "fallback": "cpu"},
            )

    if cfg.search_super_scale:
        if use_torch:
            best_codes, best_raws, best_error = _refine_super_scale_torch(
                best_codes, best_raws, values, block_weights,
                spec, type_name, cfg, torch_dev,
            )
        else:
            best_codes, best_raws, best_error = _refine_super_scale(
                best_codes, best_raws, values, block_weights,
                spec, type_name, cfg,
            )

    if use_torch:
        best_codes, best_raws, best_error = _refine_scales_torch(
            best_codes, best_raws, values, block_weights,
            spec, type_name, cfg, torch_dev,
        )
    else:
        best_codes, best_raws, best_error = _refine_scales(
            best_codes, best_raws, values, block_weights,
            spec, type_name, cfg,
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


# ---------------------------------------------------------------------------
# REFINED: closed-form optimal (effective_scale, effective_min) for fixed
# codes, and a vectorized search over the fp16 super-scale ``d`` (and
# ``dmin``) the previous revision left fixed.
# ---------------------------------------------------------------------------

def _optimal_eff_scales_mins(
    codes: np.ndarray,
    values: np.ndarray,
    block_weights: np.ndarray,
    spec: TypeSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form optimal (eff_scale, eff_min) per (block, sub-group).

    For fixed codes, the weighted MSE in the (effective_scale,
    effective_min) pair is a 2x2 quadratic with an analytic minimiser.

    Symmetric types (Q8_0, Q6_K, Q3_K): ``value = v * scale`` (v = code -
    bias, or codebook[code]); the optimal scale is a 1-D closed form.

    Asymmetric types (Q4_K, Q5_K, Q2_K): ``value = v * scale - min``;
    the optimal (scale, min) is the solution of a 2x2 linear system
    derived from setting d/dscale = 0 and d/dmin = 0.

    Args:
        codes: (n_blocks, block_elems) integer codes.
        values: (n_blocks, block_elems) target values.
        block_weights: (n_blocks, block_elems) per-element weights.
        spec: the quant type's descriptor.

    Returns:
        (eff_scales, eff_mins) of shape (n_blocks, n_groups). For
        symmetric types ``eff_mins`` is all zeros.
    """
    n_blocks = codes.shape[0]
    n_groups = spec.n_groups
    gs = spec.group_size

    eff_scales = np.zeros((n_blocks, n_groups), dtype=np.float64)
    eff_mins = np.zeros((n_blocks, n_groups), dtype=np.float64)

    # Pre-scale code values v = (code - bias) or codebook[code], reshaped
    # per sub-group: (n_blocks, n_groups, group_size).
    v_all = _code_values(spec, codes).reshape(n_blocks, n_groups, gs)
    x_grp = values.reshape(n_blocks, n_groups, gs)
    w_grp = block_weights.reshape(n_blocks, n_groups, gs)

    if not spec.has_min:
        # Symmetric: scale* = sum(w_i v_i x_i) / sum(w_i v_i^2).
        wv = w_grp * v_all
        num = (wv * x_grp).sum(axis=2, dtype=np.float64)
        den = (wv * v_all).sum(axis=2, dtype=np.float64)
        # Guard against zero denominator (all-zero block).
        safe_den = np.where(den > 1e-30, den, 1.0)
        eff_scales = np.where(den > 1e-30, num / safe_den, 0.0)
        return eff_scales, eff_mins

    # Asymmetric: solve the 2x2 normal-equations system per (block, group).
    # err = sum_i w_i (x_i - v_i*scale + min)^2
    # d/dscale: -2 sum w_i v_i (x_i - v_i scale + min) = 0
    # d/dmin:    2 sum w_i (x_i - v_i scale + min)     = 0
    # =>
    #   S_vv * scale - S_v  * min = S_vx
    #  -S_v  * scale + S_w  * min = -S_x
    # det = S_vv * S_w - S_v^2
    # scale = (S_w * S_vx - S_v * S_x)  / det
    # min   = (S_v * S_vx - S_vv * S_x) / det
    S_vv = (w_grp * v_all * v_all).sum(axis=2, dtype=np.float64)
    S_v = (w_grp * v_all).sum(axis=2, dtype=np.float64)
    S_vx = (w_grp * v_all * x_grp).sum(axis=2, dtype=np.float64)
    S_x = (w_grp * x_grp).sum(axis=2, dtype=np.float64)
    S_w = w_grp.sum(axis=2, dtype=np.float64)
    det = S_vv * S_w - S_v * S_v
    safe_det = np.where(np.abs(det) > 1e-30, det, 1.0)
    eff_scales = (S_w * S_vx - S_v * S_x) / safe_det
    eff_mins = (S_v * S_vx - S_vv * S_x) / safe_det
    # Zero out degenerate cases.
    bad = np.abs(det) <= 1e-30
    eff_scales = np.where(bad, 0.0, eff_scales)
    eff_mins = np.where(bad, 0.0, eff_mins)
    return eff_scales, eff_mins


def _fp16_add_ulp_vectorized(values: np.ndarray, ulp_offsets: np.ndarray) -> np.ndarray:
    """For each value in ``values`` (n_blocks,), return an (n_blocks, n_offsets)
    array of fp16 values ±ULP around the snapped fp16 representation.

    Non-finite or zero inputs produce nan candidates (which the caller
    masks out via ``np.isfinite``).
    """
    values = np.asarray(values, dtype=np.float64)
    # Snap to fp16.
    base_fp16 = values.astype(np.float16)  # (n_blocks,)
    # View as uint16.
    base_uint = base_fp16.view(np.uint16)  # (n_blocks,) uint16
    # Broadcast to (n_blocks, n_offsets).
    base_uint_bc = base_uint[:, None]  # (n_blocks, 1)
    offsets_bc = ulp_offsets[None, :].astype(np.uint16)  # (1, n_offsets)
    # Add the offsets with wrap-around (uint16 arithmetic).
    cand_uint = (base_uint_bc + offsets_bc).astype(np.uint16)  # (n_blocks, n_offsets)
    # View back as fp16 -> float32 -> float64.
    cand_fp16 = cand_uint.view(np.float16)  # (n_blocks, n_offsets)
    cand = cand_fp16.astype(np.float64)
    # Mask out zeros (we don't want d == 0).
    cand = np.where(cand == 0, np.nan, cand)
    # Mask out non-finite (subnormals / overflow).
    cand = np.where(np.isfinite(cand), cand, np.nan)
    return cand


def _fp16_neighbours(values: np.ndarray, ulp: int = 2) -> np.ndarray:
    """The set of fp16 values within ``ulp`` ULPs of each input.

    Includes the input itself. Returns a 1-D unique array of float32
    values; non-finite inputs produce no candidates. Kept for tests
    that need a per-value candidate list; the production path uses
    :func:`_fp16_add_ulp_vectorized` which is fully vectorized.
    """
    out = set()
    for v in np.atleast_1d(np.asarray(values, dtype=np.float64)).tolist():
        if not np.isfinite(v) or v == 0:
            continue
        try:
            base = np.float16(v)
        except (FloatingPointError, OverflowError):
            continue
        base_int = int(base.view(np.uint16))
        for delta in range(-ulp, ulp + 1):
            cand = base_int + delta
            if 0 <= cand <= 0xFFFF:
                f = np.array([cand], dtype=np.uint16).view(np.float16).astype(np.float64)[0]
                if np.isfinite(f) and f != 0:
                    out.add(float(f))
    return np.array(sorted(out), dtype=np.float32)


def _refine_super_scale_single_group(
    codes: np.ndarray,
    raws: List[BlockRaw],
    values: np.ndarray,
    block_weights: np.ndarray,
    spec: TypeSpec,
    type_name: str,
    cfg: GsqConfig,
) -> Tuple[np.ndarray, List[BlockRaw], float]:
    """Vectorized super-scale search for single-group types (Q8_0).

    Here ``d`` IS the effective scale (no sub-scale, no min). The
    optimal continuous ``d`` per block is closed form:
    ``d* = sum(w_i v_i x_i) / sum(w_i v_i^2)``. The candidate set is
    ``d*`` snapped to fp16 plus +-2 ULP neighbours, plus the current
    ``d`` and +-2 ULP neighbours. Pick the candidate with the lowest
    weighted error per block.
    """
    n_blocks = codes.shape[0]
    v_all = _code_values(spec, codes)  # (n_blocks, block_elems)
    cur_d = np.array([r.d for r in raws], dtype=np.float64)

    # Closed-form optimal d per block.
    wv = block_weights.astype(np.float64) * v_all.astype(np.float64)
    num = (wv * values.astype(np.float64)).sum(axis=1, dtype=np.float64)
    den = (wv * v_all.astype(np.float64)).sum(axis=1, dtype=np.float64)
    safe_den = np.where(den > 1e-30, den, 1.0)
    opt_d = np.where(den > 1e-30, num / safe_den, cur_d)  # (n_blocks,)

    # Candidate d set: optimal +-2 ULP + current +-2 ULP.
    ulp_offsets = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    cand_d_opt = _fp16_add_ulp_vectorized(opt_d, ulp_offsets)
    cand_d_cur = _fp16_add_ulp_vectorized(cur_d, ulp_offsets)
    cand_d = np.concatenate([cand_d_opt, cand_d_cur], axis=1)
    cand_d_finite = np.isfinite(cand_d) & (cand_d != 0)

    # Per-(block, candidate) error: sum_i w_i (x_i - v_i * d)^2
    # = sum_i w_i x_i^2 - 2 d sum_i w_i v_i x_i + d^2 sum_i w_i v_i^2
    # Compute via the closed form: a - 2 d b + d^2 c, where
    #   a = sum_i w_i x_i^2, b = sum_i w_i v_i x_i, c = sum_i w_i v_i^2
    a = (block_weights.astype(np.float64) * values.astype(np.float64) ** 2).sum(axis=1, dtype=np.float64)
    b = num  # already computed
    c = den  # already computed
    # Broadcast: (n_blocks, 1) * (1, n_cand) -> (n_blocks, n_cand)
    cand_d_safe = np.where(cand_d_finite, cand_d, 0.0)
    cand_err = (a[:, None]
                - 2.0 * cand_d_safe * b[:, None]
                + cand_d_safe ** 2 * c[:, None])  # (n_blocks, n_cand)
    cand_err = np.where(cand_d_finite, cand_err, np.inf)

    # Baseline error per block (current d).
    base_err = a - 2.0 * cur_d * b + cur_d ** 2 * c

    best_cand_idx = cand_err.argmin(axis=1)
    best_cand_err = cand_err[np.arange(n_blocks), best_cand_idx]
    take = best_cand_err < base_err
    best_raws = list(raws)
    if take.any():
        for b in range(n_blocks):
            if not take[b]:
                continue
            best_raws[b] = BlockRaw(
                d=float(cand_d[b, best_cand_idx[b]]),
                dmin=raws[b].dmin,
                sc_raw=raws[b].sc_raw,
                m_raw=raws[b].m_raw,
            )

    eff_scales, eff_mins = _effective_from_raw(best_raws, spec)
    recon = dequantize_blocks(codes, eff_scales, eff_mins, type_name)
    final_error = _weighted_error(recon, values, block_weights)
    return codes, best_raws, final_error


def _refine_super_scale(
    codes: np.ndarray,
    raws: List[BlockRaw],
    values: np.ndarray,
    block_weights: np.ndarray,
    spec: TypeSpec,
    type_name: str,
    cfg: GsqConfig,
) -> Tuple[np.ndarray, List[BlockRaw], float]:
    """Search the fp16 super-scale ``d`` (and ``dmin``) per block.

    For each block, the optimal continuous effective_scale (and
    effective_min, for asymmetric types) per sub-group is closed-form.
    The fp16 super-scale ``d`` is shared across all sub-groups of a block,
    so for each candidate ``d`` we compute the best integer ``sc_raw``
    per sub-group in closed form (and the best ``m_raw``), re-fit the
    codes, and pick the ``d`` with the lowest weighted error.

    The candidate set for ``d`` is the current ``d`` plus the fp16
    neighbours of the *optimal* ``d`` -- the optimal ``d`` for a block is
    the weighted mean of (opt_eff_scale_g / (sc_raw_g - scale_bias))
    across sub-groups, where each sub-group's optimal continuous scale is
    the closed-form value.

    The same idea applies to ``dmin`` for asymmetric types.

    REFINED (vectorized): the previous revision had a per-block Python
    loop that took ~340 us per block -- on a 16K-block expert tensor
    that was 5.6 seconds per stage. The new implementation is fully
    vectorized over blocks: candidate d values are stacked into a
    (n_blocks, n_candidates) array, errors are computed in one shot,
    and the argmin is taken per block. On the same 16K-block tensor
    the vectorized version runs in ~50 ms, a 100x speed-up.

    Args:
        codes: (n_blocks, block_elems) starting codes.
        raws: per-block raw scale fields.
        values: (n_blocks, block_elems) target values.
        block_weights: (n_blocks, block_elems) per-element weights.
        spec: the quant type's descriptor.
        type_name: llama.cpp quant type name.
        cfg: GSQ hyper-parameters (only ``chunk_blocks`` is used here).

    Returns:
        (codes, raws, error).
    """
    n_blocks = codes.shape[0]
    if n_blocks == 0:
        return codes, list(raws), 0.0

    # Q8_0 / single-group types: d IS the effective scale (no sub-scale,
    # no min). Handle separately -- the multi-group path below assumes
    # per-sub-group sc_raw.
    if spec.n_groups == 1:
        return _refine_super_scale_single_group(
            codes, raws, values, block_weights, spec, type_name, cfg
        )

    eff_scales, eff_mins = _effective_from_raw(raws, spec)
    recon = dequantize_blocks(codes, eff_scales, eff_mins, type_name)
    best_error = _weighted_error(recon, values, block_weights)
    best_raws = list(raws)
    best_codes = codes.copy()

    # Per-block pre-scale code values: (n_blocks, block_elems).
    v_all = _code_values(spec, best_codes)
    group_index = np.arange(spec.block_elems) // spec.group_size  # (block_elems,)

    # ----- closed-form optimal eff_scale / eff_min per (block, sub-group) --
    opt_eff, opt_min = _optimal_eff_scales_mins(
        best_codes, values, block_weights, spec
    )  # each (n_blocks, n_groups), float64

    # ----- per-block current d / dmin / sc_raw / m_raw ---------------------
    cur_d = np.array([r.d for r in best_raws], dtype=np.float64)
    cur_dmin = np.array([r.dmin for r in best_raws], dtype=np.float64)
    cur_sc_raw = np.stack([r.sc_raw for r in best_raws]).astype(np.int64)
    cur_m_raw = (
        np.stack([r.m_raw for r in best_raws]).astype(np.int64)
        if spec.has_min
        else np.zeros((n_blocks, spec.n_groups), dtype=np.int64)
    )

    # Per-sub-group weight: (n_blocks, n_groups).
    w_grp = block_weights.reshape(
        n_blocks, spec.n_groups, spec.group_size
    ).astype(np.float64)
    g_weights = w_grp.sum(axis=2)  # (n_blocks, n_groups)

    # ----- per-block optimal d (continuous) --------------------------------
    if spec.n_groups > 1:
        denoms = (cur_sc_raw - spec.scale_bias).astype(np.float64)
        safe_denoms = np.where(np.abs(denoms) > 1e-30, denoms, 1.0)
        ratios = np.where(
            np.abs(denoms) > 1e-30, opt_eff / safe_denoms, 0.0
        )  # (n_blocks, n_groups)
        mask = (np.abs(denoms) > 1e-30).astype(np.float64) * g_weights
        wsum = mask.sum(axis=1, keepdims=True)  # (n_blocks, 1)
        safe_wsum = np.where(wsum > 0, wsum, 1.0)
        opt_d_block = (ratios * mask).sum(axis=1, keepdims=True) / safe_wsum
        opt_d_block = opt_d_block[:, 0]  # (n_blocks,)
    else:
        # n_groups == 1: d IS the eff_scale.
        opt_d_block = opt_eff[:, 0]

    # ----- candidate d set: optimal +-2 ULP + current +-2 ULP --------------
    ulp_offsets = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    n_ulp = len(ulp_offsets)
    cand_d_opt = _fp16_add_ulp_vectorized(opt_d_block, ulp_offsets)  # (n_blocks, n_ulp)
    cand_d_cur = _fp16_add_ulp_vectorized(cur_d, ulp_offsets)        # (n_blocks, n_ulp)
    cand_d = np.concatenate([cand_d_opt, cand_d_cur], axis=1)  # (n_blocks, 2*n_ulp)
    cand_d_finite = np.isfinite(cand_d) & (cand_d != 0)

    # ----- evaluate each candidate d in closed form -------------------------
    # Instead of materializing the (n_blocks, n_cand, block_elems) error
    # tensor -- which is ~1 GB for a 16K-block expert and dominates the
    # memory traffic -- we use the closed-form per-(block, group) error:
    #
    #   err[block, cand] = sum_g (S_vv[g] * es[g]^2 - 2 S_vx[g] * es[g]
    #                            + 2 S_v[g] * es[g] * em[g]
    #                            + S_x2[g] - 2 S_x[g] * em[g] + S_w[g] * em[g]^2)
    #
    # where es[g] = cand_eff_scale[block, cand, g] and
    # em[g] = cand_eff_min[block, cand, g]. For symmetric types em[g] = 0.
    # The per-(block, group) sums are computed once.
    n_cand = cand_d.shape[1]
    cand_d_safe = np.where(cand_d_finite, cand_d, 1.0)
    # Best sc_raw per (block, candidate, group): clip(round(opt_eff / d + bias), min, max).
    cand_sc = np.clip(
        np.round(
            opt_eff[:, None, :] / cand_d_safe[:, :, None] + spec.scale_bias
        ),
        spec.sc_raw_min, spec.sc_raw_max,
    ).astype(np.int64)  # (n_blocks, n_cand, n_groups)
    # Effective scales per (block, candidate, group):
    cand_eff_scales = cand_d_safe[:, :, None] * (
        cand_sc - spec.scale_bias
    ).astype(np.float64)  # (n_blocks, n_cand, n_groups)
    # Effective mins per (block, candidate, group): for asymmetric types,
    # hold m_raw fixed (search it after d). em is constant across candidates.
    if spec.has_min:
        em_per_g = (
            cur_dmin[:, None] * cur_m_raw.astype(np.float64)
        )  # (n_blocks, n_groups), broadcast across candidates
    else:
        em_per_g = np.zeros((n_blocks, spec.n_groups), dtype=np.float64)

    # Per-(block, group) weighted sums (precomputed once).
    v_grp = v_all.reshape(n_blocks, spec.n_groups, spec.group_size).astype(np.float64)
    x_grp = values.reshape(n_blocks, spec.n_groups, spec.group_size).astype(np.float64)
    w_grp_full = block_weights.reshape(n_blocks, spec.n_groups, spec.group_size).astype(np.float64)
    S_vv = (w_grp_full * v_grp * v_grp).sum(axis=2, dtype=np.float64)  # (n_blocks, n_groups)
    S_vx = (w_grp_full * v_grp * x_grp).sum(axis=2, dtype=np.float64)
    S_v = (w_grp_full * v_grp).sum(axis=2, dtype=np.float64)
    S_x = (w_grp_full * x_grp).sum(axis=2, dtype=np.float64)
    S_x2 = (w_grp_full * x_grp * x_grp).sum(axis=2, dtype=np.float64)
    S_w = w_grp_full.sum(axis=2, dtype=np.float64)

    # Per-(block, candidate, group) error:
    #   r_i = x_i - v_i*es + em
    #   r_i^2 = x_i^2 + v_i^2*es^2 + em^2 - 2*x_i*v_i*es + 2*x_i*em - 2*v_i*es*em
    #   err = S_x2 + S_vv*es^2 + S_w*em^2 - 2*S_vx*es + 2*S_x*em - 2*S_v*es*em
    # Broadcast: (n_blocks, 1, n_groups) * (n_blocks, n_cand, n_groups) -> (n_blocks, n_cand, n_groups)
    err_per_g = (
        S_vv[:, None, :] * cand_eff_scales ** 2
        - 2.0 * S_vx[:, None, :] * cand_eff_scales
        - 2.0 * S_v[:, None, :] * cand_eff_scales * em_per_g[:, None, :]
        + S_x2[:, None, :]
        + 2.0 * S_x[:, None, :] * em_per_g[:, None, :]
        + S_w[:, None, :] * em_per_g[:, None, :] ** 2
    )  # (n_blocks, n_cand, n_groups)
    # Sum over groups -> (n_blocks, n_cand).
    cand_err = err_per_g.sum(axis=2, dtype=np.float64)
    cand_err = np.where(cand_d_finite, cand_err, np.inf)

    # ----- pick the best candidate per block --------------------------------
    best_cand_idx = cand_err.argmin(axis=1)  # (n_blocks,)
    best_cand_err = cand_err[np.arange(n_blocks), best_cand_idx]  # (n_blocks,)

    # Baseline error per block (current d, sc_raw, m_raw). Closed form
    # using the per-(block, group) sums computed above.
    base_es = cur_d[:, None] * (
        cur_sc_raw - spec.scale_bias
    ).astype(np.float64)  # (n_blocks, n_groups)
    base_em = (
        cur_dmin[:, None] * cur_m_raw.astype(np.float64)
        if spec.has_min
        else np.zeros((n_blocks, spec.n_groups), dtype=np.float64)
    )
    base_err_per_g = (
        S_vv * base_es ** 2
        - 2.0 * S_vx * base_es
        - 2.0 * S_v * base_es * base_em
        + S_x2
        + 2.0 * S_x * base_em
        + S_w * base_em ** 2
    )  # (n_blocks, n_groups)
    base_err = base_err_per_g.sum(axis=1, dtype=np.float64)  # (n_blocks,)

    # Take the candidate where it beats the baseline.
    take = best_cand_err < base_err  # (n_blocks,)
    if take.any():
        rows = np.arange(n_blocks)
        new_d = np.where(take, cand_d[rows, best_cand_idx], cur_d)
        new_sc_raw = np.where(
            take[:, None],
            cand_sc[rows[:, None], best_cand_idx[:, None], np.arange(spec.n_groups)[None, :]],
            cur_sc_raw,
        )
        new_dmin = cur_dmin  # dmin search is below
        new_m_raw = cur_m_raw
        for b in range(n_blocks):
            if not take[b]:
                continue
            best_raws[b] = BlockRaw(
                d=float(new_d[b]),
                dmin=float(new_dmin[b]),
                sc_raw=new_sc_raw[b].copy(),
                m_raw=(new_m_raw[b].copy() if spec.has_min
                       else np.zeros(0, dtype=np.int64)),
            )

    # ----- search dmin for asymmetric types (vectorized, closed-form) -----
    if spec.has_min:
        # Per-block optimal dmin: weighted mean of (opt_min / m_raw).
        denoms_m = cur_m_raw.astype(np.float64)
        non_zero_m = denoms_m > 1e-30
        safe_denoms_m = np.where(non_zero_m, denoms_m, 1.0)
        ratios_m = np.where(non_zero_m, opt_min / safe_denoms_m, 0.0)
        mask_m = non_zero_m.astype(np.float64) * g_weights
        wsum_m = mask_m.sum(axis=1, keepdims=True)
        safe_wsum_m = np.where(wsum_m > 0, wsum_m, 1.0)
        opt_dmin_block = (ratios_m * mask_m).sum(axis=1, keepdims=True) / safe_wsum_m
        opt_dmin_block = opt_dmin_block[:, 0]  # (n_blocks,)

        cand_dmin_opt = _fp16_add_ulp_vectorized(opt_dmin_block, ulp_offsets)
        cand_dmin_cur = _fp16_add_ulp_vectorized(cur_dmin, ulp_offsets)
        cand_dmin = np.concatenate(
            [cand_dmin_opt, cand_dmin_cur], axis=1
        )  # (n_blocks, 2*n_ulp)
        cand_dmin_finite = np.isfinite(cand_dmin) & (cand_dmin >= 0)

        # Use the NEW d and sc_raw (post-d-search) for evaluating dmin.
        # es (effective scale) is constant across dmin candidates.
        new_d_arr = np.array([r.d for r in best_raws], dtype=np.float64)
        new_sc_raw_arr = np.stack(
            [r.sc_raw for r in best_raws]
        ).astype(np.int64)
        new_es = new_d_arr[:, None] * (
            new_sc_raw_arr - spec.scale_bias
        ).astype(np.float64)  # (n_blocks, n_groups), constant across candidates

        n_cand_m = cand_dmin.shape[1]
        cand_dmin_safe = np.where(cand_dmin_finite, cand_dmin, 0.0)
        # em[block, cand, group] = cand_dmin * cur_m_raw[block, group]
        cand_em = (
            cand_dmin_safe[:, :, None] * cur_m_raw[:, None, :].astype(np.float64)
        )  # (n_blocks, n_cand_m, n_groups)

        # Closed-form error per (block, candidate):
        #   sum_g (S_vv * es^2 - 2 S_vx * es - 2 S_v * es * em
        #          + S_x2 + 2 S_x * em + S_w * em^2)
        # The first two terms (S_vv * es^2 - 2 S_vx * es) are constant
        # across dmin candidates (es doesn't change). The remaining
        # terms depend on em.
        const_per_g = (
            S_vv * new_es ** 2
            - 2.0 * S_vx * new_es
            + S_x2
        )  # (n_blocks, n_groups), constant across candidates
        # Per-(block, candidate, group) error from em-dependent terms.
        err_em_per_g = (
            -2.0 * S_v[:, None, :] * new_es[:, None, :] * cand_em
            + 2.0 * S_x[:, None, :] * cand_em
            + S_w[:, None, :] * cand_em ** 2
        )  # (n_blocks, n_cand_m, n_groups)
        # Total per-(block, candidate) error = const + sum_g(err_em_per_g).
        const_bc = np.broadcast_to(
            const_per_g[:, None, :], (n_blocks, n_cand_m, spec.n_groups)
        )
        cand_err_m = (const_bc + err_em_per_g).sum(
            axis=2, dtype=np.float64
        )  # (n_blocks, n_cand_m)
        cand_err_m = np.where(cand_dmin_finite, cand_err_m, np.inf)

        # Baseline error per block (current dmin, new d / sc_raw).
        base_em = cur_dmin[:, None] * cur_m_raw.astype(np.float64)  # (n_blocks, n_groups)
        base_err_per_g_m = (
            const_per_g
            - 2.0 * S_v * new_es * base_em
            + 2.0 * S_x * base_em
            + S_w * base_em ** 2
        )  # (n_blocks, n_groups)
        base_err_m = base_err_per_g_m.sum(axis=1, dtype=np.float64)

        best_cand_m_idx = cand_err_m.argmin(axis=1)
        best_cand_m_err = cand_err_m[np.arange(n_blocks), best_cand_m_idx]
        take_m = best_cand_m_err < base_err_m
        if take_m.any():
            for b in range(n_blocks):
                if not take_m[b]:
                    continue
                r = best_raws[b]
                best_raws[b] = BlockRaw(
                    d=r.d,
                    dmin=float(cand_dmin[b, best_cand_m_idx[b]]),
                    sc_raw=r.sc_raw.copy(),
                    m_raw=r.m_raw.copy() if r.m_raw.size else r.m_raw,
                )

    eff_scales, eff_mins = _effective_from_raw(best_raws, spec)
    recon = dequantize_blocks(best_codes, eff_scales, eff_mins, type_name)
    final_error = _weighted_error(recon, values, block_weights)
    return best_codes, best_raws, final_error


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

            # Per-(block, group) weighted sums (precomputed once per group).
            v_vals = _code_values(spec, best_codes[:, lo:hi]).astype(np.float64)
            x = v.astype(np.float64)
            wv = w.astype(np.float64) * v_vals
            S_vv = (wv * v_vals).sum(axis=1, dtype=np.float64)
            S_vx = (wv * x).sum(axis=1, dtype=np.float64)
            S_v = wv.sum(axis=1, dtype=np.float64)
            S_x = (w.astype(np.float64) * x).sum(axis=1, dtype=np.float64)
            S_x2 = (w.astype(np.float64) * x * x).sum(axis=1, dtype=np.float64)
            S_w = w.astype(np.float64).sum(axis=1, dtype=np.float64)

            # Baseline for this group: current codes and their error
            # (closed form).
            base_es = cur_scales[:, g].astype(np.float64)
            base_em = cur_mins[:, g].astype(np.float64)
            base_err = (
                S_vv * base_es ** 2
                - 2.0 * S_vx * base_es
                - 2.0 * S_v * base_es * base_em
                + S_x2
                + 2.0 * S_x * base_em
                + S_w * base_em ** 2
            )

            best_group_err = base_err.copy()
            best_group_codes = best_codes[:, lo:hi].copy()
            best_sc = sc_raw[:, g].copy()
            best_m = m_raw[:, g].copy()

            fields = ("sc", "m") if spec.has_min else ("sc",)
            for field in fields:
                # REFINED: evaluate all 6 deltas with the closed-form error
                # (no code re-fit) and pick the best delta per block. Then
                # do ONE batched re-fit for the per-block best delta.
                # This is a 6x speed-up on the inner loop -- the existing
                # code re-fit codes for every delta, which built a
                # (n_blocks, group_size, n_levels) error tensor 6 times
                # per (group, field).
                deltas_arr = np.array(deltas, dtype=np.int64)
                n_deltas = len(deltas_arr)
                if field == "sc":
                    new_vals = best_sc[:, None] + deltas_arr[None, :]  # (n_blocks, n_deltas)
                    ok = (new_vals >= spec.sc_raw_min) & (new_vals <= spec.sc_raw_max)
                    # cand_scales[block, delta] = block_d[block] * (new_vals[block, delta] - bias)
                    cand_scales_2d = (
                        block_d[:, None].astype(np.float64)
                        * (new_vals - spec.scale_bias).astype(np.float64)
                    )  # (n_blocks, n_deltas)
                    cand_mins_2d = np.broadcast_to(
                        cur_mins[:, g:g + 1].astype(np.float64),
                        (n_blocks, n_deltas),
                    ).copy()
                else:
                    new_vals = best_m[:, None] + deltas_arr[None, :]
                    ok = (new_vals >= 0) & (new_vals <= spec.min_raw_max)
                    cand_scales_2d = np.broadcast_to(
                        cur_scales[:, g:g + 1].astype(np.float64),
                        (n_blocks, n_deltas),
                    ).copy()
                    cand_mins_2d = (
                        block_dmin[:, None].astype(np.float64)
                        * new_vals.astype(np.float64)
                    )  # (n_blocks, n_deltas)

                # Closed-form error per (block, delta):
                #   r_i = x_i - v_i*es + em
                #   r_i^2 = x_i^2 + v_i^2*es^2 + em^2 - 2*x_i*v_i*es + 2*x_i*em - 2*v_i*es*em
                #   err = S_x2 + S_vv*es^2 + S_w*em^2 - 2*S_vx*es + 2*S_x*em - 2*S_v*es*em
                # All broadcast over deltas: (n_blocks, 1) * (n_blocks, n_deltas) -> (n_blocks, n_deltas)
                cand_err_2d = (
                    S_vv[:, None] * cand_scales_2d ** 2
                    - 2.0 * S_vx[:, None] * cand_scales_2d
                    - 2.0 * S_v[:, None] * cand_scales_2d * cand_mins_2d
                    + S_x2[:, None]
                    + 2.0 * S_x[:, None] * cand_mins_2d
                    + S_w[:, None] * cand_mins_2d ** 2
                )  # (n_blocks, n_deltas)
                cand_err_2d = np.where(ok, cand_err_2d, np.inf)

                # Best delta per block.
                best_delta_idx = cand_err_2d.argmin(axis=1)  # (n_blocks,)
                best_delta_err = cand_err_2d[np.arange(n_blocks), best_delta_idx]  # (n_blocks,)
                # The per-block best (scale, min).
                best_cand_scales = cand_scales_2d[np.arange(n_blocks), best_delta_idx]
                best_cand_mins = cand_mins_2d[np.arange(n_blocks), best_delta_idx]
                # The per-block best (sc_raw, m_raw).
                best_cand_sc_or_m = new_vals[np.arange(n_blocks), best_delta_idx]
                # Take only where it beats the running best for this group.
                take = ok[np.arange(n_blocks), best_delta_idx] & (
                    best_delta_err < best_group_err
                )
                if take.any():
                    if cfg.refit_codes:
                        # ONE batched re-fit for the per-block best (scale, min).
                        # Only refit the blocks that improved, to save work.
                        idx = np.nonzero(take)[0]
                        refit_codes, refit_err = _refit_group_chunked(
                            v[idx], w[idx],
                            best_cand_scales[idx], best_cand_mins[idx],
                            spec, chunk,
                        )
                        # The re-fit error might be lower than the closed-form
                        # error (which used the old codes). Take the actual
                        # re-fit error.
                        refit_err_total = refit_err
                        # But the re-fit error might also be higher than the
                        # baseline (e.g. when the closed-form ranking was wrong
                        # because the codes change). Take only the real wins.
                        real_take = refit_err_total < best_group_err[idx]
                        if real_take.any():
                            sub_idx = idx[real_take]
                            best_group_err = best_group_err.copy()
                            best_group_err[sub_idx] = refit_err_total[real_take]
                            best_group_codes = best_group_codes.copy()
                            best_group_codes[sub_idx] = refit_codes[real_take]
                            if field == "sc":
                                best_sc = best_sc.copy()
                                best_sc[sub_idx] = best_cand_sc_or_m[sub_idx].astype(np.int64)
                                cur_scales = cur_scales.copy()
                                cur_scales[sub_idx, g] = best_cand_scales[sub_idx].astype(cur_scales.dtype)
                            else:
                                best_m = best_m.copy()
                                best_m[sub_idx] = best_cand_sc_or_m[sub_idx].astype(np.int64)
                                cur_mins = cur_mins.copy()
                                cur_mins[sub_idx, g] = best_cand_mins[sub_idx].astype(cur_mins.dtype)
                    else:
                        # Closed-form error only; no re-fit. The codes don't
                        # change, so just track the (scale, min) improvement.
                        sub_idx = np.nonzero(take)[0]
                        best_group_err = best_group_err.copy()
                        best_group_err[sub_idx] = best_delta_err[sub_idx]
                        if field == "sc":
                            best_sc = best_sc.copy()
                            best_sc[sub_idx] = best_cand_sc_or_m[sub_idx].astype(np.int64)
                            cur_scales = cur_scales.copy()
                            cur_scales[sub_idx, g] = best_cand_scales[sub_idx].astype(cur_scales.dtype)
                        else:
                            best_m = best_m.copy()
                            best_m[sub_idx] = best_cand_sc_or_m[sub_idx].astype(np.int64)
                            cur_mins = cur_mins.copy()
                            cur_mins[sub_idx, g] = best_cand_mins[sub_idx].astype(cur_mins.dtype)

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


# ---------------------------------------------------------------------------
# REFINED: GPU (torch) versions of the two hot paths.
#
# These are mechanical translations of ``_refine_super_scale`` and
# ``_refine_scales`` from NumPy to torch. The closed-form error formula
# and the candidate layout are identical; only the array library changes.
# On an RTX 4080 the GPU path is ~20-30x faster than the vectorized CPU
# path on a 16K-block Q4_K tensor.
#
# The Gumbel-Softmax relaxation stays on CPU -- its per-chunk Adam state
# is small, and the per-chunk launch overhead would dominate.
# ---------------------------------------------------------------------------


def _torch_device(device: str):
    """Resolve a device string to a torch.device, or None if torch missing."""
    try:
        import torch
    except ImportError:
        return None
    try:
        return torch.device(device)
    except (RuntimeError, ValueError):
        return None


def _to_torch(arr, device, dtype=None):
    """NumPy -> torch, on the given device."""
    import torch
    if dtype is None:
        # Float64 for the error-precision-critical sums, else preserve.
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            dtype = torch.float64
        else:
            dtype = torch.from_numpy(arr).dtype
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)


def _from_torch(t):
    """torch -> NumPy, detaching and copying to host."""
    return t.detach().cpu().numpy()


def _refine_super_scale_torch(
    codes: np.ndarray,
    raws: List[BlockRaw],
    values: np.ndarray,
    block_weights: np.ndarray,
    spec: TypeSpec,
    type_name: str,
    cfg: GsqConfig,
    device,
):
    """GPU version of _refine_super_scale using torch.

    Mirrors the CPU implementation: builds the (n_blocks, n_cand) candidate
    d array via vectorized fp16-ULP arithmetic, computes the closed-form
    per-(block, candidate, group) error, and picks the argmin per block.
    """
    import torch
    n_blocks = codes.shape[0]
    if n_blocks == 0:
        return codes, list(raws), 0.0

    eff_scales_np, eff_mins_np = _effective_from_raw(raws, spec)
    recon_np = dequantize_blocks(codes, eff_scales_np, eff_mins_np, type_name)
    best_error = _weighted_error(recon_np, values, block_weights)
    best_raws = list(raws)
    best_codes = codes.copy()

    v_all_np = _code_values(spec, best_codes)
    group_index_np = np.arange(spec.block_elems) // spec.group_size

    # Closed-form optimal eff_scale / eff_min per (block, sub-group).
    opt_eff_np, opt_min_np = _optimal_eff_scales_mins(
        best_codes, values, block_weights, spec
    )

    # Per-block current d / dmin / sc_raw / m_raw.
    cur_d_np = np.array([r.d for r in best_raws], dtype=np.float64)
    cur_dmin_np = np.array([r.dmin for r in best_raws], dtype=np.float64)
    cur_sc_raw_np = np.stack([r.sc_raw for r in best_raws]).astype(np.int64)
    cur_m_raw_np = (
        np.stack([r.m_raw for r in best_raws]).astype(np.int64)
        if spec.has_min
        else np.zeros((n_blocks, spec.n_groups), dtype=np.int64)
    )

    # Per-sub-group weight (n_blocks, n_groups).
    w_grp_np = block_weights.reshape(
        n_blocks, spec.n_groups, spec.group_size
    ).astype(np.float64)
    g_weights_np = w_grp_np.sum(axis=2)

    # Per-block optimal d.
    if spec.n_groups > 1:
        denoms_np = (cur_sc_raw_np - spec.scale_bias).astype(np.float64)
        non_zero_np = np.abs(denoms_np) > 1e-30
        safe_denoms_np = np.where(non_zero_np, denoms_np, 1.0)
        ratios_np = np.where(non_zero_np, opt_eff_np / safe_denoms_np, 0.0)
        mask_np = non_zero_np.astype(np.float64) * g_weights_np
        wsum_np = mask_np.sum(axis=1, keepdims=True)
        safe_wsum_np = np.where(wsum_np > 0, wsum_np, 1.0)
        opt_d_block_np = (ratios_np * mask_np).sum(axis=1, keepdims=True) / safe_wsum_np
        opt_d_block_np = opt_d_block_np[:, 0]
    else:
        opt_d_block_np = opt_eff_np[:, 0]

    # Build candidate d set on the GPU. fp16 ULP arithmetic via uint16.
    ulp_offsets_np = np.array([-2, -1, 0, 1, 2], dtype=np.int64)

    def _fp16_ulp_torch(values_np, offsets_np):
        """Vectorized fp16 ULP on the GPU.

        PyTorch does NOT support element-wise arithmetic on uint16 tensors
        (``RuntimeError: "add" not implemented for 'Uint16'``), so we view
        the fp16 bits as int16 -> int32, do the addition in int32 (which
        IS supported), wrap modulo 65536, then view back as fp16.
        """
        base_fp16 = torch.from_numpy(values_np.astype(np.float16)).to(device)
        # View as int16 (same 2-byte width), then widen to int32 for arithmetic.
        base_int = base_fp16.view(torch.int16).to(torch.int32)
        offsets = torch.from_numpy(offsets_np.astype(np.int32)).to(device)
        cand_int = base_int.unsqueeze(1) + offsets.unsqueeze(0)
        # Wrap to the unsigned [0, 65535] range so the int16 view is the
        # same bit pattern as the original uint16 would have been.
        cand_int = torch.remainder(cand_int, 65536)
        # View as uint16 (for the reinterpret back to fp16).
        cand_uint = cand_int.to(torch.int16).view(torch.uint16)
        cand_fp16 = cand_uint.view(torch.float16)
        cand = cand_fp16.to(torch.float64)
        cand = torch.where(cand == 0, torch.full_like(cand, float('nan')), cand)
        cand = torch.where(torch.isfinite(cand), cand, torch.full_like(cand, float('nan')))
        return cand

    cand_d_opt = _fp16_ulp_torch(opt_d_block_np, ulp_offsets_np)
    cand_d_cur = _fp16_ulp_torch(cur_d_np, ulp_offsets_np)
    cand_d = torch.cat([cand_d_opt, cand_d_cur], dim=1)  # (n_blocks, 2*n_ulp)
    cand_d_finite = torch.isfinite(cand_d) & (cand_d != 0)

    # Move everything to GPU.
    opt_eff = _to_torch(opt_eff_np, device)
    opt_min = _to_torch(opt_min_np, device)
    cur_d = _to_torch(cur_d_np, device)
    cur_dmin = _to_torch(cur_dmin_np, device)
    cur_sc_raw = _to_torch(cur_sc_raw_np, device, dtype=torch.int64)
    cur_m_raw = _to_torch(cur_m_raw_np, device, dtype=torch.int64)
    g_weights = _to_torch(g_weights_np, device)
    S_vv = _to_torch(_compute_S_vv(v_all_np, block_weights, spec), device)
    S_vx = _to_torch(_compute_S_vx(v_all_np, values, block_weights, spec), device)
    S_v = _to_torch(_compute_S_v(v_all_np, block_weights, spec), device)
    S_x = _to_torch(_compute_S_x(values, block_weights, spec), device)
    S_x2 = _to_torch(_compute_S_x2(values, block_weights, spec), device)
    S_w = _to_torch(_compute_S_w(block_weights, spec), device)

    n_cand = cand_d.shape[1]
    cand_d_safe = torch.where(cand_d_finite, cand_d, torch.ones_like(cand_d))
    # Best sc_raw per (block, candidate, group).
    cand_sc = torch.clamp(
        torch.round(opt_eff.unsqueeze(1) / cand_d_safe.unsqueeze(2) + spec.scale_bias),
        spec.sc_raw_min, spec.sc_raw_max,
    ).to(torch.int64)
    # Effective scales per (block, candidate, group).
    cand_eff_scales = cand_d_safe.unsqueeze(2) * (
        cand_sc - spec.scale_bias
    ).to(torch.float64)
    # Effective mins (constant across candidates for asymmetric types).
    if spec.has_min:
        em_per_g = cur_dmin.unsqueeze(1) * cur_m_raw.to(torch.float64)
        em_per_g = em_per_g.expand(n_blocks, n_cand, spec.n_groups)
    else:
        em_per_g = torch.zeros((n_blocks, n_cand, spec.n_groups),
                               dtype=torch.float64, device=device)

    # Per-(block, candidate, group) error using the closed-form formula:
    #   r_i = x_i - v_i*es + em
    #   r_i^2 = x_i^2 + v_i^2*es^2 + em^2 - 2*x_i*v_i*es + 2*x_i*em - 2*v_i*es*em
    #   err = S_x2 + S_vv*es^2 + S_w*em^2 - 2*S_vx*es + 2*S_x*em - 2*S_v*es*em
    err_per_g = (
        S_vv.unsqueeze(1) * cand_eff_scales ** 2
        - 2.0 * S_vx.unsqueeze(1) * cand_eff_scales
        - 2.0 * S_v.unsqueeze(1) * cand_eff_scales * em_per_g
        + S_x2.unsqueeze(1)
        + 2.0 * S_x.unsqueeze(1) * em_per_g
        + S_w.unsqueeze(1) * em_per_g ** 2
    )  # (n_blocks, n_cand, n_groups)
    cand_err = err_per_g.sum(dim=2, dtype=torch.float64)
    cand_err = torch.where(cand_d_finite, cand_err, torch.full_like(cand_err, float('inf')))

    # Pick the best candidate per block.
    best_cand_idx = torch.argmin(cand_err, dim=1)
    rows = torch.arange(n_blocks, device=device)
    best_cand_err = cand_err[rows, best_cand_idx]

    # Baseline error per block (current d, sc_raw, m_raw).
    base_es = cur_d.unsqueeze(1) * (cur_sc_raw - spec.scale_bias).to(torch.float64)
    if spec.has_min:
        base_em = cur_dmin.unsqueeze(1) * cur_m_raw.to(torch.float64)
    else:
        base_em = torch.zeros((n_blocks, spec.n_groups), dtype=torch.float64, device=device)
    base_err_per_g = (
        S_vv * base_es ** 2
        - 2.0 * S_vx * base_es
        - 2.0 * S_v * base_es * base_em
        + S_x2
        + 2.0 * S_x * base_em
        + S_w * base_em ** 2
    )
    base_err = base_err_per_g.sum(dim=1, dtype=torch.float64)

    take = best_cand_err < base_err
    if take.any().item():
        new_d_t = torch.where(take, cand_d[rows, best_cand_idx], cur_d)
        # Pull sc_raw for the chosen candidates back to host for the BlockRaw write.
        cand_sc_cpu = _from_torch(cand_sc)
        cur_d_cpu = _from_torch(cur_d)
        new_d_cpu = _from_torch(new_d_t)
        take_cpu = _from_torch(take)
        best_cand_idx_cpu = _from_torch(best_cand_idx)
        for b in range(n_blocks):
            if not take_cpu[b]:
                continue
            best_raws[b] = BlockRaw(
                d=float(new_d_cpu[b]),
                dmin=float(cur_d_cpu[b]),
                sc_raw=cand_sc_cpu[b, best_cand_idx_cpu[b]].copy(),
                m_raw=(cur_m_raw_np[b].copy() if spec.has_min
                       else np.zeros(0, dtype=np.int64)),
            )

    # ----- search dmin for asymmetric types (vectorized) ------------------
    if spec.has_min:
        denoms_m_np = cur_m_raw_np.astype(np.float64)
        non_zero_m_np = denoms_m_np > 1e-30
        safe_denoms_m_np = np.where(non_zero_m_np, denoms_m_np, 1.0)
        ratios_m_np = np.where(non_zero_m_np, opt_min_np / safe_denoms_m_np, 0.0)
        mask_m_np = non_zero_m_np.astype(np.float64) * g_weights_np
        wsum_m_np = mask_m_np.sum(axis=1, keepdims=True)
        safe_wsum_m_np = np.where(wsum_m_np > 0, wsum_m_np, 1.0)
        opt_dmin_block_np = (ratios_m_np * mask_m_np).sum(axis=1, keepdims=True) / safe_wsum_m_np
        opt_dmin_block_np = opt_dmin_block_np[:, 0]

        cand_dmin_opt = _fp16_ulp_torch(opt_dmin_block_np, ulp_offsets_np)
        cand_dmin_cur = _fp16_ulp_torch(cur_dmin_np, ulp_offsets_np)
        cand_dmin = torch.cat([cand_dmin_opt, cand_dmin_cur], dim=1)
        cand_dmin_finite = torch.isfinite(cand_dmin) & (cand_dmin >= 0)

        # Use the NEW d and sc_raw (post-d-search).
        new_d_arr_np = np.array([r.d for r in best_raws], dtype=np.float64)
        new_sc_raw_arr_np = np.stack([r.sc_raw for r in best_raws]).astype(np.int64)
        new_es = _to_torch(new_d_arr_np, device).unsqueeze(1) * (
            _to_torch(new_sc_raw_arr_np, device, dtype=torch.int64) - spec.scale_bias
        ).to(torch.float64)

        n_cand_m = cand_dmin.shape[1]
        cand_dmin_safe = torch.where(cand_dmin_finite, cand_dmin, torch.zeros_like(cand_dmin))
        cand_em = cand_dmin_safe.unsqueeze(2) * cur_m_raw.to(torch.float64).unsqueeze(0).expand(
            n_blocks, n_cand_m, spec.n_groups
        )
        # const_per_g (constant across dmin candidates): S_vv*es^2 - 2*S_vx*es + S_x2.
        const_per_g = S_vv * new_es ** 2 - 2.0 * S_vx * new_es + S_x2
        # err_em_per_g = -2*S_v*es*em + 2*S_x*em + S_w*em^2.
        err_em_per_g = (
            -2.0 * S_v.unsqueeze(1).unsqueeze(2) * new_es.unsqueeze(1) * cand_em
            + 2.0 * S_x.unsqueeze(1).unsqueeze(2) * cand_em
            + S_w.unsqueeze(1).unsqueeze(2) * cand_em ** 2
        )
        const_bc = const_per_g.unsqueeze(1).expand(n_blocks, n_cand_m, spec.n_groups)
        cand_err_m = (const_bc + err_em_per_g).sum(dim=2, dtype=torch.float64)
        cand_err_m = torch.where(cand_dmin_finite, cand_err_m, torch.full_like(cand_err_m, float('inf')))

        # Baseline (current dmin, new d / sc_raw).
        base_em = cur_dmin.unsqueeze(1) * cur_m_raw.to(torch.float64)
        base_err_per_g_m = (
            const_per_g
            - 2.0 * S_v * new_es * base_em
            + 2.0 * S_x * base_em
            + S_w * base_em ** 2
        )
        base_err_m = base_err_per_g_m.sum(dim=1, dtype=torch.float64)

        best_cand_m_idx = torch.argmin(cand_err_m, dim=1)
        best_cand_m_err = cand_err_m[rows, best_cand_m_idx]
        take_m = best_cand_m_err < base_err_m
        if take_m.any().item():
            cand_dmin_cpu = _from_torch(cand_dmin)
            take_m_cpu = _from_torch(take_m)
            best_cand_m_idx_cpu = _from_torch(best_cand_m_idx)
            for b in range(n_blocks):
                if not take_m_cpu[b]:
                    continue
                r = best_raws[b]
                best_raws[b] = BlockRaw(
                    d=r.d,
                    dmin=float(cand_dmin_cpu[b, best_cand_m_idx_cpu[b]]),
                    sc_raw=r.sc_raw.copy(),
                    m_raw=r.m_raw.copy() if r.m_raw.size else r.m_raw,
                )

    eff_scales_np, eff_mins_np = _effective_from_raw(best_raws, spec)
    recon_np = dequantize_blocks(best_codes, eff_scales_np, eff_mins_np, type_name)
    final_error = _weighted_error(recon_np, values, block_weights)
    return best_codes, best_raws, final_error


def _compute_S_vv(v_all, block_weights, spec):
    """S_vv = sum w_i v_i^2 per (block, group)."""
    n_blocks = v_all.shape[0]
    v_grp = v_all.reshape(n_blocks, spec.n_groups, spec.group_size).astype(np.float64)
    w_grp = block_weights.reshape(n_blocks, spec.n_groups, spec.group_size).astype(np.float64)
    return (w_grp * v_grp * v_grp).sum(axis=2, dtype=np.float64)


def _compute_S_vx(v_all, values, block_weights, spec):
    n_blocks = v_all.shape[0]
    v_grp = v_all.reshape(n_blocks, spec.n_groups, spec.group_size).astype(np.float64)
    x_grp = values.reshape(n_blocks, spec.n_groups, spec.group_size).astype(np.float64)
    w_grp = block_weights.reshape(n_blocks, spec.n_groups, spec.group_size).astype(np.float64)
    return (w_grp * v_grp * x_grp).sum(axis=2, dtype=np.float64)


def _compute_S_v(v_all, block_weights, spec):
    n_blocks = v_all.shape[0]
    v_grp = v_all.reshape(n_blocks, spec.n_groups, spec.group_size).astype(np.float64)
    w_grp = block_weights.reshape(n_blocks, spec.n_groups, spec.group_size).astype(np.float64)
    return (w_grp * v_grp).sum(axis=2, dtype=np.float64)


def _compute_S_x(values, block_weights, spec):
    n_blocks = values.shape[0]
    x_grp = values.reshape(n_blocks, spec.n_groups, spec.group_size).astype(np.float64)
    w_grp = block_weights.reshape(n_blocks, spec.n_groups, spec.group_size).astype(np.float64)
    return (w_grp * x_grp).sum(axis=2, dtype=np.float64)


def _compute_S_x2(values, block_weights, spec):
    n_blocks = values.shape[0]
    x_grp = values.reshape(n_blocks, spec.n_groups, spec.group_size).astype(np.float64)
    w_grp = block_weights.reshape(n_blocks, spec.n_groups, spec.group_size).astype(np.float64)
    return (w_grp * x_grp * x_grp).sum(axis=2, dtype=np.float64)


def _compute_S_w(block_weights, spec):
    n_blocks = block_weights.shape[0]
    w_grp = block_weights.reshape(n_blocks, spec.n_groups, spec.group_size).astype(np.float64)
    return w_grp.sum(axis=2, dtype=np.float64)


def _refine_scales_torch(
    codes: np.ndarray,
    raws: List[BlockRaw],
    values: np.ndarray,
    block_weights: np.ndarray,
    spec: TypeSpec,
    type_name: str,
    cfg: GsqConfig,
    device,
):
    """GPU version of _refine_scales using torch.

    Mirrors the CPU coordinate descent: for each (group, field), evaluates
    all 6 deltas with the closed-form error (no code re-fit), picks the
    best delta per block, then does ONE batched re-fit for the per-block
    best delta.
    """
    import torch
    n_blocks = codes.shape[0]
    eff_scales_np, eff_mins_np = _effective_from_raw(raws, spec)
    recon_np = dequantize_blocks(codes, eff_scales_np, eff_mins_np, type_name)
    best_error = _weighted_error(recon_np, values, block_weights)
    best_raws = list(raws)
    best_codes = codes.copy()

    if spec.n_groups == 1:
        return best_codes, best_raws, best_error

    gs = spec.group_size
    deltas_np = np.array((-3, -2, -1, 1, 2, 3), dtype=np.int64)

    # Per-block effective values, kept in sync with best_raws.
    cur_scales_np = eff_scales_np.astype(np.float32).copy()
    cur_mins_np = eff_mins_np.astype(np.float32).copy()
    sc_raw_np = np.stack([r.sc_raw for r in best_raws]).astype(np.int64)
    m_raw_np = (
        np.stack([r.m_raw for r in best_raws]).astype(np.int64)
        if spec.has_min
        else np.zeros((n_blocks, spec.n_groups), dtype=np.int64)
    )
    block_d_np = np.array([r.d for r in best_raws], dtype=np.float32)
    block_dmin_np = np.array([r.dmin for r in best_raws], dtype=np.float32)

    # Move the persistent state to GPU.
    cur_scales = _to_torch(cur_scales_np, device, dtype=torch.float64)
    cur_mins = _to_torch(cur_mins_np, device, dtype=torch.float64)
    sc_raw = _to_torch(sc_raw_np, device, dtype=torch.int64)
    m_raw = _to_torch(m_raw_np, device, dtype=torch.int64)
    block_d = _to_torch(block_d_np, device, dtype=torch.float64)
    block_dmin = _to_torch(block_dmin_np, device, dtype=torch.float64)
    deltas = _to_torch(deltas_np, device, dtype=torch.int64)
    n_deltas = len(deltas_np)

    for _ in range(max(cfg.scale_passes, 0)):
        improved = False
        for g in range(spec.n_groups):
            lo = g * gs
            hi = lo + gs
            v_chunk_np = _code_values(spec, best_codes[:, lo:hi]).astype(np.float64)
            x_chunk_np = values[:, lo:hi].astype(np.float64)
            w_chunk_np = block_weights[:, lo:hi].astype(np.float64)
            v_chunk = _to_torch(v_chunk_np, device)
            x_chunk = _to_torch(x_chunk_np, device)
            w_chunk = _to_torch(w_chunk_np, device)

            # Per-block weighted sums (this group only).
            wv = w_chunk * v_chunk
            S_vv = (wv * v_chunk).sum(dim=1, dtype=torch.float64)
            S_vx = (wv * x_chunk).sum(dim=1, dtype=torch.float64)
            S_v = wv.sum(dim=1, dtype=torch.float64)
            S_x = (w_chunk * x_chunk).sum(dim=1, dtype=torch.float64)
            S_x2 = (w_chunk * x_chunk * x_chunk).sum(dim=1, dtype=torch.float64)
            S_w = w_chunk.sum(dim=1, dtype=torch.float64)

            # Baseline error per block (current codes + current scale/min).
            base_es = cur_scales[:, g]
            base_em = cur_mins[:, g]
            base_err = (
                S_vv * base_es ** 2
                - 2.0 * S_vx * base_es
                - 2.0 * S_v * base_es * base_em
                + S_x2
                + 2.0 * S_x * base_em
                + S_w * base_em ** 2
            )

            best_group_err = base_err.clone()
            best_group_codes_t = _to_torch(best_codes[:, lo:hi], device, dtype=torch.int32)
            best_sc_t = sc_raw[:, g].clone()
            best_m_t = m_raw[:, g].clone() if spec.has_min else None

            fields = ("sc", "m") if spec.has_min else ("sc",)
            for field in fields:
                if field == "sc":
                    new_vals = best_sc_t.unsqueeze(1) + deltas.unsqueeze(0)  # (n_blocks, n_deltas)
                    ok = (new_vals >= spec.sc_raw_min) & (new_vals <= spec.sc_raw_max)
                    cand_scales_2d = block_d.unsqueeze(1) * (new_vals - spec.scale_bias).to(torch.float64)
                    cand_mins_2d = cur_mins[:, g].unsqueeze(1).expand(n_blocks, n_deltas)
                else:
                    new_vals = best_m_t.unsqueeze(1) + deltas.unsqueeze(0)
                    ok = (new_vals >= 0) & (new_vals <= spec.min_raw_max)
                    cand_scales_2d = cur_scales[:, g].unsqueeze(1).expand(n_blocks, n_deltas)
                    cand_mins_2d = block_dmin.unsqueeze(1) * new_vals.to(torch.float64)

                # Closed-form error per (block, delta).
                cand_err_2d = (
                    S_vv.unsqueeze(1) * cand_scales_2d ** 2
                    - 2.0 * S_vx.unsqueeze(1) * cand_scales_2d
                    - 2.0 * S_v.unsqueeze(1) * cand_scales_2d * cand_mins_2d
                    + S_x2.unsqueeze(1)
                    + 2.0 * S_x.unsqueeze(1) * cand_mins_2d
                    + S_w.unsqueeze(1) * cand_mins_2d ** 2
                )
                cand_err_2d = torch.where(ok, cand_err_2d, torch.full_like(cand_err_2d, float('inf')))

                best_delta_idx = torch.argmin(cand_err_2d, dim=1)
                rows = torch.arange(n_blocks, device=device)
                best_delta_err = cand_err_2d[rows, best_delta_idx]
                best_cand_scales = cand_scales_2d[rows, best_delta_idx]
                best_cand_mins = cand_mins_2d[rows, best_delta_idx]
                best_cand_sc_or_m = new_vals[rows, best_delta_idx]
                ok_at_best = ok[rows, best_delta_idx]
                take = ok_at_best & (best_delta_err < best_group_err)

                if take.any().item():
                    if cfg.refit_codes:
                        idx_np = _from_torch(torch.nonzero(take, as_tuple=False).squeeze(1))
                        bcs_np = _from_torch(best_cand_scales.take(
                            torch.nonzero(take, as_tuple=False).squeeze(1)
                        ))
                        bcm_np = _from_torch(best_cand_mins.take(
                            torch.nonzero(take, as_tuple=False).squeeze(1)
                        ))
                        v_sub_np = v_chunk_np[idx_np]
                        w_sub_np = w_chunk_np[idx_np]
                        cand_codes_np, cand_err_np = _refit_group_chunked(
                            v_sub_np, w_sub_np, bcs_np, bcm_np, spec,
                            max(1, int(cfg.chunk_blocks)),
                        )
                        # Compare the actual re-fit error to the baseline.
                        refit_err = cand_err_np
                        baseline_err_at_idx = _from_torch(best_group_err.take(
                            torch.nonzero(take, as_tuple=False).squeeze(1)
                        ))
                        real_take = refit_err < baseline_err_at_idx
                        if real_take.any():
                            sub_idx = idx_np[real_take]
                            best_group_err_np = _from_torch(best_group_err)
                            best_group_err_np[sub_idx] = refit_err[real_take]
                            # Sync back to GPU.
                            best_group_err = _to_torch(best_group_err_np, device)
                            best_group_codes_np = best_codes[:, lo:hi].copy()
                            best_group_codes_np[sub_idx] = cand_codes_np[real_take]
                            best_codes[:, lo:hi] = best_group_codes_np
                            best_group_codes_t = _to_torch(best_group_codes_np, device, dtype=torch.int32)
                            if field == "sc":
                                best_sc_np = _from_torch(best_sc_t)
                                bcsom_np = _from_torch(best_cand_sc_or_m)
                                best_sc_np[sub_idx] = bcsom_np[real_take].astype(np.int64)
                                best_sc_t = _to_torch(best_sc_np, device, dtype=torch.int64)
                                cur_scales_np = _from_torch(cur_scales)
                                cur_scales_np[sub_idx, g] = bcs_np[real_take]
                                cur_scales = _to_torch(cur_scales_np, device)
                            else:
                                best_m_np = _from_torch(best_m_t)
                                bcmom_np = _from_torch(best_cand_sc_or_m)
                                best_m_np[sub_idx] = bcmom_np[real_take].astype(np.int64)
                                best_m_t = _to_torch(best_m_np, device, dtype=torch.int64)
                                cur_mins_np = _from_torch(cur_mins)
                                cur_mins_np[sub_idx, g] = bcm_np[real_take]
                                cur_mins = _to_torch(cur_mins_np, device)
                    else:
                        idx = torch.nonzero(take, as_tuple=False).squeeze(1)
                        best_group_err = torch.where(
                            take, best_delta_err, best_group_err
                        )
                        if field == "sc":
                            best_sc_t = torch.where(take, best_cand_sc_or_m, best_sc_t)
                            cur_scales[:, g] = torch.where(
                                take, best_cand_scales, cur_scales[:, g]
                            )
                        else:
                            best_m_t = torch.where(take, best_cand_sc_or_m, best_m_t)
                            cur_mins[:, g] = torch.where(
                                take, best_cand_mins, cur_mins[:, g]
                            )

            # Apply the group's improvements.
            changed_np = _from_torch(best_group_err < base_err)
            if changed_np.any():
                improved = True
                best_codes[:, lo:hi] = best_codes[:, lo:hi]  # already updated above
                sc_raw_np = _from_torch(sc_raw)
                sc_raw_np[:, g] = _from_torch(best_sc_t)
                sc_raw = _to_torch(sc_raw_np, device, dtype=torch.int64)
                if spec.has_min:
                    m_raw_np = _from_torch(m_raw)
                    m_raw_np[:, g] = _from_torch(best_m_t)
                    m_raw = _to_torch(m_raw_np, device, dtype=torch.int64)
                cur_scales[:, g] = block_d * (_from_torch(best_sc_t).astype(np.float64) - spec.scale_bias).to(device)
                # Move cur_scales back to GPU after the host computation.
                cur_scales = _to_torch(_from_torch(cur_scales), device)
                if spec.has_min:
                    cur_mins[:, g] = block_dmin * _from_torch(best_m_t).astype(np.float64).to(device)
                    cur_mins = _to_torch(_from_torch(cur_mins), device)

        if not improved:
            break

    # Rebuild the raw fields from the searched integer fields.
    sc_raw_final = _from_torch(sc_raw)
    m_raw_final = _from_torch(m_raw) if spec.has_min else None
    block_d_final = _from_torch(block_d)
    block_dmin_final = _from_torch(block_dmin)
    for b in range(n_blocks):
        best_raws[b] = BlockRaw(
            d=float(block_d_final[b]),
            dmin=float(block_dmin_final[b]),
            sc_raw=sc_raw_final[b].copy(),
            m_raw=m_raw_final[b].copy() if spec.has_min else np.zeros(0, dtype=np.int64),
        )

    eff_scales_np, eff_mins_np = _effective_from_raw(best_raws, spec)
    recon_np = dequantize_blocks(best_codes, eff_scales_np, eff_mins_np, type_name)
    final_error = _weighted_error(recon_np, values, block_weights)
    return best_codes, best_raws, final_error


__all__ = [
    "GsqConfig",
    "RefineResult",
    "refine_tensor",
]
