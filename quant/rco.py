"""
@file toffee/quant/rco.py
   RCO: Riemannian Constrained Optimization for exact-budget quantization.

Ported from IST-DASLab/RCO (Apache-2.0), adapted to TOFFEE's GGUF pipeline.

## PARKED -- experimental, not on by default, not proven to help

This is a faithful port of upstream RCO's manifold, retraction and
budget-constrained DP. It is **not** enabled in the standard quantization
flow: APEX/ONYX is the policy layer and RCO only refines its assignment.

Measured on a same-family model (2026-10-21): the RCO map came out
**+2.53% worse** than the APEX map it started from, and every one of its
three changes was ~15x worse. Four root causes were found and fixed:

1. The DP's value had no measured-distortion term -- it maximized ``log p``
   with ``p`` seeded from a binary edge/bulk flag, so under a byte budget
   it downgraded arbitrarily. Upstream RCO seeds ``alpha_init`` from the
   measured per-bitwidth distortion.
2. The retraction computed its achievable range over *all* options
   (including F32 at 32 bpw) instead of only the *allowed* ones, so an
   unreachable target looked reachable and the bisection diverged to
   logits of ~1e18.
3. The budget was in bits/weight while the real cost is bytes with per-block
   overheads -- the nominal estimate said 24.453 MB where the file was
   24.554 MB, so the budget was *below* the map it started from.
4. A role's ladder did not include the type APEX already picked, so the
   search was *forced* to move a well-chosen Q8_0 tensor to Q6_K (10x
   more error).

After the fixes the *mechanism* is sound -- the search now makes sensible
trades (it keeps the Q8_0 tensors and downgrades Q6_K -> Q5_K where it
must) -- but a quality win is **still not demonstrated** (last
measurement: +3.3% error vs APEX). Until it is, this stays parked.

The code is worth keeping: the exact-byte-cost path (llama.cpp's own
``ggml_row_size``) is correct and reusable, and the codecs/bridge are the
correctness oracle for everything else.

## Attribution

The problem
-----------
A quantizer must pick one quant type per tensor, and the *total* model size
must land on a target budget. The objective (how much accuracy each choice
costs) does not factor across tensors, so greedy or per-tensor heuristics
miss the global optimum, and penalty/Lagrangian methods only satisfy the
budget approximately.

RCO's answer
------------
Relax the discrete choice into per-tensor softmax probabilities over the
candidate quant types, then optimize on the *budget manifold*

    M = { alpha : C(alpha) = B },
    C(alpha) = sum_i w_i * sum_k p_ik * c_k,

where ``w_i`` is the parameter count of tensor *i* and ``c_k`` the size cost
of quant type *k*. Three operations keep the iterate on the manifold:

  * tangent projection  -- remove the gradient component along the budget
    normal, so the Adam step cannot drift off the constraint;
  * retraction          -- shift all logits along the cost vector and
    binary-search the shift so C(alpha) = B exactly again;
  * vector transport    -- re-project Adam's first moment onto the rotated
    tangent plane so momentum stays budget-neutral.

Relationship to the APEX tier
-----------------------------
APEX remains the *policy*: it decides which quant types are legal for each
tensor role and what the target budget is. RCO refines APEX's assignment by
searching over APEX's own option set, so the result is still an APEX-shaped
tensor-type file -- just one that hits the size budget exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budget manifold primitives (ported from RCO's src/manifold.py)
# ---------------------------------------------------------------------------

def budget_normal(
    alpha: np.ndarray,
    costs: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normal vector to the budget surface at ``alpha``.

    ``dC/dalpha_ik = w_i * p_ik * (c_k - E_i[c])`` with ``E_i[c] = sum_k p_ik c_k``.

    Args:
        alpha: (n_groups, n_options) logits.
        costs: (n_options,) cost per option (e.g. bytes per parameter).
        weights: optional (n_groups,) per-group weight.

    Returns:
        (n_groups, n_options) normal vector.
    """
    probs = _softmax(alpha)
    expected = (probs * costs).sum(axis=-1, keepdims=True)
    normal = probs * (costs - expected)
    if weights is not None:
        normal = normal * weights[:, None]
    return normal


class _Logits:
    """A logits array with a ``.grad`` slot, mirroring torch's Parameter.

    The RCO optimizer steps are written against a tensor-like object, so the
    budget-manifold primitives only need ``alpha`` and ``alpha.grad``.
    """

    def __init__(self, values: np.ndarray) -> None:
        self.data = np.asarray(values, dtype=np.float64)
        self.grad = np.zeros_like(self.data)

    def __array__(self, dtype=None) -> np.ndarray:
        return self.data if dtype is None else self.data.astype(dtype)

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value) -> None:
        self.data[key] = value

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


def project_gradient(
    alpha: np.ndarray,
    costs: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Project the gradient onto the budget-preserving subspace, in place.

    Removes the component of ``alpha.grad`` along the budget normal.

    Args:
        alpha: (n_groups, n_options) logits. May be a plain array (in which
            case the gradient is taken from ``alpha.grad`` if present) or an
            object exposing a writable ``.grad``.
        costs: (n_options,) cost per option.
        weights: optional (n_groups,) per-group weight.

    Returns:
        (projection coefficient, raw gradient norm, projected gradient norm).
    """
    grad = getattr(alpha, "grad", None)
    if grad is None:
        raise ValueError(
            "project_gradient needs a gradient; pass an object with a "
            "writable .grad attribute"
        )
    normal = budget_normal(np.asarray(alpha), costs, weights)
    n_flat = normal.reshape(-1)
    g_flat = grad.reshape(-1)
    raw_norm = float(np.linalg.norm(g_flat))
    denom = float(n_flat @ n_flat) + 1e-12
    coeff = float((g_flat @ n_flat) / denom)
    grad -= coeff * normal
    proj_norm = float(np.linalg.norm(grad.reshape(-1)))
    return coeff, raw_norm, proj_norm


def retraction(
    alpha: np.ndarray,
    costs: np.ndarray,
    target: float,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 60,
    tol: float = 1e-4,
    allowed: Optional[np.ndarray] = None,
) -> float:
    """Shift ``alpha`` along the cost vector so the budget hits ``target``.

    The map ``t -> C(alpha + t*c)`` is monotone, so a bisection converges to
    the exact budget in O(log) steps. Modifies ``alpha`` in place.

    The target is first clamped to the *achievable* range. The cost is a
    convex combination of the option costs, so it can only lie between the
    cheapest and the dearest option:

        C_min = sum_i w_i * min_k c_k
        C_max = sum_i w_i * max_k c_k

    A target outside that range is unreachable, and bisecting for it
    diverges -- the bracket doubles without bound and the logits blow up
    to ~1e18. That is not hypothetical: the free groups' share of a tier's
    own size budget can exceed their maximum precision, because the
    *fixed* groups (norms, routers) are stored more precisely.

    Args:
        alpha: (n_groups, n_options) logits, modified in place.
        costs: (n_options,) cost of each option.
        target: the budget to hit.
        weights: optional (n_groups,) per-group weight.
        max_iter: bisection iterations.
        tol: convergence tolerance.
        allowed: optional (n_groups, n_options) mask of the options a
            group may select. Masked-out options cannot be part of the
            achievable range.

    Returns:
        The achieved budget.
    """
    def cost_at(shift: float) -> float:
        probs = _softmax(alpha + shift * costs)
        row = (probs * costs).sum(axis=-1)
        return float((row * weights).sum() if weights is not None else row.sum())

    # Clamp the target to the achievable range. The achievable range is
    # over the options a group may actually *select* -- an option that is
    # masked out contributes a probability of zero, so it cannot be part
    # of the range. Using the unmasked range makes an unreachable target
    # look reachable, and the bisection then diverges (the logits blow up
    # to ~1e7-1e18).
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
    else:
        w = np.ones(len(alpha), dtype=np.float64)
    if allowed is not None:
        masked = np.where(allowed, costs[None, :], np.inf)
        c_min = float((w * np.nanmin(np.where(np.isfinite(masked), masked, np.inf),
                                     axis=1)).sum())
        c_max = float((w * np.nanmax(np.where(np.isfinite(masked), masked, -np.inf),
                                     axis=1)).sum())
    else:
        c_min = float((w * costs.min()).sum())
        c_max = float((w * costs.max()).sum())
    if target > c_max:
        target = c_max
    elif target < c_min:
        target = c_min

    current = cost_at(0.0)
    if abs(current - target) < tol:
        return current

    lo, hi = -1.0, 1.0
    for _ in range(60):
        if cost_at(lo) <= target <= cost_at(hi):
            break
        if cost_at(hi) < target:
            hi *= 2.0
        if cost_at(lo) > target:
            lo *= 2.0
    else:
        # The bracket never closed: the target is at the very edge of the
        # achievable range. Take the best endpoint rather than diverging.
        lo = hi = hi if abs(cost_at(hi) - target) < abs(cost_at(lo) - target) else lo

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        value = cost_at(mid)
        if abs(value - target) < tol:
            lo = hi = mid
            break
        if value > target:
            hi = mid
        else:
            lo = mid

    alpha += 0.5 * (lo + hi) * costs
    return cost_at(0.0)


def vector_transport(
    alpha: np.ndarray,
    costs: np.ndarray,
    moment: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> None:
    """Re-project an Adam first moment onto the current tangent plane.

    Modifies ``moment`` in place. The second moment is a variance estimate
    without a direction and is left untouched.
    """
    normal = budget_normal(alpha, costs, weights)
    n_flat = normal.reshape(-1)
    nsq = float(n_flat @ n_flat) + 1e-12
    coeff = float((moment.reshape(-1) @ n_flat) / nsq)
    moment -= coeff * normal


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def gumbel_perturb(
    logits: np.ndarray,
    tau: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Perturb logits with Gumbel noise, as upstream RCO does.

    Upstream's search (``optimize_projected_gumbel``) feeds the budget DP
    *Gumbel-perturbed* logits rather than the raw ones. The perturbation
    turns the smooth softmax into a sample from the categorical
    distribution, so the DP's value ordering reflects a concrete
    assignment rather than the mode of every row -- which, for a shared
    budget, is what lets the search trade precision between groups.

    Args:
        logits: (n_groups, n_options) logits.
        tau: temperature of the perturbation.
        rng: optional random generator, for reproducibility.

    Returns:
        The perturbed logits, same shape as the input.
    """
    generator = rng if rng is not None else np.random.default_rng()
    u = generator.random(logits.shape)
    # Guard against log(0) and log(1).
    u = np.clip(u, 1e-20, 1.0 - 1e-20)
    gumbel = -np.log(-np.log(u))
    return (logits + gumbel) / max(tau, 1e-6)


# ---------------------------------------------------------------------------
# Budget-constrained discrete assignment
# ---------------------------------------------------------------------------

def budget_constrained_argmax(
    logits: np.ndarray,
    group_weights: np.ndarray,
    option_costs: np.ndarray,
    target: float,
    resolution: int = 500,
    allowed: Optional[np.ndarray] = None,
    byte_costs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Exact multiple-choice knapsack: best assignment under a cost budget.

    Solves ``max sum_i log p[i, a_i]`` subject to
    ``sum_i w_i * c_{a_i} <= B``, by dynamic programming over a discretized
    budget. Complexity O(n_groups * n_options * budget).

    The budget and the per-group costs are in the *same* units: the cost of
    assigning group *i* the option *k* is ``w_i * c_k``. The discretization
    factor only scales both, so the solution is invariant to it.

    Args:
        logits: (n_groups, n_options) assignment logits.
        group_weights: (n_groups,) weight of each group (e.g. parameter count).
        option_costs: (n_options,) cost of each option (e.g. bytes/param).
        target: budget to spend, in the same units as ``w_i * c_k``.
        resolution: budget discretization granularity.
        allowed: optional (n_groups, n_options) boolean mask. Options that
            are not allowed for a group are never selected; they also do not
            contribute to that group's cost.
        byte_costs: optional (n_groups, n_options) *absolute* byte cost of
            each (group, option), from llama.cpp's ``ggml_row_size``.
            When given, the budget ``target`` is in **bytes** and the
            costs are used as-is -- no bits/weight estimate, which
            underestimates the real file size.

    Returns:
        (n_groups,) integer index of the chosen option per group.
    """
    n_groups, n_options = logits.shape
    values = logits.astype(np.float64)

    # Scale the weights so the budget stays a small integer.
    total_weight = float(group_weights.sum())
    if total_weight <= 0:
        return np.zeros(n_groups, dtype=np.int64)
    norm_weights = group_weights / total_weight

    # The objective is a *weighted* sum of the per-group log-probabilities:
    #
    #     max sum_i (w_i / W) * log p[i, a_i]
    #
    # The weight matters because a group's contribution to the model's
    # reconstruction error scales with its parameter count. Without it the
    # search trades a 268M-parameter expert tensor the same as a
    # 2M-parameter attention tensor, and it can end up making the *big*
    # tensors worse to pay for the *small* ones.
    values = values * norm_weights[:, None]

    budget = int(round(target * resolution))
    if budget < 0:
        budget = 0
    if byte_costs is not None:
        # Absolute byte costs: the budget is in bytes too. But the DP
        # allocates one entry per budget unit, so a byte budget of 1e9
        # makes the table 1e9 wide -- a 10 GB array. Scale the costs down
        # so the table stays at ~resolution units wide; the *relative*
        # ordering of spends is preserved, which is all the DP needs.
        scale = max(1.0, float(target) / resolution)
        costs = np.round(
            np.asarray(byte_costs, dtype=np.float64) / scale
        ).astype(np.int64)
        budget = int(round(target / scale))
    else:
        costs = np.round(
            np.outer(norm_weights, option_costs) * resolution
        ).astype(np.int64)

    # The budget must be clamped to the *achievable* range. The weights are
    # normalized (they sum to 1), so the total cost of an assignment can
    # only lie between the cheapest and the dearest option:
    #
    #     sum_i norm_w_i * min_k c_k  ..  sum_i norm_w_i * max_k c_k
    #
    # A target above that range is unreachable -- which happens when the
    # budget is the tier's own size and the *fixed* groups (norms,
    # routers) are stored more precisely than the free groups can be.
    # Without the clamp the DP would return an arbitrary assignment, and
    # the "exact budget" guarantee would be a fiction.
    cheapest = int(costs[:, :].min(axis=1).sum())
    dearest = int(costs[:, :].max(axis=1).sum())
    if budget > dearest:
        budget = dearest
    elif budget < cheapest:
        budget = cheapest

    neg_inf = -1e30
    dp = np.full(budget + 1, neg_inf, dtype=np.float64)
    dp[0] = 0.0
    backtrack = np.zeros((n_groups, budget + 1), dtype=np.int64)

    # A state is reachable when its value is finite; unreachable states stay
    # at ``neg_inf``. Using an explicit mask avoids treating "neg_inf plus a
    # logit" as a real value.
    reachable = np.zeros(budget + 1, dtype=bool)
    reachable[0] = True

    for g in range(n_groups):
        new_dp = np.full(budget + 1, neg_inf, dtype=np.float64)
        new_reachable = np.zeros(budget + 1, dtype=bool)
        for b in range(n_options):
            if allowed is not None and not bool(allowed[g, b]):
                continue
            c = int(costs[g, b])
            if c < 0:
                c = 0
            if c > budget:
                continue
            prev = dp[: budget - c + 1]
            prev_ok = reachable[: budget - c + 1]
            candidate = np.where(prev_ok, prev + values[g, b], neg_inf)
            target_slice = new_dp[c:]
            better = (candidate > target_slice) & prev_ok
            if np.any(better):
                idx = np.nonzero(better)[0]
                target_slice[idx] = candidate[idx]
                new_reachable[c + idx] = True
                backtrack[g, c + idx] = b
        dp = new_dp
        reachable = new_reachable

    # Select the budget to spend. The DP tracks the best value for each
    # *exact* spend, so the optimal assignment under the constraint
    # ``sum_i w_i c_{a_i} <= B`` is the best value over all spends at or
    # below the budget. Among equally good options, the largest spend is
    # taken, so the budget is used as fully as the assignment allows.
    if not np.any(reachable):
        # Nothing fits the budget. Return the jointly cheapest assignment, so
        # the overshoot is as small as possible. The cost is computed in
        # exact float arithmetic, because the DP's discretized costs cannot
        # distinguish options whose prices differ by less than one unit.
        #
        # Use the *absolute* byte costs when they are available -- the
        # bits/weight cost vector and the byte cost vector rank options the
        # same way in general, but the fallback's job is to minimise the
        # overshoot, and the overshoot is in bytes.
        exact = np.outer(norm_weights, option_costs)
        if byte_costs is not None:
            exact = np.asarray(byte_costs, dtype=np.float64)
        assignment = np.zeros(n_groups, dtype=np.int64)
        for g in range(n_groups):
            choices = [
                b for b in range(n_options)
                if allowed is None or bool(allowed[g, b])
            ]
            if not choices:
                choices = list(range(n_options))
            assignment[g] = min(choices, key=lambda b: (exact[g, b], b))
        return assignment

    reachable_values = np.where(reachable, dp, neg_inf)
    best_value = float(reachable_values.max())
    candidates = np.nonzero(
        reachable & (reachable_values >= best_value - 1e-9)
    )[0]
    best_j = int(candidates.max())

    assignment = np.zeros(n_groups, dtype=np.int64)
    j = best_j
    for g in range(n_groups - 1, -1, -1):
        b = int(backtrack[g, j])
        assignment[g] = b
        j -= int(costs[g, b])
        if j < 0:
            break
    return assignment


# ---------------------------------------------------------------------------
# Quant-type options
# ---------------------------------------------------------------------------

@dataclass
class QuantOption:
    """One candidate quant type for a tensor group.

    Attributes:
        name: llama.cpp type name, e.g. ``Q4_K``.
        bits_per_weight: effective bits per weight, used as the budget cost.
        allowed: whether the option may be selected.
    """

    name: str
    bits_per_weight: float
    allowed: bool = True

    def __post_init__(self) -> None:
        if self.bits_per_weight <= 0:
            raise ValueError(f"Option {self.name!r} needs positive bits/weight")


@dataclass
class GroupSpec:
    """A set of tensors that must share one quant type.

    Attributes:
        name: human-readable group label.
        tensors: tensor names in the group.
        options: candidate quant types, in preference order.
    """

    name: str
    tensors: List[str] = field(default_factory=list)
    options: List[QuantOption] = field(default_factory=list)

    @property
    def allowed_options(self) -> List[QuantOption]:
        return [o for o in self.options if o.allowed]

    def option_names(self) -> List[str]:
        return [o.name for o in self.options]

    def bits(self) -> np.ndarray:
        return np.array([o.bits_per_weight for o in self.options], dtype=np.float64)

    def allowed_mask(self) -> np.ndarray:
        return np.array([o.allowed for o in self.options], dtype=bool)


def option_cost_table(
    groups: Sequence[GroupSpec],
    param_counts: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
    """Build the (n_groups, n_options) cost table for the budget DP.

    Cost is measured in *bytes of model*, so the budget is a real file size.

    Args:
        groups: group specifications.
        param_counts: parameter count per tensor name.

    Returns:
        (costs, weights, option_names_per_group) where ``costs[g][k]`` is the
        total bytes group *g* would occupy with option *k*.
    """
    n_groups = len(groups)
    n_options = max((len(g.options) for g in groups), default=0)
    costs = np.zeros((n_groups, n_options), dtype=np.float64)
    weights = np.zeros(n_groups, dtype=np.float64)
    names: List[List[str]] = []

    for gi, group in enumerate(groups):
        params = sum(param_counts.get(t, 0) for t in group.tensors)
        weights[gi] = float(params)
        names.append([o.name for o in group.options])
        for oi, opt in enumerate(group.options):
            # bits/weight * params / 8 = bytes
            costs[gi, oi] = opt.bits_per_weight * params / 8.0
    return costs, weights, names


def assign_from_options(
    groups: Sequence[GroupSpec],
    assignment: np.ndarray,
) -> Dict[str, str]:
    """Expand a per-group option assignment into a per-tensor type map."""
    out: Dict[str, str] = {}
    for gi, group in enumerate(groups):
        opt = group.options[int(assignment[gi])]
        for tensor in group.tensors:
            out[tensor] = opt.name
    return out


def allowed_option_mask(groups: Sequence[GroupSpec]) -> np.ndarray:
    """Build the (n_groups, n_options) mask of selectable options.

    Args:
        groups: group specifications.

    Returns:
        A boolean array; ``mask[g, k]`` is True when group *g* may select
        option *k*.
    """
    n_options = max((len(g.options) for g in groups), default=0)
    mask = np.zeros((len(groups), n_options), dtype=bool)
    for gi, group in enumerate(groups):
        for oi, opt in enumerate(group.options):
            mask[gi, oi] = bool(opt.allowed)
    return mask


def budget_aware_init(
    groups: Sequence[GroupSpec],
    param_counts: Dict[str, int],
    target_bytes: float,
    sensitivity: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Initial logits that put the model on the budget surface.

    High-sensitivity tensors start biased toward their highest-precision
    option; the shared budget is then enforced by a retraction.

    Args:
        groups: group specifications.
        param_counts: parameter count per tensor.
        target_bytes: total model size budget in bytes.
        sensitivity: optional per-tensor sensitivity score (higher = more
            sensitive = keep more precision).

    Returns:
        (n_groups, n_options) logits.
    """
    n_options = max((len(g.options) for g in groups), default=0)
    logits = np.zeros((len(groups), n_options), dtype=np.float64)
    mask = allowed_option_mask(groups)

    total_params = float(
        sum(sum(param_counts.get(t, 0) for t in g.tensors) for g in groups)
    )

    for gi, group in enumerate(groups):
        n_opt = len(group.options)
        if n_opt == 0:
            continue
        if sensitivity:
            score = float(np.mean([sensitivity.get(t, 0.0) for t in group.tensors] or [0.0]))
        else:
            score = 0.0
        # The option axis is sorted most-precise first, so a *decreasing*
        # bias favours the most precise option.
        bias = np.linspace(1.0, 0.0, n_opt) * (1.0 + score)
        logits[gi] = bias - bias.max()
        # Options the group may not select are pushed out of the softmax.
        logits[gi][~mask[gi]] = -1e9

    if len(groups) and n_options and total_params > 0:
        # The retraction works in upstream RCO's units:
        #   C(alpha) = sum_i w_i * sum_k p_ik * c_k
        # with w_i the parameter *fraction* of group i and c_k the
        # bits/weight of option k. That keeps the budget, the cost vector
        # and the weights in one consistent unit system.
        fractions = np.array(
            [
                sum(param_counts.get(t, 0) for t in g.tensors) / total_params
                for g in groups
            ],
            dtype=np.float64,
        )
        option_bits = np.array(
            [
                float(
                    np.mean(
                        [
                            g.options[k].bits_per_weight
                            for g in groups
                            if len(g.options) > k
                        ]
                    )
                )
                for k in range(n_options)
            ],
            dtype=np.float64,
        )
        target_bpw = target_bytes * 8.0 / total_params
        retraction(
            logits, option_bits, target_bpw, fractions, tol=1e-4,
            allowed=mask,
        )
    return logits


__all__ = [
    "GroupSpec",
    "QuantOption",
    "allowed_option_mask",
    "assign_from_options",
    "budget_aware_init",
    "budget_constrained_argmax",
    "budget_normal",

    "option_cost_table",
    "project_gradient",
    "retraction",
    "vector_transport",
]
