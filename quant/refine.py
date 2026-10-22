"""
@file toffee/quant/refine.py
   Orchestrators that combine the APEX tier with RCO and GSQ.

The APEX tier is the *policy* layer: it decides which quant type each tensor
role gets and, implicitly, what the model costs. This module adds two
refinements underneath it, both of which preserve APEX's semantics:

  * :func:`build_rco_tensor_map` keeps APEX's per-role quant *family* as the
    option set and lets RCO pick the exact member of each family under an
    **exact size budget**. The result is still a llama.cpp
    ``--tensor-type-file`` and still obeys APEX's role rules.

  * :func:`refine_gguf_with_gsq` takes a GGUF that ``llama-quantize`` already
    produced and re-learns, in format, the codes and per-group scales of the
    selected tensors. The file stays a valid GGUF.

Both are opt-in: without them, TOFFEE quantizes exactly as before.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from toffee.quant.ggml_bridge import GgmlBridge, type_id_for
from toffee.quant.gguf_io import (
    GgufFile,
    group_by_block,
)
from toffee.quant.gguf_kquant import (
    decode_blocks,
    dequantize_blocks,
    encode_blocks,
    is_supported,
)
from toffee.quant.gsq import GsqConfig, refine_tensor
from toffee.quant.rco import (
    GroupSpec,
    QuantOption,
    allowed_option_mask,
    budget_aware_init,
    budget_constrained_argmax,
    gumbel_perturb,
    option_cost_table,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quant-type metadata
# ---------------------------------------------------------------------------

#: Effective bits per weight for the scalar quant types TOFFEE can emit.
#: These are the *storage* cost of a tensor, used as the RCO budget cost.
BITS_PER_WEIGHT: Dict[str, float] = {
    "Q8_0": 8.5,
    "Q6_K": 6.5625,
    "Q5_K": 5.5,
    "Q4_K": 4.5,
    "Q3_K": 3.4375,
    "Q2_K": 2.625,
    "IQ4_XS": 4.25,
    "IQ4_NL": 4.5,
    "IQ3_S": 3.4375,
    "IQ3_XXS": 3.25,
    "IQ2_S": 2.5,
    "IQ2_XS": 2.25,
    "IQ2_XXS": 2.125,
    "IQ1_S": 1.75,
    "IQ1_M": 1.75,
    "TQ1_0": 1.6875,
    "TQ2_0": 2.0625,
    "F32": 32.0,
    "F16": 16.0,
}


def bits_for(type_name: str) -> float:
    """Effective bits per weight of a llama.cpp quant type."""
    try:
        return BITS_PER_WEIGHT[type_name.upper()]
    except KeyError:
        raise KeyError(
            f"Unknown bits/weight for {type_name!r}; add it to BITS_PER_WEIGHT"
        ) from None


# ---------------------------------------------------------------------------
# APEX tier -> RCO option sets
# ---------------------------------------------------------------------------

@dataclass
class ApexRole:
    """One APEX role and the quant types it may use.

    Attributes:
        name: the APEX role name (e.g. ``experts``, ``shared``, ``attention``).
        patterns: regular expressions matching the tensor names of the role.
        options: the quant types the role may use, best first.
    """

    name: str
    patterns: List[str]
    options: List[str]


def _role_from_regex(pattern: str) -> str:
    """Derive a role label from a tensor-name regex."""
    return re.sub(r"[^a-z0-9]+", "_", pattern.lower()).strip("_") or "misc"


def build_role_groups(
    tensor_map: Dict[str, str],
    role_patterns: Sequence[Tuple[str, str]],
) -> List[GroupSpec]:
    """Group tensors into APEX roles from a set of (regex, quant type) rules.

    Every tensor in ``tensor_map`` is assigned to the **first** matching
    pattern, mirroring how llama.cpp itself resolves a tensor-type file
    (first regex match wins). Tensors that match nothing keep their
    original type and form singleton groups.

    The group's role name is derived from the tensor *family* (the name with
    the block number replaced by a placeholder), so all blocks of the same
    kind end up in one role: ``blk.0.ffn_gate_exps.weight`` and
    ``blk.9.ffn_gate_exps.weight`` share the role ``ffn_gate_exps``.

    Args:
        tensor_map: the APEX tier's tensor -> quant type map.
        role_patterns: ordered (regex, quant type) rules, best role first.

    Returns:
        The role groups, each carrying the quant types its members may use.
    """
    groups: List[GroupSpec] = []
    assigned: Dict[str, int] = {}
    group_by_role: Dict[str, int] = {}

    for pattern, quant_type in role_patterns:
        rx = re.compile(pattern)
        members = [
            name for name in tensor_map
            if name not in assigned and rx.search(name)
        ]
        if not members:
            continue
        # The role comes from the family of the first member, so the group
        # covers every block of that kind.
        role = _role_of(_family_of(members[0]))
        if role in group_by_role:
            # Merge into the existing role group.
            gi = group_by_role[role]
            groups[gi].tensors.extend(members)
            groups[gi].tensors.sort()
            for name in members:
                assigned[name] = gi
            continue
        group = GroupSpec(
            name=role,
            tensors=sorted(members),
            options=[QuantOption(quant_type, bits_for(quant_type))],
        )
        groups.append(group)
        group_by_role[role] = len(groups) - 1
        for name in members:
            assigned[name] = len(groups) - 1

    # Everything else keeps the type APEX gave it.
    for name, quant_type in sorted(tensor_map.items()):
        if name in assigned:
            continue
        groups.append(
            GroupSpec(
                name=f"fixed_{_role_of(_family_of(name))}",
                tensors=[name],
                options=[QuantOption(quant_type, bits_for(quant_type))],
            )
        )
        assigned[name] = len(groups) - 1

    return groups


def expand_role_options(
    groups: Sequence[GroupSpec],
    ladders: Dict[str, List[str]],
) -> List[GroupSpec]:
    """Widen each group's option set to a precision ladder.

    All groups end up with the **same** number of options, drawn from the
    union of every ladder. RCO's budget manifold (softmax over the option
    axis, a shared cost vector, and the knapsack DP) assumes a rectangular
    ``(n_groups, n_options)`` problem, exactly like the upstream
    implementation, which gives every group the same global bitwidth axis.
    A role that does not offer a given type simply cannot select it: the
    option is marked as not allowed, and the search masks it out.

    Args:
        groups: the role groups.
        ladders: role name -> list of quant types the role may use, from the
            most precise to the least.

    Returns:
        New groups whose option sets come from the ladders.
    """
    # The global option axis: every type any ladder mentions, plus every
    # group's own type. The latter matters: a role with no ladder keeps
    # APEX's type, and that type (Q8_0, F32) is usually *not* in any
    # ladder. Without it on the axis the group would have no allowed
    # option at all -- so it would contribute nothing to the budget
    # while really costing bytes, and the budget arithmetic would be
    # wrong.
    universe: List[str] = []
    for ladder in ladders.values():
        for name in ladder:
            if name not in universe:
                universe.append(name)
    if not universe:
        return list(groups)
    for group in groups:
        for opt in group.options:
            if opt.name not in universe:
                universe.append(opt.name)
    universe.sort(key=lambda n: -bits_for(n))

    out: List[GroupSpec] = []
    for group in groups:
        ladder = ladders.get(group.name)
        if not ladder:
            # The role keeps APEX's type, but it still needs the full axis.
            keep = group.options[0].name if group.options else None
            options = [
                QuantOption(name, bits_for(name), allowed=(name == keep))
                for name in universe
            ]
            out.append(
                GroupSpec(name=group.name, tensors=list(group.tensors), options=options)
            )
            continue
        allowed = set(ladder)
        # The type APEX already assigned must stay selectable. Otherwise
        # the search is *forced* to move the tensor to some other type --
        # and if APEX chose it well (e.g. Q8_0 for an attention output),
        # every alternative is worse. Measured: that made the model 3%
        # worse than the APEX map it started from.
        for opt in group.options:
            if opt.allowed:
                allowed.add(opt.name)
        options = [
            QuantOption(name, bits_for(name), allowed=(name in allowed))
            for name in universe
        ]
        out.append(
            GroupSpec(name=group.name, tensors=list(group.tensors), options=options)
        )
    return out


# ---------------------------------------------------------------------------
# RCO refinement of the APEX tier
# ---------------------------------------------------------------------------

@dataclass
class RcoResult:
    """Outcome of the RCO mixed-precision search.

    Attributes:
        tensor_map: the refined tensor -> quant type map.
        bytes_used: the model size implied by the assignment.
        target_bytes: the budget that was requested.
        groups: the groups that were searched.
        assignment: the chosen option index per group.
    """

    tensor_map: Dict[str, str]
    bytes_used: float
    target_bytes: float
    groups: List[GroupSpec]
    assignment: np.ndarray

    @property
    def within_budget(self) -> bool:
        """True when the assignment respects the requested budget."""
        return self.bytes_used <= self.target_bytes * 1.001

    @property
    def relative_error(self) -> float:
        """Relative overshoot of the budget."""
        if self.target_bytes <= 0:
            return 0.0
        return (self.bytes_used - self.target_bytes) / self.target_bytes


def measure_role_distortion(
    groups: Sequence[GroupSpec],
    source: Optional[Dict[str, np.ndarray]],
    bridge: Optional[GgmlBridge] = None,
    max_rows: int = 2048,
) -> Dict[str, Dict[str, float]]:
    """Measure each group's distortion at each of its candidate types.

    This is the signal that makes RCO's search meaningful. Upstream RCO
    initializes its logits from the *measured* distortion of each
    bitwidth (``alpha_init = -log(distortion + eps)``), so the DP's
    ``log p`` value reflects real accuracy. A binary edge/bulk flag
    cannot: under a byte budget it makes the search downgrade whatever
    fits, arbitrarily -- measured on a real model, that made the model
    **2.5% worse**, not better.

    The distortion is the normalized mean squared error of quantizing the
    group's weights with the candidate type and decoding them back, using
    llama.cpp's own quantizer and decoder.

    Args:
        groups: the role groups.
        source: tensor name -> the f16 weights, as float32. Groups whose
            tensors are absent are skipped.
        bridge: optional pre-built llama.cpp bridge.
        max_rows: how many rows of each tensor to sample.

    Returns:
        A mapping of group name -> {quant type: relative MSE}.
    """
    if not source:
        return {}
    br = bridge or GgmlBridge.get()
    out: Dict[str, Dict[str, float]] = {}
    for group in groups:
        per_type: Dict[str, float] = {}
        for tensor in group.tensors:
            weights = source.get(tensor)
            if weights is None or weights.size == 0:
                continue
            npr = weights.shape[-1]
            rows = min(max_rows, weights.shape[0])
            sample = np.ascontiguousarray(
                weights[:rows].reshape(rows, npr), dtype=np.float32
            )
            ref_energy = float((sample.reshape(-1) ** 2).sum())
            if ref_energy <= 0:
                continue
            for opt in group.options:
                try:
                    type_id = type_id_for(opt.name)
                except KeyError:
                    continue
                try:
                    raw = br.quantize(type_id, sample, npr, None)
                    got = br.dequantize(type_id, raw, npr).reshape(-1)
                except Exception:  # noqa: BLE001 - a bad option is skipped
                    continue
                err = float(((got - sample.reshape(-1)) ** 2).sum())
                per_type.setdefault(opt.name, 0.0)
                per_type[opt.name] += err / ref_energy
        if per_type:
            # Average over the group's tensors.
            n_used = max(1, len(
                [t for t in group.tensors if t in source]
            ))
            out[group.name] = {k: v / n_used for k, v in per_type.items()}
    return out


def exact_tensor_bytes(
    tensor: str,
    type_name: str,
    n_elements: int,
    n_cols: int,
    bridge: Optional[GgmlBridge] = None,
) -> float:
    """The exact number of bytes a tensor occupies when stored as a type.

    llama.cpp's ``ggml_row_size`` gives the real size of a row, including
    the per-block overheads (sub-scales, sub-minimums, the fp16
    super-scale). The nominal bits/weight does not: on a 24.5 MB model it
    said 24.453 MB where the file was 24.554 MB.

    Args:
        tensor: the tensor's name (for diagnostics).
        type_name: the llama.cpp quant type.
        n_elements: the tensor's element count.
        n_cols: the tensor's elements per row.
        bridge: optional pre-built llama.cpp bridge.

    Returns:
        The byte cost.
    """
    if n_elements <= 0:
        return 0.0
    try:
        type_id = type_id_for(type_name)
    except KeyError:
        return bits_for(type_name) * n_elements / 8.0
    if n_cols <= 0:
        return bits_for(type_name) * n_elements / 8.0
    br = bridge or GgmlBridge.get()
    try:
        size = br.row_size(type_id, n_cols)
    except Exception:  # noqa: BLE001 - fall back to the estimate
        size = 0
    if not size:
        return bits_for(type_name) * n_elements / 8.0
    return (n_elements // n_cols) * size


def _exact_byte_costs(
    groups: Sequence[GroupSpec],
    param_counts: Dict[str, int],
    cols: Dict[str, int],
    bridge: Optional[GgmlBridge] = None,
) -> List[List[float]]:
    """The exact byte cost of each (group, option).

    llama.cpp's ``ggml_row_size`` gives the real number of bytes a row
    occupies in a given quant type, including the per-block overheads
    (sub-scales, sub-minimums, the fp16 super-scale). The nominal
    bits/weight does not: measured on a 24.5 MB model, the estimate said
    24.453 MB and the file was 24.554 MB.

    The DP's cost vector must use the real costs, or the budget it
    enforces is not the budget the file ends up at.

    Args:
        groups: the role groups.
        param_counts: parameter count per tensor.
        cols: tensor name -> elements per row (the GGUF's last dimension).
        bridge: optional pre-built llama.cpp bridge.

    Returns:
        One list of per-option byte costs per group.
    """
    br = bridge or GgmlBridge.get()
    out: List[List[float]] = []
    for group in groups:
        row: List[float] = []
        for opt in group.options:
            total = 0.0
            for tensor in group.tensors:
                n = param_counts.get(tensor, 0)
                if n <= 0:
                    continue
                try:
                    type_id = type_id_for(opt.name)
                except KeyError:
                    total += opt.bits_per_weight * n / 8.0
                    continue
                # The number of columns of the tensor: llama.cpp stores a
                # row-major flattening, so the row length is the number of
                # elements per row.
                ncols = cols.get(tensor, 0)
                if ncols <= 0:
                    total += opt.bits_per_weight * n / 8.0
                    continue
                rows = n // ncols
                try:
                    size = br.row_size(type_id, ncols)
                except Exception:  # noqa: BLE001 - fall back to the estimate
                    size = 0
                if size:
                    total += rows * size
                else:
                    total += opt.bits_per_weight * n / 8.0
            row.append(total)
        out.append(row)
    return out


def build_rco_tensor_map(
    tensor_map: Dict[str, str],
    param_counts: Dict[str, int],
    target_bytes: float,
    ladders: Optional[Dict[str, List[str]]] = None,
    sensitivity: Optional[Dict[str, float]] = None,
    samples: int = 64,
    tau: float = 0.0,
    seed: int = 0,
    source: Optional[Dict[str, np.ndarray]] = None,
    bridge: Optional[GgmlBridge] = None,
    cols: Optional[Dict[str, int]] = None,
) -> RcoResult:
    """Refine an APEX tier map with an exactly budget-constrained RCO search.

    The APEX map decides the *role* of every tensor; RCO then chooses, within
    the role's precision ladder, the quant type that best fits the budget.
    When no ladder is given for a role, the role keeps APEX's type.

    Args:
        tensor_map: the APEX tier's tensor -> quant type map.
        param_counts: parameter count per tensor.
        target_bytes: the model size budget in bytes.
        ladders: role name -> list of quant types the role may use.
        sensitivity: optional per-tensor sensitivity (higher = keep more
            precision). Used to seed the search.
        samples: number of Gumbel samples drawn for the assignment.
        tau: temperature of the Gumbel perturbation.
        seed: seed for the random generator.

    Returns:
        An :class:`RcoResult` with the refined map.
    """
    groups = build_role_groups(tensor_map, _patterns_from_map(tensor_map))
    if ladders:
        groups = expand_role_options(groups, ladders)

    # The *exact* byte cost of each (group, option), from llama.cpp's own
    # `ggml_row_size`. The nominal bits/weight underestimates the real file
    # size, because every block also carries scales, minimums and an fp16
    # super-scale. Measured on a 24.5MB model: the estimate said 24.453MB
    # and the file was 24.554MB. That 0.4% gap is enough to make the
    # budget *smaller* than the map it started from, which forces the
    # search to downgrade something it should have kept.
    byte_costs = _exact_byte_costs(
        groups, param_counts, cols or {}, bridge
    )

    # Groups with a single option are fixed; they still count against the
    # budget, so they are excluded from the search.
    fixed_bytes = 0.0
    free_groups: List[GroupSpec] = []
    # A group is *fixed* when it has at most one *allowed* option -- the
    # no-ladder branch gives those groups the full axis with only their
    # own type selectable, and counting the axis length instead would
    # classify them as free. That matters: a "free" group still counts
    # against the budget, and a misclassified one makes the search
    # believe it has a choice it does not.
    free_groups: List[GroupSpec] = []
    free_byte_index: List[int] = []
    for gi, group in enumerate(groups):
        n_allowed = sum(1 for o in group.options if o.allowed)
        if n_allowed <= 1:
            # The group's only selectable option is its own type, which
            # may sit anywhere on the axis (Q8_0 is not at index 0).
            own = group.options[0].name
            own_idx = next(
                (oi for oi, o in enumerate(group.options) if o.name == own),
                0,
            )
            fixed_bytes += byte_costs[gi][own_idx]
        else:
            free_groups.append(group)
            free_byte_index.append(gi)

    if not free_groups:
        logger.info(
            "rco_no_free_groups",
            extra={"fixed_bytes": fixed_bytes, "target": target_bytes},
        )
        return RcoResult(
            tensor_map=dict(tensor_map),
            bytes_used=fixed_bytes,
            target_bytes=target_bytes,
            groups=groups,
            assignment=np.zeros(len(groups), dtype=np.int64),
        )

    budget = max(target_bytes - fixed_bytes, 0.0)
    costs, weights, _ = option_cost_table(free_groups, param_counts)

    # The DP's cost vector is the *exact* byte cost of each (group, option),
    # from llama.cpp's ggml_row_size -- not the nominal bits/weight, which
    # underestimates the real file by ~0.4%.
    free_byte_costs = np.array(
        [byte_costs[gi] for gi in free_byte_index], dtype=np.float64
    )

    # Seed the logits with the sensitivity signal, then pull them onto the
    # budget surface with a retraction.
    logits = budget_aware_init(
        free_groups, param_counts, budget, sensitivity=None
    )

    # The measured distortion is the signal that makes the search mean
    # anything: upstream RCO initializes its logits from the real
    # per-bitwidth error (``alpha_init = -log(distortion + eps)``), so the
    # DP's ``log p`` value is an accuracy signal. Without it the search
    # only knows that "more precision is better", and under a byte budget
    # it downgrades arbitrarily.
    distortion = measure_role_distortion(
        free_groups, source, bridge=bridge
    )
    if distortion:
        for gi, group in enumerate(free_groups):
            per_type = distortion.get(group.name)
            if not per_type:
                continue
            row = logits[gi]
            for oi, opt in enumerate(group.options):
                d = per_type.get(opt.name)
                if d is None:
                    continue
                # -log(distortion): lower error -> higher logit.
                row[oi] = -float(np.log(max(d, 1e-12)))
            # Normalize within the row so the scale is comparable across
            # groups, and mask the options the group may not select.
            finite = row[np.isfinite(row)]
            if finite.size:
                row -= finite.mean()
            row[~allowed_option_mask([group])[0]] = -1e9
        logger.info(
            "rco_distortion_seeded",
            extra={"groups": len(distortion)},
        )

    if sensitivity:
        for gi, group in enumerate(free_groups):
            score = float(np.mean([sensitivity.get(t, 0.0) for t in group.tensors] or [0.0]))
            # More sensitive groups lean towards the *first* (most precise)
            # option; the budget retraction then balances the whole model.
            mask = allowed_option_mask([group])[0]
            allowed_idx = np.nonzero(mask)[0]
            if allowed_idx.size == 0:
                continue
            bias = np.zeros(len(group.options), dtype=np.float64)
            # Rank the allowed options from most to least precise and give
            # the most precise one the largest logit.
            best = allowed_idx[0]
            bias[best] = score
            logits[gi][~mask] = -1e9
            logits[gi] += bias

    # The DP's budget and costs are in **bytes**, from llama.cpp's own
    # ``ggml_row_size``, so the budget the search enforces is the budget
    # the file ends up at. (The nominal bits/weight underestimates the
    # real size by ~0.4%, which is enough to make the search downgrade
    # something it should have kept.)
    total_params = float(weights.sum())
    fractions = weights / max(total_params, 1.0)
    target_bpw = budget
    free_byte_costs = np.array(
        [byte_costs[gi] for gi in free_byte_index], dtype=np.float64
    )
    mask = allowed_option_mask(free_groups)

    # Upstream samples the assignment with Gumbel-Softmax and then lets the
    # knapsack DP repair it onto the budget surface.
    rng = np.random.default_rng(seed)
    best_assignment: Optional[np.ndarray] = None
    best_fill = -1.0
    best_value = -np.inf
    for _ in range(max(samples, 1)):
        # The DP's value is ``log p`` over the option axis. With the
        # distortion-seeded logits that is an *accuracy* signal, so the
        # perturbation must not swamp it: a full-strength Gumbel noise on
        # logits that differ by several nats randomizes the ordering, and
        # the search then picks by fill rather than by accuracy.
        #
        # So the perturbation is scaled down, and the *best* candidate is
        # chosen by the DP's own value -- the total log-probability --
        # not by how much of the budget it happens to fill.
        noisy = gumbel_perturb(logits, tau=tau, rng=rng) if tau > 0 else logits
        candidate = budget_constrained_argmax(
            noisy,
            fractions,
            free_byte_costs,
            target_bpw,
            resolution=10000,
            allowed=mask,
            byte_costs=free_byte_costs,
        )
        # How much of the budget this candidate uses, and how accurate it is.
        # The value is the *weighted* sum of the chosen logits, matching the
        # DP's own objective.
        fill = float(free_byte_costs[np.arange(len(candidate)), candidate].sum())
        value = float(
            (fractions * logits[np.arange(len(candidate)), candidate]).sum()
        )
        if value > best_value or (value == best_value and fill > best_fill):
            best_value = value
            best_fill = fill
            best_assignment = candidate
    if best_assignment is None:
        best_assignment = np.zeros(len(free_groups), dtype=np.int64)
    assignment = best_assignment

    # Build the final map.
    new_map = dict(tensor_map)
    for gi, group in enumerate(free_groups):
        chosen = group.options[int(assignment[gi])].name
        for tensor in group.tensors:
            new_map[tensor] = chosen

    # The reported size must be computed the same way the budget was
    # enforced -- with llama.cpp's real row sizes -- or the "within
    # budget" claim is a fiction. Sum per *group*, because
    # ``byte_costs[gi][oi]`` is the group's total cost for choosing option
    # oi -- it already sums over every tensor in the group.
    _gi_of = {name: gi for gi, g in enumerate(groups) for name in g.tensors}
    used = 0.0
    seen: set = set()
    for name, chosen in new_map.items():
        gi = _gi_of[name]
        if gi in seen:
            continue
        seen.add(gi)
        own_idx = next(
            oi for oi, o in enumerate(groups[gi].options) if o.name == chosen
        )
        used += byte_costs[gi][own_idx]
    logger.info(
        "rco_map_built",
        extra={
            "target_bytes": target_bytes,
            "used_bytes": used,
            "free_groups": len(free_groups),
            "fixed_groups": len(groups) - len(free_groups),
        },
    )
    return RcoResult(
        tensor_map=new_map,
        bytes_used=used,
        target_bytes=target_bytes,
        groups=groups,
        assignment=assignment,
    )


def _patterns_from_map(tensor_map: Dict[str, str]) -> List[Tuple[str, str]]:
    """Derive (regex, quant type) rules from an APEX tensor map.

    The rules are ordered so that more specific patterns win, mirroring
    llama.cpp's own "first match wins" resolution. Tensors that belong to
    the same *family* (same kind, different block) share one rule, so RCO
    can move all of them together.
    """
    # Collect the distinct types and the patterns that produce them.
    by_type: Dict[str, List[str]] = {}
    for name, quant_type in tensor_map.items():
        by_type.setdefault(quant_type, []).append(name)

    rules: List[Tuple[str, str]] = []
    for quant_type, names in by_type.items():
        # Group the names into families so that a whole role shares one rule.
        families: Dict[str, List[str]] = {}
        for name in names:
            family = _family_of(name)
            families.setdefault(family, []).append(name)
        for family, members in sorted(families.items()):
            rules.append((_family_pattern(family, members), quant_type))
    return rules


def _family_of(name: str) -> str:
    """Reduce a tensor name to its role family.

    The block number is replaced by a placeholder so that every block's
    tensors share one family, e.g. ``blk.3.ffn_gate_exps.weight`` and
    ``blk.7.ffn_gate_exps.weight`` both become
    ``blk.N.ffn_gate_exps.weight``.
    """
    return re.sub(r"\bblk\.\d+", "blk.N", name)


def _role_of(family: str) -> str:
    """Derive a stable role label from a tensor family.

    The block placeholder and the ``.weight`` suffix are dropped, so the
    role reads as the tensor's *kind*: ``ffn_gate_exps``, ``attn_q``, ...
    """
    stem = re.sub(r"^blk\.N\.", "", family)
    stem = re.sub(r"\.weight$", "", stem)
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_") or "misc"


def _family_pattern(family: str, members: Sequence[str]) -> str:
    """Build a regex matching every member of a family.

    The family already has the block number replaced by a placeholder, so
    the pattern matches *every* block of that kind -- which is what lets RCO
    move a whole role at once.
    """
    return re.escape(family).replace(re.escape("blk.N"), r"blk\.\d+")


# ---------------------------------------------------------------------------
# GSQ refinement of a GGUF
# ---------------------------------------------------------------------------

@dataclass
class GsqRefineReport:
    """Summary of a GSQ refinement pass over a GGUF.

    Attributes:
        tensors_refined: number of tensors that were refined.
        tensors_skipped: number of tensors that were left untouched.
        initial_error: summed weighted error before refinement.
        final_error: summed weighted error after refinement.
        per_tensor: per-tensor (name, initial, final) error triples.
    """

    tensors_refined: int = 0
    tensors_skipped: int = 0
    initial_error: float = 0.0
    final_error: float = 0.0
    per_tensor: List[Tuple[str, float, float]] = field(default_factory=list)

    @property
    def relative_improvement(self) -> float:
        """Fractional reduction of the summed weighted error."""
        if self.initial_error <= 0:
            return 0.0
        return (self.initial_error - self.final_error) / self.initial_error


def _load_imatrix(imatrix_path: Optional[Path]) -> Dict[str, np.ndarray]:
    """Read a TOFFEE imatrix file into per-tensor weight vectors."""
    if imatrix_path is None or not imatrix_path.exists():
        return {}
    import gguf as _gguf

    reader = _gguf.GGUFReader(str(imatrix_path))
    out: Dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        if tensor.name.endswith(".in_sum2"):
            name = tensor.name[: -len(".in_sum2")]
            out[name] = np.asarray(tensor.data, dtype=np.float32).reshape(-1)
    return out


def _budget_exceeded(started: float, budget_minutes: Optional[float]) -> bool:
    """Whether the wall-clock budget has been used up.

    Args:
        started: the monotonic timestamp the run started at.
        budget_minutes: the budget in minutes, or None for no budget.

    Returns:
        True when the budget is set and has been exceeded.
    """
    if budget_minutes is None:
        return False
    return (time.monotonic() - started) / 60.0 >= budget_minutes


def _load_completed_blocks(state_path: Optional[Path]) -> set:
    """Read the set of blocks already refined, from a checkpoint file.

    Args:
        state_path: the checkpoint file, if any.

    Returns:
        The set of completed block ids.
    """
    if state_path is None or not Path(state_path).exists():
        return set()
    try:
        with open(state_path, "r") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("gsq_state_load_failed", extra={"error": str(exc)})
        return set()
    return {int(b) for b in data.get("completed_blocks", [])}


def _save_completed_blocks(state_path: Optional[Path], block_id: int) -> None:
    """Record a completed block in the checkpoint file.

    The file is rewritten atomically, so a crash mid-write cannot leave a
    corrupt checkpoint behind.

    Args:
        state_path: the checkpoint file, if any.
        block_id: the block that just finished.
    """
    if state_path is None:
        return
    done = _load_completed_blocks(state_path)
    done.add(int(block_id))
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump({"completed_blocks": sorted(done)}, f, indent=2)
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("gsq_state_save_failed", extra={"error": str(exc)})


def _agrees_with_llama(
    bridge: GgmlBridge,
    type_name: str,
    payload: bytes,
    n_per_row: int,
    result,
) -> bool:
    """Check that llama.cpp decodes the payload exactly as the codec claims.

    The refinement is written into the GGUF verbatim, so the bytes must
    decode identically under llama.cpp's own ``to_float``. This check makes
    a codec mismatch a skipped tensor instead of a corrupted file.

    Args:
        bridge: the llama.cpp bridge.
        type_name: llama.cpp quant type name.
        payload: the encoded bytes.
        n_per_row: elements per row.
        result: the refinement result (for its effective scales/mins).

    Returns:
        True when llama.cpp and the codec agree.
    """
    try:
        type_id = type_id_for(type_name)
    except KeyError:
        return False
    try:
        via_llama = bridge.dequantize(type_id, payload, n_per_row)
        c2, s2, m2, _ = decode_blocks(payload, type_name)
        via_codec = dequantize_blocks(c2, s2, m2, type_name).reshape(via_llama.shape)
    except Exception as exc:  # noqa: BLE001 - a mismatch is a skip, not a crash
        logger.warning("gsq_verify_error", extra={"error": str(exc)})
        return False
    return bool(np.abs(via_llama - via_codec).max() == 0.0)


def refine_gguf_with_gsq(
    gguf_path: Path,
    f16_gguf_path: Path,
    imatrix_path: Optional[Path] = None,
    config: Optional[GsqConfig] = None,
    tensor_types: Optional[Dict[str, str]] = None,
    max_tensors: Optional[int] = None,
    bridge: Optional[GgmlBridge] = None,
    state_path: Optional[Path] = None,
    resume: bool = False,
    time_budget_minutes: Optional[float] = None,
) -> GsqRefineReport:
    """Refine the quantized tensors of a GGUF in place, with GSQ.

    The f16 GGUF holds the original weights; the quantized GGUF holds the
    codes and scales. For every tensor that is stored in a format TOFFEE can
    decode, the codes and per-group scales are re-learned and written back
    into the quantized file. The result is a valid GGUF that llama.cpp loads
    unchanged, with (in expectation) a lower reconstruction error.

    The pass is **streamed by block**: tensors are grouped by their
    ``blk.N.*`` index and one block is processed at a time, so only a single
    layer's weights are resident in RAM. Both files are opened once, so the
    GGUF header is parsed once rather than once per tensor.

    On a large model this is slow -- the routed experts of a 36B MoE are
    33B parameters and take ~22 hours -- so ``time_budget_minutes`` bounds
    the run. The work is checkpointed per block, so a budget-stopped run
    can be resumed and picks up where it left off.

    Args:
        gguf_path: the quantized GGUF to refine (patched in place).
        f16_gguf_path: the f16 GGUF holding the original weights.
        imatrix_path: optional imatrix, used to weight the objective.
        config: GSQ hyper-parameters.
        tensor_types: optional tensor -> quant type map; by default the
            types are read from the quantized GGUF itself.
        max_tensors: optional cap on how many tensors to refine.
        bridge: optional pre-built llama.cpp bridge.
        state_path: optional file to checkpoint progress into, so a long run
            can be resumed.
        resume: when True, skip blocks already recorded in ``state_path``.
        time_budget_minutes: optional wall-clock budget. The pass stops
            cleanly (with the work so far saved) when it is exceeded.

    Returns:
        A :class:`GsqRefineReport` describing what was refined.
    """
    cfg = config or GsqConfig()
    # Resolve the bridge up front: it both validates that llama.cpp is
    # available and provides the reference dequantization.
    br = bridge or GgmlBridge.get()
    report = GsqRefineReport()
    started = time.monotonic()

    imatrix = _load_imatrix(imatrix_path)

    done_blocks = _load_completed_blocks(state_path) if resume else set()

    with GgufFile(gguf_path, writable=True) as qfile, \
            GgufFile(f16_gguf_path) as f16file:
        index = qfile.index
        f16_index = f16file.index

        if tensor_types is None:
            tensor_types = {
                name: info.type_name for name, info in index.items()
            }

        # One block at a time: only a single layer's weights are resident.
        blocks = group_by_block(index.keys())
        # The non-block group (-1: the embeddings and the output head) is
        # processed *last*, so the per-layer blocks -- which are the bulk
        # of the work -- are the ones a time budget or a tensor cap lands
        # on. The key ``(is_negative, id)`` puts the real blocks first
        # (False < True) and the -1 group last.
        ordered_blocks = sorted(
            blocks.items(), key=lambda kv: (kv[0] < 0, kv[0])
        )

        refined = 0
        for block_id, names in ordered_blocks:
            if max_tensors is not None and refined >= max_tensors:
                break
            if block_id in done_blocks:
                continue
            # The wall-clock budget is checked *between* blocks, so a
            # budget-stopped run always leaves a consistent file and a
            # checkpoint that a later run can resume from.
            if _budget_exceeded(started, time_budget_minutes):
                logger.info(
                    "gsq_time_budget_reached",
                    extra={
                        "elapsed_minutes": round(
                            (time.monotonic() - started) / 60.0, 2
                        ),
                        "budget_minutes": time_budget_minutes,
                        "refined": report.tensors_refined,
                    },
                )
                break

            logger.info(
                "gsq_block_start",
                extra={"block": block_id, "tensors": len(names)},
            )
            for name in names:
                if max_tensors is not None and refined >= max_tensors:
                    break
                # The budget is also checked between tensors: a single
                # block of a 36B MoE takes ~17 minutes, so a budget
                # checked only between blocks would overshoot by a huge
                # margin. Checking per tensor bounds the overshoot to one
                # tensor.
                if _budget_exceeded(started, time_budget_minutes):
                    logger.info(
                        "gsq_time_budget_reached",
                        extra={
                            "elapsed_minutes": round(
                                (time.monotonic() - started) / 60.0, 2
                            ),
                            "budget_minutes": time_budget_minutes,
                            "refined": report.tensors_refined,
                            "partial_block": block_id,
                        },
                    )
                    break
                info = index[name]
                if name not in f16_index:
                    report.tensors_skipped += 1
                    continue
                type_name = info.type_name.upper()
                if not is_supported(type_name):
                    report.tensors_skipped += 1
                    continue
                if info.n_rows < 1 or info.n_per_row < 1:
                    report.tensors_skipped += 1
                    continue

                try:
                    weights = f16file.read_f32(name)
                except Exception as exc:  # noqa: BLE001 - skip unreadable tensors
                    logger.warning(
                        "gsq_skip_tensor", extra={"tensor": name, "error": str(exc)}
                    )
                    report.tensors_skipped += 1
                    continue

                f16 = f16_index[name]
                if f16.n_rows != info.n_rows or f16.n_per_row != info.n_per_row:
                    report.tensors_skipped += 1
                    weights = None
                    continue

                weights = weights.reshape(info.n_rows, info.n_per_row)
                im = imatrix.get(name)
                if im is not None and im.shape[0] != info.n_per_row:
                    im = None

                raw_bytes = qfile.read_bytes(name)
                codes, _, _, raws = decode_blocks(raw_bytes, type_name)

                try:
                    result = refine_tensor(
                        weights,
                        type_name,
                        imatrix=im,
                        config=cfg,
                        initial_codes=codes,
                        initial_raw=raws,
                    )
                except Exception as exc:  # noqa: BLE001 - never break the run
                    logger.warning(
                        "gsq_refine_failed", extra={"tensor": name, "error": str(exc)}
                    )
                    report.tensors_skipped += 1
                    weights = None
                    continue
                # Release the source weights before the next tensor.
                weights = None

                if result.final_error < result.initial_error:
                    new_bytes = encode_blocks(result.codes, type_name, result.raw)
                    if len(new_bytes) == len(raw_bytes):
                        # Cross-check the new bytes against llama.cpp's own
                        # decoder, so a codec mismatch can never be written
                        # into the file.
                        if not _agrees_with_llama(
                            br, type_name, new_bytes, info.n_per_row, result
                        ):
                            logger.warning(
                                "gsq_verify_failed", extra={"tensor": name}
                            )
                            report.tensors_skipped += 1
                            continue
                        qfile.write_bytes(name, new_bytes)
                        refined += 1
                        report.tensors_refined += 1
                        report.initial_error += result.initial_error
                        report.final_error += result.final_error
                        report.per_tensor.append(
                            (name, result.initial_error, result.final_error)
                        )
                        continue
                report.tensors_skipped += 1

            # A block interrupted by the time budget is *not* complete: it
            # must stay unmarked so a later run resumes it from the start.
            if _budget_exceeded(started, time_budget_minutes):
                logger.info(
                    "gsq_block_partial",
                    extra={"block": block_id, "refined": report.tensors_refined},
                )
                break

            _save_completed_blocks(state_path, block_id)
            logger.info(
                "gsq_block_complete",
                extra={
                    "block": block_id,
                    "refined": report.tensors_refined,
                    "skipped": report.tensors_skipped,
                },
            )

    logger.info(
        "gsq_refine_complete",
        extra={
            "refined": report.tensors_refined,
            "skipped": report.tensors_skipped,
            "improvement": report.relative_improvement,
        },
    )
    return report


__all__ = [
    "ApexRole",
    "BITS_PER_WEIGHT",
    "GsqRefineReport",
    "RcoResult",
    "bits_for",
    "build_rco_tensor_map",
    "build_role_groups",
    "expand_role_options",
    "refine_gguf_with_gsq",
]
