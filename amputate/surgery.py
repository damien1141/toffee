"""
@file toffee/amputate/surgery.py
   The no-training math: partition, router fit, scale calibration.

All functions are pure (stats in -> decision out) so they can be unit
checked against the synthetic model and re-run with different knobs
without touching the 54 GB checkpoint.

Importance convention
---------------------
A neuron's importance is its OUTPUT energy, not its activation
magnitude: ``sum_t act_t[i]^2 * ||W_down[:, i]||^2``. Activation-only
importance (classic MoEfication) would rank loud neurons whose down
column is ~0 -- dead weight we would then faithfully route and
reconstruct. The down-projection column norms are available during
harvest (the layer is resident), so the energy weighting is free.

Router
------
The router is fit by ridge regression from the FFN input x (the
post-attention LayerNorm output) to per-expert output-energy scores.
The sufficient statistics (XtX, XtY) are accumulated streamingly in
sweep A, so the fit is a single (H+1)-dimensional solve per layer --
no token is ever stored. The emitted checkpoint carries no router bias
(the HF/llama.cpp MoE layout has none); the bias is kept in the stats
file for diagnostics. Because the regression target is a SUM of I/E
per-neuron log1p energies, the raw logits are far outside softmax
range on real models; the fit therefore rescales the emitted weights
so the logits have a fixed pooled std (temperature normalization,
see ``fit_router``).

Calibration
-----------
llama.cpp MoE forward is: probs = softmax(router_logits) over ALL
experts, then top-k experts are summed, each weighted by its softmax
probability. That weighting is not what the dense FFN did, so the
reconstruction is re-scaled by per-expert scales c_e and a shared
scale s, solved exactly as a joint (E+1)-dimensional ridge least
squares from co-selection statistics accumulated in sweep B. The
scales are baked into the expert ``down_proj`` columns at build time:
zero runtime cost, zero format extensions.
"""

from __future__ import annotations

from typing import Dict, Tuple

import math

import torch

EPS = 1e-6


def make_partition(
    importance: torch.Tensor, num_experts: int, strategy: str = "balanced"
) -> torch.Tensor:
    """Assign every neuron to exactly one routed expert.

    ``balanced`` deals the importance-sorted neurons round-robin, so
    every expert gets exactly I/E neurons and roughly equal total
    importance (load balance). ``block`` uses contiguous blocks of the
    sorted order (kept for experiments; no load-balance guarantee).

    Returns an int16 tensor of expert ids, shape [I].
    """
    imp = importance.detach().to(torch.float32)
    I = int(imp.numel())
    if I % num_experts != 0:
        raise ValueError(f"I={I} not divisible by E={num_experts}")
    order = torch.argsort(imp, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(I, device=imp.device)
    if strategy == "balanced":
        assignment = ranks % num_experts
    elif strategy == "block":
        assignment = torch.div(ranks, I // num_experts, rounding_mode="floor")
    else:
        raise ValueError(f"unknown partition strategy {strategy!r}")
    return assignment.to(torch.int16)


def expert_index_lists(partition: torch.Tensor) -> list:
    """Per-expert sorted neuron index lists (list of LongTensors)."""
    partition = partition.long()
    return [torch.nonzero(partition == e, as_tuple=False).flatten()
            for e in range(int(partition.max().item()) + 1)]


def shared_indices(importance: torch.Tensor, shared_intermediate: int) -> torch.Tensor:
    """The top-K neurons by importance; they are duplicated into the
    always-on shared expert (and stay in the routed pool too)."""
    imp = importance.detach().to(torch.float32)
    return torch.topk(imp, int(shared_intermediate)).indices


def fit_router(
    XtX: torch.Tensor,        # [(H+1), (H+1)], bias column LAST
    XtZ: torch.Tensor,        # [(H+1), I], z = log1p(per-neuron output energy)
    ntok: int,
    partition: torch.Tensor,
    num_experts: int,
    ridge_alpha: float = 1.0,
    logit_scale: float | None = 2.0,
) -> Dict[str, torch.Tensor]:
    """Ridge-fit the router from streaming sufficient statistics.

    Solves ``(XtX + lambda*I) W = XtZ_e`` for all experts at once,
    where ``XtZ_e`` is the per-expert aggregated ``log1p`` output-energy
    score (the target chosen by experiment; see harvest._SweepA).
    Returns the router weight [E, H] (no bias), the bias row [E]
    (diagnostics only) and the effective lambda.

    Temperature normalization
    -------------------------
    The raw fit's logits live in the target's units: a SUM of I/E
    per-neuron ``log1p`` energies. On a real FFN (I/E = 544, large
    activations) that puts logits in the hundreds-to-thousands, so
    llama.cpp's ``softmax`` degenerates to a hard argmax: the top-k
    weights collapse to ``[1, 0, 0, 0]``-ish, three of the four selected
    experts contribute nothing, the calibration scales blow past their
    clamp, and the router/truth correlation the offline canary measures
    dies (observed: 0.023 mean on the 27B dense source).

    ``logit_scale`` fixes the gauge: the emitted (bias-less) logits are
    rescaled so their DISCRIMINATIVE pooled std -- the std that remains
    after removing the per-token mean across experts, which is all
    softmax ever sees -- equals ``logit_scale`` over the calibration
    tokens. The std is computed EXACTLY from the sufficient statistics
    -- ``var(l_e - l_mean) = wc_e' Cov(x) wc_e`` with ``Cov`` formed
    from ``XtX`` and ``wc`` the expert-centred weight rows -- so no
    token is stored and no extra pass is run. Softmax top-k ORDER is
    untouched (a positive rescaling never reorders logits); only the
    weights (and thus the calibration) are put back in a sane range.
    ``None`` disables the normalization (legacy behaviour, kept for
    A/B).
    """
    dev = XtX.device
    H1 = XtX.shape[0]
    idx_lists = expert_index_lists(partition)
    if len(idx_lists) != num_experts:
        raise ValueError(
            f"partition has {len(idx_lists)} experts, expected {num_experts}"
        )
    targets = torch.stack(
        [XtZ[:, idx].sum(dim=1) for idx in idx_lists], dim=1
    ).to(dev)                                                    # [(H+1), E]

    diag = torch.diagonal(XtX[: H1 - 1, : H1 - 1])
    scale = float(diag.mean().item()) if diag.numel() else 1.0
    lam = max(ridge_alpha * scale, EPS * max(scale, 1.0) * ntok)
    A = XtX.clone()
    A.diagonal().add_(lam)

    try:
        W = torch.linalg.solve(A, targets)                       # [(H+1), E]
    except Exception:
        W = torch.linalg.lstsq(A, targets).solution

    router_w = W[: H1 - 1].T.contiguous()                        # [E, H]
    router_b = W[H1 - 1].contiguous()                            # [E]

    # -- temperature normalization (exact moments from sufficient stats) --
    logit_std_raw = None
    logit_std = None
    if logit_scale is not None and logit_scale > 0:
        # softmax only sees the DISCRIMINATIVE part of the logits: a
        # per-token shift common to all experts cancels. Centering the
        # weight rows across experts gives exactly those logits, so the
        # gauge below is the softmax-relevant std, independent of how
        # much common mode the fit carries.
        W64, XtX64 = W.double(), XtX.double()
        n = max(int(ntok), 1)
        mu = XtX64[:-1, -1] / n                                  # [H] mean of x
        cov = XtX64[:-1, :-1] / n - mu[:, None] * mu[None, :]    # [H, H]
        w64 = router_w.double()
        wc = w64 - w64.mean(dim=0, keepdim=True)                 # centered rows
        var_e = ((wc @ cov) * wc).sum(-1).clamp_min(0.0)         # [E]
        pooled = float(var_e.mean().sqrt().item())               # RMS across experts
        logit_std_raw = pooled
        if pooled > 1e-9:
            mult = float(logit_scale) / pooled
            router_w = (router_w.double() * mult).float()
            router_b = (router_b.double() * mult).float()
            logit_std = float(logit_scale)

    return {"router_w": router_w, "router_b": router_b, "lambda": lam,
            "logit_std_raw": logit_std_raw, "logit_std": logit_std}


def solve_scales(
    Ce: torch.Tensor,   # [E, E] or flat [E*E]: sum_t sum_{e,e' in sel} w w' <d_e, d_e'>
    Cs: torch.Tensor,   # [E]     sum_t sum_{e in sel}   w   <d_e, shared>
    Ss: float,          #         sum_t <shared, shared>
    Cd: torch.Tensor,   # [E]     sum_t sum_{e in sel}   w   <d_e, dense>
    Csh: float,         #         sum_t <shared, dense>
    clamp: float = 5.0,
) -> Tuple[torch.Tensor, float, Dict[str, int]]:
    """Exact joint least-squares for per-expert scales and the shared scale.

    Minimises ``|| sum_e c_e w_e d_e + s * shared - dense ||^2`` over
    (c, s) in closed form from the co-selection statistics, then
    clamps for numerical safety. Returns (c [E], s, clamped counts).
    """
    Ce = Ce.detach().double()
    if Ce.dim() == 1:
        E = int(round(math.sqrt(Ce.numel())))
        Ce = Ce.reshape(E, E)
    else:
        E = Ce.shape[0]
    A = torch.zeros(E + 1, E + 1, dtype=torch.float64)
    A[:E, :E] = Ce.double()
    A[:E, E] = Cs.double()
    A[E, :E] = Cs.double()
    A[E, E] = Ss if torch.is_tensor(Ss) else torch.tensor(Ss, dtype=torch.float64)
    b = torch.zeros(E + 1, dtype=torch.float64)
    b[:E] = Cd.double()
    b[E] = Csh if torch.is_tensor(Csh) else float(Csh)

    ridge = max(EPS * float(A.diagonal().abs().max().item() + 1.0), 1e-9)
    A.diagonal().add_(ridge)
    try:
        x = torch.linalg.solve(A, b)
    except Exception:
        x = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)

    c = x[:E].clamp(min=0.0, max=clamp).float()
    s = float(x[E].item())
    s = max(0.0, min(s, clamp))
    n_clamped_c = int(((x[:E] < 0.0) | (x[:E] > clamp)).sum().item())
    return c, s, {"clamped_c": n_clamped_c, "clamped_s": int(s != x[E].item())}


def posthoc_cosine(
    Ce: torch.Tensor, Cs: torch.Tensor, Ss: float,
    Cd: torch.Tensor, Csh: float, dnorm2: float,
    c: torch.Tensor, s: float,
) -> float:
    """cos(reconstruction, dense FFN output) from co-selection statistics.

    ``<recon, dense>  = sum_e c_e Cd[e] + s * Csh``
    ``||recon||^2     = c' Ce c + 2 s Cs' c + s^2 Ss``
    No token is stored; this is exact for the linear-in-scales model.
    """
    Ce = Ce.detach().double()
    if Ce.dim() == 1:
        Ce = Ce.reshape(int(round(math.sqrt(Ce.numel()))), -1)
    c64 = c.double()
    num = float((c64 @ Cd.double()).item()) + s * float(Csh)
    ce = float((c64 @ Ce.double() @ c64).item())
    cross = 2.0 * s * float((Cs.double() @ c64).item())
    den = ce + cross + s * s * float(Ss)
    dn = float(dnorm2)
    if den <= 0.0 or dn <= 0.0:
        return 0.0
    return num / max(float(torch.sqrt(torch.tensor(den * dn)).item()), EPS)


def pearson_from_moments(
    sum_a: torch.Tensor, sum_a2: torch.Tensor,
    sum_b: torch.Tensor, sum_b2: torch.Tensor,
    sum_ab: torch.Tensor, n: int,
) -> torch.Tensor:
    """Per-expert Pearson correlation from streaming moments, [E]."""
    n = max(n, 1)
    sa, sa2 = sum_a.double(), sum_a2.double()
    sb, sb2 = sum_b.double(), sum_b2.double()
    sab = sum_ab.double()
    cov = sab - sa * sb / n
    va = sa2 - sa * sa / n
    vb = sb2 - sb * sb / n
    denom = torch.sqrt((va * vb).clamp_min(0.0))
    corr = torch.where(denom > 0, cov / denom.clamp_min(EPS), torch.zeros_like(cov))
    return corr.float()
