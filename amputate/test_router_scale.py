"""Regression test: router logit scale vs softmax saturation.

The router's ridge target is a SUM of I/E per-neuron log1p energies.
On a real FFN (I/E = 544, large activations) the raw fit's logits land
in the hundreds-to-thousands and llama.cpp's softmax degenerates to a
hard argmax: top-k weights collapse to [1,0,0,0], the calibration
scales blow past their clamp and the offline canary's router_corr dies
(0.023 mean on the real 27B dense source, 2026-10-31).

The tiny synthetic selftest CANNOT catch this (its target scale is
small), so this test drives ``surgery.fit_router`` directly at two
scales -- synthetic-tiny and real-model-magnitude -- and asserts:

  1. with ``logit_scale`` set, the emitted logits' pooled
     std equals the requested scale at BOTH magnitudes, and the
     softmax stays well off the hard-argmax regime -- checked at the
     two temperatures that matter: 2.0 (pre-v1.0.4 default, kept for
     A/B) and 1.0 (the current default);
  2. the top-k ORDER is untouched by the normalization (a positive
     rescaling never reorders logits);
  3. with ``logit_scale=None`` (legacy) the real-magnitude fit IS
     saturated -- documents the failure mode being guarded against;
  4. the exact-moment std matches a direct computation on held data.

Run:  python -m toffee.amputate.test_router_scale
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from toffee.amputate import surgery

H, I, E, N = 192, 1536, 32, 4000
LOGIT_SCALES = (2.0, 1.0)          # 2.0 = pre-v1.0.4 default (A/B); 1.0 = default


def _make_stats(energy_mult: float, seed: int = 0):
    """Sufficient statistics for a synthetic layer at a given magnitude.

    ``energy_mult`` scales the per-neuron output energy -- 1.0 is
    selftest-tiny, ~30-100 emulates a real FFN's log1p sums.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, H, generator=g)
    x[:, 0] *= 20            # outlier dims, as in real post-LN states
    x[:, 5] *= 9
    x = x / x.pow(2).mean(-1, keepdim=True).sqrt()
    # per-neuron gains: heavy-tailed, and a token-cluster structure so
    # WHICH neurons are hot varies per token (real FFNs; a uniform
    # energy scaling would only shift log1p by a constant)
    n_clusters = 8
    clus = torch.randint(0, n_clusters, (N,), generator=g)
    cdirs = F.normalize(torch.randn(n_clusters, H, generator=g), dim=1)
    gain = torch.exp(1.5 * torch.randn(I, generator=g))          # [I]
    circ = torch.randint(0, I, (256 * n_clusters,), generator=g)  # circuits
    gate_w = torch.randn(I, H, generator=g) * 0.08 * gain[:, None]
    up_w = torch.randn(I, H, generator=g) * 0.08 * gain[:, None]
    for c in range(n_clusters):
        idx = circ[c * 256:(c + 1) * 256]
        gate_w[idx] += 0.5 * cdirs[c][None, :]
        up_w[idx] += 0.5 * cdirs[c][None, :]
    down_w = torch.randn(H, I, generator=g) * 0.02
    dn2 = down_w.pow(2).sum(0)
    act = F.silu(x @ gate_w.T) * (x @ up_w.T)
    y = act.pow(2) * dn2 * energy_mult           # [N, I] output energy

    xb = torch.cat([x, torch.ones(N, 1)], dim=1)
    z = torch.log1p(y)
    XtX = xb.T @ xb
    XtZ = xb.T @ z
    imp = y.sum(0)
    return x, XtX, XtZ, imp


def _pooled_std(x, router_w):
    """Discriminative pooled std: logits centered across experts per
    token (the per-token common mode cancels in softmax), then RMS of
    the per-expert stds."""
    logits = x @ router_w.T
    centered = logits - logits.mean(dim=-1, keepdim=True)
    sd = centered.std(dim=0)
    return sd.mul(sd).mean().sqrt().item()


def _top1_prob(x, router_w):
    return torch.softmax(x @ router_w.T, dim=-1).max(-1).values.mean().item()


def main() -> int:
    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail else ""))
        if not cond:
            failures.append(name)

    for tag, mult in (("selftest-scale", 1.0), ("real-model-scale", 1e12)):
        x, XtX, XtZ, imp = _make_stats(mult)
        partition = surgery.make_partition(imp, E, "balanced")

        # -- legacy fit: no temperature normalization -------------------
        legacy = surgery.fit_router(XtX, XtZ, N, partition, E,
                                    logit_scale=None)
        legacy_std = _pooled_std(x, legacy["router_w"])
        legacy_p1 = _top1_prob(x, legacy["router_w"])

        for LOGIT_SCALE in LOGIT_SCALES:
            stag = f"{tag} s{LOGIT_SCALE:g}"
            # -- fixed fit --------------------------------------------
            fixed = surgery.fit_router(XtX, XtZ, N, partition, E,
                                       logit_scale=LOGIT_SCALE)

            fixed_std = _pooled_std(x, fixed["router_w"])
            fixed_p1 = _top1_prob(x, fixed["router_w"])

            # exact-moment std (from sufficient stats) vs direct
            check(f"{stag}: moment std matches direct",
                  abs(fixed["logit_std_raw"] - legacy_std)
                  <= 1e-3 * max(legacy_std, 1.0),
                  f"moment {fixed['logit_std_raw']:.4g} vs direct {legacy_std:.4g}")

            check(f"{stag}: normalized std == {LOGIT_SCALE}",
                  abs(fixed_std - LOGIT_SCALE) <= 0.05 * LOGIT_SCALE,
                  f"{fixed_std:.4f}")

            # top-k order preserved (positive rescaling)
            kl = (x @ legacy["router_w"].T).topk(4, -1).indices
            kf = (x @ fixed["router_w"].T).topk(4, -1).indices
            check(f"{stag}: top-k order preserved",
                  bool((kl == kf).all()), "")

            if mult == 1.0:
                # tiny scale: legacy may already be healthy; only sanity
                check(f"{stag}: fixed softmax healthy",
                      fixed_p1 <= 0.90, f"p1 {fixed_p1:.3f}")
            else:
                # real magnitude: legacy MUST be saturated (documents
                # the bug), fixed MUST be healthy
                check(f"{stag}: legacy softmax saturated (the bug)",
                      legacy_p1 > 0.90,
                      f"p1 {legacy_p1:.3f} (logit std {legacy_std:.0f})")
                check(f"{stag}: fixed softmax healthy",
                      fixed_p1 <= 0.90, f"p1 {fixed_p1:.3f}")

    print()
    if failures:
        print(f"test_router_scale: {len(failures)} FAILURES: {failures}")
        return 1
    print("test_router_scale: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
