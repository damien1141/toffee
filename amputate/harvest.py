"""
@file toffee/amputate/harvest.py
   Two-sweep, layer-streaming activation harvest for the amputation.

Same skeleton as ``toffee.quant.passes.pass2_harvest`` (stream one
layer's weights into VRAM at a time, run disk-cached batches through
it, propagate hidden states) with two changes:

Sweep A collects, per layer, the sufficient statistics for the whole
surgery in ONE pass over the corpus:

  * per-neuron output-energy importance ``sum_t act_t^2 * ||W_down[:,i]||^2``
  * router sufficient statistics ``XtX`` ((H+1)^2, bias column last)
    and ``XtY`` ((H+1) x I against per-neuron energy)

Sweep B re-runs the same batches through the SAME layer weights with
the just-fitted partition + router, and accumulates the calibration
co-selection statistics (exact joint least-squares data for per-expert
scales), the selection-quality metrics (router correlation, top-k
Jaccard vs true importance, mass recall vs oracle) and the
reconstruction cosine -- all streaming, all without storing a token.

The only per-layer state that survives is a few MB of statistics, so a
64-layer 27B harvest runs in the same VRAM envelope as pass2 and
writes ``surgery_stats.pt`` (re-written after every layer, so a crash
loses at most one layer of hidden-state propagation).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from toffee.amputate.preset import AmputationPreset
from toffee.amputate import surgery
from toffee.quant.config import (
    get_hidden_size,
    get_intermediate_size,
    get_layer_types,
    get_num_hidden_layers,
    load_config_with_fallback,
)
from toffee.quant.masks import build_4d_causal_mask, layer_needs_rope, layer_wants_2d_mask

logger = logging.getLogger(__name__)

def _log(logger, level, event, **kw):
    """structlog-style event logging on a stdlib logger."""
    if kw:
        rendered = " ".join(f"{k}={v}" for k, v in kw.items())
        getattr(logger, level)(f"{event} {rendered}")
    else:
        getattr(logger, level)(event)


# ---------------------------------------------------------------------------
# device-aware weight injection (adapted from toffee.quant.passes)
# ---------------------------------------------------------------------------

def _inject_weights(
    module, weights_dict: Dict[str, torch.Tensor], prefix: str, device: str
) -> Tuple[int, List[str]]:
    """Inject checkpoint weights into a module on ``device``.

    Adapted from ``toffee.quant.passes._inject_weights`` (same key
    matching), parameterised by device so the harvest also runs on CPU
    (synthetic selftest, machines without CUDA).
    """
    assigned = 0
    missing = []
    available_keys = list(weights_dict.keys())

    def find_weight(full_key: str, name: str) -> Optional[torch.Tensor]:
        if full_key in weights_dict:
            return weights_dict[full_key]
        for k in available_keys:
            if k.endswith(full_key):
                return weights_dict[k]
            if k == name:
                return weights_dict[k]
        return None

    for name, param in module.named_parameters(recurse=True):
        full_key = prefix + name
        tensor = find_weight(full_key, name)
        if tensor is not None:
            if tensor.dtype != param.dtype:
                tensor = tensor.to(param.dtype)
            if tensor.shape != param.shape and tensor.numel() == param.numel():
                tensor = tensor.reshape(param.shape)
            param.data = tensor.to(device)
            assigned += 1
        else:
            # Missing norms: weight -> 1, bias -> 0 (a bias filled with 1
            # would silently distort every activation; pass2's original
            # filled both with 1).
            if name.endswith(".bias") or name == "bias":
                param.data.fill_(0.0)
            elif "norm" in name or "ln" in name or "layernorm" in name:
                param.data.fill_(1.0)
            else:
                param.data.fill_(0.0)
            missing.append(full_key)

    for name, buf in module.named_buffers(recurse=True):
        full_key = prefix + name
        tensor = find_weight(full_key, name)
        if tensor is not None:
            if tensor.dtype != buf.dtype:
                tensor = tensor.to(buf.dtype)
            buf.data = tensor.to(device)
            assigned += 1
        else:
            if name.endswith(".bias") or name == "bias":
                buf.data.fill_(0.0)
            elif "norm" in name or "ln" in name or "layernorm" in name:
                buf.data.fill_(1.0)
            else:
                buf.data.fill_(0.0)

    return assigned, missing


def _to_meta(module) -> None:
    """Release a module's weights (mirror of pass2's _apply meta trick)."""
    module._apply(lambda t: torch.empty_like(t, device="meta"))


# ---------------------------------------------------------------------------
# dense-source enforcement + model construction
#   (mirrors pass2: real transformers, minimal fallback -- but the
#   source is pinned DENSE: config sanitization, tensor-truth check,
#   and a structural assertion on the built model)
# ---------------------------------------------------------------------------

# Family-level MoE fields that must describe the OUTPUT of the
# amputation, never its INPUT. They may appear on a dense checkpoint's
# config object through:
#   * transformers config class __init__ defaults (Qwen3.5 is a
#     hybrid-MoE family; a dense config.json that omits num_experts
#     still gets one filled in),
#   * CLASS attributes of the local qwen4_exp_text fallback config
#     (num_experts=180, moe_intermediate_size=512), which plain
#     ``getattr`` sees even though the instance never set them,
#   * leftover keys in config.json itself.
_MOE_CONFIG_FIELDS = (
    "num_experts", "n_routed_experts", "num_local_experts",
    "moe_intermediate_size", "shared_expert_intermediate_size",
)


def _sanitize_dense_config(config: Any) -> None:
    """Force the config object to describe a DENSE FFN.

    Sets the MoE fields to ``None`` at INSTANCE level (an instance
    ``None`` shadows any class-level default), so no downstream
    ``getattr`` can hallucinate experts for a dense checkpoint. Only
    call this once the checkpoint's tensors have been verified dense
    (``_assert_dense_checkpoint``) -- which is exactly what harvest
    does, since amputation is defined on dense sources.
    """
    targets = []
    tc = getattr(config, "text_config", None)
    if isinstance(tc, dict):
        for f in _MOE_CONFIG_FIELDS:
            tc.pop(f, None)
    elif tc is not None and tc is not config:
        targets.append(tc)
    targets.append(config)
    for t in targets:
        for f in _MOE_CONFIG_FIELDS:
            if getattr(t, f, None) is not None:
                _log(logger, "info", "amputate_config_moe_field_cleared",
                     field=f, was=getattr(t, f))
            try:
                setattr(t, f, None)
            except Exception:
                pass


def _assert_dense_checkpoint(weight_map: Dict[str, str],
                             num_layers: int) -> None:
    """Checkpoint tensor names are the ground truth for FFN layout.

    The config object must not decide this (see ``_sanitize_dense_config``):
    a dense checkpoint whose config looks MoE through leaked defaults is
    still dense, and an already-amputated checkpoint must be rejected.
    """
    moe_tensors = sorted(k for k in weight_map
                         if ".mlp.experts." in k
                         or k.endswith(".mlp.gate.weight"))
    if moe_tensors:
        raise RuntimeError(
            "checkpoint is already MoE (found e.g. "
            f"{moe_tensors[0]}); `toffee amputate` converts a DENSE "
            "hybrid checkpoint, it cannot re-amputate an MoE")
    missing = []
    for i in range(num_layers):
        have = {k.rsplit(".", 2)[-2] for k in weight_map
                if f"layers.{i}.mlp." in k}
        if not {"gate_proj", "up_proj", "down_proj"} <= have:
            missing.append(i)
    if missing:
        raise RuntimeError(
            f"layers {missing} carry no dense FFN tensors "
            "(mlp.gate/up/down_proj); amputate harvest requires a dense "
            "hybrid checkpoint")


def _ffn_is_dense(layer: Any) -> bool:
    """True if the built layer's MLP is a dense gate/up/down FFN."""
    names = [n for n, _ in layer.named_modules()]
    return (any(n.endswith("mlp.gate_proj") for n in names)
            and not any(".mlp.experts." in n or n.endswith("mlp.gate")
                        for n in names))


def _build_text_model(config: Any, model_impl: str = "auto"):
    """Build the text model hierarchy for hooking.

    Returns ``(shim, used_real: bool, text_cfg)``. ``model_impl``:
    ``real`` forces the transformers classes, ``minimal`` forces the
    local stand-in, ``auto`` tries real then falls back.

    The source is DENSE by definition: ``_sanitize_dense_config`` runs
    first so family-level MoE defaults (transformers class defaults or
    the qwen4_exp_text fallback's class attributes) cannot turn the
    harvest model into an MoE, and the built model is structurally
    verified -- a real-impl build that still comes out MoE falls back
    to the minimal model instead of failing at injection time.
    """
    from accelerate import init_empty_weights
    from toffee.quant.hybrid_model import MinimalHybridTextModel, TextModelShim

    model_type = getattr(config, "model_type", None)
    text_cfg = getattr(config, "text_config", None) or config

    if model_type in ("qwen3_5_moe", "qwen3_5_moe_text"):
        raise ValueError(
            "amputate harvest expects a DENSE hybrid checkpoint; "
            f"{model_type} is already MoE"
        )

    _sanitize_dense_config(config)

    used_real = False
    if model_impl in ("auto", "real") and model_type in ("qwen3_5", "qwen3_5_text"):
        try:
            from transformers.models.qwen3_5 import Qwen3_5TextConfig, Qwen3_5TextModel

            cfg = text_cfg if isinstance(text_cfg, Qwen3_5TextConfig) \
                else Qwen3_5TextConfig.from_dict(text_cfg.to_dict())
            with init_empty_weights():
                text = Qwen3_5TextModel(cfg)
            if not _ffn_is_dense(text.layers[0]):
                raise RuntimeError(
                    "transformers built an MoE FFN despite the sanitized "
                    "dense config (family defaults)")
            used_real = True
        except Exception as exc:
            if model_impl == "real":
                raise
            _log(logger, "info", "amputate_minimal_fallback_build", reason=str(exc)[:200])
    if not used_real:
        with init_empty_weights():
            text = MinimalHybridTextModel(text_cfg)
        if not _ffn_is_dense(text.layers[0]):
            raise RuntimeError(
                "minimal hybrid model built an MoE FFN for a dense "
                "checkpoint -- config sanitization failed; this is a "
                "toffee bug, please report it")
    return TextModelShim(text), used_real, text_cfg


# ---------------------------------------------------------------------------
# sweep hooks
# ---------------------------------------------------------------------------

class _SweepA:
    """Forward hooks collecting importance + router statistics.

    Router target: per-neuron ``log1p`` output energy, aggregated per
    expert at fit time. Chosen by direct experiment on the synthetic
    model (scripts/router_target_exp.py): vs raw energy it lifts top-4
    Jaccard 0.13->0.23 and selection mass 0.886->0.906, and unlike the
    log1p-of-aggregate / indicator targets it stays computable from
    per-neuron streaming statistics.
    """

    def __init__(self, hidden: int, inter: int, device: str, down_w: torch.Tensor):
        self.H, self.I, self.dev = hidden, inter, device
        # Per-neuron down-column energy: importance is OUTPUT energy.
        self.dn2 = down_w.detach().float().pow(2).sum(dim=0).to(device)   # [I]
        self.imp = torch.zeros(inter, dtype=torch.float64, device=device)
        self.XtX = torch.zeros(hidden + 1, hidden + 1, dtype=torch.float64,
                               device=device)
        self.XtZ = torch.zeros(hidden + 1, inter, dtype=torch.float64, device=device)
        self.ntok = 0
        self._x = None
        self._g = None

    # -- hook plumbing ------------------------------------------------------

    def pre_gate(self, module, inputs):
        if inputs and isinstance(inputs[0], torch.Tensor):
            self._x = inputs[0]

    def post_gate(self, module, args, output):
        if isinstance(output, torch.Tensor):
            self._g = output

    def post_up(self, module, args, output):
        x, g, u = self._x, self._g, output
        self._x = self._g = None
        if x is None or g is None or not isinstance(u, torch.Tensor):
            return
        if x.shape[:-1] != u.shape[:-1]:
            return
        act = F.silu(g.float()) * u.float()
        act = torch.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)
        y = act.pow(2) * self.dn2                                    # output energy
        xb = x.float().reshape(-1, self.H)
        xb = torch.cat([xb, torch.ones(xb.shape[0], 1, device=xb.device)], dim=1)
        yb = y.reshape(-1, self.I)
        self.XtX += (xb.T @ xb).double()
        self.XtZ += (xb.T @ torch.log1p(yb)).double()
        self.imp += yb.sum(dim=0).double()
        self.ntok += int(xb.shape[0])

    def register(self, gate_mod, up_mod):
        self._h1 = gate_mod.register_forward_pre_hook(self.pre_gate)
        self._h2 = gate_mod.register_forward_hook(self.post_gate)
        self._h3 = up_mod.register_forward_hook(self.post_up)

    def unregister(self):
        for attr in ("_h1", "_h2", "_h3"):
            h = getattr(self, attr, None)
            if h is not None:
                h.remove()
                setattr(self, attr, None)

    def snapshot(self):
        return (self.imp.clone(), self.XtX.clone(), self.XtZ.clone(), self.ntok)

    def restore(self, snap):
        self.imp, self.XtX, self.XtZ, self.ntok = (
            snap[0].clone(), snap[1].clone(), snap[2].clone(), snap[3])


class _SweepB:
    """Forward hooks collecting calibration + verification statistics.

    Replicates llama.cpp MoE semantics exactly: softmax over ALL
    experts, top-k selected, each weighted by its softmax probability.
    """

    def __init__(self, hidden, inter, device, down_w, partition, shared_idx,
                 router_w, num_experts, experts_per_tok):
        self.H, self.I, self.dev = hidden, inter, device
        self.E, self.K = num_experts, experts_per_tok
        self.Wd = down_w.detach().float().to(device)                   # [H, I]
        self.dn2 = self.Wd.pow(2).sum(dim=0)                           # [I]
        self.idx_lists = surgery.expert_index_lists(partition)
        self.shared_idx = shared_idx.to(device).long()
        self.router_w = router_w.detach().float().to(device)           # [E, H]
        # calibration accumulators (float64 on CPU for determinism);
        # Ce is kept FLAT [E*E] so co-selection pairs scatter-add cleanly
        self.Ce = torch.zeros(self.E * self.E, dtype=torch.float64)
        self.Cs = torch.zeros(self.E, dtype=torch.float64)
        self.Cd = torch.zeros(self.E, dtype=torch.float64)
        self.Ss = 0.0
        self.Csh = 0.0
        self.dnorm2 = 0.0
        # verification accumulators
        self.mass_total = 0.0
        self.mass_ours = 0.0
        self.mass_oracle = 0.0
        self.budget = self.K * (self.I // self.E) + int(self.shared_idx.numel())
        # router-vs-truth moments
        self.sp = torch.zeros(self.E, dtype=torch.float64)
        self.sp2 = torch.zeros(self.E, dtype=torch.float64)
        self.ss = torch.zeros(self.E, dtype=torch.float64)
        self.ss2 = torch.zeros(self.E, dtype=torch.float64)
        self.sps = torch.zeros(self.E, dtype=torch.float64)
        # same moments in LOGIT space: Pearson(z, s) is invariant to the
        # softmax nonlinearity (and to --router-logit-scale), so it is
        # the attenuation-free measure of "can the router see expert
        # importance". prob-space corr (the gated metric) is a softmax-
        # attenuated lower bound on it.
        self.sz = torch.zeros(self.E, dtype=torch.float64)
        self.sz2 = torch.zeros(self.E, dtype=torch.float64)
        self.szs = torch.zeros(self.E, dtype=torch.float64)
        self.jacc_sum = 0.0
        # softmax saturation: mean top-1 probability (1.0 = hard argmax,
        # the degenerate regime the temperature normalization prevents)
        self.p1_sum = 0.0
        self.ntok = 0
        self._x = None
        self._g = None
        # neuron masks for mass accounting
        rm = torch.zeros(self.E, self.I, device=device)
        rm[partition.long().to(device), torch.arange(self.I, device=device)] = 1.0
        self.routed_mask = rm
        sm = torch.zeros(self.I, device=device)
        sm[self.shared_idx] = 1.0
        self.shared_mask = sm

    def _stats(self, x, g, u):
        act = F.silu(g.float()) * u.float()
        act = torch.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)
        act = act.reshape(-1, self.I)                              # [N, I]
        x = x.float().reshape(-1, self.H)                          # [N, H]
        y = act.pow(2) * self.dn2                                      # [N, I]
        N = y.shape[0]
        dense_out = act @ self.Wd.T                                    # [N, H]
        shared_out = act[:, self.shared_idx] @ self.Wd[:, self.shared_idx].T
        logits = x @ self.router_w.T                                   # [N, E]
        probs = torch.softmax(logits.float(), dim=-1)
        topv, topi = probs.topk(self.K, dim=-1)                        # [N, K]

        # selected expert outputs, gathered as [N, K, H]
        D = torch.zeros(N, self.K, self.H, device=y.device)
        pos_of_e = torch.full((N, self.E), self.K, dtype=torch.long, device=y.device)
        pos_of_e.scatter_(1, topi, torch.arange(self.K, device=y.device)[None, :].expand(N, -1))
        for e in range(self.E):
            tok = (pos_of_e[:, e] < self.K)
            if not bool(tok.any()):
                continue
            cols = self.idx_lists[e].to(y.device)
            d_e = act[tok][:, cols] @ self.Wd[:, cols].T               # [n_e, H]
            D[tok, pos_of_e[tok, e]] = d_e

        Dw = D * topv[..., None]                                       # [N, K, H]
        # joint-calibration statistics
        gram = torch.einsum("nkh,nmh->nkm", Dw, Dw)                    # [N, K, K]
        pair = (topi[:, :, None] * self.E + topi[:, None, :]).reshape(-1)
        self.Ce.index_add_(0, pair.cpu(), gram.reshape(-1).double().cpu())
        cd_k = torch.einsum("nkh,nh->nk", Dw, dense_out)
        self.Cd.index_add_(0, topi.reshape(-1).cpu(), cd_k.reshape(-1).double().cpu())
        cs_k = torch.einsum("nkh,nh->nk", Dw, shared_out)
        self.Cs.index_add_(0, topi.reshape(-1).cpu(), cs_k.reshape(-1).double().cpu())
        self.Ss += float(shared_out.pow(2).sum().item())
        self.Csh += float((shared_out * dense_out).sum().item())
        self.dnorm2 += float(dense_out.pow(2).sum().item())

        # selection quality
        onehot = torch.zeros(N, self.E, device=y.device)
        onehot.scatter_(1, topi, 1.0)
        nmask = ((onehot @ self.routed_mask) > 0) | (self.shared_mask[None] > 0)
        self.mass_total += float(y.sum().item())
        self.mass_ours += float((y * nmask).sum().item())
        self.mass_oracle += float(y.topk(min(self.budget, self.I), dim=-1)
                                  .values.sum().item())
        # Correlate in the router's own target space (log1p energy): a
        # log-space predictor is unfairly punished by raw-energy tails.
        true_scores = torch.log1p(y @ self.routed_mask.T)              # [N, E]
        self.p1_sum += float(probs.max(dim=-1).values.sum().item())
        self.sp += probs.double().sum(dim=0).cpu()
        self.sp2 += probs.double().pow(2).sum(dim=0).cpu()
        self.ss += true_scores.double().sum(dim=0).cpu()
        self.ss2 += true_scores.double().pow(2).sum(dim=0).cpu()
        self.sps += (probs.double() * true_scores.double()).sum(dim=0).cpu()
        z64 = logits.double()
        self.sz += z64.sum(dim=0).cpu()
        self.sz2 += z64.pow(2).sum(dim=0).cpu()
        self.szs += (z64 * true_scores.double()).sum(dim=0).cpu()
        ttop = true_scores.topk(self.K, dim=-1).indices
        a = torch.zeros_like(onehot).scatter_(1, topi, 1.0)
        b = torch.zeros_like(onehot).scatter_(1, ttop, 1.0)
        inter = (a * b).sum(dim=-1)
        self.jacc_sum += float((inter / (2 * self.K - inter).clamp_min(1)).sum().item())
        self.ntok += N

    def pre_gate(self, module, inputs):
        if inputs and isinstance(inputs[0], torch.Tensor):
            self._x = inputs[0]

    def post_gate(self, module, args, output):
        if isinstance(output, torch.Tensor):
            self._g = output

    def post_up(self, module, args, output):
        x, g, u = self._x, self._g, output
        self._x = self._g = None
        if x is None or g is None or not isinstance(u, torch.Tensor):
            return
        if x.shape[:-1] != u.shape[:-1]:
            return
        self._stats(x, g, u)

    def register(self, gate_mod, up_mod):
        self._h1 = gate_mod.register_forward_pre_hook(self.pre_gate)
        self._h2 = gate_mod.register_forward_hook(self.post_gate)
        self._h3 = up_mod.register_forward_hook(self.post_up)

    def unregister(self):
        for h in (getattr(self, "_h1", None), getattr(self, "_h2", None),
                  getattr(self, "_h3", None)):
            if h is not None:
                h.remove()
        self._h1 = self._h2 = self._h3 = None

    def snapshot(self):
        return tuple(t.clone() for t in (self.Ce, self.Cs, self.Cd)) + (
            self.Ss, self.Csh, self.dnorm2, self.mass_total, self.mass_ours,
            self.mass_oracle, self.jacc_sum, self.p1_sum, self.ntok,
            self.sp.clone(), self.sp2.clone(), self.ss.clone(), self.ss2.clone(),
            self.sps.clone(), self.sz.clone(), self.sz2.clone(),
            self.szs.clone())

    def restore(self, snap):
        (self.Ce, self.Cs, self.Cd, self.Ss, self.Csh, self.dnorm2,
         self.mass_total, self.mass_ours, self.mass_oracle, self.jacc_sum,
         self.p1_sum, self.ntok, self.sp, self.sp2, self.ss, self.ss2,
         self.sps, self.sz, self.sz2, self.szs) = snap


# ---------------------------------------------------------------------------
# the harvest
# ---------------------------------------------------------------------------

def harvest(
    model_name: str,
    batches: List[Any],
    output_dir: Path,
    preset: AmputationPreset,
    device: str = "auto",
    model_impl: str = "auto",
    vram_limit_gb: Optional[float] = None,
) -> Path:
    """Run the two-sweep harvest; write ``surgery_stats.pt``.

    Args:
        model_name: path to the unpacked dense checkpoint directory.
        batches: ``BatchRecord`` list (same objects pass2 uses).
        output_dir: directory for ``surgery_stats.pt``.
        preset: amputation preset (experts / top-k / shared width).
        device: ``auto`` (cuda if available), ``cuda`` or ``cpu``.
        model_impl: ``auto``, ``real`` (transformers) or ``minimal``.
        vram_limit_gb: optional allocator cap, as in pass2.
    """
    from safetensors.torch import load_file as safetensors_load_file

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "surgery_stats.pt"

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        _log(logger, "warning", "amputate_cuda_unavailable", fallback="cpu")
    if device == "cuda" and vram_limit_gb:
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        try:
            torch.cuda.set_per_process_memory_fraction(min(1.0, vram_limit_gb / total))
        except Exception:
            pass

    config = load_config_with_fallback(model_name)
    _sanitize_dense_config(config)   # dense source: no MoE field may leak in
    num_layers = get_num_hidden_layers(config)
    hidden = get_hidden_size(config)
    inter = get_intermediate_size(config)
    preset.validate(inter)

    index_path = Path(model_name) / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    # Tensor names are the ground truth: refuse MoE inputs and pin the
    # dense layout before anything is built or streamed.
    _assert_dense_checkpoint(weight_map, num_layers)

    _log(logger, "info", "amputate_harvest_start", model=model_name, layers=num_layers, hidden=hidden, inter=inter, device=device, experts=preset.num_experts, topk=preset.experts_per_tok, shared=preset.shared_intermediate)
    model, used_real, _ = _build_text_model(config, model_impl)
    _log(logger, "info", "amputate_model_impl", real=used_real, minimal=not used_real)
    layer_types = get_layer_types(config)
    muse_layer_is_sliding = (
        [(lt == "sliding_attention") for lt in layer_types] if layer_types else []
    )
    layer_rope_theta = getattr(config, "layer_rope_theta", None)
    if layer_rope_theta is None:
        layer_rope_theta = getattr(getattr(config, "text_config", None),
                                   "layer_rope_theta", None)

    def _load_files(files, prefix_filter=None):
        out = {}
        for fn in files:
            w = safetensors_load_file(Path(model_name) / fn, device="cpu")
            for k, v in w.items():
                if prefix_filter is None or k.startswith(prefix_filter):
                    out[k] = v
            del w
        return out

    # -- embeddings phase (mirrors pass2) -----------------------------------
    embed_keys = [k for k in weight_map if k.endswith("embed_tokens.weight")]
    embed_weights = {}
    for fn in set(weight_map[k] for k in embed_keys):
        w = safetensors_load_file(Path(model_name) / fn, device="cpu")
        for k, v in w.items():
            if k.endswith("embed_tokens.weight"):
                embed_weights[k] = v
        del w
    embed_module = getattr(getattr(model, "model", model), "embed_tokens", None)
    if embed_module is None:
        raise AttributeError("no embed_tokens module found for harvest injection")
    embed_module.to_empty(device=device)
    prefix = ""
    if embed_weights:
        first = next(iter(embed_weights))
        prefix = first[: first.rfind(".") + 1]
    assigned, missing = _inject_weights(embed_module, embed_weights, prefix, device)
    if missing:
        _log(logger, "warning", "amputate_embed_missing_keys", keys=missing[:5])
    del embed_weights

    rotary = None
    for name, module in model.named_modules():
        if "RotaryEmbedding" in type(module).__name__:
            rotary = module
            break
    # Real rotary modules sometimes live on the outer text model only.
    if rotary is None and used_real:
        for name, module in model.model.named_modules():
            if "RotaryEmbedding" in type(module).__name__:
                rotary = module
                break

    batch_hidden = []
    for batch in batches:
        input_ids = batch.input_ids.unsqueeze(0).to(device)
        attn2d = batch.attention_mask.unsqueeze(0).to(device)
        T = input_ids.shape[1]
        pos = torch.arange(T, device=device).unsqueeze(0).expand_as(input_ids)
        with torch.no_grad():
            embeds = embed_module(input_ids)
            cos_sin = None
            if rotary is not None:
                try:
                    cos, sin = rotary(embeds, pos)
                    cos_sin = (cos.to(embeds.dtype), sin.to(embeds.dtype))
                except Exception:
                    cos_sin = None
        batch_hidden.append((embeds.cpu(), attn2d.cpu(), pos.cpu(), cos_sin))
    _to_meta(embed_module)
    if device == "cuda":
        torch.cuda.empty_cache()

    # -- per-layer two-sweep loop -------------------------------------------
    stats: Dict[str, Any] = {
        "preset": {
            "name": preset.name, "num_experts": preset.num_experts,
            "experts_per_tok": preset.experts_per_tok,
            "shared_intermediate": preset.shared_intermediate,
            "strategy": preset.strategy, "ridge_alpha": preset.ridge_alpha,
            "router_logit_scale": preset.router_logit_scale,
        },
        "meta": {
            "model": str(model_name), "num_layers": num_layers,
            "hidden": hidden, "inter": inter, "device": device,
            "used_real_transformers": used_real, "num_batches": len(batches),
        },
        "layers": {},
    }

    for layer_idx in range(num_layers):
        _log(logger, "info", "amputate_layer_start", layer=layer_idx, total=num_layers)
        layer_module = model.model.layers[layer_idx]
        layer_prefix = None
        for k in weight_map:
            if f"layers.{layer_idx}." in k:
                layer_prefix = k.split(f"layers.{layer_idx}.")[0] + f"layers.{layer_idx}."
                break
        if layer_prefix is None:
            raise KeyError(f"no checkpoint tensors found for layer {layer_idx}")
        layer_weights = _load_files(
            set(v for k, v in weight_map.items() if k.startswith(layer_prefix)),
            layer_prefix,
        )
        layer_module.to_empty(device=device)
        assigned, missing = _inject_weights(layer_module, layer_weights,
                                            layer_prefix, device)
        if missing:
            _log(logger, "warning", "amputate_layer_missing_keys", layer=layer_idx, keys=missing[:8])
        del layer_weights

        # locate the dense FFN linears
        gate_mod = up_mod = down_mod = None
        for name, module in layer_module.named_modules():
            if name.endswith("mlp.gate_proj"):
                gate_mod = module
            elif name.endswith("mlp.up_proj"):
                up_mod = module
            elif name.endswith("mlp.down_proj"):
                down_mod = module
        if gate_mod is None or up_mod is None or down_mod is None:
            raise RuntimeError(
                f"layer {layer_idx}: dense FFN (mlp.gate/up/down_proj) not found; "
                "amputate harvest requires a dense hybrid checkpoint"
            )
        down_w = down_mod.weight.detach()                            # [H, I]

        def _run_batch(hidden_cpu, attn2d_cpu, pos_cpu, cos_sin):
            """One layer forward; returns the new hidden states on CPU."""
            gpu_hidden = hidden_cpu.to(device)
            pos = pos_cpu.to(device)
            gpu_attn = attn2d_cpu.to(device)
            use_2d = layer_wants_2d_mask(layer_module)
            if use_2d:
                mask = gpu_attn
            else:
                win = None
                if muse_layer_is_sliding and layer_idx < len(muse_layer_is_sliding) \
                        and muse_layer_is_sliding[layer_idx]:
                    win = getattr(config, "sliding_window", None) or getattr(
                        getattr(config, "text_config", None), "sliding_window", None)
                mask = build_4d_causal_mask(gpu_attn, gpu_hidden.dtype,
                                            gpu_hidden.device, sliding_window=win)
            kwargs = {"attention_mask": mask, "position_ids": pos,
                      "position_embeddings": None}
            if cos_sin is not None and layer_needs_rope(
                    layer_module, layer_rope_theta, layer_idx):
                cos, sin = cos_sin
                kwargs["position_embeddings"] = (cos.to(device), sin.to(device))
            with torch.no_grad():
                out = layer_module(gpu_hidden, **kwargs)
            new_hidden = out[0] if isinstance(out, tuple) else out
            return new_hidden.cpu()

        def _forward_sweep(sweep, i, halves=None):
            """Run batch i through the layer; hooks live on the modules.

            On CUDA OOM the stats snapshot is restored and the batch is
            split in half along tokens (GDN state continuity across the
            split boundary is not preserved -- acceptable for statistics,
            and pass2 has the same caveat). Split halves' outputs are
            concatenated so hidden-state propagation stays correct.
            """
            hidden_cpu, attn2d_cpu, pos_cpu, cos_sin = pre_layer[i]
            snap = sweep.snapshot() if sweep is not None else None
            lo = 0 if halves is None else halves[0]
            hi = hidden_cpu.shape[1] if halves is None else halves[1]
            h = hidden_cpu[:, lo:hi]
            a = attn2d_cpu[:, lo:hi]
            p = pos_cpu[:, lo:hi]
            cs = None
            if cos_sin is not None:
                c, s = cos_sin
                cs = (c[:, lo:hi] if c.dim() > 2 else c[lo:hi], s)
            try:
                return _run_batch(h, a, p, cs)
            except torch.cuda.OutOfMemoryError:
                if sweep is not None and snap is not None:
                    sweep.restore(snap)
                torch.cuda.empty_cache()
                mid = (lo + hi) // 2
                if mid <= lo:
                    raise
                _log(logger, "warning", "amputate_oom_split", layer=layer_idx, tokens=[lo, hi])
                out_lo = _forward_sweep(sweep, i, (lo, mid))
                out_hi = _forward_sweep(sweep, i, (mid, hi))
                return torch.cat([out_lo, out_hi], dim=1)

        # pre-layer hidden states (identical input for both sweeps)
        pre_layer = [(h.clone(), a, p, cs) for (h, a, p, cs) in batch_hidden]

        # ---- sweep A: importance + router statistics ----
        # The hooked forward IS the propagation forward: its output
        # becomes the input hidden states for the next layer.
        sweepA = _SweepA(hidden, inter, device, down_w)
        sweepA.register(gate_mod, up_mod)
        try:
            for i in range(len(batches)):
                out = _forward_sweep(sweepA, i)
                batch_hidden[i] = (out, batch_hidden[i][1], batch_hidden[i][2],
                                   batch_hidden[i][3])
        finally:
            sweepA.unregister()

        # partition + router fit from sweep A
        imp = sweepA.imp.float().cpu()
        partition = surgery.make_partition(imp, preset.num_experts,
                                           preset.strategy)
        shared_idx = surgery.shared_indices(imp, preset.shared_intermediate)
        fit = surgery.fit_router(sweepA.XtX.float(), sweepA.XtZ.float(),
                                 sweepA.ntok, partition, preset.num_experts,
                                 preset.ridge_alpha,
                                 logit_scale=preset.router_logit_scale)
        del sweepA
        if device == "cuda":
            torch.cuda.empty_cache()

        # ---- sweep B ----
        sweepB = _SweepB(hidden, inter, device, down_w, partition, shared_idx,
                         fit["router_w"], preset.num_experts,
                         preset.experts_per_tok)
        sweepB.register(gate_mod, up_mod)
        try:
            for i in range(len(batches)):
                _forward_sweep(sweepB, i)     # stats only; outputs == sweep A
        finally:
            sweepB.unregister()

        c, s, clamp_info = surgery.solve_scales(
            sweepB.Ce, sweepB.Cs, sweepB.Ss, sweepB.Cd, sweepB.Csh,
            clamp=preset.calib_clamp)
        cos_cal = surgery.posthoc_cosine(
            sweepB.Ce, sweepB.Cs, sweepB.Ss, sweepB.Cd, sweepB.Csh,
            sweepB.dnorm2, c, s)
        cos_shared = (sweepB.Csh / max(
            math.sqrt(max(sweepB.Ss, 0.0) * max(sweepB.dnorm2, 0.0)), 1e-9))
        cos_shared = max(-1.0, min(1.0, cos_shared))
        corr = surgery.pearson_from_moments(
            sweepB.sp, sweepB.sp2, sweepB.ss, sweepB.ss2, sweepB.sps,
            sweepB.ntok)
        lcorr = surgery.pearson_from_moments(
            sweepB.sz, sweepB.sz2, sweepB.ss, sweepB.ss2, sweepB.szs,
            sweepB.ntok)
        metrics = {
            "router_corr_mean": float(corr.mean().item()),
            "router_corr_min": float(corr.min().item()),
            "router_logit_corr": float(lcorr.mean().item()),
            "router_top1_prob": sweepB.p1_sum / max(sweepB.ntok, 1),
            "sel_jaccard": sweepB.jacc_sum / max(sweepB.ntok, 1),
            "mass_recall": sweepB.mass_ours / max(sweepB.mass_total, 1e-9),
            "mass_oracle": sweepB.mass_oracle / max(sweepB.mass_total, 1e-9),
            "cos_calibrated": cos_cal,
            "cos_shared_only": cos_shared,
            "scale_c_min": float(c.min().item()),
            "scale_c_max": float(c.max().item()),
            "scale_s": s,
            "clamped": clamp_info,
            "ntok": sweepB.ntok,
        }
        del sweepB

        stats["layers"][str(layer_idx)] = {
            "partition": partition,
            "shared_idx": shared_idx.cpu().long(),
            "router_w": fit["router_w"].cpu(),
            "router_b": fit["router_b"].cpu(),
            "router_logit_std_raw": fit.get("logit_std_raw"),
            "router_logit_std": fit.get("logit_std"),
            "importance": imp,
            "calib_c": c,
            "calib_s": s,
            "metrics": metrics,
        }
        torch.save(stats, stats_path)
        _log(logger, "info", "amputate_layer_complete", layer=layer_idx, cos=round(cos_cal, 4), cos_shared=round(cos_shared, 4), mass=round(metrics["mass_recall"], 4), corr=round(metrics["router_corr_mean"], 4))
        _to_meta(layer_module)
        if device == "cuda":
            torch.cuda.empty_cache()

    torch.save(stats, stats_path)
    _log(logger, "info", "amputate_harvest_complete", path=str(stats_path), layers=len(stats["layers"]))
    return stats_path
