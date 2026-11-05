"""
TOFFEE XSpark training module v2 — trains the runtime-faithful draft model.

Loss structure (per draft block anchored at p, block rows j = 0..B-1):

dspark (standard-AR rows; row 0 is the seed row and its prediction is used):
    target_j   = x_{p+j+1}
    teacher_j  = softmax(lm_head(final_hidden[p+j]))          # exact target dist
    corrected  = base_logits_j + W2 @ W1[prev_j],  prev_j = x_{p+j} (teacher-forced)
    L = a_ce * CE(corrected, target) + a_tv * TV(corrected, teacher)
        + a_conf * BCE(conf_pred, 1 - TV)

dflash2 (own-position rows; the runtime lattice reads rows 1..B-1):
    target_j   = x_{p+j}          (j >= 1; row 0 is trained only with --train-seed-row)
    teacher_j  = softmax(lm_head(final_hidden[p+j-1]))
    base       = lm_head(output_norm(h_j))
    L_backbone = a_ce * CE + a_tv * TV (on the aligned rows)
    L_selector = lattice CE: for column j, scores over (predecessor, successor)
                 candidate pairs S(a,b) = U_j(b) + <B(b), A(a) * H(h_j)>, with a
                 CE target of the TRUE token for every predecessor state — this
                 directly optimizes the greedy walk used by llama.cpp (including
                 recovery from wrong states). Column 1's predecessor is the
                 anchor token (the runtime's repeat_4d semantics).

Position weights decay as exp(-k / gamma) over draft positions (DeepSpec recipe).
TV = 0.5 * ||p_d - p_t||_1 (total variation); the acceptance target for the
confidence head is 1 - TV (the exact expected acceptance probability).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from toffee.config import AppState
from toffee.hardware import get_gpu_info
from toffee.state import load_state, save_state
from toffee.xspark.model import (
    XSparkConfig,
    XSparkDraftModel,
    infer_config_from_target,
    stream_embed_and_lm_head,
)

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class XSparkDataset(Dataset):
    """Serves harvest v2 samples: input_ids [S], ctx [S, L*H], final_hidden [S, H]."""

    def __init__(self, data_dir: Path):
        self.files = sorted(Path(data_dir).glob("sample_*.pt"))
        if not self.files:
            raise FileNotFoundError(f"no sample_*.pt files under {data_dir}")
        import json
        manifest_path = Path(data_dir) / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        d = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        return {
            "input_ids": d["input_ids"].long(),
            "ctx": d["ctx"].float(),
            "final_hidden": d["final_hidden"].float(),
        }


def sample_anchors(seq_len: int, block_size: int, n_anchors: int,
                   generator: torch.Generator) -> torch.Tensor:
    """Anchors p with 1 <= p and p + block_size <= seq_len - 1, sampled uniformly."""
    lo, hi = 1, seq_len - 1 - block_size
    if hi < lo:
        hi = lo  # degenerate short sample; anchor at 1
    span = hi - lo + 1
    n = min(n_anchors, span)
    idx = torch.randperm(span, generator=generator)[:n] + lo
    return idx.sort().values


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

@dataclass
class TrainParams:
    epochs: int = 3
    lr: float = 6e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.04
    max_grad_norm: float = 1.0
    anchors_per_sample: int = 16
    alpha_ce: float = 0.1
    alpha_tv: float = 0.9
    alpha_conf: float = 1.0
    alpha_selector: float = 1.0
    loss_decay_gamma: float = 4.0
    train_seed_row: Optional[bool] = None   # default: dspark yes, dflash2 no
    grad_accum: int = 1
    val_fraction: float = 0.05
    seed: int = 42


def _tv_distance(logp: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Row-wise total variation 0.5*||p_d - p_t||_1. logp: log_softmax'd draft."""
    p_d = logp.exp()
    p_t = F.softmax(teacher_logits, dim=-1)
    return 0.5 * (p_d - p_t).abs().sum(-1)


def compute_block_losses(
    model: XSparkDraftModel,
    h: torch.Tensor,               # [A, B, H] final hidden
    input_ids: torch.Tensor,       # [S]
    final_hidden: torch.Tensor,    # [S, H]
    anchors: torch.Tensor,         # [A]
    p: TrainParams,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """All loss terms for one batch of blocks, column-wise to bound memory."""
    cfg = model.cfg
    A, B, H = h.shape
    S = input_ids.shape[0]
    device = h.device
    is_dspark = cfg.variant == "dspark"
    train_seed_row = p.train_seed_row if p.train_seed_row is not None else is_dspark

    gamma = max(p.loss_decay_gamma, 1e-6)

    losses: Dict[str, List[torch.Tensor]] = {"ce": [], "tv": [], "conf": [], "sel": []}
    metrics = {"walk_hits": 0.0, "recall_hits": 0.0, "n_sel": 0, "tv_sum": 0.0, "n_tv": 0}

    for j in range(B):
        # ---- alignment ----
        if is_dspark:
            # row j (seed row j=0 included) predicts x_{p+j+1}
            target_pos = anchors + j + 1                    # [A]
            prev_pos = anchors + j                          # teacher-forced prev
            weight = math.exp(-j / gamma)
        else:
            if j == 0 and not train_seed_row:
                continue
            # row j (mask) predicts x_{p+j} (own position)
            target_pos = anchors + j                        # [A]
            prev_pos = anchors + j - 1                      # may be p-1 for j=0 (unused)
            weight = math.exp(-(j - 1) / gamma) if j >= 1 else 1.0

        valid = (target_pos >= 0) & (target_pos < S)
        if not bool(valid.any()):
            continue
        target = input_ids[target_pos.clamp(0, S - 1)]      # [A]
        teacher_pos = (target_pos - 1).clamp(0, S - 1)
        teacher_h = final_hidden[teacher_pos]               # [A, H]

        h_j = h[:, j]                                        # [A, H]
        base = model.base_logits(h_j).float()                # [A, V]

        if is_dspark:
            prev = input_ids[prev_pos.clamp(0, S - 1)]
            base = base + model.markov_bias(prev).float()

        logp = F.log_softmax(base, dim=-1)

        # ---- CE ----
        ce = F.cross_entropy(base, target, reduction="none")           # [A]
        losses["ce"].append(weight * ce[valid].mean())

        # ---- TV (exact, full vocab) ----
        with torch.no_grad():
            teacher_logits = model.lm_head(teacher_h).float()          # [A, V]
        tv = _tv_distance(logp, teacher_logits)                        # [A]
        losses["tv"].append(weight * tv[valid].mean())
        metrics["tv_sum"] += float(tv[valid].sum())
        metrics["n_tv"] += int(valid.sum())

        # ---- confidence (dspark) ----
        if is_dspark and cfg.use_confidence_head:
            prev_emb = model.markov_w1(input_ids[prev_pos.clamp(0, S - 1)])
            conf_in = torch.cat([h_j, prev_emb], dim=-1)
            conf_pred = model.conf_head(conf_in).squeeze(-1).float()   # [A]
            conf_target = (1.0 - tv).detach().clamp(0.0, 1.0)
            bce = F.binary_cross_entropy_with_logits(conf_pred, conf_target, reduction="none")
            losses["conf"].append(weight * bce[valid].mean())

        # ---- selector lattice (dflash2) ----
        if not is_dspark and j >= 1:
            cand_vals, cands = torch.topk(base, cfg.selector_top_k, dim=-1)
            match = cands == target[:, None]                           # [A, K]
            valid_c = match.any(-1) & valid
            metrics["n_sel"] += int(valid.sum())
            metrics["recall_hits"] += float(valid_c.sum())
            if bool(valid_c.any()):
                true_idx = match.float().argmax(-1)                    # [A]
                Hh = model.selector_hidden_proj(h_j).float()           # [A, R]
                if j == 1:
                    # predecessor = the anchor token (single row, like repeat_4d)
                    prev_ids = input_ids[anchors][:, None]             # [A, 1]
                    a_prev = model.selector_prev(prev_ids).float()     # [A, 1, R]
                    anchor_col = True
                else:
                    with torch.no_grad():
                        prev_logits = model.base_logits(h[:, j - 1]).float()
                        _, cands_prev = torch.topk(prev_logits, cfg.selector_top_k, dim=-1)
                    a_prev = model.selector_prev(cands_prev).float()   # [A, K, R]
                    anchor_col = False
                b_cur = model.selector_next(cands).float()             # [A, K_succ, R]
                cond = a_prev * Hh[:, None, :]                         # [A, K_prev, R]
                s = torch.einsum("asr,apr->asp", b_cur, cond)          # [A, K_succ, K_prev]
                s = s + cand_vals[:, :, None]                          # + U(succ) (grad flows)
                lsm = F.log_softmax(s, dim=1)                          # over successors
                nll = -lsm.gather(
                    1, true_idx[:, None, None].expand(-1, 1, s.shape[2])).squeeze(1)  # [A, K_prev]
                if anchor_col:
                    nll = nll[:, 0:1]
                losses["sel"].append(weight * nll[valid_c].mean())
                with torch.no_grad():
                    # walk proxy: best successor under the true predecessor state
                    if anchor_col:
                        acc = (s[:, :, 0].argmax(-1) == true_idx) & valid_c
                    else:
                        pm = cands_prev == input_ids[(anchors + j - 1).clamp(0, S - 1)][:, None]
                        pstate = pm.float().argmax(-1)             # [A] (garbage if absent)
                        pvalid = pm.any(-1) & valid_c
                        # s: [A, K_succ, K_prev]; gather the true predecessor column
                        acc = (s.gather(2, pstate[:, None, None].expand(-1, s.shape[1], 1)
                                        ).squeeze(-1).argmax(-1) == true_idx) & pvalid
                    metrics["walk_hits"] += float(acc.sum())

    def _mean(terms: List[torch.Tensor]) -> torch.Tensor:
        if not terms:
            return h.new_zeros(())
        return torch.stack(list(terms)).mean()

    total = p.alpha_ce * _mean(losses["ce"]) + p.alpha_tv * _mean(losses["tv"])
    if is_dspark:
        total = total + p.alpha_conf * _mean(losses["conf"])
    elif losses["sel"]:
        total = total + p.alpha_selector * _mean(losses["sel"])

    m = {
        "ce": float(_mean(losses["ce"]).detach()) if losses["ce"] else 0.0,
        "tv": float(_mean(losses["tv"]).detach()) if losses["tv"] else 0.0,
        "conf": float(_mean(losses["conf"]).detach()) if losses["conf"] else 0.0,
        "sel": float(_mean(losses["sel"]).detach()) if losses["sel"] else 0.0,
        "accept_proxy": 1.0 - (metrics["tv_sum"] / max(metrics["n_tv"], 1)),
        "recall_at_k": (metrics["recall_hits"] / metrics["n_sel"]) if metrics["n_sel"] else 1.0,
        "walk_acc": (metrics["walk_hits"] / metrics["n_sel"]) if metrics["n_sel"] else 0.0,
    }
    return total, m


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_xspark(
    target_model: str,
    data_dir: Path,
    output_dir: Path,
    params: Optional[TrainParams] = None,
    variant: str = "dspark",
    config_overrides: Optional[dict] = None,
    resume: bool = True,
) -> Path:
    """Train the runtime-faithful draft head (dspark or dflash2)."""
    params = params or TrainParams()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig
    tgt_config = AutoConfig.from_pretrained(target_model, trust_remote_code=True)
    cfg = infer_config_from_target(tgt_config, variant=variant, **(config_overrides or {}))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = get_gpu_info()
    if gpu:
        logger.info("train_gpu_detected", gpu=gpu.name, vram_gb=gpu.total_memory_gb)
        if params.anchors_per_sample > 32 and gpu.total_memory_gb < 16:
            logger.warning("anchors_lowered", old=params.anchors_per_sample, new=16)
            params.anchors_per_sample = 16
    else:
        logger.warning("train_no_gpu", reason="CPU training will be very slow")

    # ---- frozen shared IO from the target ----
    logger.info("streaming_target_io")
    embed_w, lm_head_w = stream_embed_and_lm_head(target_model)
    model = XSparkDraftModel(cfg, embed_w, lm_head_w).to(device)
    del embed_w, lm_head_w

    # ---- data ----
    dataset = XSparkDataset(data_dir)
    n_val = max(1, int(len(dataset) * params.val_fraction)) if len(dataset) > 5 else 0
    n_train = len(dataset) - n_val
    train_files = dataset.files[:n_train]
    val_files = dataset.files[n_train:]
    logger.info("data_split", train=n_train, val=n_val)

    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    n_trainable = sum(p_.numel() for p_ in trainable)
    logger.info("model_built", variant=variant, trainable_params=n_trainable,
                layers=cfg.num_layers, block=cfg.block_size)

    optimizer = torch.optim.AdamW(trainable, lr=params.lr, weight_decay=params.weight_decay)
    steps_per_epoch = max(1, n_train // params.grad_accum)
    total_steps = steps_per_epoch * params.epochs
    warmup = max(1, int(total_steps * params.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16
    if gpu is not None and not gpu.supports_bf16:
        amp_dtype = torch.float16
    scaler = torch.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(params.seed)

    model.train()
    global_step = 0
    for epoch in range(params.epochs):
        order = torch.randperm(n_train, generator=gen).tolist()
        accum = 0
        optimizer.zero_grad(set_to_none=True)
        for i in order:
            d = torch.load(train_files[i], map_location="cpu", weights_only=True)
            input_ids = d["input_ids"].long()
            S = input_ids.shape[0]
            anchors = sample_anchors(S, cfg.block_size, params.anchors_per_sample, gen)
            A = anchors.shape[0]
            if A == 0:
                continue

            block_ids = torch.full((A, cfg.block_size), cfg.mask_token_id, dtype=torch.long)
            block_ids[:, 0] = input_ids[anchors]           # CPU indexing
            anchors = anchors.to(device)
            ctx = d["ctx"].to(device).float()
            fh = d["final_hidden"].to(device).float()
            input_ids_d = input_ids.to(device)

            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp):
                h = model.forward_blocks(ctx, block_ids.to(device), anchors)
                loss, metrics = compute_block_losses(
                    model, h, input_ids_d, fh, anchors, params)
                loss = loss / params.grad_accum

            scaler.scale(loss).backward()
            accum += 1
            if accum == params.grad_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, params.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accum = 0
                global_step += 1

                if global_step % 10 == 0:
                    logger.info("train_step", epoch=epoch, step=global_step,
                                loss=round(float(loss.detach()) * params.grad_accum, 4),
                                ce=round(metrics["ce"], 4), tv=round(metrics["tv"], 4),
                                sel=round(metrics["sel"], 4),
                                accept_proxy=round(metrics["accept_proxy"], 4),
                                recall_k=round(metrics["recall_at_k"], 4),
                                lr=round(scheduler.get_last_lr()[0], 6))

        # ---- validation ----
        if val_files:
            model.eval()
            with torch.no_grad():
                tot_accept, tot_recall, n = 0.0, 0.0, 0
                for vf in val_files[:8]:
                    d = torch.load(vf, map_location="cpu", weights_only=True)
                    input_ids = d["input_ids"].long()
                    S = input_ids.shape[0]
                    anchors = sample_anchors(S, cfg.block_size, min(8, params.anchors_per_sample), gen)
                    if anchors.shape[0] == 0:
                        continue
                    block_ids = torch.full((anchors.shape[0], cfg.block_size), cfg.mask_token_id, dtype=torch.long)
                    block_ids[:, 0] = input_ids[anchors]   # CPU indexing
                    anchors = anchors.to(device)
                    ctx = d["ctx"].to(device).float()
                    fh = d["final_hidden"].to(device).float()
                    with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp):
                        h = model.forward_blocks(ctx, block_ids.to(device), anchors)
                        _, m = compute_block_losses(
                            model, h, input_ids.to(device), fh, anchors, params)
                    tot_accept += m["accept_proxy"]
                    tot_recall += m["recall_at_k"]
                    n += 1
                if n:
                    logger.info("val", epoch=epoch,
                                accept_proxy=round(tot_accept / n, 4),
                                recall_at_k=round(tot_recall / n, 4))
            model.train()

        save_state(output_dir, AppState(
            current_pass=3,
            last_completed_step=f"epoch_{epoch}",
            started_at=datetime.now(timezone.utc).isoformat(),
            model_name=target_model,
        ))

    # ---- save ----
    output_path = output_dir / ("dspark_head.pt" if variant == "dspark" else "dflash2_head.pt")
    model.save_head(output_path)
    cfg.save(output_dir / "xspark_config.json")
    logger.info("train_complete", output=str(output_path))
    return output_path
