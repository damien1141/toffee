"""
TOFFEE XSpark training module.

Trains the XSpark draft model (Parallel Backbone + Markov Head + Confidence Head).
Uses the 3-term loss function (CE, TV distance, Confidence) from the DSpark specification.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer

from toffee.config import AppState
from toffee.hardware import get_gpu_info
from toffee.state import load_state, save_state

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight Streaming Utility
# ---------------------------------------------------------------------------

def stream_specific_weights(model_path: str, target_keywords: List[str]) -> dict:
    """Reads only the requested tensors from safetensors files without loading the whole model."""
    index_path = Path(model_path) / "model.safetensors.index.json"
    with open(index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]

    files_to_load = set()
    for k, file in weight_map.items():
        if any(keyword in k for keyword in target_keywords):
            files_to_load.add(file)

    extracted = {}
    for f in files_to_load:
        w_cpu = __import__('safetensors.torch').torch.load_file(Path(model_path) / f, device="cpu")
        for k, v in w_cpu.items():
            if any(keyword in k for keyword in target_keywords):
                extracted[k] = v
    return extracted


# ---------------------------------------------------------------------------
# XSpark Model Architecture
# ---------------------------------------------------------------------------

class XSparkModel(nn.Module):
    """
    Semi-autoregressive drafter: Parallel backbone + Markov sequential head + Confidence head.
    """
    def __init__(self, config, target_model_path: str, num_ctx_layers: int = 2):
        super().__init__()
        self.num_ctx_layers = num_ctx_layers

        text_config = getattr(config, "text_config", config)
        hidden_size = getattr(config, "hidden_size", getattr(text_config, "hidden_size", None))
        vocab_size = getattr(config, "vocab_size", getattr(text_config, "vocab_size", None))

        if hidden_size is None or vocab_size is None:
            raise ValueError(f"Could not determine hidden_size or vocab_size from config: {config}")

        # 1. Shared Frozen Components (Streamed from target model)
        logger.info("streaming_target_weights", keys=["embed_tokens", "lm_head"])
        weights = stream_specific_weights(target_model_path, ["embed_tokens", "lm_head"])

        embed_key = next((k for k in weights if "embed_tokens" in k), None)
        if embed_key is None:
            raise RuntimeError("Could not find embed_tokens in safetensors files!")

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_tokens.weight.data = weights[embed_key].clone().half()
        self.embed_tokens.weight.requires_grad = False

        lm_head_key = next((k for k in weights if "lm_head" in k or "output" in k), None)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if lm_head_key is not None and weights[lm_head_key].shape == self.embed_tokens.weight.shape:
            self.lm_head.weight.data = weights[lm_head_key].clone().half()
        else:
            self.lm_head.weight = self.embed_tokens.weight

        self.lm_head.weight.requires_grad = False

        # 2. Context Projection (W_c)
        self.ctx_proj = nn.Linear(hidden_size * self.num_ctx_layers, hidden_size, bias=False)

        # 3. Parallel Backbone (2-layer Standard Transformer)
        nhead = getattr(config, "num_attention_heads", getattr(text_config, "num_attention_heads", 16))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
            activation="gelu"
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. Markov Head (W1, W2)
        self.markov_rank = 256
        self.W1 = nn.Embedding(vocab_size, self.markov_rank)
        self.W1.weight.data = self.embed_tokens.weight.data[:, :self.markov_rank].clone().float()
        self.W1.weight.requires_grad = True

        self.W2 = nn.Linear(self.markov_rank, vocab_size, bias=False)
        self.W2.weight.data = self.lm_head.weight.data[:, :self.markov_rank].clone().float()
        self.W2.weight.requires_grad = True

        # 5. Confidence Head
        self.conf_head = nn.Linear(hidden_size + self.markov_rank, 1, bias=True)

        # Norm for context features
        self.ctx_norm = nn.RMSNorm(hidden_size)

        # Final residual norm applied to the backbone hidden state before the LM head.
        self.output_norm = nn.RMSNorm(hidden_size)

        # mHC (Multi-path Hyper-Connection) head
        self.hc_fn = nn.Parameter(torch.zeros(hidden_size, 1))
        self.hc_base = nn.Parameter(torch.full((1,), 6.0))
        self.hc_scale = nn.Parameter(torch.ones(1))
        self.hc_eps = 1e-7
        self.hc_rms_eps = 1e-6

    def forward(self, ctx_features: torch.Tensor, gt_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass using Teacher Forcing.
        """
        B, S, C, H = ctx_features.shape

        # 1. Project context features dynamically based on C
        ctx_flat = ctx_features.view(B, S, C * H).float()
        ctx = self.ctx_proj(ctx_flat)
        ctx = self.ctx_norm(ctx)

        # 2. Get token embeddings
        token_embeds = self.embed_tokens(gt_tokens).float()

        # 3. Parallel Backbone forward
        hidden_states = token_embeds + ctx
        hidden_states = self.backbone(hidden_states)
        h = hidden_states

        # 4. Base Logits & Markov Bias are computed in the loss function
        prev_tokens = torch.cat([
            torch.zeros(B, 1, dtype=torch.long, device=gt_tokens.device),
            gt_tokens[:, :-1]
        ], dim=1)
        markov_embs = self.W1(prev_tokens)

        # 5. Confidence Head
        conf_input = torch.cat([h, markov_embs], dim=-1)
        c_pred_logits = self.conf_head(conf_input).squeeze(-1)

        return h, markov_embs, c_pred_logits

    def hc_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch replica of build_hc_head() in build_deepseek4.cpp (dsv4_hc_mult = hc = 1).
        """
        x_norm = F.rms_norm(x, (x.size(-1),), eps=self.hc_rms_eps)
        mixes = torch.matmul(x_norm, self.hc_fn)
        pre = torch.sigmoid(mixes * self.hc_scale + self.hc_base) + self.hc_eps
        return x * pre


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class XSparkDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.files = sorted(data_dir.glob("sample_*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)
        ctx = torch.stack(data["ctx_features"], dim=2).squeeze(0)
        ctx = ctx[:-1, :, :]  # Align with shifted targets

        return {
            "ctx_features": ctx,
            "topk_vals": data["topk_vals"].squeeze(0),
            "topk_idx": data["topk_idx"].squeeze(0),
            "gt_tokens": data["gt_tokens"].squeeze(0)
        }


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------

def position_weights(S: int, mode: str = "uniform"):
    """
    Per-position loss weights w_k (length S), applied to all three loss terms.
    """
    k = torch.arange(S, dtype=torch.float32)
    if mode == "uniform":
        return torch.ones(S)
    if mode == "survivor":
        return 1.0 + k
    if mode == "paper":
        gamma = max(S, 1)
        return torch.exp(-(k) / gamma)
    raise ValueError(f"unknown position weight mode: {mode}")


def xspark_loss_backward(
    model, scaler, h, markov_embs, c_pred, batch,
    alpha_ce=0.1, alpha_tv=0.9, alpha_conf=1.0,
    pos_w: torch.Tensor = None,
):
    """
    Computes the 3-term loss in chunks and calls backward immediately.
    """
    gt_tokens = batch["gt_tokens"]
    topk_idx = batch["topk_idx"]
    topk_vals = batch["topk_vals"]

    B, S, H = h.shape
    V = model.lm_head.out_features
    chunk_size = 128
    if pos_w is None:
        pos_w = torch.ones(S)
    pos_w = pos_w.to(h.device)

    total_loss = 0.0
    num_chunks = 0

    for i in range(0, S, chunk_size):
        end = min(i + chunk_size, S)
        w = pos_w[i:end]

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            h_chunk = h[:, i:end, :]
            markov_chunk = markov_embs[:, i:end, :]
            c_pred_chunk = c_pred[:, i:end]

            U_chunk = model.lm_head(model.output_norm(model.hc_head(h_chunk)))
            bias_chunk = model.W2(markov_chunk)
            final_logits_chunk = U_chunk + bias_chunk

            chunk_gt = gt_tokens[:, i:end]
            chunk_topk_idx = topk_idx[:, i:end, :]
            chunk_topk_vals = topk_vals[:, i:end, :]

            # 1. CE (weighted per-position)
            L_ce = F.cross_entropy(
                final_logits_chunk.reshape(-1, V), chunk_gt.reshape(-1),
                reduction="none").reshape(B, -1)
            L_ce = (L_ce * w).mean()

            # 2. TV
            logits_topk = torch.gather(final_logits_chunk, -1, chunk_topk_idx)
            logZ = torch.logsumexp(final_logits_chunk, dim=-1, keepdim=True)
            p_d_topk = torch.exp(logits_topk - logZ)
            p_t_topk = F.softmax(chunk_topk_vals, dim=-1)

            tv_topk = (p_d_topk - p_t_topk).abs().sum(dim=-1) / 2.0
            tv_outside = (1.0 - p_d_topk.sum(dim=-1)) / 2.0
            L_tv = (((tv_topk + tv_outside) * w).mean())

            # 3. Confidence
            c_star = 1.0 - (tv_topk + tv_outside).detach()
            L_conf = (F.binary_cross_entropy_with_logits(
                c_pred_chunk, c_star, reduction="none") * w).mean()

            loss_chunk = (alpha_ce * L_ce) + (alpha_tv * L_tv) + (alpha_conf * L_conf)

        is_last_chunk = (end == S)
        scaler.scale(loss_chunk).backward(retain_graph=not is_last_chunk)

        total_loss += loss_chunk.item()
        num_chunks += 1

    return total_loss / num_chunks


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_xspark(
    target_model: str,
    data_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 3,
    batch_size: int = 1,
    lr: float = 2e-5,
    fp16: bool = True,
    num_ctx_layers: int = 2,
    alpha_ce_start: float = 0.3,
    alpha_ce_end: float = 0.05,
    alpha_tv_start: float = 0.5,
    alpha_tv_end: float = 1.0,
    alpha_conf: float = 1.0,
    pos_weight: str = "uniform",
    resume: bool = True,
) -> Path:
    """Train XSpark draft head."""
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu = get_gpu_info()
    if gpu:
        logger.info(
            "train_gpu_detected",
            extra={"gpu_name": gpu.name, "vram": gpu.total_memory_gb},
        )
    else:
        logger.warning(
            "train_no_gpu",
            extra={"reason": "Training will be slow on CPU"},
        )

    # State / resume
    state = load_state(output_dir) if resume else None
    start_epoch = 0
    if state and state.last_completed_step:
        try:
            start_epoch = int(state.last_completed_step.split("_")[1]) + 1
        except (IndexError, ValueError):
            start_epoch = 0

    config = AutoConfig.from_pretrained(target_model, trust_remote_code=True)
    model = XSparkModel(config, target_model, num_ctx_layers=num_ctx_layers).cuda()

    # Freeze shared components
    for param in model.embed_tokens.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False

    dataset = XSparkDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=50, num_training_steps=len(dataloader) * epochs
    )

    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    model.train()
    logger.info("train_start", epochs=epochs, steps=len(dataloader), start_epoch=start_epoch)

    total_steps = len(dataloader) * epochs
    global_step = 0

    for epoch in range(start_epoch, epochs):
        for step, batch in enumerate(dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}

            # Curriculum alpha annealing
            progress = global_step / max(total_steps, 1)
            alpha_ce = alpha_ce_start + (alpha_ce_end - alpha_ce_start) * progress
            alpha_tv = alpha_tv_start + (alpha_tv_end - alpha_tv_start) * progress

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
                inputs = torch.cat([
                    torch.zeros(batch_size, 1, dtype=torch.long, device=batch["gt_tokens"].device),
                    batch["gt_tokens"][:, :-1]
                ], dim=1)
                h, markov_embs, c_pred = model(batch["ctx_features"], inputs)

            pos_w = position_weights(h.shape[1], mode=pos_weight).to(h.device)
            loss = xspark_loss_backward(
                model, scaler, h, markov_embs, c_pred, batch,
                alpha_ce=alpha_ce, alpha_tv=alpha_tv, alpha_conf=alpha_conf,
                pos_w=pos_w,
            )

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            if step % 10 == 0:
                logger.info("train_step", epoch=epoch, step=step, loss=loss,
                            alpha_ce=round(float(alpha_ce), 4), alpha_tv=round(float(alpha_tv), 4))

        # Save state
        save_state(output_dir, AppState(
            current_pass=2,
            last_completed_step=f"epoch_{epoch}",
            started_at=datetime.now(timezone.utc).isoformat(),
            model_name=target_model,
        ))

    # Save final model
    output_path = output_dir / "xspark_head.pt"
    torch.save({
        "ctx_proj": model.ctx_proj.state_dict(),
        "ctx_norm": model.ctx_norm.state_dict(),
        "output_norm": model.output_norm.state_dict(),
        "hc_fn": {"weight": model.hc_fn.detach().clone()},
        "hc_base": {"weight": model.hc_base.detach().clone()},
        "hc_scale": {"weight": model.hc_scale.detach().clone()},
        "backbone": model.backbone.state_dict(),
        "W1": model.W1.state_dict(),
        "W2": model.W2.state_dict(),
        "conf_head": model.conf_head.state_dict(),
    }, output_path)

    logger.info("train_complete", output=str(output_path))
    return output_path
