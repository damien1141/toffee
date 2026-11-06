"""
TOFFEE XSpark offline acceptance evaluation.

Replays the exact ik_llama.cpp drafting semantics on held-out harvest samples
(no C++ build needed):

  dspark : autoregressive bigram-corrected chain (build_dspark_logits)
  dflash2: greedy lattice walk over top-K candidates (llama.cpp decode readback)

Metrics (per anchor, averaged):
  * accept_len  — committed tokens per verify round = accepted + 1 (the bonus
                  token the verifier always produces), matching the standard
                  speculative-decoding definition.
  * accept@k    — conditional acceptance rate at each draft position.
  * recall@K    — how often the true token is in the top-K candidates
                  (dflash2; the blog's "selection headroom" metric).
  * oracle_len  — acceptance length if the walk always picked the true token
                  from the candidate lists (the recall ceiling).

Usage:
  python -m toffee.xspark.evalx --head xspark_output/dflash2_head.pt \
      --target-model ./Sage-A1.5 --data-dir xspark_data/harvest --anchors 256

Add ``--quant q8_0``/``--quant q6_k`` to fake-quantize the head's 2-D weights
(with the exporter's exact per-tensor policy) before evaluating, so the
reported acceptance includes the quantization error of a quantized GGUF.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from toffee.hardware import get_gpu_info
from toffee.xspark.model import XSparkDraftModel, stream_embed_and_lm_head

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    head_path: str,
    target_model: str,
    data_dir: str,
    max_anchors: int = 256,
    block_size: Optional[int] = None,
    batch_anchors: int = 32,
    device: Optional[str] = None,
    seed: int = 1234,
    quant: Optional[str] = None,
) -> Dict[str, float]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # frozen IO in bf16 on CUDA: the fp32 embed/lm_head pair of a 200k+-vocab
    # target is ~8 GB by itself and would OOM 12 GB-class GPUs at eval time;
    # the walk only needs top-k readback precision, which bf16 provides
    gpu = get_gpu_info()
    io_dtype = torch.float32
    if device == "cuda" and (gpu is None or gpu.supports_bf16):
        io_dtype = torch.bfloat16
    embed_w, lm_head_w = stream_embed_and_lm_head(target_model, dtype=io_dtype)
    model = XSparkDraftModel.load_head(head_path, embed_w, lm_head_w,
                                       io_dtype=io_dtype)
    if quant is not None:
        # pre-flight: apply the exporter's quantization policy in fake-quant
        # form (quantize -> dequantize, CPU fp32 math) BEFORE moving to the
        # eval device, so accept_len reflects a quantized GGUF's weights
        from toffee.xspark.quantize import apply_fake_quant
        applied = apply_fake_quant(model, quant)
        worst = max(applied, key=lambda s: s.rel_rmse) if applied else None
        logger.info("eval_fake_quant", quant=quant, tensors=len(applied),
                    worst_tensor=worst.name if worst else None,
                    worst_rel_rmse_pct=round(worst.rel_rmse * 100, 4) if worst else None,
                    note="acceptance measured against quantized-dequantized "
                         "2-D weights (same policy as xspark export --quant)")
    model = model.to(device).eval()
    cfg = model.cfg
    B = block_size or cfg.block_size
    del embed_w, lm_head_w

    files = sorted(Path(data_dir).glob("sample_*.pt"))
    if not files:
        raise FileNotFoundError(f"no sample_*.pt under {data_dir}")

    gen = torch.Generator()
    gen.manual_seed(seed)

    is_dspark = cfg.variant == "dspark"
    n_draft = B if is_dspark else B - 1
    per_pos_accept = [0] * n_draft
    per_pos_recall = [0] * n_draft
    n_blocks = 0
    accept_lens: List[int] = []
    oracle_lens: List[int] = []

    for f in files:
        if n_blocks >= max_anchors:
            break
        d = torch.load(f, map_location="cpu", weights_only=True)
        input_ids = d["input_ids"].long()
        S = input_ids.shape[0]
        lo, hi = 1, S - 1 - B
        if hi < lo:
            continue
        span = hi - lo + 1
        anchors = (torch.randperm(span, generator=gen)[:min(span, max_anchors)] + lo).sort().values

        for c0 in range(0, len(anchors), batch_anchors):
            if n_blocks >= max_anchors:
                break
            chunk = anchors[c0:c0 + batch_anchors]
            A = len(chunk)
            block_ids = torch.full((A, B), cfg.mask_token_id, dtype=torch.long)
            block_ids[:, 0] = input_ids[chunk]
            anchors_d = chunk.to(device)
            ctx = d["ctx"].to(device).float()

            h = model.forward_blocks(ctx, block_ids.to(device), anchors_d)
            base = model.base_logits(h).float()               # [A, B, V]

            if is_dspark:
                # ---- dspark: autoregressive chain per anchor ----
                drafts = []
                for a in range(A):
                    prev = int(block_ids[a, 0])
                    row_tokens = []
                    for j in range(B):
                        bias = model.markov_bias(torch.tensor(prev, device=device))
                        tok = int((base[a, j] + bias).argmax())
                        row_tokens.append(tok)
                        prev = tok
                    drafts.append(row_tokens)
                cands = None
            else:
                # ---- dflash2: candidates + greedy lattice walk ----
                cand_vals, cands = torch.topk(base, cfg.selector_top_k, dim=-1)  # [A, B, K]
                Hh = model.selector_hidden_proj(h).float()     # [A, B, R]
                drafts = []
                for a in range(A):
                    prev = 0
                    row_tokens = []
                    for pos in range(1, B):
                        if pos == 1:
                            a_prev = model.selector_prev(block_ids[a, 0:1].to(device)).float()  # [1, R]
                        else:
                            a_prev = model.selector_prev(cands[a, pos - 1]).float()            # [K, R]
                        cond = a_prev * Hh[a, pos][None, :]                 # [K_prev, R]
                        b_cur = model.selector_next(cands[a, pos]).float()   # [K, R]
                        s = b_cur @ cond.t()                                 # [K_succ, K_prev]
                        s = s + cand_vals[a, pos][:, None]
                        row = s[:, prev if pos > 1 else 0]                   # anchor col at pos 1
                        idx = int(row.argmax())
                        row_tokens.append(int(cands[a, pos, idx]))
                        prev = idx
                    drafts.append(row_tokens)

            # ---- acceptance vs ground truth ----
            for a in range(A):
                p = int(chunk[a])
                draft = drafts[a]
                acc = 0
                for k in range(n_draft):
                    truth = int(input_ids[p + 1 + k])
                    if k < len(draft) and draft[k] == truth:
                        acc += 1
                        per_pos_accept[k] += 1
                    else:
                        break
                accept_lens.append(acc + 1)

                if cands is not None:
                    ora = 0
                    for k in range(n_draft):
                        truth = int(input_ids[p + 1 + k])
                        # draft k comes from lattice column k+1 = row k+1's candidates
                        if bool((cands[a, k + 1] == truth).any()):
                            ora += 1
                            per_pos_recall[k] += 1
                        else:
                            break
                    oracle_lens.append(ora + 1)
                n_blocks += 1

    result = {
        "variant": cfg.variant,
        "n_blocks": n_blocks,
        "block_size": B,
        "accept_len": sum(accept_lens) / max(len(accept_lens), 1),
        "oracle_len": (sum(oracle_lens) / max(len(oracle_lens), 1)) if oracle_lens else 0.0,
    }
    for k in range(n_draft):
        result[f"accept@{k}"] = per_pos_accept[k] / max(n_blocks, 1)
        if not is_dspark:
            result[f"recall@{k}"] = per_pos_recall[k] / max(n_blocks, 1)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="XSpark offline acceptance eval")
    parser.add_argument("--head", type=str, required=True)
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--anchors", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--batch-anchors", type=int, default=32)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    result = evaluate(
        head_path=args.head,
        target_model=args.target_model,
        data_dir=args.data_dir,
        max_anchors=args.anchors,
        block_size=args.block_size,
        batch_anchors=args.batch_anchors,
    )
    print(json.dumps(result, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
