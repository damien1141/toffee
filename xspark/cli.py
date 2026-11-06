"""
TOFFEE XSpark CLI v2.

Subcommands:
  prepare   Fetch datasets via HF datasets-server API
  harvest   Extract teacher features (target layers + final hidden)
  train     Train the runtime-faithful draft head (dspark or dflash2)
  export    Export the trained head to an ik_llama.cpp-compatible GGUF
  eval      Offline acceptance evaluation (replays runtime semantics)
  run-all   Execute the full pipeline sequentially
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from toffee.hardware import get_gpu_info
from toffee.xspark.data import build_dataset, prepare_all
from toffee.xspark.train import TrainParams, train_xspark

try:
    import structlog

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    logger = structlog.get_logger()
except ImportError:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


def _parse_target_layers(spec: str, num_layers: int) -> List[int]:
    """Parse --target-layers: 'auto', '5', or '6,12,18,24,30'."""
    spec = spec.strip()
    if spec == "auto" or spec == "":
        n = 5
    elif "," in spec:
        ids = sorted({int(x) for x in spec.split(",")})
        if not ids:
            raise ValueError("--target-layers produced an empty layer list")
        return ids
    else:
        n = int(spec)
    if n < 1:
        raise ValueError(
            f"--target-layers count must be >= 1 (got {n!r}); for explicit "
            "0-based ids use the comma form, e.g. '6,12,18,24,30'")
    # 5 evenly spaced interior layers (deduped for very small models)
    ids = sorted({max(0, (num_layers * i) // (n + 1) - 1) for i in range(1, n + 1)})
    if not ids:
        raise ValueError("--target-layers produced an empty layer list")
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toffee xspark",
        description="TOFFEE XSpark v2: runtime-faithful speculative decoding draft training.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # prepare
    p_prep = subparsers.add_parser("prepare", help="Fetch datasets via HF datasets-server API")
    p_prep.add_argument("--out", type=str, default="xspark_data/parquet")
    p_prep.add_argument("--only", type=str, default=None)
    p_prep.add_argument("--max-rows", type=int, default=None)
    p_prep.add_argument("--no-resume", action="store_true")

    # harvest
    p_harvest = subparsers.add_parser("harvest", help="Extract teacher features + final hidden")
    p_harvest.add_argument("--model", type=str, required=True, help="HF model path")
    p_harvest.add_argument("--output-dir", type=str, default="xspark_data/harvest")
    p_harvest.add_argument("--samples", type=int, default=1000)
    p_harvest.add_argument("--batch-size", type=int, default=1)
    p_harvest.add_argument("--seq-len", type=int, default=2048)
    p_harvest.add_argument("--target-layers", type=str, default="auto",
                           help="'auto', a count like '5', or comma-separated 0-based ids")
    p_harvest.add_argument("--balance-domains", action="store_true", default=True)
    p_harvest.add_argument("--vram-limit", type=float, default=None,
                           help="GPU budget in GB for one streamed layer + activations "
                                "(allocator-capped; fp32 layers auto-downcast to bf16 "
                                "to fit, like the other toffee modules)")
    p_harvest.add_argument("--ram-budget", type=float, default=None,
                           help="system-RAM budget in GB per sample chunk "
                                "(default: 45%% of installed RAM, capped at 8 GB)")
    p_harvest.add_argument("--dtype", type=str, default="auto",
                           choices=["auto", "bf16", "fp16", "fp32"],
                           help="compute dtype for the streamed layers "
                                "(auto = checkpoint dtype, recommended)")

    # train
    p_train = subparsers.add_parser("train", help="Train the draft head")
    p_train.add_argument("--variant", type=str, default="dspark", choices=["dspark", "dflash2"])
    p_train.add_argument("--target-model", type=str, required=True)
    p_train.add_argument("--data-dir", type=str, default="xspark_data/harvest")
    p_train.add_argument("--output-dir", type=str, default="xspark_output")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--lr", type=float, default=6e-4)
    p_train.add_argument("--grad-accum", type=int, default=1)
    p_train.add_argument("--anchors", type=int, default=16,
                         help="anchor blocks sampled per sequence per step")
    p_train.add_argument("--block-size", type=int, default=8)
    p_train.add_argument("--num-layers", type=int, default=4)
    p_train.add_argument("--intermediate-size", type=int, default=None)
    p_train.add_argument("--num-kv-heads", type=int, default=None)
    p_train.add_argument("--target-layers", type=str, default="auto")
    p_train.add_argument("--alpha-ce", type=float, default=0.1)
    p_train.add_argument("--alpha-tv", type=float, default=0.9)
    p_train.add_argument("--alpha-conf", type=float, default=1.0)
    p_train.add_argument("--alpha-selector", type=float, default=1.0)
    p_train.add_argument("--loss-decay-gamma", type=float, default=4.0)
    p_train.add_argument("--train-seed-row", type=str, default=None, choices=[None, "yes", "no"])
    p_train.add_argument("--markov-rank", type=int, default=256)
    p_train.add_argument("--selector-rank", type=int, default=256)
    p_train.add_argument("--selector-top-k", type=int, default=16)
    p_train.add_argument("--conv-kernel-size", type=int, default=2)
    p_train.add_argument("--conv-group-size", type=int, default=16)
    p_train.add_argument("--mask-token-id", type=int, default=None)
    p_train.add_argument("--val-fraction", type=float, default=0.05)
    p_train.add_argument("--vram-limit", type=float, default=None,
                         help="GPU budget in GB for training (allocator cap; "
                              "the frozen embed/lm_head are kept in bf16 and "
                              "the draft FFN width/depth auto-shrink to fit, "
                              "like the other toffee modules)")

    # export
    p_export = subparsers.add_parser("export", help="Export head to GGUF")
    p_export.add_argument("--head", type=str, required=True)
    p_export.add_argument("--target-model", type=str, required=True)
    p_export.add_argument("--target-gguf", type=str, default=None)
    p_export.add_argument("--output", type=str, default=None)
    p_export.add_argument("--mask-token-id", type=int, default=None)
    p_export.add_argument("--self-contained", action="store_true")
    p_export.add_argument("--dtype", type=str, default="f32", choices=["f32", "f16", "bf16"],
                          help="storage dtype for 2-D matmul weights; 1-D tensors "
                               "and conv bases always stay F32 (ik_llama.cpp "
                               "kernel requirement)")
    p_export.add_argument("--quant", type=str, default="none",
                          choices=["none", "q8", "q8_0", "q6", "q6_k"],
                          help="block-quantize 2-D matmul weights at export: "
                               "q8_0 ~53%% of F16 size (near-lossless), q6_k "
                               "~41%%; get_rows tensors (selector_*, "
                               "token_embd) cap at Q8_0 (CUDA kernel has no "
                               "k-quant path); markov_w1 (dspark) never "
                               "quantized")

    # eval
    p_eval = subparsers.add_parser("eval", help="Offline acceptance evaluation")
    p_eval.add_argument("--head", type=str, required=True)
    p_eval.add_argument("--target-model", type=str, required=True)
    p_eval.add_argument("--data-dir", type=str, required=True)
    p_eval.add_argument("--anchors", type=int, default=256)
    p_eval.add_argument("--block-size", type=int, default=None)
    p_eval.add_argument("--json-out", type=str, default=None)
    p_eval.add_argument("--quant", type=str, default="none",
                        choices=["none", "q8", "q8_0", "q6", "q6_k"],
                        help="fake-quantize the head's 2-D weights (same policy "
                             "as export) before evaluating, to measure the "
                             "acceptance impact of a quantized GGUF")

    # run-all
    p_all = subparsers.add_parser("run-all", help="Run harvest -> train -> export")
    p_all.add_argument("--model", type=str, required=True)
    p_all.add_argument("--target-gguf", type=str, required=True)
    p_all.add_argument("--output-dir", type=str, default="xspark_output")
    p_all.add_argument("--data-dir", type=str, default="xspark_data/harvest")
    p_all.add_argument("--variant", type=str, default="dspark", choices=["dspark", "dflash2"])
    p_all.add_argument("--samples", type=int, default=1000)
    p_all.add_argument("--batch-size", type=int, default=1)
    p_all.add_argument("--seq-len", type=int, default=2048)
    p_all.add_argument("--target-layers", type=str, default="auto")
    p_all.add_argument("--epochs", type=int, default=3)
    p_all.add_argument("--lr", type=float, default=6e-4)
    p_all.add_argument("--anchors", type=int, default=16)
    p_all.add_argument("--block-size", type=int, default=8)
    p_all.add_argument("--num-layers", type=int, default=4)
    p_all.add_argument("--alpha-ce", type=float, default=0.1)
    p_all.add_argument("--alpha-tv", type=float, default=0.9)
    p_all.add_argument("--alpha-selector", type=float, default=1.0)
    p_all.add_argument("--mask-token-id", type=int, default=None)
    p_all.add_argument("--dtype", type=str, default="f32", choices=["f32", "f16", "bf16"])
    p_all.add_argument("--quant", type=str, default="none",
                       choices=["none", "q8", "q8_0", "q6", "q6_k"],
                       help="block-quantize 2-D matmul weights in the export pass")
    p_all.add_argument("--vram-limit", type=float, default=None,
                       help="GPU budget in GB (harvest: one streamed layer; "
                            "train: allocator cap + draft auto-fit)")
    p_all.add_argument("--ram-budget", type=float, default=None,
                       help="system-RAM budget in GB per sample chunk (harvest pass)")
    p_all.add_argument("--harvest-dtype", type=str, default="auto",
                       choices=["auto", "bf16", "fp16", "fp32"],
                       help="compute dtype for the streamed layers (harvest pass)")

    return parser


def _train_overrides(parsed) -> dict:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(parsed.target_model, trust_remote_code=True)
    n_layers = int(getattr(cfg, "num_hidden_layers",
                           getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers", 32)))
    overrides = {
        "block_size": parsed.block_size,
        "num_layers": parsed.num_layers,
        "target_layer_ids": _parse_target_layers(parsed.target_layers, n_layers),
    }
    if getattr(parsed, "intermediate_size", None):
        overrides["intermediate_size"] = parsed.intermediate_size
    if getattr(parsed, "num_kv_heads", None):
        overrides["num_kv_heads"] = parsed.num_kv_heads
    if getattr(parsed, "markov_rank", None):
        overrides["markov_rank"] = parsed.markov_rank
    if getattr(parsed, "selector_rank", None):
        overrides["selector_rank"] = parsed.selector_rank
    if getattr(parsed, "selector_top_k", None):
        overrides["selector_top_k"] = parsed.selector_top_k
    if getattr(parsed, "conv_kernel_size", None):
        overrides["conv_kernel_size"] = parsed.conv_kernel_size
    if getattr(parsed, "conv_group_size", None):
        overrides["conv_group_size"] = parsed.conv_group_size
    if getattr(parsed, "mask_token_id", None):
        overrides["mask_token_id"] = parsed.mask_token_id
    return overrides


def main(args: Optional[List[str]] = None) -> None:
    if args is None:
        args = sys.argv[1:]
    if args and args[0] == "xspark":   # tolerate 'python -m toffee xspark ...' style tails
        args = args[1:]
    parser = build_parser()
    parsed = parser.parse_args(args)

    # must land before the first CUDA allocation (fragmentation OOM guard)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not parsed.command:
        parser.print_help()
        sys.exit(1)

    gpu = get_gpu_info()
    if gpu:
        logger.info("gpu_detected", name=gpu.name, vram_gb=gpu.total_memory_gb)
    else:
        logger.warning("no_gpu_detected", msg="XSpark will be slow on CPU")

    command = parsed.command

    if command == "prepare":
        prepare_all(Path(parsed.out), only=parsed.only, max_rows=parsed.max_rows,
                    resume=not parsed.no_resume)

    elif command == "harvest":
        from toffee.xspark.harvest import stream_and_harvest
        from transformers import AutoConfig, AutoTokenizer

        cfg = AutoConfig.from_pretrained(parsed.model, trust_remote_code=True)
        n_layers = int(getattr(cfg, "num_hidden_layers",
                               getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers", 32)))
        target_layers = _parse_target_layers(parsed.target_layers, n_layers)

        dataset = build_dataset(parsed.samples, balance_domains=parsed.balance_domains)
        tokenizer = AutoTokenizer.from_pretrained(parsed.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        batches = []
        for i in range(0, len(dataset), parsed.batch_size):
            texts = dataset[i:i + parsed.batch_size]
            tokens = tokenizer(texts, padding=True, truncation=True,
                               max_length=parsed.seq_len, return_tensors="pt")
            batches.append({"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask})

        stream_and_harvest(
            model_name=parsed.model,
            batches=batches,
            output_dir=Path(parsed.output_dir),
            target_layers=target_layers,
            vram_limit_gb=parsed.vram_limit,
            ram_budget_gb=parsed.ram_budget,
            dtype=parsed.dtype,
        )
        logger.info("harvest_complete", output_dir=parsed.output_dir)

    elif command == "train":
        params = TrainParams(
            epochs=parsed.epochs,
            lr=parsed.lr,
            anchors_per_sample=parsed.anchors,
            alpha_ce=parsed.alpha_ce,
            alpha_tv=parsed.alpha_tv,
            alpha_conf=parsed.alpha_conf,
            alpha_selector=parsed.alpha_selector,
            loss_decay_gamma=parsed.loss_decay_gamma,
            grad_accum=parsed.grad_accum,
            val_fraction=parsed.val_fraction,
        )
        if parsed.train_seed_row is not None:
            params.train_seed_row = parsed.train_seed_row == "yes"
        train_xspark(
            target_model=parsed.target_model,
            data_dir=Path(parsed.data_dir),
            output_dir=Path(parsed.output_dir),
            params=params,
            variant=parsed.variant,
            config_overrides=_train_overrides(parsed),
            vram_limit_gb=parsed.vram_limit,
        )

    elif command == "export":
        from toffee.xspark.export import export_to_gguf
        export_to_gguf(
            head_path=parsed.head,
            target_model=parsed.target_model,
            output_path=parsed.output,
            target_gguf=parsed.target_gguf,
            mask_token_id=parsed.mask_token_id,
            self_contained=parsed.self_contained,
            dtype=parsed.dtype,
            quant=parsed.quant,
        )
        logger.info("export_complete")

    elif command == "eval":
        from toffee.xspark.evalx import evaluate
        import json
        result = evaluate(
            head_path=parsed.head,
            target_model=parsed.target_model,
            data_dir=parsed.data_dir,
            max_anchors=parsed.anchors,
            block_size=parsed.block_size,
            quant=parsed.quant if parsed.quant != "none" else None,
        )
        print(json.dumps(result, indent=2))
        if parsed.json_out:
            Path(parsed.json_out).write_text(json.dumps(result, indent=2))

    elif command == "run-all":
        from toffee.state import AppState, save_state
        from toffee.xspark.export import export_to_gguf
        from toffee.xspark.harvest import stream_and_harvest
        from transformers import AutoConfig, AutoTokenizer

        output_dir = Path(parsed.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = AutoConfig.from_pretrained(parsed.model, trust_remote_code=True)
        n_layers = int(getattr(cfg, "num_hidden_layers",
                               getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers", 32)))
        target_layers = _parse_target_layers(parsed.target_layers, n_layers)

        # Pass 1: harvest
        data_dir = Path(parsed.data_dir)
        if not any(data_dir.glob("sample_*.pt")):
            logger.info("xspark_run_all_harvest_start")
            dataset = build_dataset(parsed.samples, balance_domains=True)
            tokenizer = AutoTokenizer.from_pretrained(parsed.model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            batches = []
            for i in range(0, len(dataset), parsed.batch_size):
                texts = dataset[i:i + parsed.batch_size]
                tokens = tokenizer(texts, padding=True, truncation=True,
                                   max_length=parsed.seq_len, return_tensors="pt")
                batches.append({"input_ids": tokens.input_ids,
                                "attention_mask": tokens.attention_mask})
            stream_and_harvest(
                parsed.model, batches, data_dir, target_layers,
                vram_limit_gb=parsed.vram_limit,
                ram_budget_gb=parsed.ram_budget,
                dtype=parsed.harvest_dtype,
            )
            save_state(output_dir, AppState(
                current_pass=1, last_completed_step="harvest_complete",
                started_at=datetime.now(timezone.utc).isoformat(), model_name=parsed.model))
        else:
            logger.info("xspark_run_all_harvest_skip")

        # Pass 2: train
        params = TrainParams(
            epochs=parsed.epochs, lr=parsed.lr, anchors_per_sample=parsed.anchors,
            alpha_ce=parsed.alpha_ce, alpha_tv=parsed.alpha_tv,
            alpha_selector=parsed.alpha_selector,
        )
        head = train_xspark(
            target_model=parsed.model,
            data_dir=data_dir,
            output_dir=output_dir,
            params=params,
            variant=parsed.variant,
            config_overrides=_train_overrides(parsed),
            vram_limit_gb=parsed.vram_limit,
        )
        save_state(output_dir, AppState(
            current_pass=2, last_completed_step="train_complete",
            started_at=datetime.now(timezone.utc).isoformat(), model_name=parsed.model))

        # Pass 3: export
        gguf_path = export_to_gguf(
            head_path=str(head),
            target_model=parsed.model,
            target_gguf=parsed.target_gguf,
            mask_token_id=parsed.mask_token_id,
            dtype=parsed.dtype,
            quant=parsed.quant,
        )
        save_state(output_dir, AppState(
            current_pass=3, last_completed_step="export_complete",
            started_at=datetime.now(timezone.utc).isoformat(), model_name=parsed.model))
        logger.info("xspark_run_all_complete", gguf=str(gguf_path))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
