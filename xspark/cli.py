"""
TOFFEE XSpark CLI.

Subcommands:
  prepare   Fetch datasets via HF datasets-server API
  harvest   Extract teacher activations and top-K logits
  train     Train XSpark draft head
  export    Export trained head to GGUF
  run-all   Execute all XSpark passes sequentially
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import structlog

from toffee.config import AppState
from toffee.hardware import get_gpu_info
from toffee.state import load_state, prompt_resume_or_overwrite, save_state
from toffee.xspark.data import prepare_all, prepare_source, build_dataset
from toffee.xspark.harvest import stream_and_harvest
from toffee.xspark.train import train_xspark
from toffee.xspark.export import export_to_gguf

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toffee xspark",
        description="TOFFEE XSpark: Speculative decoding draft model training.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # prepare
    p_prep = subparsers.add_parser("prepare", help="Fetch datasets via HF datasets-server API")
    p_prep.add_argument("--out", type=str, default="xspark_data/parquet")
    p_prep.add_argument("--only", type=str, default=None)
    p_prep.add_argument("--max-rows", type=int, default=None)
    p_prep.add_argument("--no-resume", action="store_true")

    # harvest
    p_harvest = subparsers.add_parser("harvest", help="Extract teacher activations")
    p_harvest.add_argument("--model", type=str, required=True, help="HF model path")
    p_harvest.add_argument("--output-dir", type=str, default="xspark_data")
    p_harvest.add_argument("--samples", type=int, default=1000)
    p_harvest.add_argument("--batch-size", type=int, default=1)
    p_harvest.add_argument("--seq-len", type=int, default=2048)
    p_harvest.add_argument("--target-layers", type=int, nargs="+", default=[15, 30, 45])
    p_harvest.add_argument("--top-k", type=int, default=1000)
    p_harvest.add_argument("--balance-domains", action="store_true", default=True)

    # train
    p_train = subparsers.add_parser("train", help="Train XSpark draft head")
    p_train.add_argument("--target-model", type=str, required=True)
    p_train.add_argument("--data-dir", type=str, default="xspark_data")
    p_train.add_argument("--output-dir", type=str, default="xspark_output")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--batch-size", type=int, default=1)
    p_train.add_argument("--lr", type=float, default=2e-5)
    p_train.add_argument("--fp16", action="store_true", default=True)
    p_train.add_argument("--num-ctx-layers", type=int, default=2)
    p_train.add_argument("--alpha-ce-start", type=float, default=0.3)
    p_train.add_argument("--alpha-ce-end", type=float, default=0.05)
    p_train.add_argument("--alpha-tv-start", type=float, default=0.5)
    p_train.add_argument("--alpha-tv-end", type=float, default=1.0)
    p_train.add_argument("--alpha-conf", type=float, default=1.0)
    p_train.add_argument("--pos-weight", type=str, default="uniform", choices=["uniform", "survivor", "paper"])

    # export
    p_export = subparsers.add_parser("export", help="Export trained head to GGUF")
    p_export.add_argument("--target-model", type=str, required=True)
    p_export.add_argument("--xspark-head", type=str, default="xspark_output/xspark_head.pt")
    p_export.add_argument("--output-dir", type=str, default="xspark_output")
    p_export.add_argument("--block-size", type=int, default=7)
    p_export.add_argument("--target-gguf", type=str, required=True)

    # run-all
    p_all = subparsers.add_parser("run-all", help="Run all XSpark passes")
    p_all.add_argument("--model", type=str, required=True, help="HF model path")
    p_all.add_argument("--target-gguf", type=str, required=True, help="Target GGUF for tokenizer")
    p_all.add_argument("--output-dir", type=str, default="xspark_output")
    p_all.add_argument("--samples", type=int, default=1000)
    p_all.add_argument("--batch-size", type=int, default=1)
    p_all.add_argument("--seq-len", type=int, default=2048)
    p_all.add_argument("--target-layers", type=int, nargs="+", default=[15, 30, 45])
    p_all.add_argument("--top-k", type=int, default=1000)
    p_all.add_argument("--epochs", type=int, default=3)
    p_all.add_argument("--lr", type=float, default=2e-5)
    p_all.add_argument("--fp16", action="store_true", default=True)
    p_all.add_argument("--num-ctx-layers", type=int, default=2)
    p_all.add_argument("--block-size", type=int, default=7)

    return parser


def main(args: Optional[List[str]] = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args)

    if not parsed.command:
        parser.print_help()
        sys.exit(1)

    gpu = get_gpu_info()
    if gpu:
        logger.info("gpu_detected", name=gpu.name, vram=gpu.total_memory_gb)
    else:
        logger.warning("no_gpu_detected", msg="XSpark will be slow on CPU")

    command = parsed.command

    if command == "prepare":
        out_root = Path(parsed.out)
        prepare_all(out_root, only=parsed.only, max_rows=parsed.max_rows, resume=not parsed.no_resume)

    elif command == "harvest":
        output_dir = Path(parsed.output_dir)
        dataset = build_dataset(parsed.samples, balance_domains=parsed.balance_domains)

        tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(parsed.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        batches = []
        for i in range(0, len(dataset), parsed.batch_size):
            texts = dataset[i:i+parsed.batch_size]
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=parsed.seq_len, return_tensors="pt")
            batches.append({"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask})

        stream_and_harvest(
            model_name=parsed.model,
            batches=batches,
            output_dir=output_dir,
            target_layers=parsed.target_layers,
            top_k=parsed.top_k,
        )
        logger.info("harvest_complete", output_dir=str(output_dir))

    elif command == "train":
        output_dir = Path(parsed.output_dir)
        train_xspark(
            target_model=parsed.target_model,
            data_dir=Path(parsed.data_dir),
            output_dir=output_dir,
            epochs=parsed.epochs,
            batch_size=parsed.batch_size,
            lr=parsed.lr,
            fp16=parsed.fp16,
            num_ctx_layers=parsed.num_ctx_layers,
            alpha_ce_start=parsed.alpha_ce_start,
            alpha_ce_end=parsed.alpha_ce_end,
            alpha_tv_start=parsed.alpha_tv_start,
            alpha_tv_end=parsed.alpha_tv_end,
            alpha_conf=parsed.alpha_conf,
            pos_weight=parsed.pos_weight,
        )

    elif command == "export":
        export_to_gguf(parsed)
        logger.info("export_complete")

    elif command == "run-all":
        output_dir = Path(parsed.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        state = load_state(output_dir)
        choice = prompt_resume_or_overwrite(state)
        if choice == "overwrite":
            import shutil
            if output_dir.exists():
                shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True)
            state = None

        # Pass 1: prepare
        if state is None or state.last_completed_step != "dataset_prepared":
            logger.info("xspark_run_all_pass1_start")
            prepare_all(Path("xspark_data/parquet"), resume=True)
            state = AppState(
                current_pass=1,
                last_completed_step="dataset_prepared",
                started_at=datetime.now(timezone.utc).isoformat(),
                model_name=parsed.model,
            )
            save_state(output_dir, state)

        # Pass 2: harvest
        if state.last_completed_step != "harvest_complete":
            logger.info("xspark_run_all_pass2_start")
            dataset = build_dataset(parsed.samples, balance_domains=True)
            tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(parsed.model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            batches = []
            for i in range(0, len(dataset), parsed.batch_size):
                texts = dataset[i:i+parsed.batch_size]
                tokens = tokenizer(texts, padding=True, truncation=True, max_length=parsed.seq_len, return_tensors="pt")
                batches.append({"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask})

            stream_and_harvest(
                model_name=parsed.model,
                batches=batches,
                output_dir=output_dir / "harvest",
                target_layers=parsed.target_layers,
                top_k=parsed.top_k,
            )
            state = AppState(
                current_pass=2,
                last_completed_step="harvest_complete",
                started_at=state.started_at if state else datetime.now(timezone.utc).isoformat(),
                model_name=parsed.model,
            )
            save_state(output_dir, state)
        else:
            logger.info("xspark_run_all_pass2_skip", msg="Harvest already complete")

        # Pass 3: train
        if state.last_completed_step != "train_complete":
            logger.info("xspark_run_all_pass3_start")
            train_xspark(
                target_model=parsed.model,
                data_dir=output_dir / "harvest",
                output_dir=output_dir,
                epochs=parsed.epochs,
                batch_size=parsed.batch_size,
                lr=parsed.lr,
                fp16=parsed.fp16,
                num_ctx_layers=parsed.num_ctx_layers,
            )
            state = AppState(
                current_pass=3,
                last_completed_step="train_complete",
                started_at=state.started_at if state else datetime.now(timezone.utc).isoformat(),
                model_name=parsed.model,
            )
            save_state(output_dir, state)
        else:
            logger.info("xspark_run_all_pass3_skip", msg="Training already complete")

        # Pass 4: export
        logger.info("xspark_run_all_pass4_start")
        class Args:
            target_model = parsed.model
            xspark_head = str(output_dir / "xspark_head.pt")
            output_dir = str(output_dir)
            block_size = parsed.block_size
            target_gguf = parsed.target_gguf

        export_to_gguf(Args())
        state = AppState(
            current_pass=4,
            last_completed_step="export_complete",
            started_at=state.started_at if state else datetime.now(timezone.utc).isoformat(),
            model_name=parsed.model,
        )
        save_state(output_dir, state)
        logger.info("xspark_run_all_complete", output_dir=str(output_dir))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
