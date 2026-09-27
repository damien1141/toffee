"""
TOFFEE CLI — main entry point.

Subcommands:
  prepare  Pass 1: dataset preparation
  train    Pass 2: LoRA fine-tuning
  export   Pass 3: merge + GGUF export
  run-all  Execute all three passes sequentially
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import structlog

from toffee.config import (
    AlignmentConfig,
    AppState,
    DatasetConfig,
    ExportConfig,
    LoRAConfig,
    TrainingConfig,
)
from toffee.data import prepare_dataset
from toffee.exporter import export
from toffee.hardware import get_gpu_info, suggest_lora_config, suggest_training_config
from toffee.merge import merge
from toffee.state import load_state, prompt_resume_or_overwrite, save_state
from toffee.trainer import train


def _run_quant_command(args: argparse.Namespace) -> None:
    from toffee.quant.cli import main as quant_main
    quant_main(args)


def _run_xspark_command(args: argparse.Namespace) -> None:
    from toffee.xspark.cli import main as xspark_main
    xspark_main(args)


# Mapping of subcommand to handler functions
_SUBCOMMAND_HANDLERS = {
    "prepare": None,  # handled inline below
    "train": None,    # handled inline below
    "export": None,   # handled inline below
    "run-all": None,  # handled inline below
    "align": None,    # handled inline below
    "merge": None,    # handled inline below
    "quant": _run_quant_command,
    "xspark": _run_xspark_command,
}

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


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--output-dir", type=str, default="toffee-output", help="Output directory")
    parser.add_argument("--vram-limit", type=float, default=None, help="VRAM limit in GB (optional)")


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name or path")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--format-template", type=str, default=None)
    parser.add_argument("--streaming", action="store_true")


def _add_lora_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", type=str, default=None, help="Comma-separated module names")
    parser.add_argument("--use-rslora", action="store_true")
    parser.add_argument("--use-dora", action="store_true")


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)


def _add_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-merge", action="store_true", help="Skip LoRA merge")
    parser.add_argument("--gguf", action="store_true", help="Export to GGUF")
    parser.add_argument("--gguf-outfile", type=str, default=None)
    parser.add_argument("--gguf-quant", type=str, default="Q4_K_M")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--hub-token", type=str, default=None)


def _add_alignment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", type=str, default="dpo", choices=["dpo", "ipo", "kto"])
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--loss-type", type=str, default="sigmoid", choices=["sigmoid", "hinge", "ipo"])
    parser.add_argument("--reference-free", action="store_true")
    parser.add_argument("--reference-model", type=str, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--max-target-length", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toffee",
        description="TOFFEE: Hardware-aware fine-tuning for Qwen3.5-MoE.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # prepare
    p_prep = subparsers.add_parser("prepare", help="Pass 1: prepare dataset")
    _add_common_args(p_prep)
    _add_dataset_args(p_prep)
    p_prep.add_argument("--batch-size", type=int, default=32, help="Tokenization batch size")

    # train
    p_train = subparsers.add_parser("train", help="Pass 2: LoRA fine-tuning")
    _add_common_args(p_train)
    _add_lora_args(p_train)
    _add_training_args(p_train)

    # export
    p_export = subparsers.add_parser("export", help="Pass 3: merge + export")
    _add_common_args(p_export)
    _add_export_args(p_export)
    p_export.add_argument("--adapter-dir", type=str, required=True, help="Path to LoRA adapter")

    # run-all
    p_all = subparsers.add_parser("run-all", help="Run all passes")
    _add_common_args(p_all)
    _add_dataset_args(p_all)
    _add_lora_args(p_all)
    _add_training_args(p_all)
    _add_export_args(p_all)

    # align
    p_align = subparsers.add_parser("align", help="Alignment: DPO/KTO fine-tuning")
    _add_common_args(p_align)
    _add_dataset_args(p_align)
    _add_lora_args(p_align)
    _add_training_args(p_align)
    _add_alignment_args(p_align)
    _add_export_args(p_align)

    # merge
    p_merge = subparsers.add_parser("merge", help="Merge LoRA adapters")
    _add_common_args(p_merge)
    p_merge.add_argument("--adapter", type=str, action="append", required=True, help="Adapter path (repeatable)")
    p_merge.add_argument("--weight", type=float, action="append", default=None, help="Adapter weight (repeatable, must match --adapter count)")
    p_merge.add_argument("--gguf", action="store_true", help="Export to GGUF after merge")
    p_merge.add_argument("--gguf-outfile", type=str, default=None)
    p_merge.add_argument("--gguf-quant", type=str, default="Q4_K_M")

    # quant
    p_quant = subparsers.add_parser("quant", help="Quantization: imatrix + GGUF export")
    _add_common_args(p_quant)
    p_quant.add_argument("--dataset", type=str, default="el4/bartowski-imatrix-v5-semantic-parquet")
    p_quant.add_argument("--dataset-split", type=str, default="train")
    p_quant.add_argument("--text-field", type=str, default="text")
    p_quant.add_argument("--max-length", type=int, default=2048)
    p_quant.add_argument("--format-template", type=str, default=None)
    p_quant.add_argument("--streaming", action="store_true")
    p_quant.add_argument("--batch-size", type=int, default=32)
    p_quant.add_argument("--samples", type=int, default=256)
    p_quant.add_argument("--tier", type=str, default="quality", choices=["quality", "balanced", "compact", "mini", "nano"])
    p_quant.add_argument("--no-imatrix", action="store_true")
    p_quant.add_argument("--gguf-quant", type=str, default="Q4_K_M")

    # xspark
    p_xspark = subparsers.add_parser("xspark", help="XSpark speculative decoding draft model")
    p_xspark_sub = p_xspark.add_subparsers(dest="xspark_command")

    # xspark prepare
    p_xspark_prep = p_xspark_sub.add_parser("prepare", help="Fetch datasets via HF datasets-server API")
    p_xspark_prep.add_argument("--out", type=str, default="xspark_data/parquet")
    p_xspark_prep.add_argument("--only", type=str, default=None)
    p_xspark_prep.add_argument("--max-rows", type=int, default=None)
    p_xspark_prep.add_argument("--no-resume", action="store_true")

    # xspark harvest
    p_xspark_harvest = p_xspark_sub.add_parser("harvest", help="Extract teacher activations")
    p_xspark_harvest.add_argument("--model", type=str, required=True)
    p_xspark_harvest.add_argument("--output-dir", type=str, default="xspark_data")
    p_xspark_harvest.add_argument("--samples", type=int, default=1000)
    p_xspark_harvest.add_argument("--batch-size", type=int, default=1)
    p_xspark_harvest.add_argument("--seq-len", type=int, default=2048)
    p_xspark_harvest.add_argument("--target-layers", type=int, nargs="+", default=[15, 30, 45])
    p_xspark_harvest.add_argument("--top-k", type=int, default=1000)
    p_xspark_harvest.add_argument("--balance-domains", action="store_true", default=True)

    # xspark train
    p_xspark_train = p_xspark_sub.add_parser("train", help="Train XSpark draft head")
    p_xspark_train.add_argument("--target-model", type=str, required=True)
    p_xspark_train.add_argument("--data-dir", type=str, default="xspark_data")
    p_xspark_train.add_argument("--output-dir", type=str, default="xspark_output")
    p_xspark_train.add_argument("--epochs", type=int, default=3)
    p_xspark_train.add_argument("--batch-size", type=int, default=1)
    p_xspark_train.add_argument("--lr", type=float, default=2e-5)
    p_xspark_train.add_argument("--fp16", action="store_true", default=True)
    p_xspark_train.add_argument("--num-ctx-layers", type=int, default=2)
    p_xspark_train.add_argument("--alpha-ce-start", type=float, default=0.3)
    p_xspark_train.add_argument("--alpha-ce-end", type=float, default=0.05)
    p_xspark_train.add_argument("--alpha-tv-start", type=float, default=0.5)
    p_xspark_train.add_argument("--alpha-tv-end", type=float, default=1.0)
    p_xspark_train.add_argument("--alpha-conf", type=float, default=1.0)
    p_xspark_train.add_argument("--pos-weight", type=str, default="uniform", choices=["uniform", "survivor", "paper"])

    # xspark export
    p_xspark_export = p_xspark_sub.add_parser("export", help="Export draft head to GGUF")
    p_xspark_export.add_argument("--target-model", type=str, required=True)
    p_xspark_export.add_argument("--xspark-head", type=str, default="xspark_output/xspark_head.pt")
    p_xspark_export.add_argument("--output-dir", type=str, default="xspark_output")
    p_xspark_export.add_argument("--block-size", type=int, default=7)
    p_xspark_export.add_argument("--target-gguf", type=str, required=True)

    # xspark run-all
    p_xspark_all = p_xspark_sub.add_parser("run-all", help="Run all XSpark passes")
    p_xspark_all.add_argument("--model", type=str, required=True)
    p_xspark_all.add_argument("--target-gguf", type=str, required=True)
    p_xspark_all.add_argument("--output-dir", type=str, default="xspark_output")
    p_xspark_all.add_argument("--samples", type=int, default=1000)
    p_xspark_all.add_argument("--batch-size", type=int, default=1)
    p_xspark_all.add_argument("--seq-len", type=int, default=2048)
    p_xspark_all.add_argument("--target-layers", type=int, nargs="+", default=[15, 30, 45])
    p_xspark_all.add_argument("--top-k", type=int, default=1000)
    p_xspark_all.add_argument("--epochs", type=int, default=3)
    p_xspark_all.add_argument("--lr", type=float, default=2e-5)
    p_xspark_all.add_argument("--fp16", action="store_true", default=True)
    p_xspark_all.add_argument("--num-ctx-layers", type=int, default=2)
    p_xspark_all.add_argument("--block-size", type=int, default=7)

    return parser


def _build_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    return DatasetConfig(
        path=args.dataset,
        split=args.dataset_split,
        text_field=args.text_field,
        max_length=args.max_length,
        format_template=args.format_template,
        streaming=args.streaming,
    )


def _build_lora_config(args: argparse.Namespace) -> LoRAConfig:
    target_modules = None
    if args.target_modules:
        target_modules = [t.strip() for t in args.target_modules.split(",")]
    return LoRAConfig(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=target_modules,
        use_rslora=args.use_rslora,
        use_dora=args.use_dora,
    )


def _build_training_config(args: argparse.Namespace, output_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        seed=args.seed,
    )


def _build_export_config(args: argparse.Namespace) -> ExportConfig:
    return ExportConfig(
        merge=not args.no_merge,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
        gguf=args.gguf,
        gguf_outfile=Path(args.gguf_outfile) if args.gguf_outfile else None,
        gguf_quant=args.gguf_quant,
    )


def _build_alignment_config(args: argparse.Namespace) -> AlignmentConfig:
    return AlignmentConfig(
        method=args.method,
        beta=args.beta,
        label_smoothing=args.label_smoothing,
        loss_type=args.loss_type,
        reference_free=args.reference_free,
        reference_model=args.reference_model,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_target_length=args.max_target_length,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if not args.model:
        parser.error("--model is required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu = get_gpu_info()
    if gpu:
        logger.info("gpu_detected", name=gpu.name, vram=gpu.total_memory_gb, supports_bf16=gpu.supports_bf16)
    else:
        logger.warning("no_gpu_detected", msg="Training will be slow on CPU.")

    command = args.command

    if command == "prepare":
        dataset_config = _build_dataset_config(args)
        tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        batches = prepare_dataset(dataset_config, tokenizer, output_dir / "data", batch_size=args.batch_size)
        logger.info("prepare_complete", batches=len(batches))
        state = AppState(
            current_pass=1,
            last_completed_step="dataset_prepared",
            started_at=datetime.now(timezone.utc).isoformat(),
            model_name=args.model,
            dataset_path=dataset_config.path,
        )
        save_state(output_dir, state)

    elif command == "train":
        state = load_state(output_dir)
        choice = prompt_resume_or_overwrite(state)
        if choice == "overwrite":
            import shutil
            if output_dir.exists():
                shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True)

        dataset_config = _build_dataset_config(args)
        lora_config = _build_lora_config(args)
        training_config = _build_training_config(args, output_dir)
        export_config = ExportConfig()  # defaults

        adapter_dir = train(
            model_name=args.model,
            dataset_config=dataset_config,
            lora_config=lora_config,
            training_config=training_config,
            export_config=export_config,
            resume=(choice == "resume"),
        )
        logger.info("train_complete", adapter_dir=adapter_dir)

    elif command == "export":
        export_config = _build_export_config(args)
        merged_dir = export(
            base_model=args.model,
            adapter_dir=Path(args.adapter_dir),
            config=export_config,
            output_dir=output_dir,
        )
        logger.info("export_complete", merged_dir=str(merged_dir))

    elif command == "run-all":
        state = load_state(output_dir)
        choice = prompt_resume_or_overwrite(state)
        if choice == "overwrite":
            import shutil
            if output_dir.exists():
                shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True)
            state = None

        dataset_config = _build_dataset_config(args)
        lora_config = _build_lora_config(args)
        training_config = _build_training_config(args, output_dir)
        export_config = _build_export_config(args)

        # Pass 1
        if state is None or state.last_completed_step != "dataset_prepared":
            logger.info("run_all_pass1_start")
            tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            prepare_dataset(dataset_config, tokenizer, output_dir / "data", batch_size=args.batch_size)
            state = AppState(
                current_pass=1,
                last_completed_step="dataset_prepared",
                started_at=datetime.now(timezone.utc).isoformat(),
                model_name=args.model,
                dataset_path=dataset_config.path,
                lora_r=lora_config.r,
            )
            save_state(output_dir, state)

        # Pass 2
        if state.last_completed_step != "training_complete":
            logger.info("run_all_pass2_start")
            adapter_dir = train(
                model_name=args.model,
                dataset_config=dataset_config,
                lora_config=lora_config,
                training_config=training_config,
                export_config=export_config,
                resume=(choice == "resume"),
            )
            state = AppState(
                current_pass=2,
                last_completed_step="training_complete",
                started_at=state.started_at if state else datetime.now(timezone.utc).isoformat(),
                model_name=args.model,
                dataset_path=dataset_config.path,
                lora_r=lora_config.r,
            )
            save_state(output_dir, state)
        else:
            adapter_dir = str(output_dir / "adapter")

        # Pass 3
        if export_config.merge or export_config.gguf:
            logger.info("run_all_pass3_start")
            merged_dir = export(
                base_model=args.model,
                adapter_dir=Path(adapter_dir),
                config=export_config,
                output_dir=output_dir,
            )
            logger.info("run_all_complete", merged_dir=str(merged_dir))

    elif command == "align":
        state = load_state(output_dir)
        choice = prompt_resume_or_overwrite(state)
        if choice == "overwrite":
            import shutil
            if output_dir.exists():
                shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True)
            state = None

        dataset_config = _build_dataset_config(args)
        lora_config = _build_lora_config(args)
        training_config = _build_training_config(args, output_dir)
        alignment_config = _build_alignment_config(args)

        adapter_dir = align(
            model_name=args.model,
            dataset_config=dataset_config,
            lora_config=lora_config,
            training_config=training_config,
            alignment_config=alignment_config,
            resume=(choice == "resume"),
        )
        logger.info("align_complete", adapter_dir=adapter_dir)

    elif command == "merge":
        adapter_dirs = [Path(p) for p in args.adapter]
        weights = args.weight
        if weights is not None and len(weights) != len(adapter_dirs):
            parser.error("--weight count must match --adapter count")

        merged_dir = merge(
            base_model=args.model,
            adapter_dirs=adapter_dirs,
            weights=weights,
            output_dir=output_dir,
            gguf=args.gguf,
            gguf_outfile=Path(args.gguf_outfile) if args.gguf_outfile else None,
            gguf_quant=args.gguf_quant,
        )
        logger.info("merge_complete", merged_dir=str(merged_dir))

    elif command == "quant":
        _run_quant_command(args)

    elif command == "xspark":
        _run_xspark_command(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
