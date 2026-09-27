"""
TOFFEE quantization CLI.

Subcommand: `toffee quant run-all`
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import structlog

from toffee.config import AppState, DatasetConfig
from toffee.data import prepare_dataset
from toffee.hardware import get_gpu_info
from toffee.quant.config import get_num_hidden_layers, load_config_with_fallback
from toffee.quant.mapping import generate_dynamic_apex_tensor_map_from_gguf
from toffee.quant.passes import pass2_harvest, pass3_quantize
from toffee.state import load_state, prompt_resume_or_overwrite, save_state

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
        prog="toffee quant",
        description="TOFFEE quantization: imatrix generation + GGUF export.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # run-all
    p_all = subparsers.add_parser("run-all", help="Run all quantization passes")
    p_all.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    p_all.add_argument("--output-dir", type=str, default="toffee-quant-output")
    p_all.add_argument("--vram-limit", type=float, default=8.0)
    p_all.add_argument("--dataset", type=str, default="el4/bartowski-imatrix-v5-semantic-parquet")
    p_all.add_argument("--dataset-split", type=str, default="train")
    p_all.add_argument("--text-field", type=str, default="text")
    p_all.add_argument("--max-length", type=int, default=2048)
    p_all.add_argument("--format-template", type=str, default=None)
    p_all.add_argument("--streaming", action="store_true")
    p_all.add_argument("--batch-size", type=int, default=32)
    p_all.add_argument("--samples", type=int, default=256)
    p_all.add_argument("--tier", type=str, default="quality", choices=["quality", "balanced", "compact", "mini", "nano"])
    p_all.add_argument("--no-imatrix", action="store_true")
    p_all.add_argument("--gguf-quant", type=str, default="Q4_K_M")

    return parser


def main(args: Optional[List[str]] = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args)

    if not parsed.command:
        parser.print_help()
        sys.exit(1)

    if not parsed.model:
        parser.error("--model is required")

    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu = get_gpu_info()
    if gpu:
        logger.info("gpu_detected", name=gpu.name, vram=gpu.total_memory_gb)
    else:
        logger.warning("no_gpu_detected", msg="Quantization will be slow on CPU.")

    if parsed.command == "run-all":
        _run_quant_all(parsed, output_dir)
    else:
        parser.print_help()
        sys.exit(1)


def _run_quant_all(args, output_dir: Path) -> None:
    from toffee.data import load_batches_from_disk

    state = load_state(output_dir)
    choice = prompt_resume_or_overwrite(state)
    if choice == "overwrite":
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
        state = None

    model_name = args.model
    vram_limit = args.vram_limit
    samples = args.samples
    batch_size = args.batch_size
    tier = args.tier
    use_imatrix = not args.no_imatrix
    gguf_quant = args.gguf_quant

    # Pass 1: Prepare dataset
    data_dir = output_dir / "data"
    if state is None or state.last_completed_step != "dataset_prepared":
        logger.info("quant_pass1_start")
        tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dataset_config = DatasetConfig(
            path=args.dataset,
            split=args.dataset_split,
            text_field=args.text_field,
            max_length=args.max_length,
            format_template=args.format_template,
            streaming=args.streaming,
        )
        prepare_dataset(dataset_config, tokenizer, data_dir, batch_size=batch_size)
        state = AppState(
            current_pass=1,
            last_completed_step="dataset_prepared",
            started_at=datetime.now(timezone.utc).isoformat(),
            model_name=model_name,
        )
        save_state(output_dir, state)
    else:
        logger.info("quant_pass1_skip", msg="Dataset already prepared.")

    # Load batches
    batches = load_batches_from_disk(data_dir, batch_size=batch_size)
    if not batches:
        logger.error("quant_no_batches", msg="No batches found. Run dataset preparation first.")
        sys.exit(1)

    # Pass 2: Imatrix harvest
    if state.last_completed_step != "imatrix_complete":
        logger.info("quant_pass2_start")
        last_layer = -1
        if state and state.last_completed_step.startswith("layer_"):
            try:
                last_layer = int(state.last_completed_step.split("_")[1])
            except (IndexError, ValueError):
                last_layer = -1

        imatrix_path = pass2_harvest(
            model_name=model_name,
            batches=batches,
            output_dir=output_dir,
            vram_limit_gb=vram_limit,
            samples=samples,
            last_completed_layer=last_layer,
            batch_size=batch_size,
        )
        state = AppState(
            current_pass=2,
            last_completed_step="imatrix_complete",
            started_at=state.started_at if state else datetime.now(timezone.utc).isoformat(),
            model_name=model_name,
        )
        save_state(output_dir, state)
    else:
        imatrix_path = output_dir / "imatrix.dat"
        logger.info("quant_pass2_skip", msg="Imatrix already complete.")

    # Pass 3: Quantize
    base_name = Path(model_name).name
    safe_name = base_name.replace("/", "-")
    f16_gguf_path = output_dir / f"{safe_name}-f16.gguf"
    tensor_type_path = output_dir / f"tensor-type-file-{tier}.txt"
    output_path = output_dir / f"{safe_name}-TOFFEE-{tier}-{gguf_quant}.gguf"

    # Load sensitivity data
    sens_path = output_dir / "sensitivity.json"
    sensitivity_data = {}
    if sens_path.exists():
        with open(sens_path, "r") as f:
            sensitivity_data = {int(k): float(v) for k, v in json.load(f).items()}
    else:
        config = load_config_with_fallback(model_name)
        num_layers = get_num_hidden_layers(config)
        sensitivity_data = {i: 1.0 if i < 5 or i >= num_layers - 5 else 0.5 for i in range(num_layers)}

    logger.info("quant_pass3_map_start")
    generate_dynamic_apex_tensor_map_from_gguf(
        f16_gguf_path=f16_gguf_path,
        output_path=tensor_type_path,
        sensitivity_data=sensitivity_data,
        tier=tier,
        imatrix_path=imatrix_path if use_imatrix else None,
    )

    logger.info("quant_pass3_quantize_start")
    pass3_quantize(
        model_path=Path(model_name),
        imatrix_path=imatrix_path,
        tensor_type_path=tensor_type_path,
        output_path=output_path,
        f16_gguf_path=f16_gguf_path,
        use_imatrix=use_imatrix,
    )

    state = AppState(
        current_pass=3,
        last_completed_step="quant_complete",
        started_at=state.started_at if state else datetime.now(timezone.utc).isoformat(),
        model_name=model_name,
    )
    save_state(output_dir, state)

    logger.info("quant_complete", output=str(output_path))


if __name__ == "__main__":
    main()
