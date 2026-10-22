"""
TOFFEE quantization CLI.

Subcommand: `toffee quant run-all`
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
    p_all.add_argument(
        "--split-max-size", type=str, default=None,
        help="Split the F16 GGUF into shards of this size (e.g. '4G'). "
             "Needed for models too large for the disk in one file.",
    )
    p_all.add_argument(
        "--llama-cpp", type=str, default=None,
        help="Path to a llama.cpp checkout (default: ./llama.cpp)",
    )

    # --- RCO: exactly budget-constrained mixed-precision refinement ---
    p_rco = subparsers.add_parser(
        "rco",
        help="Refine the APEX tier's quant map under an exact size budget",
    )
    p_rco.add_argument(
        "--experimental", action="store_true",
        help="Acknowledge that RCO is parked: measured against APEX on a "
             "same-family model it was +2.53% worse, and a quality win is "
             "still not demonstrated. Without this flag the subcommand "
             "refuses to run.",
    )
    p_rco.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    p_rco.add_argument("--output-dir", type=str, default="toffee-quant-output")
    p_rco.add_argument("--tier", type=str, default="quality",
                       choices=["quality", "balanced", "compact", "mini", "nano"])
    p_rco.add_argument(
        "--target-size-gb", type=float, default=None,
        help="Target model size in GB. Defaults to the tier's own size.",
    )
    p_rco.add_argument(
        "--ladder", type=str, action="append", default=None,
        help="Role ladder, e.g. 'experts=Q6_K,Q5_K,Q4_K'. Repeatable.",
    )
    p_rco.add_argument(
        "--no-measure", action="store_true",
        help="Skip measuring each role's distortion at every candidate "
             "type. Without it the search has no accuracy signal and "
             "can make the model worse.",
    )
    p_rco.add_argument(
        "--sample-rows", type=int, default=256,
        help="Rows of each tensor sampled for the distortion measurement",
    )
    p_rco.add_argument("--llama-cpp", type=str, default=None)

    # --- GSQ: in-format refinement of the quantized weights ---
    p_gsq = subparsers.add_parser(
        "gsq",
        help="Refine a quantized GGUF's codes and scales in place (Gumbel-Softmax)",
    )
    p_gsq.add_argument("--gguf", type=str, required=True, help="The quantized GGUF to refine")
    p_gsq.add_argument("--f16-gguf", type=str, required=True, help="The f16 GGUF with the original weights")
    p_gsq.add_argument("--imatrix", type=str, default=None, help="Imatrix used to weight the objective")
    p_gsq.add_argument(
        "--epochs", type=int, default=0,
        help="Gumbel-Softmax epochs per tensor. 0 (the default) skips the "
             "relaxation: it provably adds nothing over llama.cpp's own "
             "code assignment, and the scale search does all the work.",
    )
    p_gsq.add_argument("--scale-passes", type=int, default=2, help="Scale coordinate-descent passes")
    p_gsq.add_argument("--max-tensors", type=int, default=None, help="Cap the number of tensors refined")
    p_gsq.add_argument("--seed", type=int, default=0)
    p_gsq.add_argument("--llama-cpp", type=str, default=None)
    p_gsq.add_argument(
        "--chunk-blocks", type=int, default=2048,
        help="Blocks processed per chunk, to bound peak memory on huge tensors",
    )
    p_gsq.add_argument(
        "--state", type=str, default=None,
        help="Checkpoint file for per-block progress (enables resume)",
    )
    p_gsq.add_argument(
        "--resume", action="store_true",
        help="Resume from the checkpoint file, skipping completed blocks",
    )
    p_gsq.add_argument(
        "--time-budget-minutes", type=float, default=None,
        help="Stop cleanly after this many minutes. The work so far is "
             "checkpointed, so the run can be resumed later. On a 36B "
             "MoE the routed experts alone take ~22h.",
    )

    # --- tiers: what each tier costs ---
    p_tiers = subparsers.add_parser(
        "tiers",
        help="List the quant tiers: what they optimize and what they cost",
    )
    p_tiers.add_argument(
        "--json", action="store_true", help="Emit the tier table as JSON",
    )

    return parser


def main(args: Optional[List[str]] = None) -> None:
    if isinstance(args, argparse.Namespace):
        # Called from the top-level `toffee` CLI, which already parsed the
        # arguments (including the `quant` sub-subcommand).
        parsed = args
        command = getattr(parsed, "quant_command", None) or "run-all"
    else:
        parser = build_parser()
        parsed = parser.parse_args(args)
        command = parsed.command

    if not command:
        parser.print_help()
        sys.exit(1)

    if command == "gsq":
        # The GSQ refinement works on an existing GGUF; it needs no model.
        _run_gsq_refine(parsed)
        return

    if command == "tiers":
        _print_tiers(parsed)
        return

    if not parsed.model:
        parser.error("--model is required")
    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu = get_gpu_info()
    if gpu:
        logger.info("gpu_detected", name=gpu.name, vram=gpu.total_memory_gb)
    else:
        logger.warning("no_gpu_detected", msg="Quantization will be slow on CPU.")

    if command == "run-all":
        _run_quant_all(parsed, output_dir)
    elif command == "rco":
        if not getattr(parsed, "experimental", False):
            print(
                "RCO is parked: measured against APEX on a same-family model "
                "it was +2.53% worse, and a quality win is still not "
                "demonstrated. Pass --experimental to run it anyway.",
                file=sys.stderr,
            )
            sys.exit(2)
        _run_rco_map(parsed, output_dir)
    else:
        parser.print_help()
        sys.exit(1)


def _parse_ladders(specs: Optional[List[str]]) -> dict:
    """Parse ``role=TYPE,TYPE,...`` ladder specifications."""
    ladders: dict = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(
                f"Invalid ladder {spec!r}; expected 'role=TYPE,TYPE,...'"
            )
        role, _, types = spec.partition("=")
        ladder = [t.strip().upper() for t in types.split(",") if t.strip()]
        if not ladder:
            raise ValueError(f"Ladder {spec!r} lists no quant types")
        ladders[role.strip()] = ladder
    return ladders


def _load_role_samples(
    f16_gguf_path: Path,
    index: dict,
    max_rows: int = 256,
) -> dict:
    """Load a sample of each tensor's f16 weights, for distortion measurement.

    Only the first ``max_rows`` rows of each tensor are read, so the whole
    sample stays small even for a 67GB model.

    Args:
        f16_gguf_path: the f16 GGUF.
        index: the tensor index of that file.
        max_rows: how many rows to read per tensor.

    Returns:
        A mapping of tensor name -> float32 rows.
    """
    import numpy as np

    from toffee.quant.gguf_io import GgufFile

    out = {}
    try:
        with GgufFile(f16_gguf_path) as f:
            for name, info in index.items():
                if info.type_name not in ("F16", "F32", "BF16"):
                    continue
                if info.n_rows < 1 or info.n_per_row < 1:
                    continue
                try:
                    data = f.read_f32(name)
                except Exception:  # noqa: BLE001 - skip unreadable tensors
                    continue
                rows = min(max_rows, info.n_rows)
                out[name] = np.ascontiguousarray(
                    data[: rows * info.n_per_row].reshape(
                        rows, info.n_per_row
                    ),
                    dtype=np.float32,
                )
    except Exception as exc:  # noqa: BLE001 - the search still works without it
        logger.warning("rco_source_load_failed", msg=str(exc)[:200])
        return {}
    return out


def _run_rco_map(args, output_dir: Path) -> None:
    """Refine the APEX tier's quant map with an exactly budgeted RCO search."""
    from toffee.quant.config import get_num_hidden_layers, load_config_with_fallback
    from toffee.quant.gguf_io import read_tensor_index
    from toffee.quant.mapping import build_apex_tensor_map
    from toffee.quant.refine import build_rco_tensor_map

    model_name = args.model
    base_name = Path(model_name).name.replace("/", "-")
    f16_gguf_path = output_dir / f"{base_name}-f16.gguf"
    tensor_type_path = output_dir / f"tensor-type-file-{args.tier}-rco.txt"

    if not f16_gguf_path.exists():
        logger.error(
            "rco_missing_f16",
            msg=(
                f"{f16_gguf_path} not found. Run 'toffee quant run-all' first, or "
                f"convert the model to f16 GGUF."
            ),
        )
        sys.exit(1)

    # --- the APEX policy: the tier's own map ------------------------------
    config = load_config_with_fallback(model_name)
    num_layers = get_num_hidden_layers(config)
    sens_path = output_dir / "sensitivity.json"
    if sens_path.exists():
        with open(sens_path, "r") as f:
            sensitivity_by_layer = {int(k): float(v) for k, v in json.load(f).items()}
    else:
        sensitivity_by_layer = {
            i: (1.0 if i < 5 or i >= num_layers - 5 else 0.5)
            for i in range(num_layers)
        }

    index = read_tensor_index(f16_gguf_path)
    apex_map = build_apex_tensor_map(
        list(index.keys()),
        sensitivity_data=sensitivity_by_layer,
        tier=args.tier,
    )
    logger.info("rco_apex_map_built", tensors=len(apex_map), tier=args.tier)

    # --- parameter counts, from the f16 GGUF ------------------------------
    param_counts = {
        name: info.n_elements for name, info in index.items()
    }

    # --- the budget -------------------------------------------------------
    # The budget is the APEX tier's *actual* size, measured with
    # llama.cpp's own row sizes -- the nominal bits/weight underestimates
    # it by ~0.4% (every block also carries scales and a super-scale),
    # and a budget below the map it starts from forces the search to
    # downgrade something it should have kept.
    from toffee.quant.ggml_bridge import GgmlBridge
    from toffee.quant.refine import exact_tensor_bytes

    br = GgmlBridge.get(args.llama_cpp)
    cols = {name: info.n_per_row for name, info in index.items()}
    apex_bytes = 0.0
    for name, quant in apex_map.items():
        apex_bytes += exact_tensor_bytes(
            name, quant, param_counts.get(name, 0), cols.get(name, 0), br
        )

    if args.target_size_gb is not None:
        target_bytes = args.target_size_gb * (1024 ** 3)
    else:
        target_bytes = apex_bytes

    ladders = _parse_ladders(args.ladder)

    # --- the source weights, for measuring the distortion ------------------
    # RCO's search is only meaningful if its value function reflects real
    # accuracy. Upstream RCO initializes its logits from the *measured*
    # distortion of each bitwidth; we do the same, using llama.cpp's own
    # quantizer on a sample of each role's weights.
    source = None
    if not args.no_measure:
        source = _load_role_samples(f16_gguf_path, index, args.sample_rows)
        logger.info(
            "rco_source_loaded",
            extra={"tensors": len(source) if source else 0},
        )

    # The exact byte cost needs each tensor's row length, which only the
    # GGUF knows.
    cols = {name: info.n_per_row for name, info in index.items()}

    result = build_rco_tensor_map(
        apex_map,
        param_counts,
        target_bytes,
        ladders=ladders,
        sensitivity=None,
        source=source,
        cols=cols,
    )

    _write_map(tensor_type_path, result.tensor_map)
    logger.info(
        "rco_complete",
        output=str(tensor_type_path),
        target_gb=target_bytes / (1024 ** 3),
        used_gb=result.bytes_used / (1024 ** 3),
        within_budget=result.within_budget,
    )
    print(f"RCO tensor map written to {tensor_type_path}")
    print(
        f"  target: {target_bytes / (1024**3):.3f} GB   "
        f"used: {result.bytes_used / (1024**3):.3f} GB   "
        f"within budget: {result.within_budget}"
    )


def _write_map(path: Path, tensor_map: dict) -> None:
    """Write a tensor -> quant type map as a llama.cpp tensor-type file."""
    with open(path, "w") as f:
        for name, quant in tensor_map.items():
            f.write(f"{name}={quant}\n")


def _print_tiers(args) -> None:
    """Print the tier catalogue: what each tier optimizes and what it costs.

    The tiers are two axes, not a size ladder: `balanced` is deliberately
    bigger than `quality` because it trades the IQ types' quality-per-bit
    for a model that decodes without the imatrix tax.
    """
    from dataclasses import asdict

    from toffee.quant.mapping import TIERS

    if getattr(args, "json", False):
        import json

        print(json.dumps([asdict(t) for t in TIERS.values()], indent=2))
        return

    print("TOFFEE quantization tiers")
    print()
    print(f"{'tier':<10} {'quality':>8} {'decode':>7} {'imatrix':>8} "
          f"{'bpw':>6} {'~GB (36B)':>10}")
    print("-" * 60)
    for t in TIERS.values():
        print(
            f"{t.name:<10} {t.quality:>8} {t.decode_speed:>7} "
            f"{('yes' if t.needs_imatrix else 'no'):>8} "
            f"{t.approx_bpw:>6.2f} {t.approx_size_gb:>10.1f}"
        )
    print()
    for t in TIERS.values():
        print(f"  {t.name:<10} {t.summary}")
    print()
    print("Note: sizes are NOT monotonic across the tier list. `balanced` is")
    print("bigger than `quality` on purpose -- it avoids the IQ/TQ imatrix")
    print("decode tax, which is worth more than the extra gigabytes when the")
    print("model is decoded continuously.")


def _run_gsq_refine(args) -> None:
    """Refine a quantized GGUF in place with GSQ."""
    from toffee.quant.ggml_bridge import GgmlBridge
    from toffee.quant.gsq import GsqConfig
    from toffee.quant.refine import refine_gguf_with_gsq

    gguf_path = Path(args.gguf)
    f16_gguf_path = Path(args.f16_gguf)
    for path in (gguf_path, f16_gguf_path):
        if not path.exists():
            logger.error("gsq_missing_file", msg=f"{path} does not exist")
            sys.exit(1)

    config = GsqConfig(
        epochs=args.epochs,
        scale_passes=args.scale_passes,
        seed=args.seed,
        chunk_blocks=args.chunk_blocks,
    )
    bridge = GgmlBridge.get(args.llama_cpp)

    # A long run over a large model checkpoints per block, so a crash does
    # not throw away the work already done.
    state_path = None
    if args.state:
        state_path = Path(args.state)
    elif args.resume:
        state_path = gguf_path.with_suffix(gguf_path.suffix + ".gsq-state.json")
    if args.resume and state_path is not None and not state_path.exists():
        logger.warning(
            "gsq_resume_state_missing",
            msg=f"no checkpoint at {state_path}; starting from the beginning",
        )

    report = refine_gguf_with_gsq(
        gguf_path=gguf_path,
        f16_gguf_path=f16_gguf_path,
        imatrix_path=Path(args.imatrix) if args.imatrix else None,
        config=config,
        max_tensors=args.max_tensors,
        bridge=bridge,
        state_path=state_path,
        resume=bool(args.resume),
        time_budget_minutes=getattr(args, "time_budget_minutes", None),
    )
    print(f"GSQ refined {report.tensors_refined} tensors "
          f"({report.tensors_skipped} skipped)")
    if report.tensors_refined:
        print(
            f"  weighted error: {report.initial_error:.6e} -> "
            f"{report.final_error:.6e} "
            f"({100 * report.relative_improvement:+.3f}%)"
        )
    if state_path is not None:
        print(f"  checkpoint: {state_path}")


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
    split_max_size = getattr(args, "split_max_size", None)
    explicit_tensor_type = getattr(args, "tensor_type_file", None)

    # Step 0: Convert safetensors → F16 GGUF (so pass2 can size imatrix
    # entries to llama.cpp's exact tensor dimensions for this arch).
    base_name = Path(model_name).name
    safe_name = base_name.replace("/", "-")
    f16_gguf_path = output_dir / f"{safe_name}-f16.gguf"
    if not f16_gguf_path.exists():
        logger.info("quant_f16_convert_start", model=model_name, split=split_max_size)
        convert_cmd = [
            sys.executable, "./llama.cpp/convert_hf_to_gguf.py",
            str(model_name),
            "--outtype", "f16",
            "--outfile", str(f16_gguf_path),
        ]
        if split_max_size:
            convert_cmd += ["--split-max-size", str(split_max_size)]
        convert_result = subprocess.run(
            convert_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if convert_result.returncode != 0:
            logger.error(
                "quant_f16_convert_failed",
                returncode=convert_result.returncode,
                stdout=convert_result.stdout,
            )
            raise RuntimeError(
                f"convert_hf_to_gguf.py failed with exit code "
                f"{convert_result.returncode}:\n{convert_result.stdout}"
            )
        logger.info("quant_f16_convert_complete", path=str(f16_gguf_path))
    else:
        logger.info("quant_f16_convert_skip", msg="F16 GGUF already present.")

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
    tensor_type_path = output_dir / f"tensor-type-file-{tier}.txt"
    output_path = output_dir / f"{safe_name}-TOFFEE-{tier}-{gguf_quant}.gguf"
    # The RCO-refined map, if the RCO step has been run for this tier.
    rco_tensor_type_path = output_dir / f"tensor-type-file-{tier}-rco.txt"

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

    # An explicitly supplied tensor-type file wins: it is how the caller
    # hands TOFFEE a map that was refined elsewhere (e.g. by RCO).
    if explicit_tensor_type is not None:
        explicit_path = Path(explicit_tensor_type)
        if not explicit_path.exists():
            logger.error(
                "quant_tensor_type_file_missing",
                msg=f"{explicit_path} does not exist",
            )
            sys.exit(1)
        logger.info(
            "quant_pass3_use_explicit_map",
            path=str(explicit_path),
        )
        tensor_type_path = explicit_path
    elif rco_tensor_type_path is not None and rco_tensor_type_path.exists():
        # An RCO-refined map, if one exists, takes precedence: it is the
        # APEX tier's map with the per-role quant types chosen under an
        # exact size budget. Feeding it to llama-quantize is the whole point
        # of the RCO step, so it must not be silently ignored.
        logger.info(
            "quant_pass3_use_rco_map",
            rco=str(rco_tensor_type_path),
            apex=str(tensor_type_path),
        )
        tensor_type_path = rco_tensor_type_path

    logger.info("quant_pass3_quantize_start")
    pass3_quantize(
        model_path=Path(model_name),
        imatrix_path=imatrix_path,
        tensor_type_path=tensor_type_path,
        output_path=output_path,
        f16_gguf_path=f16_gguf_path,
        use_imatrix=use_imatrix,
        split_max_size=split_max_size,
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
