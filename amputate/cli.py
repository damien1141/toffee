"""
@file toffee/amputate/cli.py
   ``toffee amputate <report|harvest|build|verify|canary-prep|canary-run|
   plan|selftest>``.

Also runnable standalone: ``python -m toffee.amputate <cmd>``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _add_harvest_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", type=str, required=True,
                   help="Unpacked dense checkpoint directory (safetensors)")
    p.add_argument("--output-dir", type=str, default="toffee-amputate-output")
    p.add_argument("--dataset", type=str,
                   default="el4/bartowski-imatrix-v5-semantic-parquet")
    p.add_argument("--dataset-split", type=str, default="train")
    p.add_argument("--text-field", type=str, default="text")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--format-template", type=str, default=None)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--samples", type=int, default=256,
                   help="Batches to harvest (~samples x max_length tokens)")
    p.add_argument("--vram-limit", type=float, default=None)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--model-impl", type=str, default="auto",
                   choices=["auto", "real", "minimal"])
    p.add_argument("--num-experts", type=int, default=32)
    p.add_argument("--experts-per-tok", type=int, default=4)
    p.add_argument("--shared-intermediate", type=int, default=4096)
    p.add_argument("--strategy", type=str, default="balanced",
                   choices=["balanced", "block"])
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--router-logit-scale", type=float, default=1.0,
                   help="Target pooled std of router logits (temperature "
                        "normalization; 0 disables)")


def _preset_from_args(args) -> "object":
    from toffee.amputate.preset import AmputationPreset
    rls = getattr(args, "router_logit_scale", 1.0)
    return AmputationPreset(
        name="a12b" if args.num_experts == 32 and
        args.shared_intermediate == 4096 else "custom",
        num_experts=args.num_experts,
        experts_per_tok=args.experts_per_tok,
        shared_intermediate=args.shared_intermediate,
        strategy=getattr(args, "strategy", "balanced"),
        ridge_alpha=getattr(args, "ridge_alpha", 1.0),
        router_logit_scale=None if not rls else rls,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toffee amputate",
        description="No-training MoEfication: dense FFN -> sparse MoE.",
    )
    sub = parser.add_subparsers(dest="amputate_command")

    p_report = sub.add_parser(
        "report", help="Scan shards, classify tensors, write registry JSON")
    p_report.add_argument("--model", type=str, required=True)
    p_report.add_argument("--output-dir", type=str,
                          default="toffee-amputate-output")

    p_harvest = sub.add_parser(
        "harvest", help="Two-sweep activation harvest -> surgery_stats.pt")
    _add_harvest_args(p_harvest)

    p_build = sub.add_parser(
        "build", help="Slice the checkpoint into the amputated MoE directory")
    p_build.add_argument("--model", type=str, required=True)
    p_build.add_argument("--output-dir", type=str,
                         default="toffee-amputate-output")
    p_build.add_argument("--out", type=str, default=None,
                         help="Output model directory "
                              "(default: <output-dir>/amputated)")
    p_build.add_argument("--stats", type=str, default=None,
                         help="surgery_stats.pt (default: <output-dir>/surgery_stats.pt)")
    p_build.add_argument("--registry", type=str, default=None,
                         help="amputate_registry.json from `report`")
    p_build.add_argument("--num-experts", type=int, default=32)
    p_build.add_argument("--experts-per-tok", type=int, default=4)
    p_build.add_argument("--shared-intermediate", type=int, default=4096)
    p_build.add_argument("--drop-mtp", action="store_true",
                         help="Also drop mtp.veriloop_* tensors")
    p_build.add_argument("--keep-vision", action="store_true")
    p_build.add_argument("--drop-unknown", action="store_true")
    p_build.add_argument("--config-only", action="store_true",
                         help="Refresh config.json + aux files only; do NOT "
                              "re-slice shards (requires a previous full "
                              "build at --out)")
    p_build.add_argument("--no-vision-config", action="store_true",
                         help="Do not mirror the source vision_config into "
                              "the output wrapper (strictly text-only "
                              "metadata)")

    p_verify = sub.add_parser(
        "verify", help="Offline canary + PPL commands from surgery stats")
    p_verify.add_argument("--stats", type=str, default=None)
    p_verify.add_argument("--output-dir", type=str,
                          default="toffee-amputate-output")
    p_verify.add_argument("--ppl-dense", type=str, default=None,
                          help="Dense IQ2_XS GGUF for the perplexity A/B")
    p_verify.add_argument("--ppl-amputated", type=str, default=None)
    p_verify.add_argument("--ppl-corpus", type=str, default=None)

    p_plan = sub.add_parser(
        "plan", help="Print the A12B preset math and quant dials")
    p_plan.add_argument("--eff-bw-gbs", type=float, default=None)

    p_self = sub.add_parser(
        "selftest", help="Synthetic end-to-end run (no download needed)")
    p_self.add_argument("--output-dir", type=str,
                        default="toffee-amputate-selftest")
    p_self.add_argument("--device", type=str, default="cpu",
                        choices=["auto", "cuda", "cpu"])
    p_self.add_argument("--keep", action="store_true",
                        help="Keep the synthetic checkpoint on disk")

    p_cprep = sub.add_parser(
        "canary-prep",
        help="Build + freeze the PPL canary corpus (args pass through "
             "to `python -m toffee.amputate.canary prep`)")
    p_cprep.add_argument("extra", nargs=argparse.REMAINDER,
                         help=argparse.SUPPRESS)

    p_crun = sub.add_parser(
        "canary-run",
        help="llama-perplexity A/B over the frozen corpus + pre-registered "
             "verdict (args pass through to `python -m "
             "toffee.amputate.canary run`)")
    p_crun.add_argument("extra", nargs=argparse.REMAINDER,
                        help=argparse.SUPPRESS)

    return parser


def run_args(args: argparse.Namespace) -> int:
    from toffee.amputate import build as build_mod
    from toffee.amputate import harvest as harvest_mod
    from toffee.amputate import registry as reg_mod
    from toffee.amputate import verify as verify_mod
    from toffee.amputate.preset import summarize
    from toffee.amputate.selftest import run_selftest

    cmd = args.amputate_command
    out_dir = Path(getattr(args, "output_dir", "toffee-amputate-output"))

    if cmd == "report":
        reg = reg_mod.scan_checkpoint(args.model)
        out_dir.mkdir(parents=True, exist_ok=True)
        reg_path = reg_mod.save_registry(reg, out_dir / "amputate_registry.json")
        print(reg.render())
        print(f"\nregistry written: {reg_path}")
        return 0

    if cmd == "harvest":
        preset = _preset_from_args(args)
        # Same batch machinery as `toffee quant run-all`.
        from toffee.config import DatasetConfig
        from toffee.data import prepare_dataset
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        ds = DatasetConfig(
            path=args.dataset, split=args.dataset_split,
            text_field=args.text_field, max_length=args.max_length,
            format_template=args.format_template, streaming=args.streaming,
        )
        batches = prepare_dataset(ds, tokenizer, out_dir / "data",
                                  batch_size=args.batch_size)
        import math
        n_use = min(len(batches), math.ceil(args.samples / args.batch_size))
        batches = batches[:n_use]
        logger.info("amputate_batches_ready", batches=len(batches))
        stats_path = harvest_mod.harvest(
            args.model, batches, out_dir, preset, device=args.device,
            model_impl=args.model_impl, vram_limit_gb=args.vram_limit,
        )
        print(verify_mod.render_report(stats_path))
        return 0

    if cmd == "build":
        preset = _preset_from_args(args)
        stats = args.stats or str(out_dir / "surgery_stats.pt")
        out = args.out or str(out_dir / "amputated")
        build_mod.build(
            args.model, stats, out, preset, registry_json=args.registry,
            drop_mtp=args.drop_mtp, keep_vision=args.keep_vision,
            drop_unknown=args.drop_unknown,
            config_only=args.config_only,
            mirror_vision_config=not args.no_vision_config,
        )
        if args.config_only:
            print(f"config.json refreshed (shards untouched): {out}")
        else:
            print(f"amputated model written: {out}")
        print("next: toffee quant run-all --model <that dir> ...")
        return 0

    if cmd == "verify":
        stats = args.stats or str(out_dir / "surgery_stats.pt")
        print(verify_mod.render_report(
            stats, dense_gguf=args.ppl_dense,
            amputated_gguf=args.ppl_amputated, corpus=args.ppl_corpus))
        ok = verify_mod.check_thresholds(verify_mod.load_stats(stats))["pass"]
        return 0 if ok else 1

    if cmd == "plan":
        print(summarize(eff_bw_gb_s=args.eff_bw_gbs))
        return 0

    if cmd == "selftest":
        ok = run_selftest(out_dir, device=args.device, keep=args.keep)
        return 0 if ok else 1

    if cmd in ("canary-prep", "canary-run"):
        sub_cmd = "prep" if cmd == "canary-prep" else "run"
        from toffee.amputate.canary.__main__ import main as canary_main
        return canary_main([sub_cmd] + list(getattr(args, "extra", [])))

    print("unknown amputate command", file=sys.stderr)
    return 2


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )
    argv_list = list(sys.argv[1:] if argv is None else argv)
    # canary-prep / canary-run forward their raw flags to the canary
    # module CLI; going through argparse would eat unknown options
    # (REMAINDER quirk), so intercept before parsing.
    if argv_list and argv_list[0] in ("canary-prep", "canary-run"):
        from toffee.amputate.canary.__main__ import main as canary_main
        sub = "prep" if argv_list[0] == "canary-prep" else "run"
        return canary_main([sub] + argv_list[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "amputate_command", None):
        parser.print_help()
        return 2
    return run_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
