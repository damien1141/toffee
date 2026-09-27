"""
TOFFEE export pass.

Merges LoRA adapters into the base model and optionally exports to GGUF.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from toffee.config import ExportConfig


def merge_lora(
    base_model: str,
    adapter_dir: Path,
    output_dir: Path,
) -> Path:
    """Merge LoRA adapters into the base model and save as full safetensors."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TOFFEE] Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"[TOFFEE] Loading adapter from: {adapter_dir}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()

    print(f"[TOFFEE] Merging adapters...")
    out_path = output_dir / "merged"
    model.save_pretrained(out_path, safe_serialization=True)
    tokenizer.save_pretrained(out_path)

    print(f"[TOFFEE] Merged model saved to: {out_path}")
    return out_path


def export_to_gguf(
    model_dir: Path,
    outfile: Path,
    quant: str = "Q4_K_M",
) -> None:
    """Export a HuggingFace model to GGUF via llama.cpp convert script."""
    if not outfile.suffix:
        outfile = outfile.with_suffix(".gguf")

    convert_cmd = [
        sys.executable,
        "./llama.cpp/convert_hf_to_gguf.py",
        str(model_dir),
        "--outtype", "f16",
        "--outfile", str(outfile),
    ]
    print(f"[TOFFEE] Converting to GGUF: {' '.join(convert_cmd)}")
    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed: {result.stderr}")

    # Quantize
    quant_cmd = [
        "./llama.cpp/build/bin/llama-quantize",
        str(outfile),
        str(outfile.with_suffix(f".{quant}.gguf")),
        quant,
    ]
    print(f"[TOFFEE] Quantizing: {' '.join(quant_cmd)}")
    qresult = subprocess.run(quant_cmd, capture_output=True, text=True)
    if qresult.returncode != 0:
        raise RuntimeError(f"GGUF quantization failed: {qresult.stderr}")

    print(f"[TOFFEE] GGUF export complete: {outfile}")


def export(
    base_model: str,
    adapter_dir: Path,
    config: ExportConfig,
    output_dir: Path,
) -> Path:
    """Run Pass 3: merge LoRA and optionally export to GGUF."""
    merged_dir = merge_lora(base_model, adapter_dir, output_dir)

    if config.gguf:
        outfile = config.gguf_outfile or (output_dir / f"{Path(base_model).name}-toffee.gguf")
        export_to_gguf(merged_dir, outfile, quant=config.gguf_quant)

    if config.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi(token=config.hub_token)
        api.create_repo(repo_id=config.hub_model_id, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=merged_dir,
            repo_id=config.hub_model_id,
            repo_type="model",
        )
        print(f"[TOFFEE] Pushed to hub: {config.hub_model_id}")

    return merged_dir
