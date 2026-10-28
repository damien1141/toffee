"""
TOFFEE merge utilities.

Supports:
- Single adapter merge into base model
- Weighted multi-adapter merge (e.g., 0.7 * SFT + 0.3 * DPO)
- Optional GGUF export in one shot
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from toffee.config import AppState
from toffee.exporter import export_to_gguf, merge_lora
from toffee.state import save_state

logger = logging.getLogger(__name__)


def merge_single(
    base_model: str,
    adapter_dir: Path,
    output_dir: Path,
    weight: float = 1.0,
) -> Path:
    """Merge a single LoRA adapter into the base model.

    Args:
        base_model: Path or name of the base model
        adapter_dir: Path to the LoRA adapter
        output_dir: Where to save the merged model
        weight: Scaling factor for the adapter delta (default: 1.0)

    Returns:
        Path to the merged model directory
    """
    if weight <= 0:
        raise ValueError(f"Adapter weight must be positive, got {weight}")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TOFFEE] Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"[TOFFEE] Loading adapter: {adapter_dir} (weight={weight})")
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, adapter_dir)

    if weight != 1.0:
        # Scale adapter weights before merging
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data *= weight

    print("[TOFFEE] Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"[TOFFEE] Saving merged model to: {merged_dir}")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    # Save metadata
    meta = {
        "base_model": base_model,
        "adapter": str(adapter_dir),
        "weight": weight,
        "type": "single_adapter_merge",
    }
    with open(merged_dir / "toffee_merge_meta.json", "w") as f:
        import json
        json.dump(meta, f, indent=2)

    logger.info("merge_single_complete", adapter=str(adapter_dir), weight=weight, output=str(merged_dir))
    return merged_dir


def merge_multiple(
    base_model: str,
    adapters: List[Tuple[Path, float]],
    output_dir: Path,
) -> Path:
    """Merge multiple LoRA adapters with weighted averaging.

    The merge formula is:
        merged = base + sum(weight_i * (adapter_i - base))

    All adapters must be compatible with the base model architecture.

    Args:
        base_model: Path or name of the base model
        adapters: List of (adapter_dir, weight) tuples
        output_dir: Where to save the merged model

    Returns:
        Path to the merged model directory
    """
    if not adapters:
        raise ValueError("At least one adapter is required for merging")
    if len(adapters) == 1:
        return merge_single(base_model, adapters[0][0], output_dir, weight=adapters[0][1])

    # Validate weights before loading anything
    for adapter_path, weight in adapters:
        if weight <= 0:
            raise ValueError(f"Adapter weight must be positive, got {weight} for {adapter_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TOFFEE] Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Collect all adapter delta weights
    adapter_deltas = []
    total_weight = 0.0
    for adapter_path, weight in adapters:
        print(f"[TOFFEE] Loading adapter: {adapter_path} (weight={weight})")
        from peft import PeftModel
        adapter_model = PeftModel.from_pretrained(base, adapter_path)
        delta = _extract_lora_delta(adapter_model)
        adapter_deltas.append((delta, weight))
        total_weight += weight
        del adapter_model

    # Normalize weights so they sum to 1.0
    if abs(total_weight - 1.0) > 1e-6:
        print(f"[TOFFEE] Normalizing weights (sum={total_weight:.4f})")
        adapter_deltas = [(delta, w / total_weight) for delta, w in adapter_deltas]

    # Apply weighted deltas to base model
    print("[TOFFEE] Applying weighted adapter deltas...")
    merged_state = base.state_dict()
    for delta, weight in adapter_deltas:
        for key, delta_tensor in delta.items():
            if key in merged_state:
                merged_state[key] += weight * delta_tensor
            else:
                logger.warning("merge_missing_key", key=key)

    # Load merged state into base model
    base.load_state_dict(merged_state, strict=False)

    print(f"[TOFFEE] Saving merged model to: {merged_dir}")
    base.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    # Save metadata
    meta = {
        "base_model": base_model,
        "adapters": [
            {"path": str(adapter_path), "weight": weight}
            for adapter_path, weight in adapters
        ],
        "total_weight": total_weight,
        "type": "multi_adapter_merge",
    }
    with open(merged_dir / "toffee_merge_meta.json", "w") as f:
        import json
        json.dump(meta, f, indent=2)

    logger.info("merge_multiple_complete", adapters=len(adapters), output=str(merged_dir))
    return merged_dir


def _extract_lora_delta(peft_model) -> dict:
    """Extract the LoRA delta weights from a PeftModel.

    Returns a dict mapping parameter names to their LoRA deltas.
    """
    base_state = peft_model.get_base_model().state_dict()
    lora_delta = {}

    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            # LoRA parameter: delta = lora_A @ lora_B
            # The actual weight change is stored in the PEFT model's state
            base_name = name.replace("lora_", "").replace("default.", "")
            if base_name in base_state:
                delta = param.data - base_state[base_name]
                lora_delta[base_name] = delta.cpu()

    return lora_delta


def merge(
    base_model: str,
    adapter_dirs: List[Path],
    weights: Optional[List[float]],
    output_dir: Path,
    gguf: bool = False,
    gguf_outfile: Optional[Path] = None,
    gguf_quant: str = "Q4_K_M",
) -> Path:
    """Main merge entry point.

    Args:
        base_model: Path or name of the base model
        adapter_dirs: List of adapter directories to merge
        weights: Optional list of weights for each adapter. If None, equal weights.
        output_dir: Where to save the merged model
        gguf: Whether to also export to GGUF
        gguf_outfile: Optional GGUF output path
        gguf_quant: GGUF quantization type

    Returns:
        Path to the merged model directory
    """
    if not adapter_dirs:
        raise ValueError("At least one adapter directory is required")

    if weights is None:
        weights = [1.0 / len(adapter_dirs)] * len(adapter_dirs)
    if len(weights) != len(adapter_dirs):
        raise ValueError(f"Number of weights ({len(weights)}) must match adapters ({len(adapter_dirs)})")

    adapters = list(zip(adapter_dirs, weights))

    if len(adapters) == 1:
        merged_dir = merge_single(base_model, adapters[0][0], output_dir, weight=adapters[0][1])
    else:
        merged_dir = merge_multiple(base_model, adapters, output_dir)

    # Save state
    save_state(output_dir, AppState(
        current_pass=3,
        last_completed_step="merge_complete",
        model_name=base_model,
    ))

    # Optional GGUF export
    if gguf:
        outfile = gguf_outfile or (output_dir / f"{Path(base_model).name}-merged-{gguf_quant}.gguf")
        print(f"[TOFFEE] Exporting to GGUF: {outfile}")
        export_to_gguf(merged_dir, outfile, quant=gguf_quant)

    logger.info("merge_complete", output=str(merged_dir))
    return merged_dir
