"""
TOFFEE training loop.

Wraps HuggingFace Trainer with ONYX-style hardware awareness and
state management for resume/overwrite.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from toffee.config import AppState, DatasetConfig, ExportConfig, LoRAConfig, TrainingConfig
from toffee.data import prepare_dataset, load_batches_from_disk
from toffee.hardware import get_gpu_info, suggest_lora_config, suggest_training_config
from toffee.lora import apply_lora


def _load_state(output_dir: Path) -> Optional[AppState]:
    state_path = output_dir / "state.json"
    if not state_path.exists():
        return None
    with open(state_path, "r") as f:
        return AppState(**json.load(f))


def _save_state(output_dir: Path, state: AppState) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = output_dir / "state.json.tmp"
    with open(tmp, "w") as f:
        json.dump(state.model_dump(), f, indent=2)
    tmp.replace(output_dir / "state.json")


def train(
    model_name: str,
    dataset_config: DatasetConfig,
    lora_config: LoRAConfig,
    training_config: TrainingConfig,
    export_config: ExportConfig,
    *,
    resume: bool = True,
) -> str:
    """Run Pass 2: LoRA fine-tuning.

    Returns path to the final adapter directory.
    """
    output_dir = training_config.output_dir
    gpu = get_gpu_info()

    # Auto-suggest configs if they look like defaults
    if lora_config.r == 8 and gpu is not None:
        suggested = suggest_lora_config(model_name, gpu)
        if suggested.r != lora_config.r:
            lora_config = suggested

    if training_config.per_device_train_batch_size == 1 and gpu is not None:
        suggested = suggest_training_config(model_name, gpu, lora_r=lora_config.r)
        if suggested.per_device_train_batch_size != training_config.per_device_train_batch_size:
            training_config = suggested

    # State / resume
    state = _load_state(output_dir) if resume else None
    last_completed = state.last_completed_step if state else None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    data_dir = output_dir / "data"
    if last_completed != "dataset_prepared" or not list(data_dir.glob("batch_*.pt")):
        print("[TOFFEE] Preparing dataset...")
        batches = prepare_dataset(dataset_config, tokenizer, data_dir)
        print(f"[TOFFEE] Dataset prepared: {len(batches)} batches")
        _save_state(output_dir, AppState(
            current_pass=1,
            last_completed_step="dataset_prepared",
            started_at=datetime.now(timezone.utc).isoformat(),
            model_name=model_name,
            dataset_path=dataset_config.path,
            lora_r=lora_config.r,
        ))
    else:
        print("[TOFFEE] Reusing prepared dataset from disk.")

    # Load model
    print(f"[TOFFEE] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto" if gpu is not None else None,
        torch_dtype=torch.bfloat16 if (gpu and gpu.supports_bf16) else torch.float16,
    )

    # Apply LoRA
    print("[TOFFEE] Applying LoRA adapters...")
    model = apply_lora(model, lora_config)

    # Load batches
    batches = load_batches_from_disk(data_dir)
    if not batches:
        raise RuntimeError("No batches found. Run dataset preparation first.")

    # Build a simple Dataset from batches
    class _BatchDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(batches)

        def __getitem__(self, idx):
            return {
                "input_ids": batches[idx].input_ids,
                "attention_mask": batches[idx].attention_mask,
                "labels": batches[idx].input_ids,
            }

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        num_train_epochs=training_config.num_train_epochs,
        max_steps=training_config.max_steps if training_config.max_steps > 0 else -1,
        warmup_ratio=training_config.warmup_ratio,
        weight_decay=training_config.weight_decay,
        lr_scheduler_type=training_config.lr_scheduler_type,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        gradient_checkpointing=training_config.gradient_checkpointing,
        gradient_checkpointing_kwargs=training_config.gradient_checkpointing_kwargs,
        deepspeed=training_config.deepspeed,
        max_grad_norm=training_config.max_grad_norm,
        seed=training_config.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=_BatchDataset(),
        data_collator=data_collator,
    )

    print("[TOFFEE] Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume)

    # Save final adapter
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    _save_state(output_dir, AppState(
        current_pass=2,
        last_completed_step="training_complete",
        started_at=state.started_at if state else datetime.now(timezone.utc).isoformat(),
        model_name=model_name,
        dataset_path=dataset_config.path,
        lora_r=lora_config.r,
    ))

    print(f"[TOFFEE] Training complete. Adapter saved to: {adapter_dir}")
    return str(adapter_dir)
