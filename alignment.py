"""
TOFFEE alignment trainer.

Wraps TRL's DPO/KTO trainers with TOFFEE's hardware-aware memory management.
The reference model is kept on CPU by default to avoid doubling VRAM usage
on MoE models.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from toffee.config import AlignmentConfig, AppState, DatasetConfig, LoRAConfig, TrainingConfig
from toffee.data import prepare_dataset
from toffee.hardware import get_gpu_info, suggest_lora_config, suggest_training_config
from toffee.lora import apply_lora
from toffee.state import load_state, save_state


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


def _build_preference_dataset(dataset_config: DatasetConfig, tokenizer, output_dir: Path):
    """Load and tokenize a preference dataset.

    Expects a dataset with 'prompt', 'chosen', and 'rejected' columns.
    Falls back to 'instruction', 'chosen', 'rejected' if 'prompt' is absent.
    """
    try:
        ds = load_dataset(dataset_config.path, split=dataset_config.split, streaming=dataset_config.streaming)
    except Exception as e:
        raise RuntimeError(f"Failed to load preference dataset {dataset_config.path}: {e}") from e

    prompt_col = "prompt"
    if prompt_col not in ds.column_names:
        if "instruction" in ds.column_names:
            prompt_col = "instruction"
        else:
            raise ValueError(
                f"Dataset must have 'prompt' or 'instruction' column. Found: {ds.column_names}"
            )

    chosen_col = "chosen" if "chosen" in ds.column_names else "response_chosen"
    rejected_col = "rejected" if "rejected" in ds.column_names else "response_rejected"

    if chosen_col not in ds.column_names or rejected_col not in ds.column_names:
        raise ValueError(
            f"Dataset must have 'chosen' and 'rejected' columns. Found: {ds.column_names}"
        )

    def _tokenize(example):
        prompt = example[prompt_col]
        chosen = example[chosen_col]
        rejected = example[rejected_col]

        # Tokenize prompt separately to enforce max_prompt_length
        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=dataset_config.max_length // 2,
        )

        chosen_tokens = tokenizer(
            prompt + chosen,
            truncation=True,
            max_length=dataset_config.max_length,
        )
        rejected_tokens = tokenizer(
            prompt + rejected,
            truncation=True,
            max_length=dataset_config.max_length,
        )

        return {
            "prompt_input_ids": prompt_tokens["input_ids"],
            "prompt_attention_mask": prompt_tokens["attention_mask"],
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
        }

    tokenized = ds.map(_tokenize, remove_columns=ds.column_names)
    return tokenized


def align(
    model_name: str,
    dataset_config: DatasetConfig,
    lora_config: LoRAConfig,
    training_config: TrainingConfig,
    alignment_config: AlignmentConfig,
    *,
    resume: bool = True,
) -> str:
    """Run alignment fine-tuning (DPO/KTO).

    Returns path to the final adapter directory.
    """
    output_dir = training_config.output_dir
    gpu = get_gpu_info()

    # Auto-suggest configs
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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare preference dataset
    data_dir = output_dir / "data"
    preference_dataset = _build_preference_dataset(dataset_config, tokenizer, data_dir)

    # Load policy model
    print(f"[TOFFEE] Loading policy model: {model_name}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto" if gpu is not None else None,
        torch_dtype=torch.bfloat16 if (gpu and gpu.supports_bf16) else torch.float16,
    )

    # Apply LoRA
    print("[TOFFEE] Applying LoRA adapters...")
    policy_model = apply_lora(policy_model, lora_config)

    # Load reference model (CPU-offloaded by default to save VRAM)
    ref_model = None
    if not alignment_config.reference_free:
        ref_path = alignment_config.reference_model or model_name
        print(f"[TOFFEE] Loading reference model: {ref_path} (CPU)")
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_path,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.bfloat16 if (gpu and gpu.supports_bf16) else torch.float16,
        )
        # Freeze reference model
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    # Build trainer kwargs based on method
    trainer_kwargs = dict(
        model=policy_model,
        ref_model=ref_model,
        args=_build_alignment_training_args(training_config, alignment_config),
        train_dataset=preference_dataset,
        processing_class=tokenizer,
    )

    if alignment_config.method == "dpo":
        from trl import DPOConfig, DPOTrainer
        dpo_config = DPOConfig(
            beta=alignment_config.beta,
            label_smoothing=alignment_config.label_smoothing,
            loss_type=alignment_config.loss_type,
            max_length=alignment_config.max_length or dataset_config.max_length,
            max_prompt_length=alignment_config.max_prompt_length or dataset_config.max_length // 2,
            max_target_length=alignment_config.max_target_length or dataset_config.max_length // 2,
        )
        trainer = DPOTrainer(
            model=policy_model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=preference_dataset,
            processing_class=tokenizer,
        )
    elif alignment_config.method == "ipo":
        from trl import IPOConfig, IPOTrainer
        ipo_config = IPOConfig(
            beta=alignment_config.beta,
            label_smoothing=alignment_config.label_smoothing,
            max_length=alignment_config.max_length or dataset_config.max_length,
        )
        trainer = IPOTrainer(
            model=policy_model,
            ref_model=ref_model,
            args=ipo_config,
            train_dataset=preference_dataset,
            processing_class=tokenizer,
        )
    elif alignment_config.method == "kto":
        from trl import KTOConfig, KTOTrainer
        kto_config = KTOConfig(
            beta=alignment_config.beta,
            max_length=alignment_config.max_length or dataset_config.max_length,
        )
        trainer = KTOTrainer(
            model=policy_model,
            ref_model=ref_model,
            args=kto_config,
            train_dataset=preference_dataset,
            processing_class=tokenizer,
        )
    else:
        raise ValueError(f"Unknown alignment method: {alignment_config.method}")

    print(f"[TOFFEE] Starting {alignment_config.method.upper()} alignment...")
    train_result = trainer.train(resume_from_checkpoint=resume)

    # Save adapter
    adapter_dir = output_dir / "adapter"
    policy_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    _save_state(output_dir, AppState(
        current_pass=3,
        last_completed_step="alignment_complete",
        started_at=state.started_at if state else datetime.now(timezone.utc).isoformat(),
        model_name=model_name,
        dataset_path=dataset_config.path,
        alignment_method=alignment_config.method,
    ))

    print(f"[TOFFEE] Alignment complete. Adapter saved to: {adapter_dir}")
    return str(adapter_dir)


def _build_alignment_training_args(
    training_config: TrainingConfig,
    alignment_config: AlignmentConfig,
):
    """Build TrainingArguments for alignment trainer."""
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=str(training_config.output_dir / "checkpoints"),
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
        max_grad_norm=training_config.max_grad_norm,
        seed=training_config.seed,
        report_to="none",
        remove_unused_columns=False,
    )
