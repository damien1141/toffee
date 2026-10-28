"""
TOFFEE configuration models.

All tunable knobs live here as Pydantic models with validated ranges.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class DatasetConfig(BaseModel):
    """Dataset composition and tokenization settings."""

    path: str = Field(description="HuggingFace dataset name or local path")
    split: str = Field(default="train", description="Dataset split to use")
    text_field: str = Field(default="text", description="Field containing the text")
    max_length: int = Field(default=2048, ge=64, le=32768)
    format_template: Optional[str] = Field(
        default=None,
        description="Optional chat template string with {text} placeholder",
    )
    streaming: bool = Field(default=False, description="Stream dataset to avoid full download")

    model_config = {"extra": "forbid"}


class LoRAConfig(BaseModel):
    """LoRA/QLoRA adapter configuration."""

    r: int = Field(default=8, ge=1, le=256, description="LoRA rank")
    alpha: int = Field(default=16, description="LoRA alpha scaling factor")
    dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    target_modules: Optional[List[str]] = Field(
        default=None,
        description="Modules to apply LoRA to. None = auto-detect for Qwen3.5-MoE.",
    )
    bias: str = Field(default="none", pattern="^(none|all|lora_only)$")
    modules_to_save: Optional[List[str]] = Field(
        default=None,
        description="Modules to train fully (not LoRA). None = auto.",
    )
    use_rslora: bool = Field(default=False, description="Use rank-stabilized LoRA")
    use_dora: bool = Field(default=False, description="Use DoRA")

    model_config = {"extra": "forbid"}


class TrainingConfig(BaseModel):
    """Training hyperparameters and hardware settings."""

    output_dir: Path = Field(default=Path("toffee-output"))
    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    num_train_epochs: float = Field(default=3.0, gt=0.0)
    max_steps: int = Field(default=-1, description="Override epochs if > 0")
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.01, ge=0.0)
    lr_scheduler_type: str = Field(default="cosine", description="HuggingFace scheduler name")
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=500, ge=1)
    save_total_limit: int = Field(default=3, ge=1)
    fp16: bool = Field(default=False, description="Use fp16 mixed precision")
    bf16: bool = Field(default=True, description="Use bf16 mixed precision (Ampere+)")
    gradient_checkpointing: bool = Field(default=True)
    gradient_checkpointing_kwargs: dict = Field(default_factory=lambda: {"use_reentrant": False})
    deepspeed: Optional[Union[str, dict]] = Field(
        default=None,
        description="DeepSpeed config path or dict. None = no DeepSpeed.",
    )
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    seed: int = Field(default=42)

    model_config = {"extra": "forbid"}


class ExportConfig(BaseModel):
    """Export settings for Pass 3."""

    merge: bool = Field(default=True, description="Merge LoRA into base model")
    push_to_hub: bool = Field(default=False)
    hub_model_id: Optional[str] = Field(default=None)
    hub_token: Optional[str] = Field(default=None)
    gguf: bool = Field(default=False, description="Also export to GGUF")
    gguf_outfile: Optional[Path] = Field(default=None)
    gguf_quant: str = Field(default="Q4_K_M")

    model_config = {"extra": "forbid"}


class AlignmentConfig(BaseModel):
    """Alignment method configuration."""

    method: str = Field(
        default="dpo",
        pattern="^(dpo|ipo|kto)$",
        description="Alignment algorithm: dpo, ipo, or kto",
    )
    beta: float = Field(default=0.1, gt=0.0, description="DPO/IPO temperature")
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.5)
    loss_type: str = Field(default="sigmoid", description="DPO loss type: sigmoid, hinge, or ipo")
    reference_free: bool = Field(default=False, description="DPO without reference model")
    reference_model: Optional[str] = Field(
        default=None,
        description="Path to reference model. None = use base model.",
    )
    max_length: Optional[int] = Field(default=None, description="Override max prompt+response length")
    max_prompt_length: Optional[int] = Field(default=None, description="Max prompt length")
    max_target_length: Optional[int] = Field(default=None, description="Max response length")

    model_config = {"extra": "forbid"}


class AppState(BaseModel):
    """Persistent state for resume/overwrite."""

    current_pass: int = 0
    last_completed_step: str = ""
    started_at: Optional[str] = None
    model_name: Optional[str] = None
    dataset_path: Optional[str] = None
    lora_r: Optional[int] = None
    alignment_method: Optional[str] = None

    model_config = {"extra": "forbid"}
