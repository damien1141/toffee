# TOFFEE

**Trainable Output-Free Fine-tuning Execution Environment**

> *Making AI research available to everyone. Screw NVIDIA and their overpriced GPUs.*
> *NOTE: This software is experimental, it may or may not work for you out of the box, you will have to get your hands dirty with python*

TOFFEE is an all-in-one, hardware-aware toolkit for fine-tuning, quantizing, and deploying large language models — optimized for Qwen3.5-MoE hybrid architectures, but flexible enough for any model. It handles the three domains that matter:

1. **Fine-tuning** — LoRA SFT, DPO/KTO alignment, adapter merging
2. **Quantization** — imatrix generation, APEX dynamic tensor-type maps, GGUF export
3. **Speculative decoding** — "XSpark" draft model training for dspark

Everything is designed around one principle: **maximum capability per GB of VRAM**.

---

## Why TOFFEE?

Existing tools force you to choose between power and usability. Axolotl has 47 configuration knobs and a YAML file that looks like a spreadsheet. llama.cpp quantization is powerful but requires manual imatrix calibration and tensor-type-file tuning. Speculative decoding tooling is scattered across three repos and a research paper.

TOFFEE unifies all three domains under one CLI with sane defaults, automatic hardware detection, and resume-safe state management. It doesn't hold your hand — it gets out of your way.

---

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA-capable GPU (6GB+ VRAM recommended)
- `transformers`, `peft`, `trl`, `datasets`, `gguf`, `safetensors`

Install dependencies:

```bash
pip install torch transformers peft trl datasets gguf safetensors accelerate
```

---

## Quick Start

### Fine-tuning

```bash
# LoRA SFT
python -m toffee train \
  --model ./Apodex-1.1-mini \
  --dataset el4/bartowski-imatrix-v5-semantic-parquet \
  --lora-r 8 \
  --epochs 3

# DPO alignment
python -m toffee align \
  --model ./Apodex-1.1-mini \
  --dataset anthropic/hh-rlhf \
  --method dpo \
  --beta 0.1 \
  --epochs 1

# Merge multiple adapters
python -m toffee merge \
  --model ./Apodex-1.1-mini \
  --adapter ./adapter-sft --weight 0.7 \
  --adapter ./adapter-dpo --weight 0.3 \
  --gguf --gguf-quant Q4_K_M
```

### Quantization

```bash
# Full quantization pipeline with imatrix
python -m toffee quant run-all \
  --model ./Apodex-1.1-mini \
  --tier quality \
  --samples 256 \
  --gguf-quant Q4_K_M
```

Tiers: `quality`, `balanced`, `compact`, `mini`, `nano`

### Speculative Decoding (XSpark)

```bash
# Full draft model training pipeline
python -m toffee xspark run-all \
  --model ./Apodex-1.1-mini \
  --target-gguf ./Apodex-1.1-mini-TOFFEE-quality-Q4_K_M.gguf \
  --samples 1000 \
  --epochs 3 \
  --target-layers 15 30
```

---

## Hardware Awareness

TOFFEE auto-detects your GPU and adjusts:

| VRAM | LoRA Rank | Batch Size | Precision |
|------|-----------|------------|-----------|
| < 8GB | 4 | 1, grad accum 16 | fp16 |
| 8–16GB | 8 | 1–2, grad accum 8 | bf16/fp16 |
| 16–24GB | 16 | 4, grad accum 2 | bf16 |
| > 24GB | 32 | 8, grad accum 1 | bf16 |

Gradient checkpointing is auto-enabled under 24GB. You can override anything with CLI flags.

---

## Commands

| Command | Purpose |
|---------|---------|
| `toffee train` | LoRA SFT fine-tuning |
| `toffee align` | DPO/KTO alignment |
| `toffee merge` | Weighted multi-adapter merge |
| `toffee export` | Merge + GGUF export |
| `toffee quant run-all` | Imatrix + APEX GGUF quantization |
| `toffee xspark prepare` | Fetch calibration datasets |
| `toffee xspark harvest` | Extract teacher activations |
| `toffee xspark train` | Train XSpark draft head |
| `toffee xspark export` | Export draft GGUF |
| `toffee xspark run-all` | Full XSpark pipeline |

---

## Project Structure

```
toffee/
  cli.py              # Unified CLI entry point
  config.py           # Pydantic config models
  hardware.py         # GPU detection, VRAM-aware suggestions
  state.py            # Resume/overwrite state management
  data.py             # Dataset loading and tokenization

  finetune/
    lora.py           # MoE-aware LoRA target selection
    trainer.py        # HuggingFace Trainer wrapper
    alignment.py      # DPO/KTO alignment via TRL
    merge.py          # Single/multi-adapter weighted merge

  quant/
    config.py         # Config loading with fallback
    masks.py          # 4D causal mask, RoPE dispatch
    mapping.py        # GGUF tensor category inference, APEX tensor maps
    imatrix.py        # ImatrixWriter, ActivationScaleCapturer
    passes.py         # Pass 2 harvest, Pass 3 quantize
    cli.py            # Quantization CLI

  xspark/
    cli.py            # XSpark CLI
    data.py           # Dataset fetching, Parquet caching
    harvest.py        # Teacher activation streaming
    train.py          # XSpark draft head training
    export.py         # GGUF export for dflash/DSV4
```

---

## Architecture Support

TOFFEE is built for **Qwen3.5-MoE hybrid** architectures (`qwen3_5_moe` model type) with:

- `linear_attention` blocks (GatedDeltaNet / SSM)
- `full_attention` blocks (standard transformer attention)
- MoE layers with routed experts and shared experts
- APEX dynamic quantization with per-role precision bands

It ports the ONYX quantization engine and adds speculative decoding on top. The same hardware-awareness philosophy runs through all three domains.

---

## State Management

Every command writes a `state.json` to the output directory. If a run is interrupted, TOFFEE prompts:

```
[R]esume from last completed step
[O]verwrite and start fresh
[Q]uit
```

No work is lost. No datasets are re-downloaded. No layers are re-harvested.

---

## Contributing

TOFFEE is opinionated software. If you want to add a feature:

1. It must be hardware-aware or make hardware awareness easier
2. It must preserve the existing state/resume contract
3. It must not break the three-domain separation (fine-tune / quant / xspark)
4. It must have tests

---

## License

AGPL v3
