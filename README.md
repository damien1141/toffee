# TOFFEE

Fine-tune, quantize, and serve LLMs on consumer GPUs.

Three domains, one CLI:

- `train` — LoRA SFT, DPO/KTO alignment, adapter merging
- `quant` — imatrix harvest + APEX GGUF quantization
- `xspark` — speculative decoding draft model

Runs below the model's own VRAM size by streaming layers one at a time.

## Install

```bash
pip install torch transformers peft trl datasets gguf safetensors accelerate
```

## Quick start

### Quantize a model (full pipeline, one command)

```bash
python -m toffee quant --model ./Sage-A1.5 --output-dir ./out \
  --tier quality --samples 256 --gguf-quant Q4_K_M
```

This converts safetensors to f16 GGUF, harvests an imatrix by streaming
the model layer by layer, and quantizes with llama.cpp's own quantizer.

For a model too big for one file, add `--split-max-size 4G`:

```bash
python -m toffee quant --model ./Sage-A1.5 --output-dir ./out \
  --tier quality --gguf-quant Q4_K_M --split-max-size 4G
```

### Tiers

| tier | size (36B MoE) | imatrix needed | use when |
|------|----------------|----------------|----------|
| quality | 21 GB | yes | best quality per byte, model sits on disk |
| balanced | 24 GB | no | served continuously, no decode tax |
| compact | 16 GB | no | VRAM or disk is the binding constraint |
| mini | 11 GB | yes | nothing else fits |
| nano | 10 GB | yes | smallest possible |

`balanced` is deliberately bigger than `quality`: IQ/TQ types give more
quality per bit but decode slower and need an imatrix. If the model is
decoded continuously, skipping the tax beats the extra gigabytes.

### Fine-tune

```bash
python -m toffee train --model ./model --dataset ./data --lora-r 8 --epochs 3
python -m toffee align --model ./model --dataset ./pairs --method dpo
python -m toffee merge --model ./model --adapter ./a --weight 0.7 --adapter ./b --weight 0.3 --gguf
```

### Speculative decoding

```bash
python -m toffee xspark run-all --model ./model --target-gguf ./model-q4km.gguf --samples 1000
```

## Hardware awareness

TOFFEE auto-detects your GPU and picks LoRA rank, batch size, and precision:

| VRAM | rank | batch | precision |
|------|------|-------|-----------|
| < 8 GB | 4 | 1 | fp16 |
| 8-16 GB | 8 | 1-2 | bf16 |
| 16-24 GB | 16 | 4 | bf16 |
| > 24 GB | 32 | 8 | bf16 |

Everything is overridable with CLI flags.

## Resume

Every command writes `state.json` to the output dir. If a run is
interrupted, TOFFEE prompts to resume or start over. No work is lost.

## Architecture

Built for Qwen3.5-MoE hybrid models (GatedDeltaNet SSM + full attention +
routed/shared experts). The quantization engine is ported from ONYX.

```
toffee/
  cli.py          # unified CLI
  config.py       # Pydantic config
  hardware.py     # GPU detection
  state.py        # resume/overwrite
  data.py         # dataset loading
  quant/          # imatrix, APEX maps, passes
  xspark/         # speculative decoding
```

## License

AGPL v3