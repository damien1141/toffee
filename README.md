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

#### Tiers: two axes, not a size ladder

The tiers optimize **quality** and **decode speed**, and the size follows from
that. They are deliberately *not* monotonic in size:

| tier | quality | decode | imatrix tax | ~bpw | ~GB (36B MoE) | what it is for |
|------|---------|--------|-------------|------|---------------|----------------|
| `quality` | ★★★★★ | ★★ | yes | 5.03 | 21.0 | Best quality per byte; IQ types in the bulk |
| `balanced` | ★★★★ | ★★★★★ | **no** | 5.78 | 24.2 | No imatrix decode tax; K-quants only |
| `compact` | ★★★ | ★★★★ | **no** | 3.85 | 16.1 | Small *and* fast |
| `mini` | ★★ | ★★ | yes | 2.59 | 10.8 | Small; IQ types in the bulk |
| `nano` | ★ | ★ | yes | 2.33 | 9.7 | Smallest; IQ1_M in the bulk |

`balanced` is **bigger** than `quality` on purpose: the IQ/TQ types give more
quality per bit, but they decode slower and need an imatrix to be calibrated.
If the model is decoded continuously, giving up the tax is worth more than the
extra gigabytes.

```bash
# See the table for yourself
python -m toffee quant tiers
python -m toffee quant tiers --json
```

### Quantization Refinements (RCO + GSQ)

The APEX tier stays the **policy**: it decides which quant type each tensor
role gets. Two optional engines refine it *underneath* the tier — the tier is
never displaced, only sharpened.

**RCO — exactly budget-constrained mixed precision.** Keeps the tier's
per-role quant family as the option set and searches over it on the *budget
manifold* (tangent projection + retraction + vector transport), so the model
lands on the target size exactly instead of approximately. The output is
still a plain llama.cpp `--tensor-type-file`.

```bash
# Give the experts a precision ladder and fit the model to a size budget
python -m toffee quant --model ./Apodex-1.1-mini --output-dir ./toffee-quant-output \
  --tier quality \
  rco --target-size-gb 6.0 \
  --ladder ffn_gate_exps=Q6_K,Q5_K,Q4_K \
  --ladder ffn_down_exps=Q5_K,Q4_K,Q3_K
```

**GSQ — in-format weight refinement.** Re-learns the integer codes and the
per-group scales of an already-quantized GGUF, using a Gumbel-Softmax
relaxation of the code assignment plus an exact coordinate descent over the
integer scale fields. The file stays a valid GGUF that llama.cpp loads
unchanged — only the numbers inside change.

```bash
# Refine a quantized GGUF in place (needs the f16 GGUF and the imatrix)
python -m toffee quant gsq \
  --gguf ./toffee-quant-output/model-TOFFEE-quality-Q4_K_M.gguf \
  --f16-gguf ./toffee-quant-output/model-f16.gguf \
  --imatrix ./toffee-quant-output/imatrix.dat \
  --epochs 8 --scale-passes 3
```

Measured against llama.cpp's own quantizer on the same importance matrix
(64×512 weights, weighted reconstruction error):

| type | error reduction |
|------|-----------------|
| Q2_K | +3.8%           |
| Q3_K | +1.3%           |
| Q4_K | +1.6%           |
| Q5_K | +5.0%           |
| Q6_K | +2.5%           |

> The refinement calls llama.cpp's *own* quantizer through a ctypes bridge
> (`toffee/quant/ggml_bridge.py`), so "what llama.cpp would do" and "what
> TOFFEE does" can never drift apart. The K-quant block codec
> (`toffee/quant/gguf_kquant.py`) is verified against it: dequantization is
> bit-exact and re-encoding is byte-identical.

> **Note on the CLI:** the parent options come *before* the subcommand
> (`--model`, `--output-dir`, `--tier` belong to `toffee quant`), while
> `--target-size-gb` and `--ladder` belong to `toffee quant rco`.
>
> ```bash
> python -m toffee quant --model ./Apodex-1.1-mini --output-dir ./out --tier quality \
>   rco --target-size-gb 6.0 --ladder ffn_gate_exps=Q6_K,Q5_K,Q4_K
> ```
>
> A role without a ladder keeps the tier's own type. Roles that *do* have a
> ladder share one global option axis (the union of the ladders), so the
> budget can be traded between them; the search never exceeds the requested
> size and fills it as fully as the ladder allows.
>
> **Validating the result:** use `llama-completion -m <file> -n 4`. Do *not*
> use `llama-gguf` — it is a self-test tool that asserts every tensor holds
> the literal value `100 + i`, so no real model passes it.

### Large models (layer streaming)

Models that do not fit in VRAM — or whose intermediate GGUF does not fit on
disk — are handled by streaming:

| Stage | How it stays bounded |
|-------|----------------------|
| Pass 2 (imatrix) | The model is streamed **layer by layer**: one layer's weights are resident at a time, and the hidden states pass between layers through CPU RAM. |
| F16 export | `--split-max-size 4G` writes a **split** GGUF, so a 71 GB intermediate never exists as one file. `llama-quantize` reads the shards directly. |
| Pass 3 (quantize) | `llama-quantize` memory-maps the F16 input, so it does not need it in RAM. |
| GSQ refinement | Streamed **by block**, with a bounded working set per tensor, and resumable via `--state` / `--resume`. |

```bash
# A 67GB MoE on a 12GB GPU: F16 (split) -> imatrix -> RCO -> Q4_K_M
bash run_sage_quant.sh

# or by hand
python -m toffee quant --model ./Sage-A1.5 --output-dir ./sage-out \
  --vram-limit 8 --samples 256 --batch-size 32 --streaming

python -m toffee quant --model ./Sage-A1.5 --output-dir ./sage-out \
  --gguf-quant Q4_K_M --split-max-size 4G \
  --tensor-type-file ./sage-out/tensor-type-file-quality-rco.txt
```

> **Disk:** the F16 export and the quantized output must both fit. For a
> 67 GB model that is ~71 GB + ~19 GB. Note that `/tmp` is often a small
> tmpfs — put the output directory on the real disk.

> **The RCO map is applied.** Pass 3 prefers
> `tensor-type-file-{tier}-rco.txt` when it exists, so the per-role quant
> types RCO chose under the budget are the ones llama-quantize uses.

#### GSQ is a refinement, and it is opt-in

GSQ re-learns the codes and scales of an already-quantized model. It is
**not** part of the quantization pipeline: it is a separate, explicit step.

It is also **slow on a large model**. Measured on Sage-A1.5 (36B, 33B of
routed expert parameters):

| type | elements | time | gain |
|------|----------|------|------|
| IQ4_XS | 33.0B | ~22 h | +1.44% error |
| Q4_K | 1.2B | ~1.9 h | +1.55% error |
| Q6_K | 0.1B | ~0.5 h | +2.81% error |

That is ~24 hours for a ~1.4% reconstruction-error gain, so it is opt-in in
`run_sage_quant.sh` (`GSQ=1`) and always bounded by a time budget:

```bash
# Opt in, with a 2-hour cap. The run checkpoints per block, so it can be
# resumed and it stops cleanly when the budget is used up.
GSQ=1 GSQ_MAX_TENSORS=0 bash run_sage_quant.sh

# or directly
python -m toffee quant gsq \
  --gguf ./sage-out/Sage-A1.5-TOFFEE-quality-Q4_K_M.gguf \
  --f16-gguf ./sage-out/Sage-A1.5-f16.gguf \
  --imatrix ./sage-out/imatrix.dat \
  --time-budget-minutes 120 \
  --state ./sage-out/gsq-state.json --resume
```

> **The Gumbel-Softmax relaxation is off by default** (`--epochs 0`). It was
> measured to contribute exactly **0%** — llama.cpp's own code assignment is
> already optimal given its scales, and the scale search subsumes it. On
> real Sage-A1.5 weights, `epochs=8` and `epochs=0` give the same error to
> 10 decimal places, with `epochs=8` taking ~45% longer. Pass `--epochs 8`
> to run it anyway.

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
| `toffee quant rco` | Budget-constrained refinement of the APEX tier map |
| `toffee quant gsq` | In-format Gumbel-Softmax refinement of a quantized GGUF |
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
    ggml_bridge.py    # ctypes bridge to llama.cpp's own quantizer
    gguf_kquant.py    # Exact K-quant block codecs (decode/encode)
    gguf_io.py        # GGUF tensor index, read and in-place patching
    hybrid_model.py   # Minimal Qwen3.5 hybrid model for imatrix harvesting
    rco.py            # Budget manifold + budget-constrained DP (RCO)
    gsq.py            # Gumbel-Softmax code/scale refinement (GSQ)
    refine.py         # APEX + RCO / APEX + GSQ orchestrators

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

## Attribution

The quantization engines in `toffee/quant/` are adapted from two upstream
research implementations, both Apache-2.0 licensed:

- **GSQ** — *GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs
  via Gumbel-Softmax Sampling* (Dadgarnia, Tabesh, Nikdan, Helcig,
  Kurtić, Kleinegger, Alistarh). <https://github.com/IST-DASLab/GSQ>
  → `toffee/quant/gsq.py`
- **RCO** — *Model Compression with Exact Budget Constraints via Riemannian
  Manifolds* (Helcig, Alistarh). <https://github.com/IST-DASLab/RCO>
  → `toffee/quant/rco.py`

Both were adapted to TOFFEE's contract: the APEX tier remains the policy
layer, the output remains a GGUF that llama.cpp loads unchanged, and the
quantization itself is delegated to llama.cpp's own quantizer through
`toffee/quant/ggml_bridge.py`.

`toffee/quant/hybrid_model.py` is adapted from the ONYX quantization engine's
minimal hybrid text model (`_MinimalHybridDecoderLayer` /
`_MinimalHybridTextModel`), which solves the same problem for the same
architectures: declaring the real module hierarchy and weight shapes for
imatrix harvesting when the exact `transformers` classes are unavailable.

> The layer-streaming design — one layer resident at a time, hidden states
> passed between layers through CPU RAM — follows ONYX's pass 2.

---

## License

AGPL v3
