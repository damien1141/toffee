# TOFFEE

**可训练输出无关微调执行环境**

> *让 AI 研究触手可及。去他妈的 NVIDIA 和他们的天价显卡。*
> *注意：此软件尚处于实验阶段，可能无法开箱即用，你需要在 Python 层面亲自动手。*

TOFFEE 是一个一体化、硬件感知的工具包，用于微调、量化和部署大语言模型。它针对 Qwen3.5-MoE 混合架构优化，但也适用于其他模型。它覆盖三个核心领域：

1. **微调** — LoRA SFT、DPO/KTO 对齐、适配器合并
2. **量化** — imatrix 生成、APEX 动态 tensor-type 映射、GGUF 导出
3. **推测解码** — "XSpark" draft model training for dspark

一切设计都围绕一个原则：**每 GB VRAM 最大化能力**。

---

## 为什么需要 TOFFEE？

现有工具迫使你在功能和易用性之间做选择。Axolotl 有 47 个配置旋钮和一个看起来像电子表格的 YAML 文件。llama.cpp 量化功能强大，但需要手动 imatrix 校准和 tensor-type-file 调优。推测解码工具分散在三个仓库和一篇论文里。

TOFFEE 将所有三个领域统一到一个 CLI 下，提供合理的默认值、自动硬件检测和可恢复的状态管理。它不会牵着你的手 —— 它只是不挡你的路。

---

## 环境要求

- Python 3.11+
- PyTorch 2.0+
- 支持 CUDA 的 GPU（推荐 6GB+ VRAM）
- `transformers`, `peft`, `trl`, `datasets`, `gguf`, `safetensors`

安装依赖：

```bash
pip install torch transformers peft trl datasets gguf safetensors accelerate
```

---

## 快速开始

### 微调

```bash
# LoRA SFT
python -m toffee train \
  --model ./Apodex-1.1-mini \
  --dataset el4/bartowski-imatrix-v5-semantic-parquet \
  --lora-r 8 \
  --epochs 3

# DPO 对齐
python -m toffee align \
  --model ./Apodex-1.1-mini \
  --dataset anthropic/hh-rlhf \
  --method dpo \
  --beta 0.1 \
  --epochs 1

# 合并多个适配器
python -m toffee merge \
  --model ./Apodex-1.1-mini \
  --adapter ./adapter-sft --weight 0.7 \
  --adapter ./adapter-dpo --weight 0.3 \
  --gguf --gguf-quant Q4_K_M
```

### 量化

```bash
# 带 imatrix 的完整量化流程
python -m toffee quant run-all \
  --model ./Apodex-1.1-mini \
  --tier quality \
  --samples 256 \
  --gguf-quant Q4_K_M
```

可选精度：`quality`, `balanced`, `compact`, `mini`, `nano`

### 推测解码（XSpark）

```bash
# 完整的 draft model 训练流程
python -m toffee xspark run-all \
  --model ./Apodex-1.1-mini \
  --target-gguf ./Apodex-1.1-mini-TOFFEE-quality-Q4_K_M.gguf \
  --samples 1000 \
  --epochs 3 \
  --target-layers 15 30
```

---

## 硬件感知

TOFFEE 自动检测你的 GPU 并调整配置：

| 显存 | LoRA 秩 | 批次大小 | 精度 |
|------|---------|----------|------|
| < 8GB | 4 | 1，梯度累积 16 | fp16 |
| 8–16GB | 8 | 1–2，梯度累积 8 | bf16/fp16 |
| 16–24GB | 16 | 4，梯度累积 2 | bf16 |
| > 24GB | 32 | 8，梯度累积 1 | bf16 |

24GB 以下自动启用梯度检查点。所有参数都可以通过 CLI 参数覆盖。

---

## 命令参考

| 命令 | 功能 |
|------|------|
| `toffee train` | LoRA SFT 微调 |
| `toffee align` | DPO/KTO 对齐 |
| `toffee merge` | 加权多适配器合并 |
| `toffee export` | 合并 + GGUF 导出 |
| `toffee quant run-all` | imatrix + APEX GGUF 量化 |
| `toffee xspark prepare` | 获取校准数据集 |
| `toffee xspark harvest` | 提取教师激活 |
| `toffee xspark train` | 训练 XSpark draft head |
| `toffee xspark export` | 导出 draft GGUF |
| `toffee xspark run-all` | 完整 XSpark 流程 |

---

## 项目结构

```
toffee/
  cli.py              # 统一 CLI 入口
  config.py           # Pydantic 配置模型
  hardware.py         # GPU 检测、VRAM 感知建议
  state.py            # 恢复/覆盖状态管理
  data.py             # 数据集加载和分词

  finetune/
    lora.py           # 面向 MoE 的 LoRA 目标选择
    trainer.py        # HuggingFace Trainer 封装
    alignment.py      # 通过 TRL 实现 DPO/KTO 对齐
    merge.py          # 单/多适配器加权合并

  quant/
    config.py         # 带回退的配置加载
    masks.py          # 4D 因果掩码、RoPE 调度
    mapping.py        # GGUF tensor 类别推断、APEX tensor 映射
    imatrix.py        # ImatrixWriter、ActivationScaleCapturer
    passes.py         # Pass 2 提取、Pass 3 量化
    cli.py            # 量化 CLI

  xspark/
    cli.py            # XSpark CLI
    data.py           # 数据集获取、Parquet 缓存
    harvest.py        # 教师激活流式提取
    train.py          # XSpark draft head 训练
    export.py         # 用于 dflash/DSV4 的 GGUF 导出
```

---

## 架构支持

TOFFEE 针对 **Qwen3.5-MoE 混合**架构（`qwen3_5_moe` 模型类型）构建，支持：

- `linear_attention` 块（GatedDeltaNet / SSM）
- `full_attention` 块（标准 transformer attention）
- 带路由专家和共享专家的 MoE 层
- 带 per-role 精度带的 APEX 动态量化

它移植了 ONYX 量化引擎，并在其之上添加了推测解码功能。相同的硬件感知理念贯穿所有三个领域。

---

## 状态管理

每个命令都会向输出目录写入一个 `state.json`。如果运行中断，TOFFEE 会提示：

```
[R] 从上次完成的步骤继续
[O] 覆盖并重新开始
[Q] 退出
```

不会丢失任何工作。不会重新下载数据集。不会重新提取任何层。

---

## 贡献指南

TOFFEE 是一个有主见的软件。如果你想添加功能：

1. 它必须是硬件感知的，或者让硬件感知更容易
2. 它必须保持现有的状态/恢复契约
3. 它不能破坏三个领域的分离（微调 / 量化 / xspark）
4. 它必须有测试

---

## 许可证

AGPL v3
