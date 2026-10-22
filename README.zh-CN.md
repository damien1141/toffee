# TOFFEE

在消费级 GPU 上微调、量化和部署 LLM。

一个 CLI，三个领域：

- `train` — LoRA SFT、DPO/KTO 对齐、适配器合并
- `quant` — imatrix 提取 + APEX GGUF 量化
- `xspark` — 推测解码 draft 模型

逐层流式处理，可在低于模型自身显存的环境下运行。

## 安装

```bash
pip install torch transformers peft trl datasets gguf safetensors accelerate
```

## 快速开始

### 量化（一条命令跑完全流程）

```bash
python -m toffee quant --model ./Sage-A1.5 --output-dir ./out \
  --tier quality --samples 256 --gguf-quant Q4_K_M
```

依次完成：safetensors → f16 GGUF、逐层流式提取 imatrix、用 llama.cpp 自带量化器量化。

模型太大放不下单文件时加 `--split-max-size 4G`：

```bash
python -m toffee quant --model ./Sage-A1.5 --output-dir ./out \
  --tier quality --gguf-quant Q4_K_M --split-max-size 4G
```

### 精度档位

| 档位 | 大小 (36B MoE) | 需要 imatrix | 适用场景 |
|------|----------------|--------------|----------|
| quality | 21 GB | 是 | 单位字节质量最高，模型放在磁盘上 |
| balanced | 24 GB | 否 | 持续服务，无解码税 |
| compact | 16 GB | 否 | 显存或磁盘是瓶颈 |
| mini | 11 GB | 是 | 其他都放不下 |
| nano | 10 GB | 是 | 最小可能 |

`balanced` 刻意比 `quality` 大：IQ/TQ 类型单位比特质量更高，但解码慢且需要 imatrix。持续解码时，省下 imatrix 税比多出的几个 GB 更值。

### 微调

```bash
python -m toffee train --model ./model --dataset ./data --lora-r 8 --epochs 3
python -m toffee align --model ./model --dataset ./pairs --method dpo
python -m toffee merge --model ./model --adapter ./a --weight 0.7 --adapter ./b --weight 0.3 --gguf
```

### 推测解码

```bash
python -m toffee xspark run-all --model ./model --target-gguf ./model-q4km.gguf --samples 1000
```

## 硬件感知

自动检测 GPU 并调整 LoRA 秩、批次大小和精度：

| 显存 | 秩 | 批次 | 精度 |
|------|------|-------|-----------|
| < 8 GB | 4 | 1 | fp16 |
| 8-16 GB | 8 | 1-2 | bf16 |
| 16-24 GB | 16 | 4 | bf16 |
| > 24 GB | 32 | 8 | bf16 |

所有参数都可用 CLI 覆盖。

## 断点续跑

每个命令都会向输出目录写入 `state.json`。运行中断后，TOFFEE 会提示继续或重来。不会丢失任何工作。

## 架构

面向 Qwen3.5-MoE 混合模型（GatedDeltaNet SSM + 全注意力 + 路由/共享专家）。量化引擎移植自 ONYX。

```
toffee/
  cli.py          # 统一 CLI
  config.py       # Pydantic 配置
  hardware.py     # GPU 检测
  state.py        # 断点续跑
  data.py         # 数据集加载
  quant/          # imatrix、APEX 映射、各轮处理
  xspark/         # 推测解码
```

## 许可证

AGPL v3