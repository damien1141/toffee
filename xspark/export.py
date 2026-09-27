"""
TOFFEE XSpark export module.

Exports the trained XSpark draft head to a GGUF file for dflash/DSV4 runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import gguf
import torch
from transformers import AutoConfig

from toffee.hardware import get_gpu_info

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def stream_target_weights_for_gguf(model_path: str, writer: gguf.GGUFWriter):
    """Reads embed_tokens and lm_head from safetensors and writes them to GGUF."""
    logger.info("streaming_target_weights_for_gguf")
    index_path = Path(model_path) / "model.safetensors.index.json"
    with open(index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]

    files_to_load = set()
    for k, file in weight_map.items():
        if "embed_tokens" in k or "lm_head" in k or "output" in k:
            files_to_load.add(file)

    for f in files_to_load:
        w_cpu = __import__('safetensors.torch').torch.load_file(Path(model_path) / f, device="cpu")
        for k, v in w_cpu.items():
            if "embed_tokens" in k:
                writer.add_tensor("token_embd.weight", v.to(torch.float32).numpy())
            elif "lm_head" in k or "output" in k:
                writer.add_tensor("output.weight", v.to(torch.float32).numpy())


def _build_tokenizer_from_hf(target_model: str, vocab_size: int):
    """Build tokenizer metadata from HF tokenizer.json."""
    tok_dir = Path(target_model)
    tok_json_path = tok_dir / "tokenizer.json"
    if not tok_json_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {target_model}")

    with open(tok_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_vocab = data.get("model", {}).get("vocab", {})
    added = data.get("added_tokens", [])

    max_id = 0
    for i in model_vocab.values():
        max_id = max(max_id, int(i))
    for a in added:
        max_id = max(max_id, int(a["id"]))

    tokens = [None] * (max_id + 1)
    for tok, i in model_vocab.items():
        tokens[int(i)] = tok
    for a in added:
        tokens[int(a["id"])] = a["content"]
    for i, t in enumerate(tokens):
        if t is None:
            tokens[i] = f"<pad_{i}>"

    while len(tokens) < vocab_size:
        tokens.append(f"<unused_{len(tokens)}>")
    if len(tokens) > vocab_size:
        tokens = tokens[:vocab_size]

    token_types = [1] * len(tokens)
    merges = data.get("model", {}).get("merges", [])
    merges = [m if isinstance(m, str) else f"{m[0]} {m[1]}" for m in merges]

    def _id_of(content):
        for a in added:
            if a.get("content") == content:
                return int(a["id"])
        return None

    bos = _id_of("<|endoftext|>") or _id_of("<|bos|>") or 0
    eos = _id_of("") or _id_of("<|eot|>") or _id_of("<|endoftext|>") or 0
    pad = _id_of("<|endoftext|>") or _id_of("<|pad|>") or bos
    mask = _id_of("<|mask|>")

    logger.info("built_tokenizer_from_hf",
                n_tokens=len(tokens), n_merges=len(merges),
                bos=bos, eos=eos, pad=pad, mask=mask)
    return (tokens, token_types, merges, bos, eos, pad, mask)


def copy_tokenizer_from_gguf(target_gguf_path: str, writer: gguf.GGUFWriter):
    """Reads tokenizer metadata from the target GGUF and writes it to the draft GGUF."""
    logger.info("copying_tokenizer_from_target_gguf", source=target_gguf_path)
    reader = gguf.GGUFReader(target_gguf_path)

    for key, field in reader.fields.items():
        if not key.startswith("tokenizer."):
            continue

        main_type = field.types[0]
        if main_type == gguf.GGUFValueType.ARRAY:
            sub_type = field.types[-1]
            writer.add_key_value(key, field.contents(), main_type, sub_type)
        else:
            writer.add_key_value(key, field.contents(), main_type)


def export_to_gguf(args):
    config = AutoConfig.from_pretrained(args.target_model, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    hidden_size = getattr(config, "hidden_size", getattr(text_config, "hidden_size", 2048))
    logger.info("loading_xspark_weights", path=args.xspark_head)
    xspark_state = torch.load(args.xspark_head, map_location="cpu", weights_only=True)

    def _target_vocab_from_gguf(path):
        try:
            tr = gguf.GGUFReader(path)
            for t in tr.tensors:
                if t.name in ("output.weight", "token_embd.weight", "tok_embd.weight"):
                    return int(t.shape[-1])
        except Exception as e:
            logger.warning("target_gguf_read_failed", error=str(e))
        return None

    def _target_vocab_from_head(head_state):
        for comp, sd in head_state.items():
            for name, t in sd.items():
                if name in ("W1.weight", "W2.weight"):
                    return int(t.shape[0])
        return None

    vocab_size = (
        _target_vocab_from_gguf(getattr(args, "target_gguf", None))
        or _target_vocab_from_head(xspark_state)
        or getattr(config, "vocab_size", None)
        or getattr(getattr(config, "text_config", config), "vocab_size", None)
    )
    if vocab_size is None:
        raise ValueError("Could not determine vocab_size from target GGUF / trained head / config.")
    logger.info("resolved_vocab_size", vocab_size=vocab_size)

    output_path = Path(args.output_dir) / "XSpark-Draft.gguf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("gguf_write_start", output=str(output_path))
    writer = gguf.GGUFWriter(str(output_path), "dflash")

    writer.add_uint32("dflash.context_length", 2048)
    writer.add_uint32("dflash.embedding_length", hidden_size)
    writer.add_uint32("dflash.block_count", 2)
    writer.add_uint32("dflash.markov_rank", 256)
    writer.add_uint32("dflash.vocab_size", vocab_size)

    writer.add_uint32("dflash.attention.head_count", 16)
    writer.add_uint32("dflash.attention.head_count_kv", 1)
    writer.add_uint32("dflash.attention.output_group_count", 16)
    writer.add_uint32("dflash.attention.key_length", 128)
    writer.add_uint32("dflash.attention.value_length", 128)

    writer.add_uint32("dflash.expert_count", 2)
    writer.add_uint32("dflash.expert_used_count", 2)
    writer.add_uint32("dflash.expert_feed_forward_length", 8192)
    writer.add_uint32("dflash.expert_shared_count", 1)
    writer.add_uint32("dflash.expert_shared_feed_forward_length", 8192)

    writer.add_float32("dflash.expert_weights_scale", 1.0)
    writer.add_bool("dflash.expert_weights_norm", False)
    writer.add_uint32("dflash.expert_gating_func", 4)

    writer.add_uint32("dflash.rope.dimension_count", 64)
    writer.add_float32("dflash.rope.freq_base", 10000000.0)
    writer.add_string("dflash.rope.scaling.type", "none")
    writer.add_float32("dflash.rope.scaling.factor", 1.0)
    writer.add_uint32("dflash.rope.scaling.original_context_length", 32768)
    writer.add_float32("dflash.rope.scaling.yarn_beta_fast", 32.0)
    writer.add_float32("dflash.rope.scaling.yarn_beta_slow", 1.0)

    writer.add_float32("dflash.attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_array("dflash.swiglu_clamp_exp", [0.0, 0.0])
    writer.add_array("dflash.swiglu_clamp_shexp", [0.0, 0.0])
    writer.add_uint32("dflash.attention.sliding_window", 0)

    writer.add_uint32("dflash.attention.q_lora_rank", 128)
    writer.add_uint32("dflash.attention.kv_lora_rank", 128)
    writer.add_uint32("dflash.attention.output_lora_rank", hidden_size)
    writer.add_array("dflash.attention.compress_ratios", [0, 0, 0])
    writer.add_float32("dflash.attention.compress_rope_freq_base", 160000.0)

    writer.add_uint32("dflash.attention.indexer.head_count", 1)
    writer.add_uint32("dflash.attention.indexer.key_length", 1)
    writer.add_uint32("dflash.attention.indexer.value_length", 1)
    writer.add_uint32("dflash.attention.indexer.top_k", 1)
    writer.add_float32("dflash.attention.indexer.norm_eps", 1e-6)

    writer.add_uint32("dflash.hyper_connection.count", 1)
    writer.add_uint32("dflash.hyper_connection.sinkhorn_iterations", 20)
    writer.add_float32("dflash.hyper_connection.epsilon", 1e-7)
    writer.add_uint32("dflash.hash_layer_count", 0)
    writer.add_uint32("dflash.n_target_features", 4096)
    writer.add_uint32("dflash.block_size", args.block_size)
    writer.add_array("dflash.target_layers", [15, 30])

    copy_tokenizer_from_gguf(args.target_gguf, writer)

    if not any("tokenizer.ggml.mask_token_id" in d for d in writer.kv_data):
        writer.add_uint32("tokenizer.ggml.mask_token_id", 0)

    n_ff = 8192
    n_expert = 2
    hc_width = 3
    n_head = 16
    n_embd_head = 128
    output_group_count = 16
    output_lora_rank = 2048
    o_group_dim = n_head * n_embd_head // output_group_count
    o_lora_width = output_lora_rank * output_group_count

    dummy_norm_1 = torch.ones(n_embd_head, dtype=torch.float32)
    dummy_norm_hs = torch.ones(hidden_size, dtype=torch.float32)
    dummy_embd_x_1 = torch.zeros(n_embd_head, hidden_size, dtype=torch.float32)
    dummy_1_x_embdh = torch.zeros(hidden_size, n_embd_head, dtype=torch.float32)
    dummy_wo_a = torch.zeros(o_lora_width, o_group_dim, dtype=torch.float32)
    dummy_2048_sq = torch.zeros(hidden_size, o_lora_width, dtype=torch.float32)
    dummy_sinks = torch.zeros(n_head, dtype=torch.float32)
    dummy_gate_inp = torch.zeros(n_expert, hidden_size, dtype=torch.float32)
    dummy_ffn_gate_exps = torch.zeros(n_expert, n_ff, hidden_size, dtype=torch.float32)
    dummy_ffn_down_exps = torch.zeros(n_expert, hidden_size, n_ff, dtype=torch.float32)
    dummy_ffn_up_exps = torch.zeros(n_expert, n_ff, hidden_size, dtype=torch.float32)
    dummy_ffn_shexp_g = torch.zeros(n_ff, hidden_size, dtype=torch.float32)
    dummy_ffn_shexp_d = torch.zeros(hidden_size, n_ff, dtype=torch.float32)
    dummy_ffn_shexp_u = torch.zeros(n_ff, hidden_size, dtype=torch.float32)
    dummy_hc_base = torch.zeros(hc_width, dtype=torch.float32)
    dummy_hc_fn = torch.zeros(hc_width, hidden_size, dtype=torch.float32)
    dummy_hc_scale = torch.zeros(3, dtype=torch.float32)
    dummy_1_bias = torch.zeros(n_expert, dtype=torch.float32)

    for i in range(2):
        writer.add_tensor(f"blk.{i}.attn_sinks.weight", dummy_sinks.numpy())
        writer.add_tensor(f"blk.{i}.attn_q_a.weight", dummy_embd_x_1.numpy())
        writer.add_tensor(f"blk.{i}.attn_q_a_norm.weight", dummy_norm_1.numpy())
        writer.add_tensor(f"blk.{i}.attn_q_b.weight", dummy_1_x_embdh.numpy())
        writer.add_tensor(f"blk.{i}.attn_kv.weight", dummy_embd_x_1.numpy())
        writer.add_tensor(f"blk.{i}.attn_kv_a_norm.weight", dummy_norm_1.numpy())
        writer.add_tensor(f"blk.{i}.attn_output_a.weight", dummy_wo_a.numpy())
        writer.add_tensor(f"blk.{i}.attn_output_b.weight", dummy_2048_sq.numpy())

        writer.add_tensor(f"blk.{i}.hc_attn_base.weight", dummy_hc_base.numpy())
        writer.add_tensor(f"blk.{i}.hc_attn_fn.weight", dummy_hc_fn.numpy())
        writer.add_tensor(f"blk.{i}.hc_attn_scale.weight", dummy_hc_scale.numpy())
        writer.add_tensor(f"blk.{i}.hc_ffn_base.weight", dummy_hc_base.numpy())
        writer.add_tensor(f"blk.{i}.hc_ffn_fn.weight", dummy_hc_fn.numpy())
        writer.add_tensor(f"blk.{i}.hc_ffn_scale.weight", dummy_hc_scale.numpy())

        writer.add_tensor(f"blk.{i}.ffn_gate_inp.weight", dummy_gate_inp.numpy())
        writer.add_tensor(f"blk.{i}.ffn_gate_exps.weight", dummy_ffn_gate_exps.numpy())
        writer.add_tensor(f"blk.{i}.ffn_down_exps.weight", dummy_ffn_down_exps.numpy())
        writer.add_tensor(f"blk.{i}.ffn_up_exps.weight", dummy_ffn_up_exps.numpy())
        writer.add_tensor(f"blk.{i}.ffn_gate_shexp.weight", dummy_ffn_shexp_g.numpy())
        writer.add_tensor(f"blk.{i}.ffn_down_shexp.weight", dummy_ffn_shexp_d.numpy())
        writer.add_tensor(f"blk.{i}.ffn_up_shexp.weight", dummy_ffn_shexp_u.numpy())
        writer.add_tensor(f"blk.{i}.exp_probs_b.bias", dummy_1_bias.numpy())

    has_ctx_norm = False
    has_output_norm = False
    has_hc_head = False

    for component_name, state_dict in xspark_state.items():
        for layer_name, tensor in state_dict.items():
            full_name = f"{component_name}.{layer_name}" if component_name != "backbone" else f"backbone.{layer_name}"
            clean_name = full_name.replace("backbone.layers.0.", "blk.0.").replace("backbone.layers.1.", "blk.1.")

            if "ctx_proj.weight" in clean_name:
                writer.add_tensor("fc.weight", tensor.to(torch.float32).numpy())
            elif "ctx_norm.weight" in clean_name:
                writer.add_tensor("enc.output_norm.weight", tensor.to(torch.float32).numpy())
                has_ctx_norm = True
            elif "output_norm.weight" in clean_name:
                writer.add_tensor("output_norm.weight", tensor.to(torch.float32).numpy())
                has_output_norm = True
            elif "hc_fn.weight" in clean_name:
                writer.add_tensor("output_hc_fn.weight", tensor.t().to(torch.float32).numpy())
                has_hc_head = True
            elif "hc_base.weight" in clean_name:
                writer.add_tensor("output_hc_base.weight", tensor.to(torch.float32).numpy())
                has_hc_head = True
            elif "hc_scale.weight" in clean_name:
                writer.add_tensor("output_hc_scale.weight", tensor.to(torch.float32).numpy())
                has_hc_head = True
            elif "W1.weight" in clean_name:
                writer.add_tensor("markov_w1.weight", tensor.to(torch.float32).numpy())
            elif "W2.weight" in clean_name:
                writer.add_tensor("markov_w2.weight", tensor.to(torch.float32).numpy())
            elif "conf_head.weight" in clean_name:
                writer.add_tensor("conf_proj.weight", tensor.to(torch.float32).numpy())
            elif "conf_head.bias" in clean_name:
                writer.add_tensor("conf_proj.bias", tensor.to(torch.float32).numpy())
            elif "norm1.weight" in clean_name:
                blk = clean_name.split(".")[1]
                writer.add_tensor(f"blk.{blk}.attn_norm.weight", tensor.to(torch.float32).numpy())
            elif "norm2.weight" in clean_name:
                blk = clean_name.split(".")[1]
                writer.add_tensor(f"blk.{blk}.ffn_norm.weight", tensor.to(torch.float32).numpy())

    if not has_ctx_norm:
        writer.add_tensor("dflash_hidden_norm.weight", dummy_norm_hs.numpy())
    if not has_output_norm:
        writer.add_tensor("output_norm.weight", dummy_norm_hs.numpy())
    if not has_hc_head:
        writer.add_tensor("hc_head_base.weight", torch.zeros(1, dtype=torch.float32).numpy())
        writer.add_tensor("hc_head_fn.weight", torch.zeros(hidden_size, 1, dtype=torch.float32).numpy())
        writer.add_tensor("hc_head_scale.weight", torch.zeros(1, dtype=torch.float32).numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    logger.info("gguf_export_complete", path=str(output_path))


def main():
    parser = argparse.ArgumentParser(description="XSpark GGUF Exporter")
    parser.add_argument("--target-model", type=str, required=True, help="Path to target HF model")
    parser.add_argument("--xspark-head", type=str, default="xspark_output/xspark_head.pt")
    parser.add_argument("--output-dir", type=str, default="xspark_output")
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--target-gguf", type=str, required=True,
                        help="Path to the TARGET model GGUF for tokenizer copy")
    args = parser.parse_args()

    export_to_gguf(args)


if __name__ == "__main__":
    main()
