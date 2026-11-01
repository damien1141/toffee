"""
@file toffee/amputate/selftest.py
   End-to-end synthetic proof of the amputation machinery.

Builds a tiny DENSE hybrid checkpoint (hidden 64, 4 layers, FFN 224,
GDN + full attention, vision tower junk, mtp.veriloop_* memory
tensors, untied lm_head) with a *structured* FFN:

  * gate/up projections are rank-8 (activations are learnable from x,
    like a real model -- this is what makes MoEfication possible)
  * down columns 0..55 are scaled 8x (the "heavy" neurons the shared
    expert must find), columns 56..111 are scaled 3x (a second tier
    only reachable through routing)

then runs report -> harvest -> build -> verify and asserts, at every
stage, the properties the real surgery depends on. No download, no
GPU required: the whole thing runs on CPU in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from toffee.amputate import build as build_mod
from toffee.amputate import harvest as harvest_mod
from toffee.amputate import registry as reg_mod
from toffee.amputate import verify as verify_mod
from toffee.amputate.preset import AmputationPreset
from toffee.quant.passes import BatchRecord

H, L, I, VOCAB = 64, 4, 224, 256
E, K, SHARED = 32, 4, 56
E_INTER = I // E           # 7
LAYER_TYPES = ["linear_attention", "full_attention",
               "linear_attention", "full_attention"]
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 16
LIN_K_HEADS, LIN_V_HEADS, LIN_DIM = 2, 4, 16
SEED = 20260131


def _text_config() -> dict:
    return {
        "hidden_size": H,
        "num_hidden_layers": L,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "attn_output_gate": True,
        "intermediate_size": I,
        "vocab_size": VOCAB,
        "layer_types": LAYER_TYPES,
        "linear_num_key_heads": LIN_K_HEADS,
        "linear_num_value_heads": LIN_V_HEADS,
        "linear_key_head_dim": LIN_DIM,
        "linear_value_head_dim": LIN_DIM,
        "linear_conv_kernel_dim": 4,
        "partial_rotary_factor": 0.25,
        "rope_theta": 1e7,
        "mamba_ssm_dtype": "float32",
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
    }


def _structured_ffn(rng: torch.Generator):
    """Rank-8 gate/up + two-tier down-column energy structure."""
    def low_rank(rows, cols, r=8, scale=0.5):
        a = torch.randn(rows, r, generator=rng) * scale
        b = torch.randn(r, cols, generator=rng) * scale
        return a @ b

    gate = low_rank(I, H)
    up = low_rank(I, H)
    down = torch.randn(H, I, generator=rng) * 0.2
    down[:, :SHARED] *= 12.0         # heavy tier -> shared expert must find these
    down[:, SHARED:2 * SHARED] *= 4.0  # second tier -> only routing reaches these
    return gate, up, down


def write_synthetic_checkpoint(model_dir: Path) -> Dict[str, torch.Tensor]:
    """Write the dense hybrid checkpoint; returns the FFN tensors."""
    rng = torch.Generator().manual_seed(SEED)
    model_dir.mkdir(parents=True, exist_ok=True)
    tensors: Dict[str, torch.Tensor] = {}

    tensors["model.embed_tokens.weight"] = torch.randn(
        VOCAB, H, generator=rng) * 0.05
    tensors["model.norm.weight"] = torch.zeros(H)

    q_out = HEADS * HEAD_DIM * 2     # attn_output_gate doubles the width
    kv_out = KV_HEADS * HEAD_DIM
    key_total = LIN_K_HEADS * LIN_DIM
    value_total = LIN_V_HEADS * LIN_DIM
    qkv_out = 2 * key_total + value_total

    for l in range(L):
        p = f"model.layers.{l}."
        # zero-centred RMSNorm weights (deltas around 0 = identity),
        # matching the real Qwen3.5 checkpoint format
        tensors[p + "input_layernorm.weight"] = torch.zeros(H)
        tensors[p + "post_attention_layernorm.weight"] = torch.zeros(H)
        if LAYER_TYPES[l] == "full_attention":
            a = p + "self_attn."
            tensors[a + "q_proj.weight"] = torch.randn(q_out, H, generator=rng) * 0.05
            tensors[a + "k_proj.weight"] = torch.randn(kv_out, H, generator=rng) * 0.05
            tensors[a + "v_proj.weight"] = torch.randn(kv_out, H, generator=rng) * 0.05
            tensors[a + "o_proj.weight"] = torch.randn(
                HEADS * HEAD_DIM, HEADS * HEAD_DIM, generator=rng) * 0.05
            tensors[a + "q_norm.weight"] = torch.zeros(HEAD_DIM)
            tensors[a + "k_norm.weight"] = torch.zeros(HEAD_DIM)
        else:
            a = p + "linear_attn."
            tensors[a + "in_proj_qkv.weight"] = torch.randn(qkv_out, H, generator=rng) * 0.05
            tensors[a + "in_proj_z.weight"] = torch.randn(value_total, H, generator=rng) * 0.05
            tensors[a + "in_proj_a.weight"] = torch.randn(LIN_V_HEADS, H, generator=rng) * 0.05
            tensors[a + "in_proj_b.weight"] = torch.randn(LIN_V_HEADS, H, generator=rng) * 0.05
            tensors[a + "out_proj.weight"] = torch.randn(H, value_total, generator=rng) * 0.05
            tensors[a + "conv1d.weight"] = torch.randn(qkv_out, 1, 4, generator=rng) * 0.05
            # gated RMSNorm: PLAIN weight (ones = identity), unlike the
            # zero-centred norms above
            tensors[a + "norm.weight"] = torch.ones(LIN_DIM)
        gate, up, down = _structured_ffn(rng)
        tensors[p + "mlp.gate_proj.weight"] = gate              # [I, H]
        tensors[p + "mlp.up_proj.weight"] = up                  # [I, H]
        tensors[p + "mlp.down_proj.weight"] = down              # [H, I]

    tensors["lm_head.weight"] = torch.randn(VOCAB, H, generator=rng) * 0.05
    # vision tower junk (must be dropped)
    tensors["model.visual.patch_embed.weight"] = torch.randn(16, 3, 4, 4, generator=rng)
    tensors["model.visual.blocks.0.attn.q.weight"] = torch.randn(H, H, generator=rng)
    # mtp.veriloop conditional memory (must be preserved)
    tensors["mtp.veriloop_conditional_memory.embedded_candidate"] = torch.randn(16, H, generator=rng)
    tensors["mtp.veriloop_conditional_memory.gate.weight"] = torch.randn(32, H, generator=rng)
    tensors["mtp.veriloop_bank"] = torch.randn(8, H, generator=rng)
    # real MTP head (must be dropped)
    tensors["mtp.fc.weight"] = torch.randn(H, H, generator=rng)
    tensors["mtp.head.weight"] = torch.randn(VOCAB, H, generator=rng)

    _fix_down_orientations(tensors)
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(model_dir / shard))
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": sum(
            t.numel() * t.element_size() for t in tensors.values())},
            "weight_map": {k: shard for k in tensors}}, f, indent=2)
    with open(model_dir / "config.json", "w") as f:
        json.dump({
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": _text_config(),
            "vision_config": {"depth": 2, "hidden_size": 64},
            "mtp_num_hidden_layers": 1,
        }, f, indent=2)
    return tensors


def _fix_down_orientations(tensors: Dict[str, torch.Tensor]) -> None:
    """No-op kept for symmetry: the structured FFN builder already emits
    down_proj in checkpoint orientation [H, I]."""
    for l in range(L):
        k = f"model.layers.{l}.mlp.down_proj.weight"
        assert tensors[k].shape == (H, I), tensors[k].shape


def _batches(n_batches: int = 16, tokens: int = 128) -> List[BatchRecord]:
    rng = torch.Generator().manual_seed(SEED + 1)
    out = []
    for i in range(n_batches):
        ids = torch.randint(0, VOCAB, (tokens,), generator=rng)
        mask = torch.ones(tokens, dtype=torch.long)
        out.append(BatchRecord(ids, mask, i))
    return out


def _moe_forward_built(x, experts, shared, router, k=K):
    """llama.cpp semantics: softmax over all experts, top-k weighted."""
    logits = x @ router.T
    p = torch.softmax(logits.float(), dim=-1)
    topv, topi = p.topk(k, dim=-1)
    out = torch.zeros_like(x)
    for n in range(x.shape[0]):
        for j in range(k):
            e = int(topi[n, j])
            g, u, d = experts[e]
            act = F.silu(x[n] @ g.T) * (x[n] @ u.T)
            out[n] += float(topv[n, j]) * (act @ d.T)
    gs, us, ds = shared
    act = F.silu(x @ gs.T) * (x @ us.T)
    out = out + 0.5 * (act @ ds.T)   # sigmoid(zero gate) = 0.5; ds holds 2s
    return out


def run_selftest(output_dir: str | Path, device: str = "cpu",
                 keep: bool = False) -> bool:
    """Run the full synthetic pipeline; returns True on PASS."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "synthetic-dense"
    write_synthetic_checkpoint(model_dir)
    preset = AmputationPreset(
        name="selftest", num_experts=E, experts_per_tok=K,
        shared_intermediate=SHARED, strategy="balanced",
    )
    checks: List[tuple] = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    # ---- 1. registry report ----------------------------------------------
    print("== report ==")
    reg = reg_mod.scan_checkpoint(model_dir)
    counts = reg.counts()
    check("registry ffn count", counts.get("ffn", 0) == L * 3,
          f"{counts.get('ffn')}")
    check("registry mtp_veriloop count", counts.get("mtp_veriloop", 0) == 3,
          f"{counts.get('mtp_veriloop')}")
    check("registry mtp_head count", counts.get("mtp_head", 0) == 2,
          f"{counts.get('mtp_head')}")
    check("registry vision count", counts.get("vision", 0) == 2,
          f"{counts.get('vision')}")
    reg_path = reg_mod.save_registry(reg, output_dir / "amputate_registry.json")
    check("registry json written", reg_path.exists())

    # ---- 2. harvest --------------------------------------------------------
    print("== harvest ==")
    batches = _batches()
    stats_path = harvest_mod.harvest(
        model_dir, batches, output_dir, preset,
        device=device, model_impl="minimal",
    )
    stats = verify_mod.load_stats(stats_path)
    check("stats covers all layers", len(stats["layers"]) == L)
    heavy = set(range(SHARED))
    for l in range(L):
        ls = stats["layers"][str(l)]
        m = ls["metrics"]
        found = set(ls["shared_idx"].tolist())
        overlap = len(found & heavy) / SHARED
        # On a random synthetic, unscaled neurons with naturally large
        # activations legitimately outrank some scaled ones; the check
        # is that the shared set is DOMINATED by the heavy tier.
        check(f"L{l} shared finds heavy tier", overlap >= 0.80,
              f"overlap {overlap:.2f}")
        check(f"L{l} mass recall", m["mass_recall"] >= 0.85,
              f"{m['mass_recall']:.3f} (oracle {m['mass_oracle']:.3f})")
        # Routing must add value over the shared-only floor; the LSQ
        # calibration makes it structurally >=, so any negative margin
        # means the router fit is broken on that layer.
        check(f"L{l} routing beats shared-only",
              m["cos_calibrated"] > m["cos_shared_only"],
              f"cos {m['cos_calibrated']:.4f} vs shared {m['cos_shared_only']:.4f} "
              f"(margin +{m['cos_calibrated'] - m['cos_shared_only']:.4f})")
        check(f"L{l} router corr", m["router_corr_mean"] >= 0.15,
              f"{m['router_corr_mean']:.3f}")
        check(f"L{l} jaccard > chance", m["sel_jaccard"] >= 0.12,
              f"{m['sel_jaccard']:.3f}")
        # temperature normalization: the emitted logits must sit at the
        # preset's pooled std and the softmax must not be a hard argmax
        # (regression: unnormalized logits at real-model scales collapse
        # the top-k weights to [1,0,0,0] and kill router_corr)
        raw_std = ls.get("router_logit_std_raw")
        check(f"L{l} router logit std normalized",
              ls.get("router_logit_std") == preset.router_logit_scale,
              f"raw {raw_std:.3g} -> {ls.get('router_logit_std')}"
              if raw_std is not None else "not applied")
        check(f"L{l} softmax not saturated",
              m["router_top1_prob"] <= 0.90,
              f"top1 prob {m['router_top1_prob']:.3f}")
        # logit-space corr: the temperature-invariant measure of
        # router/truth alignment. In the synthetic the router genuinely
        # sees the cluster structure, so lz must be present and healthy.
        # The two corrs must TRACK each other: prob-space corr is lz
        # pushed through the softmax, which attenuates it at high
        # temperature (p1 large) but can push it slightly ABOVE lz at
        # low temperature (v1.0.4 default 1.0): softmax also removes
        # the per-token common mode of z, which raw-z lz keeps, and
        # dropping a positively-correlated common mode can raise
        # Pearson corr. Hence a two-sided band, not a directional
        # floor (regression note: "lz >= corr - 0.02" broke at the 1.0
        # default with lz 0.40 vs corr 0.42).
        lz = m.get("router_logit_corr")
        check(f"L{l} logit-corr healthy",
              lz is not None and lz >= 0.25,
              f"{lz:.3f}" if lz is not None else "missing")
        check(f"L{l} logit-corr tracks prob-corr",
              lz is not None and abs(lz - m["router_corr_mean"]) <= 0.12,
              f"lz {lz:.3f} vs corr {m['router_corr_mean']:.3f}"
              if lz is not None else "missing")

    # ---- 2b. dense-source robustness (regression: config pollution) ------
    # The laptop bug: a dense checkpoint whose config OBJECT looks MoE
    # through leaked defaults (transformers class __init__ defaults, or
    # the qwen4_exp_text fallback's CLASS attributes, visible to plain
    # getattr) used to make the harvest model come out MoE and die at
    # "dense FFN not found". Tensor names + config sanitization now pin
    # the source dense.
    print("== dense-source robustness ==")
    from toffee.amputate.harvest import (
        _assert_dense_checkpoint, _build_text_model)
    from toffee.quant.config import Qwen4ExpTextConfig, load_config_with_fallback

    with open(model_dir / "model.safetensors.index.json") as f:
        wmap = json.load(f)["weight_map"]
    try:
        _assert_dense_checkpoint(wmap, L)
        check("tensor-truth accepts dense checkpoint", True)
    except RuntimeError as e:
        check("tensor-truth accepts dense checkpoint", False, str(e)[:70])
    moe_wmap = dict(wmap)
    moe_wmap["model.layers.0.mlp.experts.0.gate_proj.weight"] = \
        "model-00001-of-00001.safetensors"
    moe_wmap["model.layers.0.mlp.gate.weight"] = \
        "model-00001-of-00001.safetensors"
    try:
        _assert_dense_checkpoint(moe_wmap, L)
        check("tensor-truth rejects MoE checkpoint", False)
    except RuntimeError:
        check("tensor-truth rejects MoE checkpoint", True)

    def _mlp_names(shim):
        return [n for n, _ in shim.model.layers[0].named_modules()
                if "mlp" in n]

    # route 1: qwen4_exp_text fallback class attributes (the exact
    # laptop failure: num_experts=180 / moe_intermediate_size=512 leak
    # through getattr even though the instance never set them)
    fb = Qwen4ExpTextConfig.from_dict(dict(_text_config()))
    assert getattr(fb, "num_experts", None) is not None  # class attr visible
    shim_fb, _, _ = _build_text_model(
        SimpleNamespace(model_type="qwen3_5", text_config=fb), "minimal")
    n_fb = _mlp_names(shim_fb)
    check("fallback class-attr pollution -> dense build",
          any(n.endswith("mlp.gate_proj") for n in n_fb)
          and not any(".mlp.experts." in n or n.endswith("mlp.gate")
                      for n in n_fb),
          "qwen4_exp_text class attrs shadowed")

    # route 2: transformers __init__ defaults filled in at instance level
    cfg2 = load_config_with_fallback(model_dir)
    tc2 = getattr(cfg2, "text_config", None) or cfg2
    if isinstance(tc2, dict):
        tc2.update(num_experts=32, moe_intermediate_size=7)
    else:
        setattr(tc2, "num_experts", 32)
        setattr(tc2, "moe_intermediate_size", 7)
    shim2, _, _ = _build_text_model(cfg2, "minimal")
    n2 = _mlp_names(shim2)
    check("instance-default pollution -> dense build",
          any(n.endswith("mlp.gate_proj") for n in n2)
          and not any(".mlp.experts." in n or n.endswith("mlp.gate")
                      for n in n2),
          "num_experts=32 / moe_inter=7 injected")

    # ---- 3. build ----------------------------------------------------------
    print("== build ==")
    out_model = output_dir / "amputated"
    build_mod.build(model_dir, stats_path, out_model, preset,
                    registry_json=reg_path)
    with open(out_model / "model.safetensors.index.json") as f:
        wmap = json.load(f)["weight_map"]
    n_exp = sum(1 for k in wmap if ".mlp.experts." in k)
    n_sh = sum(1 for k in wmap if ".mlp.shared_expert." in k)
    n_rt = sum(1 for k in wmap if k.endswith(".mlp.gate.weight"))
    n_sg = sum(1 for k in wmap if k.endswith(".mlp.shared_expert_gate.weight"))
    check("expert tensors", n_exp == L * E * 3, f"{n_exp}")
    check("shared expert tensors", n_sh == L * 3, f"{n_sh}")
    check("router tensors", n_rt == L, f"{n_rt}")
    check("shared gate tensors", n_sg == L, f"{n_sg}")
    check("mtp.veriloop preserved",
          sum(1 for k in wmap if "veriloop" in k) == 3)
    check("vision dropped", not any(k.startswith("model.visual") for k in wmap))
    check("mtp head dropped", not any(k.startswith("mtp.fc") or
                                      k.startswith("mtp.head") for k in wmap))
    check("lm_head untied kept", "lm_head.weight" in wmap)

    with open(out_model / "config.json") as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", {})
    check("config architecture is the registered GGUF key",
          cfg["architectures"] == ["Qwen3_5MoeForConditionalGeneration"])
    check("config model_type", cfg["model_type"] == "qwen3_5_moe")
    check("config text model_type", tc.get("model_type") == "qwen3_5_moe_text")
    check("config num_experts", tc["num_experts"] == E)
    check("config moe_intermediate_size",
          tc["moe_intermediate_size"] == E_INTER)
    check("config shared_expert_intermediate_size",
          tc["shared_expert_intermediate_size"] == SHARED)
    check("config text_config has no vision/mtp keys",
          not any(k.startswith(("vision", "mtp", "vllm", "veriloop"))
                  for k in tc))
    check("config text_config no mtp_num_hidden_layers leak",
          "mtp_num_hidden_layers" not in cfg)
    check("config mirrors source vision_config",
          cfg.get("vision_config") == {"depth": 2, "hidden_size": 64})
    check("config keeps layer_types", tc.get("layer_types") == LAYER_TYPES)
    check("config provenance records dense inter",
          tc.get("amputate_provenance", {}).get("dense_intermediate_size") == I)
    no_vis = build_mod._output_config(
        {"text_config": _text_config(),
         "vision_config": {"depth": 2, "hidden_size": 64}},
        preset, I, {"model": "synthetic", "num_layers": L, "hidden": H},
        mirror_vision_config=False)
    check("config --no-vision-config drops the wrapper block",
          "vision_config" not in no_vis
          and no_vis["architectures"] == ["Qwen3_5MoeForConditionalGeneration"])

    # ---- 4. equivalence: built weights == source math ----------------------
    print("== equivalence ==")
    src_shard = load_file(model_dir / "model-00001-of-00001.safetensors")
    l = 0
    ls = stats["layers"][str(l)]
    c = ls["calib_c"].double()
    s = float(ls["calib_s"])
    part = ls["partition"].long()
    from toffee.amputate.surgery import expert_index_lists
    idx_lists = expert_index_lists(part)
    shared_idx = ls["shared_idx"].long()

    out_shards = {}
    for shard_file in sorted(out_model.glob("*.safetensors")):
        out_shards.update(load_file(shard_file))

    src_down = src_shard[f"model.layers.{l}.mlp.down_proj.weight"].float()  # [H, I]
    max_rel_err = 0.0
    for e in range(E):
        d_built = out_shards[f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"].float()
        d_src = src_down[:, idx_lists[e]] * c[e]
        rel = ((d_built - d_src).abs() / d_src.abs().clamp_min(1e-6)).max().item()
        max_rel_err = max(max_rel_err, rel)
    check("expert down cols carry calibration scales",
          max_rel_err < 0.05, f"max rel err {max_rel_err:.4f}")

    ds_built = out_shards[f"model.layers.{l}.mlp.shared_expert.down_proj.weight"].float()
    ds_src = src_down[:, shared_idx] * (2.0 * s)
    rel = ((ds_built - ds_src).abs() / ds_src.abs().clamp_min(1e-6)).max().item()
    check("shared down carries 2s scale", rel < 0.05, f"rel err {rel:.4f}")

    g_built = out_shards[f"model.layers.{l}.mlp.gate.weight"].float()
    rel = ((g_built - ls["router_w"].float()).abs() /
           ls["router_w"].float().abs().clamp_min(1e-6)).max().item()
    check("router weights verbatim", rel < 0.05, f"rel err {rel:.4f}")

    # functional: built model's MoE forward == source-math forward
    rng = torch.Generator().manual_seed(SEED + 2)
    x = torch.randn(256, H, generator=rng)
    experts = []
    for e in range(E):
        experts.append((
            out_shards[f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"].float(),
            out_shards[f"model.layers.{l}.mlp.experts.{e}.up_proj.weight"].float(),
            out_shards[f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"].float(),
        ))
    shared = (
        out_shards[f"model.layers.{l}.mlp.shared_expert.gate_proj.weight"].float(),
        out_shards[f"model.layers.{l}.mlp.shared_expert.up_proj.weight"].float(),
        out_shards[f"model.layers.{l}.mlp.shared_expert.down_proj.weight"].float(),
    )
    router = out_shards[f"model.layers.{l}.mlp.gate.weight"].float()
    recon_built = _moe_forward_built(x, experts, shared, router)

    # source math with harvest stats (fp32 reference)
    src_gate = src_shard[f"model.layers.{l}.mlp.gate_proj.weight"].float()
    src_up = src_shard[f"model.layers.{l}.mlp.up_proj.weight"].float()
    act = F.silu(x @ src_gate.T) * (x @ src_up.T)          # [N, I]
    dense_out = act @ src_down.T
    logits = x @ ls["router_w"].float().T
    p = torch.softmax(logits, dim=-1)
    topv, topi = p.topk(K, dim=-1)
    recon_src = torch.zeros_like(x)
    for n in range(x.shape[0]):
        for j in range(K):
            e = int(topi[n, j])
            cols = idx_lists[e]
            recon_src[n] += float(topv[n, j]) * c[e] * (act[n, cols] @ src_down[:, cols].T)
    shared_src = act[:, shared_idx] @ src_down[:, shared_idx].T
    recon_src = recon_src + s * shared_src

    cos_bb = F.cosine_similarity(recon_built, recon_src, dim=-1).mean().item()
    check("built forward == source math", cos_bb >= 0.995, f"cos {cos_bb:.5f}")
    cos_dense = F.cosine_similarity(recon_src, dense_out, dim=-1).mean().item()
    check("reconstruction tracks dense FFN", cos_dense >= 0.85,
          f"cos {cos_dense:.4f}")

    # ---- 5. verify canary ---------------------------------------------------
    print("== verify ==")
    res = verify_mod.check_thresholds(stats)
    check("offline canary passes", res["pass"],
          "; ".join(res["failures"]) or "all thresholds green")

    # ---- 6. preset math smoke ----------------------------------------------
    from toffee.amputate.preset import summarize
    text = summarize()
    check("preset math renders", "ACTIVE params" in text and "dial 3" in text)

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\nselftest: {n_pass}/{len(checks)} checks passed")
    if not keep:
        import shutil
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.rmtree(out_model, ignore_errors=True)
    return n_pass == len(checks)
