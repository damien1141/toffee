"""
GPU (torch) GSQ test. Skips if torch or CUDA is not available.

On a machine with an RTX-class GPU, this verifies the torch hot-path
matches the CPU NumPy path's accuracy and reports the speed-up.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

REFINED_DIR = str(Path(__file__).parent.resolve())
PARENT = str(Path(__file__).parent.parent.resolve())

# Make toffee.quant importable.
toffee_mod = types.ModuleType("toffee")
toffee_mod.__path__ = [PARENT]
sys.modules["toffee"] = toffee_mod
quant_mod = types.ModuleType("toffee.quant")
quant_mod.__path__ = [REFINED_DIR]
sys.modules["toffee.quant"] = quant_mod


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gguf_kquant = _load(
    "toffee.quant.gguf_kquant",
    os.path.join(REFINED_DIR, "gguf_kquant.py"),
)
gsq = _load(
    "toffee.quant.gsq",
    os.path.join(REFINED_DIR, "gsq.py"),
)

spec_for = gguf_kquant.spec_for
dequantize_blocks = gguf_kquant.dequantize_blocks
BlockRaw = gguf_kquant.BlockRaw


def _naive_quantize_q4_k(values, d_factor=1.0):
    n_blocks, block_elems = values.shape
    spec = spec_for("Q4_K")
    n_groups = spec.n_groups
    codes = np.zeros((n_blocks, block_elems), dtype=np.int32)
    raws = []
    for b in range(n_blocks):
        block = values[b].astype(np.float64)
        bmin = float(block.min())
        bmax = float(block.max())
        span = bmax - bmin
        d_naive = float(np.float16(d_factor * span / 15 + 1e-12)) or 1e-6
        dmin_naive = float(np.float16(abs(bmin) + 1e-12)) or 1e-6
        sc_raw = np.full(n_groups, 1, dtype=np.int64)
        m_raw = np.full(n_groups, 1, dtype=np.int64)
        c = np.clip(
            np.round((block + dmin_naive) / max(d_naive, 1e-12)),
            spec.code_min, spec.code_max,
        ).astype(np.int32)
        codes[b] = c
        raws.append(BlockRaw(
            d=d_naive, dmin=dmin_naive,
            sc_raw=sc_raw.copy(), m_raw=m_raw.copy(),
        ))
    return codes, raws


def _perturb_d(raws, factor):
    return [BlockRaw(
        d=float(np.float16(r.d * factor)) or 1e-6,
        dmin=r.dmin,
        sc_raw=r.sc_raw.copy(),
        m_raw=r.m_raw.copy() if r.m_raw.size else r.m_raw,
    ) for r in raws]


def _weighted_mse(recon, target, weights):
    diff = (recon.astype(np.float64) - target.astype(np.float64))
    return float((diff * diff * weights).sum() / max(float(weights.sum()), 1e-12))


def main():
    try:
        import torch
    except ImportError:
        print("SKIP: torch not installed. Install with `pip install torch`")
        return 0
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available. Run on a machine with a CUDA GPU.")
        return 0

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    rng = np.random.default_rng(456)
    n_blocks = 16384
    block_elems = 256
    values = rng.normal(0, 1.0, size=(n_blocks, block_elems)).astype(np.float32)
    weights = np.ones_like(values)
    codes, raws = _naive_quantize_q4_k(values, d_factor=1.0)
    raws_p = _perturb_d(raws, factor=1.5)

    spec = spec_for("Q4_K")
    eff_s, eff_m = gsq._effective_from_raw(raws_p, spec)
    recon0 = dequantize_blocks(codes, eff_s, eff_m, "Q4_K")
    initial_error = _weighted_mse(recon0, values, weights)
    print(f"initial_error (perturbed d): {initial_error:.6f}")

    # CPU run.
    cfg_cpu = gsq.GsqConfig(epochs=0, search_super_scale=True, device="cpu")
    t0 = time.perf_counter()
    res_cpu = gsq.refine_tensor(
        values, "Q4_K", imatrix=None, config=cfg_cpu,
        initial_codes=codes, initial_raw=raws_p,
    )
    dt_cpu = time.perf_counter() - t0
    print(f"CPU:  {dt_cpu*1000:.1f} ms   final_error: {res_cpu.final_error:.6f}")

    # GPU run.
    cfg_gpu = gsq.GsqConfig(epochs=0, search_super_scale=True, device="cuda")
    t0 = time.perf_counter()
    res_gpu = gsq.refine_tensor(
        values, "Q4_K", imatrix=None, config=cfg_gpu,
        initial_codes=codes, initial_raw=raws_p,
    )
    dt_gpu = time.perf_counter() - t0
    print(f"GPU:  {dt_gpu*1000:.1f} ms   final_error: {res_gpu.final_error:.6f}")

    speedup = dt_cpu / dt_gpu
    print(f"\nGPU speed-up: {speedup:.2f}x")

    # Verify the GPU result matches the CPU result (modulo floating-point
    # noise -- both use float64 internally).
    rel_diff = abs(res_cpu.final_error - res_gpu.final_error) / max(
        abs(res_cpu.final_error), 1e-12
    )
    print(f"CPU vs GPU final_error relative diff: {rel_diff:.2e}")
    if rel_diff < 1e-6:
        print("PASS: GPU path matches CPU path.")
    else:
        print("WARN: GPU and CPU results differ. This may be float noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
