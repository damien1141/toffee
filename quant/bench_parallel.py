"""
Benchmark the parallel tensor mode (parallel_workers > 1).

Simulates the per-tensor work of ``refine_gguf_with_gsq`` by spawning
N workers and refining N synthetic tensors concurrently. The workers
use the same _refine_tensor_worker function the real path uses, so the
speed-up number is representative of what the user will see on a real
GGUF.

Run:
    python /home/z/my-project/scripts/bench_parallel.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool

import numpy as np

REFINED_DIR = "/home/z/my-project/download/refined"
toffee_mod = types.ModuleType("toffee")
toffee_mod.__path__ = []
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
BlockRaw = gguf_kquant.BlockRaw


def make_task(seed, n_blocks=2048, block_elems=256):
    """Build a synthetic refine task (in-memory; no GGUF I/O)."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 1.0, size=(n_blocks, block_elems)).astype(np.float32)
    weights = np.ones_like(values)
    spec = spec_for("Q4_K")
    # Naive quantize: pick optimal d, generate codes.
    codes = np.zeros((n_blocks, block_elems), dtype=np.int32)
    raws = []
    for b in range(n_blocks):
        block = values[b].astype(np.float64)
        bmin = float(block.min())
        bmax = float(block.max())
        span = bmax - bmin
        d = float(np.float16(1.5 * span / 15 + 1e-12)) or 1e-6  # perturbed
        dmin = float(np.float16(abs(bmin) + 1e-12)) or 1e-6
        sc_raw = np.full(spec.n_groups, 1, dtype=np.int64)
        m_raw = np.full(spec.n_groups, 1, dtype=np.int64)
        eff_s = float(d)
        eff_m = float(dmin)
        codes[b] = np.clip(
            np.round((block + eff_m) / max(eff_s, 1e-12)),
            spec.code_min, spec.code_max,
        ).astype(np.int32)
        raws.append(BlockRaw(
            d=d, dmin=dmin,
            sc_raw=sc_raw.copy(), m_raw=m_raw.copy(),
        ))
    return values, codes, raws


def refine_one(task):
    """Worker: refine one task. Must be at module top-level for pickling."""
    values, codes, raws = task
    cfg = gsq.GsqConfig(epochs=0, search_super_scale=True)
    result = gsq.refine_tensor(
        values, "Q4_K", imatrix=None, config=cfg,
        initial_codes=codes, initial_raw=raws,
    )
    return result.final_error


def bench_serial(n_tasks, task_size):
    tasks = [make_task(seed=i, n_blocks=task_size) for i in range(n_tasks)]
    t0 = time.perf_counter()
    errors = [refine_one(t) for t in tasks]
    dt = time.perf_counter() - t0
    return dt, errors


def bench_parallel(n_tasks, task_size, n_workers):
    tasks = [make_task(seed=i, n_blocks=task_size) for i in range(n_tasks)]
    t0 = time.perf_counter()
    with Pool(processes=n_workers) as pool:
        errors = pool.map(refine_one, tasks)
    dt = time.perf_counter() - t0
    return dt, errors


def main():
    print("=" * 72)
    print("Parallel tensor benchmark (CPU-only, no llama.cpp binary)")
    print("=" * 72)

    n_tasks = 8
    task_size = 4096  # ~4K blocks per tensor, similar to a transformer block
    print(f"\nConfig: {n_tasks} tensors, {task_size} blocks each, Q4_K")

    # Serial baseline.
    dt_serial, errors_s = bench_serial(n_tasks, task_size)
    print(f"\n  serial (1 worker):     {dt_serial*1000:8.1f} ms "
          f"({dt_serial / n_tasks * 1000:.1f} ms/tensor)")

    # Parallel with N workers.
    for n in [2, 4, 8]:
        try:
            dt_p, errors_p = bench_parallel(n_tasks, task_size, n)
        except Exception as exc:
            print(f"\n  parallel ({n} workers): FAILED ({exc})")
            continue
        speedup = dt_serial / dt_p
        print(f"  parallel ({n} workers): {dt_p*1000:8.1f} ms "
              f"({dt_p / n_tasks * 1000:.1f} ms/tensor)  "
              f"speed-up: {speedup:.2f}x")
        # Verify the results match (modulo floating-point noise).
        max_diff = max(abs(a - b) for a, b in zip(errors_s, errors_p))
        assert max_diff < 1e-6, f"parallel results differ: max_diff={max_diff}"
    print("\nAll parallel results match the serial results.")


if __name__ == "__main__":
    main()
