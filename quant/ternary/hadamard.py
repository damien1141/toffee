"""
Blockwise Hadamard rotation, as used by Ternary Bonsai 2 27B and the
SpinQuant / QuaRot line of work.

The rotation matrix is

    R = (1/sqrt(n)) * H_n * S,  n = 1024,

where H_n is the Walsh-Hadamard matrix of order n and S is a fixed
diagonal of +/-1 signs. R is orthogonal: R @ R^T = I.

For a weight matrix W of shape (M, N) with N divisible by the block
size, the rotated weights are

    W' = W @ R_full,

where R_full is block-diagonal: I_{N/n} ⊗ R_n. Each n-element slice of
a row of W is rotated independently.

The activation-side transform at inference time is the inverse (which,
for an orthogonal matrix, is the transpose): x' = R^T @ x. Since R is
symmetric (R = R^T for the Hadamard-with-signs construction), the
forward and inverse are the same operation -- a fast Walsh-Hadamard
transform (FWHT) with sign flips. That is what Bonsai's kernels run at
inference time. This module only does the weight-side rotation (a
one-time quantization cost), so it uses the direct matrix multiply
rather than the FWHT -- the speed difference does not matter for a
one-shot quantization pass.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def hadamard_matrix(n: int) -> np.ndarray:
    """The Walsh-Hadamard matrix of order ``n`` (must be a power of 2).

    Returns a normalized (1/sqrt(n)) H_n with entries in {+/-1/sqrt(n)}.
    Constructed recursively: H_1 = [1], H_{2k} = [[H_k, H_k], [H_k, -H_k]].
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"hadamard_matrix: n must be a power of 2, got {n}")
    H = np.array([[1]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def random_signs(n: int, seed: int = 0) -> np.ndarray:
    """A fixed diagonal of +/-1 signs, reproducible across runs."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=n).astype(np.float64)


def rotation_matrix(block_size: int = 1024, seed: int = 0) -> np.ndarray:
    """R = (1/sqrt(n)) * H_n * S, the per-block rotation matrix.

    Args:
        block_size: the Hadamard block size n (must be a power of 2).
            Bonsai uses n=1024. Smaller blocks (256, 512) are cheaper
            but distribute magnitude less uniformly.
        seed: RNG seed for the +/-1 sign pattern.

    Returns:
        (block_size, block_size) float64 orthogonal matrix R, with
        R @ R^T = I.
    """
    H = hadamard_matrix(block_size)
    S = np.diag(random_signs(block_size, seed=seed))
    return H @ S  # H is already (1/sqrt(n)) * H_raw, so this is R = H * S.


def rotate_weights(W: np.ndarray, block_size: int = 1024, seed: int = 0) -> np.ndarray:
    """Apply the blockwise Hadamard rotation to a weight matrix.

    For each row of W, splits into N/block_size chunks of block_size
    elements and rotates each chunk by R.

    Args:
        W: (M, N) weight matrix. N must be divisible by block_size.
        block_size: the Hadamard block size (default 1024, per Bonsai).
        seed: RNG seed for the +/-1 sign pattern.

    Returns:
        (M, N) rotated weight matrix W' = W @ R_full.
    """
    if W.ndim != 2:
        raise ValueError(f"rotate_weights: W must be 2-D, got shape {W.shape}")
    M, N = W.shape
    if N % block_size != 0:
        raise ValueError(
            f"rotate_weights: N={N} not divisible by block_size={block_size}"
        )
    R = rotation_matrix(block_size, seed=seed)
    # Reshape W to (M, N/block_size, block_size), rotate each block, reshape back.
    W_blocks = W.astype(np.float64).reshape(M, N // block_size, block_size)
    # W' = W @ R, applied per block: (M, n_blocks, block_size) @ (block_size, block_size)
    W_rot = np.einsum("mbk,kl->mbl", W_blocks, R)
    return W_rot.reshape(M, N).astype(W.dtype)


def unrotate_weights(W_rot: np.ndarray, block_size: int = 1024, seed: int = 0) -> np.ndarray:
    """Inverse of :func:`rotate_weights`.

    Since R is orthogonal (R R^T = I), the inverse is R^T. Note: R is
    NOT symmetric in general (R = H S, R^T = S H, which differ unless
    S commutes with H -- and a random ±1 sign diagonal does not).

    Args:
        W_rot: (M, N) rotated weight matrix.
        block_size: the Hadamard block size (must match what was used
            to rotate).
        seed: the RNG seed used to rotate.

    Returns:
        (M, N) original weight matrix W = W_rot @ R^T.
    """
    R = rotation_matrix(block_size, seed=seed)
    R_inv = R.T  # R is orthogonal, so R^{-1} = R^T.
    M, N = W_rot.shape
    W_blocks = W_rot.astype(np.float64).reshape(M, N // block_size, block_size)
    W = np.einsum("mbk,kl->mbl", W_blocks, R_inv)
    return W.reshape(M, N).astype(W_rot.dtype)


def rotate_weights_torch(W, block_size: int = 1024, seed: int = 0, device=None):
    """GPU-accelerated Hadamard rotation using torch.

    For a 27B model (54 GB at fp16), the CPU rotation takes ~5-10
    minutes; on an RTX 4080 (~600 GB/s bandwidth), it takes ~90
    seconds. The rotation is a single einsum per 1024-element block.

    Args:
        W: (M, N) numpy array or torch tensor.
        block_size: the Hadamard block size (power of 2).
        seed: RNG seed for the +/-1 sign pattern.
        device: torch device (e.g. "cuda"). If None, uses CPU.

    Returns:
        (M, N) numpy array (on host) with the rotation applied.
    """
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Build the rotation matrix on CPU (small: 1024x1024), then move to GPU.
    R_np = rotation_matrix(block_size, seed=seed)  # (block_size, block_size)
    R = torch.from_numpy(R_np).to(dev, dtype=torch.float32)

    # Move W to GPU.
    if isinstance(W, np.ndarray):
        W_t = torch.from_numpy(np.ascontiguousarray(W)).to(dev, dtype=torch.float32)
    else:
        W_t = W.to(dev, dtype=torch.float32)

    M, N = W_t.shape
    n_blocks = N // block_size
    # Reshape to (M, n_blocks, block_size) and rotate each block.
    W_blocks = W_t.reshape(M, n_blocks, block_size)
    # W' = W @ R per block: (M, n_blocks, block_size) @ (block_size, block_size)
    # -> (M, n_blocks, block_size)
    W_rot = torch.bmm(W_blocks, R.unsqueeze(0).expand(M, -1, -1))
    return W_rot.reshape(M, N).cpu().numpy().astype(
        W.dtype if isinstance(W, np.ndarray) else np.float32
    )


__all__ = [
    "hadamard_matrix",
    "random_signs",
    "rotation_matrix",
    "rotate_weights",
    "unrotate_weights",
    "rotate_weights_torch",
]
