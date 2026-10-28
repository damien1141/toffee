"""TOFFEE ternary quantization module.

Provides the PTQ1_0 and PQ2_0 ternary codecs (matching the PrismML
llama.cpp fork) and a pipeline that takes an f16 GGUF, applies the
blockwise Hadamard rotation, and produces a ternary GGUF loadable by
the PrismML runtime.
"""

from toffee.quant.ternary.codec import (
    TernaryBlockRaw,
    QK_PTQ1_0,
    QK_PQ2_0,
    encode_pq2_0_row,
    decode_pq2_0_row,
    encode_ptq1_0_row,
    decode_ptq1_0_row,
    block_size_bytes,
    block_elems,
    bits_per_weight,
)
from toffee.quant.ternary.hadamard import (
    hadamard_matrix,
    random_signs,
    rotation_matrix,
    rotate_weights,
    unrotate_weights,
)

__all__ = [
    "TernaryBlockRaw",
    "QK_PTQ1_0",
    "QK_PQ2_0",
    "encode_pq2_0_row",
    "decode_pq2_0_row",
    "encode_ptq1_0_row",
    "decode_ptq1_0_row",
    "block_size_bytes",
    "block_elems",
    "bits_per_weight",
    "hadamard_matrix",
    "random_signs",
    "rotation_matrix",
    "rotate_weights",
    "unrotate_weights",
]
