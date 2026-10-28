"""
@file toffee/quant/ggml_bridge.py
   Thin ctypes bridge to llama.cpp's own ggml quantizer.

Why this exists
---------------
TOFFEE's quantizer produces GGUF files that must be byte-compatible with
llama.cpp. Re-implementing the K-quant block layouts in Python would be a
second source of truth that silently drifts from llama.cpp. Instead we call
the *same* code llama.cpp calls (`ggml_quantize_chunk`), so:

  * "quantize this tensor" means exactly what llama.cpp means;
  * byte-for-byte equality with `llama-quantize` is testable (and tested);
  * the imatrix weighting convention is llama.cpp's, not ours.

The bridge is read-only w.r.t. the shared library: it never mutates model
state, it only exposes quantization primitives.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ggml type ids (mirrors enum ggml_type in ggml/include/ggml.h)
# ---------------------------------------------------------------------------
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_IQ1_M = 29
GGML_TYPE_BF16 = 30
GGML_TYPE_TQ1_0 = 34
GGML_TYPE_TQ2_0 = 35
GGML_TYPE_MXFP4 = 39
GGML_TYPE_NVFP4 = 40
GGML_TYPE_Q1_0 = 41
GGML_TYPE_Q2_0 = 42
GGML_TYPE_COUNT = 43

#: ggml type id -> canonical llama.cpp type name.
GGML_TYPE_NAMES: dict[int, str] = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_Q4_0: "Q4_0",
    GGML_TYPE_Q4_1: "Q4_1",
    GGML_TYPE_Q5_0: "Q5_0",
    GGML_TYPE_Q5_1: "Q5_1",
    GGML_TYPE_Q8_0: "Q8_0",
    GGML_TYPE_Q8_1: "Q8_1",
    GGML_TYPE_Q2_K: "Q2_K",
    GGML_TYPE_Q3_K: "Q3_K",
    GGML_TYPE_Q4_K: "Q4_K",
    GGML_TYPE_Q5_K: "Q5_K",
    GGML_TYPE_Q6_K: "Q6_K",
    GGML_TYPE_Q8_K: "Q8_K",
    GGML_TYPE_IQ2_XXS: "IQ2_XXS",
    GGML_TYPE_IQ2_XS: "IQ2_XS",
    GGML_TYPE_IQ3_XXS: "IQ3_XXS",
    GGML_TYPE_IQ1_S: "IQ1_S",
    GGML_TYPE_IQ4_NL: "IQ4_NL",
    GGML_TYPE_IQ3_S: "IQ3_S",
    GGML_TYPE_IQ2_S: "IQ2_S",
    GGML_TYPE_IQ4_XS: "IQ4_XS",
    GGML_TYPE_IQ1_M: "IQ1_M",
    GGML_TYPE_BF16: "BF16",
    GGML_TYPE_TQ1_0: "TQ1_0",
    GGML_TYPE_TQ2_0: "TQ2_0",
    GGML_TYPE_MXFP4: "MXFP4",
    GGML_TYPE_NVFP4: "NVFP4",
    GGML_TYPE_Q1_0: "Q1_0",
    GGML_TYPE_Q2_0: "Q2_0",
}

#: Reverse map: canonical name -> ggml type id. Accepts aliases used by
#: llama.cpp's ftype names (e.g. "Q4_K_M" is a *mixture*, not a type).
GGML_TYPE_IDS: dict[str, int] = {name: tid for tid, name in GGML_TYPE_NAMES.items()}

#: Quant types that llama.cpp refuses to run without an importance matrix.
IMATRIX_REQUIRED_TYPES: frozenset[int] = frozenset({
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ1_S,
})


class GgmlBridgeError(RuntimeError):
    """Raised when the llama.cpp shared library is missing or unusable."""


class _TypeTraits(ctypes.Structure):
    _fields_ = [
        ("type_name", ctypes.c_char_p),
        ("blck_size", ctypes.c_int64),
        ("blck_size_interleave", ctypes.c_int64),
        ("type_size", ctypes.c_size_t),
        ("is_quantized", ctypes.c_bool),
        ("to_float", ctypes.c_void_p),
        ("from_float_ref", ctypes.c_void_p),
    ]


class GgmlBridge:
    """ctypes wrapper around the ggml quantization API.

    The shared library is located relative to a llama.cpp checkout. A single
    instance is cached per resolved library path.
    """

    _instances: dict[str, "GgmlBridge"] = {}

    def __init__(self, lib_path: Path) -> None:
        self.lib_path = lib_path
        self._lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

        self._lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self._lib.ggml_quantize_chunk.argtypes = [
            ctypes.c_int,                        # type
            ctypes.POINTER(ctypes.c_float),      # src
            ctypes.c_void_p,                     # dst
            ctypes.c_int64,                      # start
            ctypes.c_int64,                      # nrows
            ctypes.c_int64,                      # n_per_row
            ctypes.POINTER(ctypes.c_float),      # imatrix
        ]
        self._lib.ggml_row_size.restype = ctypes.c_size_t
        self._lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self._lib.ggml_type_size.restype = ctypes.c_size_t
        self._lib.ggml_type_size.argtypes = [ctypes.c_int]
        self._lib.ggml_blck_size.restype = ctypes.c_int64
        self._lib.ggml_blck_size.argtypes = [ctypes.c_int]
        self._lib.ggml_get_type_traits.restype = ctypes.c_void_p
        self._lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
        self._lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        self._lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
        self._lib.ggml_quantize_init.restype = None
        self._lib.ggml_quantize_init.argtypes = [ctypes.c_int]

    # -- construction -------------------------------------------------------

    @classmethod
    def locate(cls, llama_cpp_dir: Optional[os.PathLike | str] = None) -> Path:
        """Find a usable llama.cpp shared library.

        Search order:
          1. explicit ``llama_cpp_dir``
          2. ``$TOFFEE_LLAMA_CPP`` env var
          3. ``./llama.cpp`` relative to CWD
          4. repository-relative ``toffee/../llama.cpp``

        Raises GgmlBridgeError when nothing usable is found.
        """
        candidates: list[Path] = []
        if llama_cpp_dir is not None:
            candidates.append(Path(llama_cpp_dir))
        env = os.environ.get("TOFFEE_LLAMA_CPP")
        if env:
            candidates.append(Path(env))
        candidates.append(Path.cwd() / "llama.cpp")
        candidates.append(Path(__file__).resolve().parents[2] / "llama.cpp")

        for root in candidates:
            found = cls._find_library(root)
            if found is not None:
                return found

        raise GgmlBridgeError(
            "Could not locate llama.cpp's libggml. Pass --llama-cpp PATH, set "
            "TOFFEE_LLAMA_CPP, or build llama.cpp at ./llama.cpp."
        )

    @staticmethod
    def _find_library(root: Path) -> Optional[Path]:
        """Locate libggml under a llama.cpp checkout."""
        if not root.is_dir():
            return None
        build_dirs = [root / "build" / "bin", root / "build", root / "bin"]
        names = ["libggml.so", "libggml.dylib", "libggml.dll"]
        for d in build_dirs:
            for n in names:
                p = d / n
                if p.exists():
                    return p.resolve()
        return None

    @classmethod
    def get(cls, llama_cpp_dir: Optional[os.PathLike | str] = None) -> "GgmlBridge":
        """Return a cached bridge instance for the given llama.cpp directory."""
        lib_path = cls.locate(llama_cpp_dir)
        key = str(lib_path)
        inst = cls._instances.get(key)
        if inst is None:
            inst = cls(lib_path)
            cls._instances[key] = inst
            logger.info("ggml_bridge_loaded", path=key)
        return inst

    @classmethod
    def available(cls, llama_cpp_dir: Optional[os.PathLike | str] = None) -> bool:
        """True when a usable llama.cpp library can be found."""
        try:
            cls.locate(llama_cpp_dir)
            return True
        except GgmlBridgeError:
            return False

    # -- traits -------------------------------------------------------------

    def type_name(self, type_id: int) -> str:
        """Canonical ggml type name (e.g. ``q4_K``)."""
        traits = self._traits(type_id)
        if traits is None or not traits.type_name:
            return GGML_TYPE_NAMES.get(type_id, f"type_{type_id}")
        return traits.type_name.decode("ascii")

    def blck_size(self, type_id: int) -> int:
        """Number of elements per block (256 for K-quants, 32 for Q4_0)."""
        traits = self._traits(type_id)
        if traits is None:
            return int(self._lib.ggml_blck_size(type_id))
        return int(traits.blck_size)

    def type_size(self, type_id: int) -> int:
        """Bytes per *block* (not per row)."""
        return int(self._lib.ggml_type_size(type_id))

    def row_size(self, type_id: int, n_per_row: int) -> int:
        """Bytes per row of ``n_per_row`` elements."""
        return int(self._lib.ggml_row_size(type_id, n_per_row))

    def is_quantized(self, type_id: int) -> bool:
        traits = self._traits(type_id)
        return bool(traits.is_quantized) if traits is not None else False

    def requires_imatrix(self, type_id: int) -> bool:
        """True when llama.cpp refuses this type without an importance matrix."""
        return bool(self._lib.ggml_quantize_requires_imatrix(type_id))

    def _traits(self, type_id: int) -> Optional[_TypeTraits]:
        ptr = self._lib.ggml_get_type_traits(type_id)
        if not ptr:
            return None
        return ctypes.cast(ptr, ctypes.POINTER(_TypeTraits)).contents

    # -- quantization -------------------------------------------------------

    def quantize(
        self,
        type_id: int,
        src: np.ndarray,
        n_per_row: int,
        imatrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Quantize ``src`` into ggml's layout for ``type_id``.

        Args:
            type_id: ggml type id (e.g. ``GGML_TYPE_Q4_K``).
            src: float32 array of shape ``(nrows, n_per_row)``, C-contiguous.
            n_per_row: elements per row.
            imatrix: optional float32 array of length ``n_per_row`` holding
                per-column importance weights (llama.cpp convention).

        Returns:
            uint8 array of the quantized bytes, length ``nrows * row_size``.
        """
        if src.dtype != np.float32:
            src = src.astype(np.float32)
        src = np.ascontiguousarray(src)
        if src.ndim == 1:
            src = src.reshape(1, -1)
        nrows = src.shape[0]
        if src.shape[1] != n_per_row:
            raise ValueError(
                f"src has {src.shape[1]} columns but n_per_row={n_per_row}"
            )

        block = self.blck_size(type_id)
        if n_per_row % block != 0:
            raise ValueError(
                f"n_per_row={n_per_row} is not a multiple of block size {block} "
                f"for {self.type_name(type_id)}"
            )

        row_bytes = self.row_size(type_id, n_per_row)
        out = ctypes.create_string_buffer(row_bytes * nrows)

        im_ptr = None
        if imatrix is not None:
            im = np.ascontiguousarray(imatrix, dtype=np.float32).reshape(-1)
            if im.shape[0] != n_per_row:
                raise ValueError(
                    f"imatrix has {im.shape[0]} entries but n_per_row={n_per_row}"
                )
            im_ptr = im.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        written = self._lib.ggml_quantize_chunk(
            type_id,
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(out),
            0,
            nrows,
            n_per_row,
            im_ptr,
        )
        if written != row_bytes * nrows:
            raise RuntimeError(
                f"ggml_quantize_chunk wrote {written} bytes, expected "
                f"{row_bytes * nrows}"
            )
        return np.frombuffer(out.raw[:written], dtype=np.uint8).copy()

    def dequantize(self, type_id: int, data: np.ndarray, n_per_row: int) -> np.ndarray:
        """Dequantize ggml-layout bytes back to float32.

        Args:
            type_id: ggml type id.
            data: uint8 buffer of quantized bytes.
            n_per_row: elements per row.

        Returns:
            float32 array of shape ``(nrows, n_per_row)``.
        """
        traits = self._traits(type_id)
        if traits is None or not traits.to_float:
            raise GgmlBridgeError(
                f"{self.type_name(type_id)} has no dequantization function"
            )
        if isinstance(data, (bytes, bytearray, memoryview)):
            data = np.frombuffer(data, dtype=np.uint8)
        else:
            data = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
        row_bytes = self.row_size(type_id, n_per_row)
        if data.size % row_bytes != 0:
            raise ValueError(
                f"data size {data.size} is not a multiple of row size {row_bytes}"
            )
        nrows = data.size // row_bytes

        fn = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64
        )(traits.to_float)

        out = np.zeros((nrows, n_per_row), dtype=np.float32)
        buf = ctypes.create_string_buffer(row_bytes)
        for r in range(nrows):
            buf.raw = data[r * row_bytes:(r + 1) * row_bytes].tobytes()
            fn(
                ctypes.byref(buf),
                out[r].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                n_per_row,
            )
        return out


def type_id_for(name: str) -> int:
    """Resolve a llama.cpp type name to its ggml type id.

    Accepts the canonical names (``Q4_K``) and the common ftype mixtures
    (``Q4_K_M``) by falling back to the mixture's dominant component.
    """
    key = name.strip().upper()
    if key in GGML_TYPE_IDS:
        return GGML_TYPE_IDS[key]
    # Mixtures: use the dominant (lowest-bit) component of the mixture.
    mixtures = {
        "Q2_K_S": GGML_TYPE_Q2_K,
        "Q2_K": GGML_TYPE_Q2_K,
        "Q3_K_S": GGML_TYPE_Q3_K,
        "Q3_K_M": GGML_TYPE_Q3_K,
        "Q3_K_L": GGML_TYPE_Q3_K,
        "Q4_K_S": GGML_TYPE_Q4_K,
        "Q4_K_M": GGML_TYPE_Q4_K,
        "Q5_K_S": GGML_TYPE_Q5_K,
        "Q5_K_M": GGML_TYPE_Q5_K,
        "Q6_K": GGML_TYPE_Q6_K,
        "IQ1_S": GGML_TYPE_IQ1_S,
        "IQ1_M": GGML_TYPE_IQ1_M,
        "IQ2_XXS": GGML_TYPE_IQ2_XXS,
        "IQ2_XS": GGML_TYPE_IQ2_XS,
        "IQ2_S": GGML_TYPE_IQ2_S,
        "IQ3_XXS": GGML_TYPE_IQ3_XXS,
        "IQ3_S": GGML_TYPE_IQ3_S,
        "IQ4_XS": GGML_TYPE_IQ4_XS,
        "IQ4_NL": GGML_TYPE_IQ4_NL,
    }
    if key in mixtures:
        return mixtures[key]
    raise KeyError(f"Unknown llama.cpp quant type: {name!r}")


def type_name_for(type_id: int) -> str:
    """Canonical (uppercase) llama.cpp name for a ggml type id."""
    return GGML_TYPE_NAMES.get(type_id, f"type_{type_id}")


__all__ = [
    "GgmlBridge",
    "GgmlBridgeError",
    "GGML_TYPE_IDS",
    "GGML_TYPE_NAMES",
    "IMATRIX_REQUIRED_TYPES",
    "type_id_for",
    "type_name_for",
]
