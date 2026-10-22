"""
@file toffee/quant/gguf_io.py
   GGUF reading and in-place tensor patching.

TOFFEE's GSQ refinement works *in format*: it takes a GGUF that
llama-quantize already produced, re-derives better codes/scales for selected
tensors, and writes the new bytes back into the same file. That requires:

  * reading a tensor's bytes and its exact offset inside the file;
  * replacing bytes **in place** without disturbing the GGUF header,
    the KV metadata or any other tensor.

`gguf.GGUFReader` gives us offsets but is read-only. We therefore open the
file in `r+b` mode and seek to the tensor's data offset.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GgufTensorInfo:
    """Metadata about one tensor inside a GGUF file."""

    name: str
    type_name: str
    shape: List[int]
    n_elements: int
    n_bytes: int
    data_offset: int

    @property
    def n_rows(self) -> int:
        """Number of rows (all but the last dimension)."""
        if not self.shape:
            return 0
        n = 1
        for d in self.shape[1:]:
            n *= int(d)
        return n

    @property
    def n_per_row(self) -> int:
        """Elements in one row (the last dimension)."""
        return int(self.shape[0]) if self.shape else 0


def _gguf_module():
    """Import the `gguf` package, falling back to llama.cpp's bundled copy."""
    try:
        import gguf  # noqa: F401
        return gguf
    except ImportError:
        import sys

        for candidate in (
            Path.cwd() / "llama.cpp" / "gguf-py",
            Path(__file__).resolve().parents[2] / "llama.cpp" / "gguf-py",
        ):
            if candidate.is_dir():
                sys.path.insert(0, str(candidate))
                import gguf  # noqa: F401

                return gguf
        raise


def read_tensor_index(path: Path) -> Dict[str, GgufTensorInfo]:
    """Index every tensor in a GGUF file by name."""
    gguf = _gguf_module()
    reader = gguf.GGUFReader(str(path))
    out: Dict[str, GgufTensorInfo] = {}
    for t in reader.tensors:
        out[t.name] = GgufTensorInfo(
            name=t.name,
            type_name=str(t.tensor_type.name).replace("GGMLQuantizationType.", ""),
            shape=[int(d) for d in t.shape],
            n_elements=int(t.n_elements),
            n_bytes=int(t.n_bytes),
            data_offset=int(t.data_offset),
        )
    return out


def read_tensor_bytes(path: Path, name: str) -> np.ndarray:
    """Read the raw bytes of a single tensor."""
    gguf = _gguf_module()
    reader = gguf.GGUFReader(str(path))
    for t in reader.tensors:
        if t.name == name:
            return np.frombuffer(t.data.tobytes(), dtype=np.uint8).copy()
    raise KeyError(f"Tensor {name!r} not found in {path}")


def read_tensor_f32(path: Path, name: str) -> np.ndarray:
    """Read a tensor and interpret it as float32 (F32/F16/BF16 only)."""
    gguf = _gguf_module()
    reader = gguf.GGUFReader(str(path))
    for t in reader.tensors:
        if t.name != name:
            continue
        data = t.data
        if data.dtype == np.float32:
            return np.array(data, dtype=np.float32)
        if data.dtype == np.float16:
            return np.array(data, dtype=np.float32)
        raise ValueError(
            f"Tensor {name!r} has dtype {data.dtype}; only float tensors can be "
            f"read as float32."
        )
    raise KeyError(f"Tensor {name!r} not found in {path}")


def write_tensor_bytes(path: Path, name: str, data: np.ndarray) -> None:
    """Overwrite a tensor's bytes **in place**.

    The replacement must be exactly the same size as the original tensor,
    because GGUF stores no per-tensor length metadata beyond the shape and
    type -- a size change would corrupt the file.

    Args:
        path: GGUF file to patch.
        name: tensor name.
        data: uint8 bytes of the same length as the tensor's current payload.
    """
    index = read_tensor_index(path)
    if name not in index:
        raise KeyError(f"Tensor {name!r} not found in {path}")
    info = index[name]
    if isinstance(data, (bytes, bytearray, memoryview)):
        payload = np.frombuffer(data, dtype=np.uint8)
    else:
        payload = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
    if payload.size != info.n_bytes:
        raise ValueError(
            f"Replacement for {name!r} is {payload.size} bytes but the tensor "
            f"occupies {info.n_bytes} bytes; in-place patching cannot resize."
        )
    with open(path, "r+b") as f:
        f.seek(info.data_offset)
        f.write(payload.tobytes())


def iter_tensors(path: Path) -> Iterator[GgufTensorInfo]:
    """Iterate over every tensor's metadata."""
    yield from read_tensor_index(path).values()


class GgufFile:
    """A GGUF file opened for streaming reads and in-place writes.

    ``gguf.GGUFReader`` is memmapped, so it does not pull a whole file into
    RAM -- but constructing one re-parses the header, the KV block and the
    tensor-info table. Rebuilding a reader per tensor is therefore
    O(n_tensors^2) in work, which dominates the runtime on a large model
    (measured: 10 ms per tensor read on a 281-tensor file).

    This class opens the reader **once** and reuses it, caching the tensor
    index and keeping a single writable file handle for in-place patches.

    Args:
        path: the GGUF file.
        writable: whether the file will be patched in place.
    """

    def __init__(self, path: Path, writable: bool = False) -> None:
        self.path = Path(path)
        self.writable = writable
        gguf = _gguf_module()
        # A writable reader maps the file 'r+' so the memmap sees our writes.
        mode = "r+" if writable else "r"
        self._reader = gguf.GGUFReader(str(self.path), mode=mode)
        self._index: Dict[str, GgufTensorInfo] = {}
        for t in self._reader.tensors:
            self._index[t.name] = GgufTensorInfo(
                name=t.name,
                type_name=str(t.tensor_type.name).replace(
                    "GGMLQuantizationType.", ""
                ),
                shape=[int(d) for d in t.shape],
                n_elements=int(t.n_elements),
                n_bytes=int(t.n_bytes),
                data_offset=int(t.data_offset),
            )
        self._handle = None

    def __enter__(self) -> "GgufFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Release the file handle."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @property
    def index(self) -> Dict[str, GgufTensorInfo]:
        """The cached tensor index, by name."""
        return self._index

    def names(self) -> List[str]:
        """The tensor names, in file order."""
        return [t.name for t in self._reader.tensors]

    def read_bytes(self, name: str) -> np.ndarray:
        """Read one tensor's raw bytes.

        Args:
            name: tensor name.

        Returns:
            A uint8 array with the tensor's payload.
        """
        info = self._index.get(name)
        if info is None:
            raise KeyError(f"Tensor {name!r} not found in {self.path}")
        for t in self._reader.tensors:
            if t.name == name:
                return np.frombuffer(t.data.tobytes(), dtype=np.uint8).copy()
        raise KeyError(f"Tensor {name!r} not found in {self.path}")

    def read_f32(self, name: str) -> np.ndarray:
        """Read one tensor as float32 (F32/F16/BF16 only).

        Args:
            name: tensor name.

        Returns:
            The tensor's values as float32.
        """
        info = self._index.get(name)
        if info is None:
            raise KeyError(f"Tensor {name!r} not found in {self.path}")
        for t in self._reader.tensors:
            if t.name != name:
                continue
            data = t.data
            if data.dtype in (np.float32, np.float16):
                return np.array(data, dtype=np.float32)
            raise ValueError(
                f"Tensor {name!r} has dtype {data.dtype}; only float tensors "
                f"can be read as float32."
            )
        raise KeyError(f"Tensor {name!r} not found in {self.path}")

    def write_bytes(self, name: str, data: np.ndarray) -> None:
        """Overwrite one tensor's bytes **in place**.

        The replacement must be exactly the same size as the original
        tensor, because GGUF stores no per-tensor length metadata beyond
        the shape and type -- a size change would corrupt the file.

        Args:
            name: tensor name.
            data: uint8 bytes of the same length as the current payload.
        """
        info = self._index.get(name)
        if info is None:
            raise KeyError(f"Tensor {name!r} not found in {self.path}")
        if isinstance(data, (bytes, bytearray, memoryview)):
            payload = np.frombuffer(data, dtype=np.uint8)
        else:
            payload = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
        if payload.size != info.n_bytes:
            raise ValueError(
                f"Replacement for {name!r} is {payload.size} bytes but the "
                f"tensor occupies {info.n_bytes} bytes; in-place patching "
                f"cannot resize."
            )
        if self._handle is None:
            self._handle = open(self.path, "r+b")
        self._handle.seek(info.data_offset)
        self._handle.write(payload.tobytes())
        self._handle.flush()


def group_by_block(names: Iterable[str]) -> Dict[int, List[str]]:
    """Group GGUF tensor names by the block index they belong to.

    ``blk.12.ffn_gate.weight`` -> block 12. Tensors outside the block
    namespace (``token_embd.weight``, ``output.weight``) are grouped under
    block ``-1``, so a streaming pass can keep them together without
    inventing an ordering.

    Args:
        names: the tensor names.

    Returns:
        A mapping of block index to the tensor names in that block.
    """
    groups: Dict[int, List[str]] = {}
    for name in names:
        match = re.match(r"^blk\.(\d+)\.", name)
        groups.setdefault(int(match.group(1)) if match else -1, []).append(name)
    for key in groups:
        groups[key].sort()
    return groups


def tensor_map_lines(pairs: Dict[str, str]) -> str:
    """Render a ``name=TYPE`` tensor-type file body."""
    return "".join(f"{name}={qtype}\n" for name, qtype in pairs.items())


__all__ = [
    "GgufFile",
    "GgufTensorInfo",
    "group_by_block",
    "iter_tensors",
    "read_tensor_bytes",
    "read_tensor_f32",
    "read_tensor_index",
    "tensor_map_lines",
    "write_tensor_bytes",
]
