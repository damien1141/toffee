"""
TOFFEE quantization imatrix.

Ported from ONYX: ImatrixWriter, ActivationScaleCapturer.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional

import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger()

class ActivationScaleCapturer:
    """PyTorch forward_pre_hook that captures activation scales for a module."""

    def __init__(self, tensor_name: str) -> None:
        self.tensor_name = tensor_name
        self._accum: Optional[torch.Tensor] = None
        self._ncall = 0
        self._handles: List[Any] = []

    def pre_hook(
        self,
        module: nn.Module,
        inputs: Tuple[Any, ...],
    ) -> None:
        """Capture the input activation (x in y = x @ W^T), computing scale statistics."""
        if not inputs:
            return
        x = inputs[0]
        if not isinstance(x, torch.Tensor) or x.dim() < 2:
            return
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        # For a Linear the input is [B, ..., D] and the feature dim is the LAST
        # dim, so summing over all-but-last gives a per-feature scale of length
        # D (stable across batches). For a Conv1d/Conv2d/Conv3d the input is
        # [B, C, *spatial] and the FEATURE dim is dim 1, while the spatial dims
        # (incl. the last) vary with sequence length — summing over all-but-last
        # there produced a per-T-length accumulation that crashed when T changed
        # between batches. Reduce over (batch, *spatial) instead so the channel
        # dim is stable.
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            reduce_dims = (0,) + tuple(range(2, x.dim()))
        else:
            reduce_dims = tuple(range(x.dim() - 1))
        sq = x.pow(2).sum(dim=reduce_dims).cpu()
        if self._accum is None:
            self._accum = sq
        else:
            self._accum = self._accum + sq
        self._ncall += 1

    def register(self, module: nn.Module) -> None:
        """Register the forward_pre_hook on the given module."""
        handle = module.register_forward_pre_hook(self.pre_hook)
        self._handles.append(handle)

    def unregister(self) -> None:
        """Unregister all forward_pre_hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get_statistics(self) -> "ImatrixEntry":
        """Compute imatrix statistics from captured activations."""
        if self._accum is None:
            return ImatrixEntry(
                tensor_name=self.tensor_name,
                num_elements=1,
                squared_sums=[1.0],
                ncall=0,
            )
        return ImatrixEntry(
            tensor_name=self.tensor_name,
            num_elements=int(self._accum.numel()),
            squared_sums=self._accum.tolist(),
            ncall=self._ncall,
        )

@dataclass

class ImatrixEntry:
    """An imatrix entry for a single tensor."""
    tensor_name: str
    num_elements: int
    squared_sums: List[float] = field(default_factory=list)
    ncall: int = 1

    def to_binary(self) -> bytes:
        """Serialize to llama.cpp imatrix binary format."""
        data = bytearray()
        name_bytes = self.tensor_name.encode("utf-8")
        data.extend(struct.pack("<i", len(name_bytes)))
        data.extend(name_bytes)
        data.extend(struct.pack("<i", self.ncall))
        n_values = len(self.squared_sums)
        data.extend(struct.pack("<i", n_values))
        for val in self.squared_sums:
            data.extend(struct.pack("<f", val))
        return bytes(data)

class ImatrixWriter:
    """Writes imatrix data as a GGUF-wrapped importance matrix.

    Each captured tensor becomes two GGUF tensors: `name.in_sum2` (F32
    [nval, 1], the accumulated squared sums) and `name.counts` (F32 [1, 1], the
    number of accumulations).

    The file is rewritten on each flush to avoid the fragile GGUF append path.
    """

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.entries: List[ImatrixEntry] = []  # per-layer staging buffer
        self._all_entries: List[ImatrixEntry] = []  # all layers, kept for rewrite
        self._on_disk: int = 0
        self._names_on_disk: set = set()
        self._chunk_count: int = 0
        self._load_existing()

    def _load_existing(self) -> None:
        """Reload previously-written entries so a resumed run does not duplicate."""
        if not self.output_path.exists():
            return
        try:
            import gguf as _gguf
            reader = _gguf.GGUFReader(str(self.output_path))
        except Exception as e:
            logger.warning("imatrix_load_existing_failed", error=str(e))
            return

        loaded: List[ImatrixEntry] = []
        for t in reader.tensors:
            base = t.name[:-len(".in_sum2")] if t.name.endswith(".in_sum2") else None
            if base is None:
                continue
            sums = t.data
            counts_t = next((c for c in reader.tensors
                             if c.name == base + ".counts"), None)
            ncall = int(counts_t.data.flatten()[0]) if counts_t is not None else 1
            loaded.append(ImatrixEntry(
                tensor_name=base,
                num_elements=len(sums),
                squared_sums=sums.tolist(),
                ncall=ncall,
            ))
        self.entries = loaded
        self._all_entries = loaded
        self._on_disk = len(loaded)
        self._names_on_disk = {e.tensor_name for e in loaded}

    def add_entry(self, entry: ImatrixEntry) -> None:
        """Add or update an entry in the staging buffer and accumulated store."""
        for i, e in enumerate(self.entries):
            if e.tensor_name == entry.tensor_name:
                self.entries[i] = entry
                break
        else:
            self.entries.append(entry)
        for i, e in enumerate(self._all_entries):
            if e.tensor_name == entry.tensor_name:
                self._all_entries[i] = entry
                break
        else:
            self._all_entries.append(entry)

    def _write_gguf(self, entries: List[ImatrixEntry], chunk_count: int,
                   chunk_size: int, datasets: List[str]) -> None:
        import gguf as _gguf
        import numpy as _np

        writer = _gguf.GGUFWriter(str(self.output_path), "imatrix")
        writer.add_string("general.type", "imatrix")
        writer.add_array("imatrix.datasets", datasets)
        writer.add_uint32("imatrix.chunk_count", chunk_count)
        writer.add_uint32("imatrix.chunk_size", chunk_size)

        for entry in sorted(entries, key=lambda e: e.tensor_name):
            vals = _np.asarray(entry.squared_sums, dtype=_np.float32)
            writer.add_tensor(f"{entry.tensor_name}.in_sum2",
                             vals.reshape(-1, 1))
            writer.add_tensor(f"{entry.tensor_name}.counts",
                             _np.array([float(entry.ncall)], dtype=_np.float32).reshape(1, 1))

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for entry in sorted(entries, key=lambda e: e.tensor_name):
            vals = _np.asarray(entry.squared_sums, dtype=_np.float32)
            writer.write_tensor_data(vals.reshape(-1, 1))
            writer.write_tensor_data(
                _np.array([float(entry.ncall)], dtype=_np.float32).reshape(1, 1))
        writer.close()

    def write(self) -> None:
        """Persist accumulated entries to the GGUF imatrix, then drop them from RAM."""
        if not self.entries:
            return

        appending = len(self.entries)
        logger.info(
            "imatrix_write_start",
            on_disk=self._on_disk,
            appending=appending,
            path=str(self.output_path),
        )

        self._chunk_count += 1
        datasets = ["onyx-calibration"]
        chunk_size = 2048

        all_entries = list(self._all_entries)
        self._write_gguf(all_entries, self._chunk_count, chunk_size, datasets)

        self._on_disk = len(all_entries)
        self._names_on_disk.update(e.tensor_name for e in all_entries)
        self.entries = []
        logger.info("imatrix_write_complete", on_disk=self._on_disk, path=str(self.output_path))

    def flush(self) -> None:
        """Alias for write()."""
        self.write()

