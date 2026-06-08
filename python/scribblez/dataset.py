"""Training dataset backed by .slog files via the native C++ DataLoader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .ffi import (
    NativeDataLoader,
    get_input_shapes,
    get_target_shapes,
    read_file_header,
    row_size_floats,
)


class SlogDataset:
    """Loads all .slog files from a directory and provides epoch iteration.

    For v1, all data is loaded into RAM at construction. Shuffling and
    batching are handled in Python. The C++ DataLoader handles parallel
    file I/O and game-replay decoding.
    """

    def __init__(
        self,
        data_dir: str | Path,
        post_move: bool = True,
        apply_symmetry: bool = True,
        memory_budget: int = 512 * 1024 * 1024,
        num_workers: int = 4,
        num_prefetch: int = 2,
    ):
        self.data_dir = Path(data_dir)
        self.post_move = post_move
        self.apply_symmetry = apply_symmetry

        # Discover and register .slog files.
        slog_files = sorted(self.data_dir.glob("*.slog"))
        if not slog_files:
            raise FileNotFoundError(f"No .slog files in {self.data_dir}")

        self._loader = NativeDataLoader(memory_budget, num_workers, num_prefetch)
        total = 0
        for path in slog_files:
            num_pos, file_size = read_file_header(path)
            self._loader.add_file(path, num_pos, file_size)
            total += num_pos

        self._total = total
        self._row_floats = row_size_floats()

        # Compute tensor slicing offsets from the shape info.
        input_shapes = get_input_shapes()
        target_shapes = get_target_shapes()

        # Input region: all floats before the first target.
        input_total = sum(_prod(s.dims) for s in input_shapes)
        self._input_end = input_total

        # Target regions: consecutive after inputs.
        self._targets: list[tuple[str, int, int, tuple[int, ...]]] = []
        offset = input_total
        for ts in target_shapes:
            size = _prod(ts.dims)
            self._targets.append((ts.name, offset, offset + size, ts.dims))
            offset += size

        self._data: Optional[np.ndarray] = None

    @property
    def num_samples(self) -> int:
        return self._total

    def load(self) -> None:
        """Load all data into RAM (one big C++ load() call)."""
        if self._data is not None:
            return
        self._data = self._loader.load(
            0, self._total, self.post_move, self.apply_symmetry
        )

    def get_tensors(self) -> dict[str, torch.Tensor]:
        """Return the full dataset as a dict of named torch tensors."""
        self.load()
        assert self._data is not None
        data = self._data

        result: dict[str, torch.Tensor] = {}

        # Inputs: split into spatial + scalar based on shape info.
        input_shapes = get_input_shapes()
        offset = 0
        for s in input_shapes:
            size = _prod(s.dims)
            arr = data[:, offset : offset + size]
            result[s.name] = torch.from_numpy(arr.reshape(-1, *s.dims))
            offset += size

        # Targets.
        for name, start, end, dims in self._targets:
            arr = data[:, start:end]
            result[name] = torch.from_numpy(arr.reshape(-1, *dims))

        return result

    def iter_batches(
        self, batch_size: int, shuffle: bool = True, drop_last: bool = True
    ):
        """Yield (batch_dict) for one epoch over the loaded data."""
        self.load()
        assert self._data is not None

        n = self._total
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        end = (n // batch_size) * batch_size if drop_last else n
        for start in range(0, end, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_data = self._data[batch_idx]
            yield self._slice_batch(batch_data)

    def _slice_batch(self, batch_data: np.ndarray) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        input_shapes = get_input_shapes()
        offset = 0
        for s in input_shapes:
            size = _prod(s.dims)
            arr = batch_data[:, offset : offset + size]
            result[s.name] = torch.from_numpy(arr.reshape(-1, *s.dims).copy())
            offset += size
        for name, start, end, dims in self._targets:
            arr = batch_data[:, start:end]
            result[name] = torch.from_numpy(arr.reshape(-1, *dims).copy())
        return result


def _prod(dims: tuple[int, ...]) -> int:
    r = 1
    for d in dims:
        r *= d
    return r
