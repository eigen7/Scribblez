"""Training dataset backed by .slog files via the native C++ DataLoader."""

from pathlib import Path

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
    """Streams training data from .slog files via the C++ epoch-based DataLoader.

    Data is loaded on-demand in batch-sized chunks, with memory-budget-
    constrained LRU eviction. Shuffling and symmetry augmentation are
    deterministic for a given seed.
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

    @property
    def num_samples(self) -> int:
        return self._total

    @property
    def input_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-input tensor shapes (channel/feature dims), keyed by name."""
        return {s.name: s.dims for s in get_input_shapes()}

    def iter_batches(
        self,
        batch_size: int,
        seed: int = 42,
        post_move: bool | None = None,
        apply_symmetry: bool | None = None,
    ):
        """Yield batch dicts for one epoch, streaming from disk.

        All data is loaded on-demand with LRU eviction. Deterministic for
        a given seed.
        """
        pm = post_move if post_move is not None else self.post_move
        sym = apply_symmetry if apply_symmetry is not None else self.apply_symmetry
        self._loader.epoch_start(batch_size, pm, sym, seed)
        while True:
            batch_data = self._loader.load_batch()
            if batch_data is None:
                return
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
