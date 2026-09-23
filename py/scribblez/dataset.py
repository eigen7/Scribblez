"""Training dataset backed by .slog files via the native C++ DataLoader."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from .ffi import (
    NativeDataLoader,
    get_input_shapes,
    get_max_move_per_lane_input_shapes,
    get_max_move_per_lane_target_shapes,
    get_target_shapes,
    read_file_header,
)


def row_layout(input_shapes=None, target_shapes=None):
    """How a flat (B, row_floats) training batch maps to named tensors.

    Returns (input_shapes, targets): the input ShapeInfo list in row order, and
    one (name, start, end, dims) entry per target, where start/end are float
    offsets into the row. Inputs come first, then targets. The shapes default to
    the position-evaluation task's; pass explicit lists for another task (e.g.
    max-move-per-lane).
    """
    input_shapes = get_input_shapes() if input_shapes is None else input_shapes
    target_shapes = get_target_shapes() if target_shapes is None else target_shapes
    input_total = sum(int(np.prod(s.dims)) for s in input_shapes)
    targets: list[tuple[str, int, int, tuple[int, ...]]] = []
    offset = input_total
    for ts in target_shapes:
        size = int(np.prod(ts.dims))
        targets.append((ts.name, offset, offset + size, ts.dims))
        offset += size
    return input_shapes, targets


def slice_row_batch(batch_2d: np.ndarray, input_shapes, targets) -> dict[str, torch.Tensor]:
    """Split a (B, row_floats) float array into named (B, *dims) tensors,
    each a contiguous copy of its column range."""
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for s in input_shapes:
        size = int(np.prod(s.dims))
        arr = batch_2d[:, offset : offset + size]
        result[s.name] = torch.from_numpy(arr.reshape(-1, *s.dims).copy())
        offset += size
    for name, start, end, dims in targets:
        arr = batch_2d[:, start:end]
        result[name] = torch.from_numpy(arr.reshape(-1, *dims).copy())
    return result


class SlogDataset:
    """Streams training rows from .slog files through the C++ DataLoader.

    Games are decoded on demand under a memory budget with LRU eviction.
    Shuffling and symmetry augmentation are deterministic for a given seed.
    """

    def __init__(
        self,
        data_dir: str | Path | Iterable[str | Path],
        task: str = "position_eval",
        post_move: bool = True,
        apply_symmetry: bool = True,
        memory_budget: int = 512 * 1024 * 1024,
        num_workers: int = 4,
        num_prefetch: int = 2,
    ):
        # A multi-directory dataset is the union of every directory's .slog files.
        # A lone str/Path is one directory, not an iterable of characters.
        if isinstance(data_dir, (str, Path)):
            self.data_dirs = [Path(data_dir)]
        else:
            self.data_dirs = [Path(d) for d in data_dir]
        self.task = task
        self.post_move = post_move
        self.apply_symmetry = apply_symmetry

        slog_files = sorted(f for d in self.data_dirs for f in d.glob("*.slog"))
        if not slog_files:
            dirs = ", ".join(str(d) for d in self.data_dirs)
            raise FileNotFoundError(f"No .slog files in {dirs}")

        self._loader = NativeDataLoader(memory_budget, num_workers, num_prefetch, task=task)
        self._num_games = 0
        for path in slog_files:
            num_games, file_size = read_file_header(path)
            self._loader.add_file(path, num_games, file_size)
            self._num_games += num_games

        # One row per eligible turn, so the row count exceeds the game count.
        self._total = self._loader.num_positions
        self._row_floats = self._loader.row_floats
        if task == "max_move_per_lane":
            self._input_shapes = get_max_move_per_lane_input_shapes()
            target_shapes = get_max_move_per_lane_target_shapes()
        else:
            self._input_shapes = get_input_shapes()
            target_shapes = get_target_shapes()
        self._input_layout, self._targets = row_layout(self._input_shapes, target_shapes)

    @property
    def num_samples(self) -> int:
        return self._total

    @property
    def num_games(self) -> int:
        """Total games across all files. An all-turns epoch yields one row per
        eligible turn; a turns_per_game=1 epoch yields one row per game."""
        return self._num_games

    @property
    def input_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-sample input shapes, keyed by input name."""
        return {s.name: s.dims for s in self._input_shapes}

    def iter_batches(
        self,
        batch_size: int,
        seed: int = 42,
        post_move: bool | None = None,
        apply_symmetry: bool | None = None,
        turns_per_game: int = 0,
        epoch_index: int = 0,
        drop_last: bool = False,
    ):
        """Yield one epoch of batch dicts, deterministic for a given seed.

        turns_per_game: 0 iterates every eligible turn of every game; k > 0
        draws k turns per game. Pass a distinct epoch_index per epoch so
        successive epochs draw distinct turns.
        drop_last: skip the trailing short batch so every batch has the same
        shape.
        """
        pm = post_move if post_move is not None else self.post_move
        sym = apply_symmetry if apply_symmetry is not None else self.apply_symmetry
        self._loader.epoch_start(batch_size, pm, sym, seed, turns_per_game, epoch_index)
        while True:
            batch_data = self._loader.load_batch()
            if batch_data is None:
                return
            if drop_last and batch_data.shape[0] < batch_size:
                continue
            yield self._slice_batch(batch_data)

    def _slice_batch(self, batch_data: np.ndarray) -> dict[str, torch.Tensor]:
        return slice_row_batch(batch_data, self._input_layout, self._targets)
