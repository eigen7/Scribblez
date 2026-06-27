"""Training dataset backed by .slog files via the native C++ DataLoader."""

from collections.abc import Iterable
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


def row_layout() -> tuple[list, list[tuple[str, int, int, tuple[int, ...]]]]:
    """Describe the contiguous training row.

    Returns (input_shapes, targets): the list of input ShapeInfo (in row order)
    and a list of (name, start, end, dims) slices for each target head, where
    start/end are float offsets into the row. Single source of truth for how a
    flat (B, row_floats) batch maps to named tensors, shared by the disk
    DataLoader and the streaming source.
    """
    input_shapes = get_input_shapes()
    target_shapes = get_target_shapes()
    input_total = sum(int(np.prod(s.dims)) for s in input_shapes)
    targets: list[tuple[str, int, int, tuple[int, ...]]] = []
    offset = input_total
    for ts in target_shapes:
        size = int(np.prod(ts.dims))
        targets.append((ts.name, offset, offset + size, ts.dims))
        offset += size
    return input_shapes, targets


def slice_row_batch(batch_2d: np.ndarray, input_shapes, targets) -> dict[str, torch.Tensor]:
    """Split a (B, row_floats) float array into named input/target tensors.

    Each named region is reshaped to its (B, *dims) tensor; a copy is taken so
    the result outlives the source buffer (important for the streaming source,
    whose slot is overwritten once released).
    """
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
    """Streams training data from .slog files via the C++ epoch-based DataLoader.

    Data is loaded on-demand in batch-sized chunks, with memory-budget-
    constrained LRU eviction. Shuffling and symmetry augmentation are
    deterministic for a given seed.
    """

    def __init__(
        self,
        data_dir: str | Path | Iterable[str | Path],
        post_move: bool = True,
        apply_symmetry: bool = True,
        memory_budget: int = 512 * 1024 * 1024,
        num_workers: int = 4,
        num_prefetch: int = 2,
    ):
        # Accept a single directory or several; a multi-directory dataset is the
        # union of every directory's .slog files (e.g. data accumulated across
        # separate generation runs). A lone str/Path is one directory, not an
        # iterable of characters.
        if isinstance(data_dir, (str, Path)):
            self.data_dirs = [Path(data_dir)]
        else:
            self.data_dirs = [Path(d) for d in data_dir]
        self.post_move = post_move
        self.apply_symmetry = apply_symmetry

        # Discover and register .slog files across every directory.
        slog_files = sorted(f for d in self.data_dirs for f in d.glob("*.slog"))
        if not slog_files:
            dirs = ", ".join(str(d) for d in self.data_dirs)
            raise FileNotFoundError(f"No .slog files in {dirs}")

        self._loader = NativeDataLoader(memory_budget, num_workers, num_prefetch)
        self._num_games = 0
        for path in slog_files:
            num_games, file_size = read_file_header(path)
            self._loader.add_file(path, num_games, file_size)
            self._num_games += num_games

        # num_samples is the loader's EXPANDED row count (one per eligible turn
        # across every game), not the game count -- read it back from the loader,
        # which derives it from each file's header.
        self._total = self._loader.num_positions
        self._row_floats = row_size_floats()
        self._input_layout, self._targets = row_layout()

    @property
    def num_samples(self) -> int:
        return self._total

    @property
    def num_games(self) -> int:
        """Total games across all files (the all-turns epoch expands each game
        into eligible_turns rows; a turns_per_game=1 epoch is one row per game)."""
        return self._num_games

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
        turns_per_game: int = 0,
        epoch_index: int = 0,
    ):
        """Yield batch dicts for one epoch, streaming from disk.

        All data is loaded on-demand with LRU eviction. Deterministic for
        a given seed.

        turns_per_game: 0 (default) iterates every eligible turn of every game.
        k > 0 draws k turns per game this epoch; pass a distinct epoch_index per
        epoch so successive epochs cover distinct turns (k == 1 makes every row
        in the epoch come from a different game).
        """
        pm = post_move if post_move is not None else self.post_move
        sym = apply_symmetry if apply_symmetry is not None else self.apply_symmetry
        self._loader.epoch_start(batch_size, pm, sym, seed, turns_per_game, epoch_index)
        while True:
            batch_data = self._loader.load_batch()
            if batch_data is None:
                return
            yield self._slice_batch(batch_data)

    def _slice_batch(self, batch_data: np.ndarray) -> dict[str, torch.Tensor]:
        return slice_row_batch(batch_data, self._input_layout, self._targets)
