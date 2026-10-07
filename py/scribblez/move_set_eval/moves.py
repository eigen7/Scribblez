"""Move-feature inputs for the move set evaluation model.

The engine owns the encoding (engine/include/training/move_set_encoder.h), so
the training dataset and the in-engine agents share one implementation. This
module re-exports its FFI bindings: `encode_moves` turns packed Move records
into per-candidate letter/blank/square/mask/scalar arrays, `cross_checks` /
`gcg_cross_checks` give each candidate's post-move cross-check entries on its
position's board, `move_encoding_dims` and `move_cross_slots` report the layout
constants the model sizes its embeddings from, and `score_diff_input_layout`
locates the score-diff scalar in the board input so the dataset can read a
position's pre-move differential from its encoded row.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from scribblez.ffi import (
    cross_checks,
    encode_moves,
    gcg_cross_checks,
    move_cross_slots,
    move_encoding_dims,
    move_encoding_version,
    score_diff_input_layout,
)

BOARD = 15

__all__ = [
    "BOARD",
    "batch_cross_checks",
    "cross_checks",
    "encode_moves",
    "gcg_cross_checks",
    "move_cross_slots",
    "move_encoding_dims",
    "move_encoding_version",
    "score_diff_input_layout",
    "synthetic_cross_checks",
]


def batch_cross_checks(
    slogs: list[Path], positions: list[tuple[int, int, int]], moves: list[np.ndarray]
) -> dict[str, np.ndarray]:
    """cross_checks for a batch whose positions span several .slog files.

    Position j is `positions[j]` = (file_id, game_index, turn_index), its
    .slog `slogs[file_id]`, and its candidates `moves[j]`. One call per file;
    the result rows follow the batch's own order, position-major."""
    counts = np.array([len(m) for m in moves], dtype=np.int64)
    starts = np.cumsum(counts) - counts
    total = int(counts.sum())
    slots = move_cross_slots()
    out = {
        "cells": np.zeros((total, slots), dtype=np.int64),
        "letters": np.zeros((total, slots * 26), dtype=np.uint8),
    }
    by_file: dict[int, list[int]] = defaultdict(list)
    for j, (file_id, _, _) in enumerate(positions):
        by_file[file_id].append(j)
    for file_id, js in by_file.items():
        got = cross_checks(
            slogs[file_id],
            np.array([positions[j][1] for j in js], dtype=np.int64),
            np.array([positions[j][2] for j in js], dtype=np.int64),
            counts[js],
            np.concatenate([moves[j] for j in js]),
        )
        rows = np.concatenate([np.arange(starts[j], starts[j] + counts[j]) for j in js])
        for key, values in got.items():
            out[key][rows] = values
    return out


def synthetic_cross_checks(is_play: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Cross-check features shaped as the engine fills them, for fixtures and
    tests without a board: each play gets 1..3 real entries (random cells and
    legal-letter flags), every other slot and every non-play none. The model
    reads them as numbers, so they need not be consistent with any position."""
    slots = move_cross_slots()
    _, _, _, cells_per_axis = move_encoding_dims()
    rng = np.random.default_rng(seed)
    m = len(is_play)
    cells = np.zeros((m, slots), dtype=np.int64)
    letters = np.zeros((m, slots, 26), dtype=np.uint8)
    for i in np.flatnonzero(is_play):
        n = int(rng.integers(1, 4))
        axis = rng.integers(0, 2, n)
        cells[i, :n] = 1 + axis * cells_per_axis + rng.integers(0, cells_per_axis, n)
        letters[i, :n] = rng.integers(0, 2, (n, 26))
    return {"cells": cells, "letters": letters.reshape(m, slots * 26)}
