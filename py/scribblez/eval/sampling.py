"""Sample a frozen evaluation subset from the held-out test split.

Picks N games (one sampled position each) from a directory of test .slog files
and writes them into a single standalone .slog -- a curated, frozen set of
positions stored in the standard format, inspectable and loadable by the same
tooling as training data. This replaces a specialized in-memory bank: the
probes read the resulting .slog directly each generation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..ffi import read_file_header, sample_slog


def select_positions(test_dir: str | Path, num_positions: int) -> list[tuple[Path, int]]:
    """Pick `num_positions` (file, game_idx) pairs evenly across the test set."""
    test_dir = Path(test_dir)
    slog_files = sorted(test_dir.glob("*.slog"))
    if not slog_files:
        raise FileNotFoundError(f"No .slog files in {test_dir}")
    catalog: list[tuple[Path, int]] = []
    for f in slog_files:
        num_games, _ = read_file_header(f)
        catalog.extend((f, idx) for idx in range(num_games))
    if not catalog:
        raise ValueError(f"Test set {test_dir} contains no games")
    n = min(num_positions, len(catalog))
    picks = np.linspace(0, len(catalog) - 1, n, dtype=np.int64)
    return [catalog[i] for i in picks]


def build_test_subset(test_dir: str | Path, out_slog: str | Path, num_positions: int = 12) -> int:
    """Sample positions from `test_dir` into the .slog at `out_slog`.

    Selection is deterministic (evenly spaced across all test games), so the
    same test set always yields the same subset. Returns the number written.
    """
    picks = select_positions(test_dir, num_positions)
    Path(out_slog).parent.mkdir(parents=True, exist_ok=True)
    sample_slog(out_slog, picks)
    return len(picks)
