"""Frozen bank of positions for the structural monotonicity probe.

A probe bank is a small, fixed set of board positions drawn from the held-out
test split. Each position is pre-encoded into a sweep of input tensors that
differ ONLY in the active player's score differential (the sweep is performed
in C++ by GameStateEncoder, so no feature-layout knowledge lives here). The
spatial planes are identical across a position's sweep and are stored once.

The bank is built once and cached to disk; every training epoch reloads it and
runs the current model over it, so the only thing that changes across epochs is
the model -- exactly what the probe is meant to isolate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..ffi import get_input_shapes, probe_encode_sweep, read_file_header


@dataclass
class ProbeBank:
    """Pre-encoded probe positions swept over score differential.

    Attributes:
        score_diffs: (S,) the swept score differentials (active - opponent).
        spatial:     (N, C, H, W) per-position spatial planes (sweep-invariant).
        scalar:      (N, S, F) per-position, per-step scalar features.
        files:       length-N source .slog paths (provenance).
        game_idx:    (N,) source game index within each file.
    """

    score_diffs: np.ndarray
    spatial: np.ndarray
    scalar: np.ndarray
    files: list[str]
    game_idx: np.ndarray

    @property
    def num_positions(self) -> int:
        return self.spatial.shape[0]

    @property
    def num_steps(self) -> int:
        return self.score_diffs.shape[0]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            score_diffs=self.score_diffs,
            spatial=self.spatial,
            scalar=self.scalar,
            files=np.array(self.files, dtype=object),
            game_idx=self.game_idx,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ProbeBank":
        d = np.load(path, allow_pickle=True)
        return cls(
            score_diffs=d["score_diffs"],
            spatial=d["spatial"],
            scalar=d["scalar"],
            files=list(d["files"]),
            game_idx=d["game_idx"],
        )


def _select_positions(test_dir: Path, num_positions: int) -> list[tuple[Path, int]]:
    """Pick `num_positions` (file, game_idx) pairs evenly across the test set."""
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


def build_probe_bank(
    test_dir: str | Path,
    num_positions: int = 12,
    diff_lo: int = -200,
    diff_hi: int = 200,
    step: int = 1,
    post_move: bool = True,
) -> ProbeBank:
    """Build a probe bank from the held-out test split.

    Positions are sampled deterministically (evenly spaced across all test
    games), so the same test set always yields the same bank.
    """
    test_dir = Path(test_dir)
    score_diffs = np.arange(diff_lo, diff_hi + 1, step, dtype=np.float32)
    positions = _select_positions(test_dir, num_positions)

    spatial_shape = next(s.dims for s in get_input_shapes() if s.name == "input_spatial")
    spatial_floats = int(np.prod(spatial_shape))

    spatial_list, scalar_list, files, idxs = [], [], [], []
    for f, idx in positions:
        sweep = probe_encode_sweep(f, idx, score_diffs, post_move=post_move)  # (S, input_floats)
        # Spatial planes are identical across the sweep; store one copy.
        spatial_list.append(sweep[0, :spatial_floats].reshape(spatial_shape))
        scalar_list.append(sweep[:, spatial_floats:])
        files.append(str(f))
        idxs.append(idx)

    return ProbeBank(
        score_diffs=score_diffs,
        spatial=np.stack(spatial_list),
        scalar=np.stack(scalar_list),
        files=files,
        game_idx=np.array(idxs, dtype=np.int64),
    )
