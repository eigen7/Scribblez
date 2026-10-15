"""The win table: P(win | spread + own leave value, tiles unseen) for the side
to move, fitted from self-play outcomes (docs/plans/deep_bot_v0.md, "The win
table").

The samples come from the engine (ffi.win_table_samples), which computes each
position's cell with the same function a sim leaf uses, so the fit and the
lookup agree on what a cell means. The fit:

1. counts outcomes per cell, with the spread clamped to the table's range;
2. pools each unseen column with its neighbors until it holds
   `min_column_samples` samples;
3. kernel-smooths the pooled column along the spread, then makes it monotone
   in spread by weighted isotonic regression;
4. averages each fitted column with its neighbors along the unseen axis.

Averaging monotone columns keeps them monotone. This module only fits: the
engine's WinPctTable owns the table format and its evaluation, and both are
reached through the FFI (ffi.win_table_write, ffi.win_table_eval), so a change
of format needs no second reader here.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scribblez import ffi
from scribblez.paths import DEFAULT_MOUNT_ROOT

# The table bundled with the Macondo checkout: the reference a fitted table is
# compared against. The engine reads the same file (WinPctTable::macondo_default).
MACONDO_WIN_TABLE = DEFAULT_MOUNT_ROOT / "macondo" / "data" / "strategy" / "default" / "winpct.csv"


@dataclass(frozen=True)
class FitParams:
    # An unseen column is pooled with its neighbors until it holds this many
    # samples (or the pool spans every column).
    min_column_samples: int = 5000
    # The Gaussian kernel's standard deviation along the spread, in points.
    spread_bandwidth: float = 4.0
    # Each fitted column is averaged with this many neighbors on each side.
    unseen_smoothing: int = 1


DEFAULT_FIT = FitParams()


@dataclass
class Samples:
    """Fitting samples: WinTableSample records (ffi.win_table_samples), one
    per position, and the number of games they came from."""

    records: np.ndarray
    games: int

    def __len__(self) -> int:
        return len(self.records)

    @property
    def spread(self) -> np.ndarray:
        return self.records["feature"]["spread"]

    @property
    def unseen(self) -> np.ndarray:
        return self.records["feature"]["tiles_unseen"]

    @property
    def outcome(self) -> np.ndarray:
        return self.records["outcome"]


def _layout() -> dict:
    return ffi.format_layout()["constants"]["win_table"]


def max_spread() -> int:
    return int(_layout()["max_spread"])


def max_unseen() -> int:
    return int(_layout()["max_unseen"])


def read_samples(slog_paths: Iterable[Path]) -> Samples:
    """Every fittable position of every game in `slog_paths`."""
    parts = []
    games = 0
    for path in slog_paths:
        parts.append(ffi.win_table_samples(path))
        games += ffi.read_file_header(path)[0]
    if not parts:
        raise ValueError("no .slog files to fit from")
    return Samples(np.concatenate(parts), games)


def _cell_counts(samples: Samples) -> tuple[np.ndarray, np.ndarray]:
    """(n, wins), each (rows, cols): row r holds spread max_spread - r, as in
    the file; column u holds u tiles unseen."""
    ms, mu = max_spread(), max_unseen()
    rows = ms - np.clip(samples.spread, -ms, ms)
    cols = np.clip(samples.unseen, 0, mu)
    shape = (2 * ms + 1, mu + 1)
    flat = rows * shape[1] + cols
    n = np.bincount(flat, minlength=shape[0] * shape[1]).reshape(shape).astype(np.float64)
    wins = np.bincount(flat, weights=samples.outcome, minlength=n.size).reshape(shape)
    return n, wins


def _pooled_columns(n: np.ndarray, wins: np.ndarray, min_samples: int):
    """Per unseen column, its counts pooled with the fewest symmetric
    neighbors that reach `min_samples`. Yields (n_col, wins_col), or None for
    a column whose widest pool is still empty."""
    cols = n.shape[1]
    totals = n.sum(axis=0)
    for u in range(cols):
        for k in range(cols):
            lo, hi = max(0, u - k), min(cols, u + k + 1)
            if totals[lo:hi].sum() >= min_samples or (lo == 0 and hi == cols):
                break
        n_col, wins_col = n[:, lo:hi].sum(axis=1), wins[:, lo:hi].sum(axis=1)
        yield (n_col, wins_col) if n_col.sum() > 0 else None


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(np.ceil(4 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return np.convolve(x, kernel, mode="same")


def _isotonic_increasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares nondecreasing fit (pool adjacent violators)."""
    means, weights, sizes = [], [], []
    for yi, wi in zip(y, w, strict=True):
        means.append(yi)
        weights.append(wi)
        sizes.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            m2, w2, s2 = means.pop(), weights.pop(), sizes.pop()
            wt = weights[-1] + w2
            means[-1] = (means[-1] * weights[-1] + m2 * w2) / wt
            weights[-1] = wt
            sizes[-1] += s2
    return np.repeat(means, sizes)


def _fit_column(n_col: np.ndarray, wins_col: np.ndarray, bandwidth: float) -> np.ndarray:
    """P(win) per row of one pooled column, monotone in spread."""
    n_s = _gaussian_smooth(n_col, bandwidth)
    wins_s = _gaussian_smooth(wins_col, bandwidth)
    supported = n_s > 1e-9
    # Fit in ascending spread, which is descending row order.
    y = np.where(supported, wins_s / np.where(supported, n_s, 1.0), 0.0)[::-1]
    w = np.where(supported, n_s, 0.0)[::-1]
    idx = np.flatnonzero(w)
    fitted = _isotonic_increasing(y[idx], w[idx])
    # Rows the kernel never reached take their nearest supported neighbor's
    # value, which keeps the column monotone.
    column = np.interp(np.arange(len(y)), idx, fitted)
    return column[::-1]


def fit(samples: Samples, params: FitParams = DEFAULT_FIT) -> np.ndarray:
    """The fitted grid: (rows, cols), row r spread max_spread - r, column u
    tiles unseen."""
    n, wins = _cell_counts(samples)
    columns = [
        None if pooled is None else _fit_column(*pooled, params.spread_bandwidth)
        for pooled in _pooled_columns(n, wins, params.min_column_samples)
    ]
    fitted = [u for u, c in enumerate(columns) if c is not None]
    if not fitted:
        raise ValueError("no samples to fit")
    # A column with no data anywhere in reach copies the nearest fitted one.
    for u, c in enumerate(columns):
        if c is None:
            columns[u] = columns[min(fitted, key=lambda f: abs(f - u))]
    grid = np.stack(columns, axis=1)
    k = params.unseen_smoothing
    smoothed = np.empty_like(grid)
    for u in range(grid.shape[1]):
        smoothed[:, u] = grid[:, max(0, u - k) : u + k + 1].mean(axis=1)
    return smoothed


def write(path: Path, grid: np.ndarray, provenance: Mapping[str, object]):
    """Writes the fitted `grid` as a win table, `provenance` its header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ffi.win_table_write(path, grid, dict(provenance))


def log_loss(
    table: Path, samples: Samples, macondo_format: bool = False, eps: float = 1e-4
) -> float:
    """Mean binary cross-entropy of the samples' outcomes under the table at
    `table` (Macondo's winpct.csv if `macondo_format`), in nats. A tie counts
    half a win."""
    p = ffi.win_table_eval(table, samples.records["feature"], macondo_format).astype(np.float64)
    p = np.clip(p, eps, 1 - eps)
    y = samples.outcome
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
