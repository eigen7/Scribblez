"""Tests for the win-table fit (scribblez.leave_inference.win_table)."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from scribblez import ffi
from scribblez.leave_inference import win_table as wt

# This checkout's own binaries, not the primary checkout's -- see the note in
# test_move_set_eval_targets.py.
_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"


def _true_p(spread: np.ndarray, unseen: np.ndarray) -> np.ndarray:
    """A known table: logistic in spread, flatter with more tiles unseen."""
    return 1 / (1 + np.exp(-spread / (10 + unseen)))


def _synthetic_samples(n: int, seed: int = 0) -> wt.Samples:
    rng = np.random.default_rng(seed)
    spread = rng.normal(0, 60, n).round().astype(np.int32)
    unseen = rng.integers(1, 90, n).astype(np.int32)
    outcome = (rng.random(n) < _true_p(spread, unseen)).astype(np.float32)
    return wt.Samples(spread, unseen, outcome, games=n // 20)


def test_fit_recovers_a_known_table_and_is_monotone():
    grid = wt.fit(_synthetic_samples(400_000))
    assert grid.shape == (2 * wt.max_spread() + 1, wt.max_unseen() + 1)
    # Monotone in spread in every column (rows run from +max_spread down).
    assert (np.diff(grid[::-1], axis=0) >= -1e-12).all()
    spread = np.array([-60, -20, 0, 20, 60])
    for u in (10, 45, 80):
        unseen = np.full_like(spread, u)
        np.testing.assert_allclose(
            wt.lookup(grid, spread, unseen), _true_p(spread, unseen), atol=0.03
        )


def test_fitted_table_beats_a_flat_one_on_fresh_samples():
    grid = wt.fit(_synthetic_samples(200_000, seed=1))
    test = _synthetic_samples(50_000, seed=2)
    assert wt.log_loss(grid, test) < wt.log_loss(np.full_like(grid, 0.5), test) - 0.05


def test_lookup_clamps_like_the_engine():
    grid = wt.fit(_synthetic_samples(50_000))
    ms, mu = wt.max_spread(), wt.max_unseen()
    np.testing.assert_array_equal(
        wt.lookup(grid, np.array([10_000, -10_000, 5]), np.array([5, 5, 500])),
        wt.lookup(grid, np.array([ms, -ms, 5]), np.array([5, 5, mu])),
    )


def test_write_read_round_trip_keeps_the_header(tmp_path):
    grid = wt.fit(_synthetic_samples(50_000))
    path = tmp_path / "t.csv"
    wt.write(path, grid, {"policy": "--type=hastybot-endgame", "generation": 3})
    back, header = wt.read(path)
    np.testing.assert_allclose(back, grid, atol=5e-7)
    layout = ffi.format_layout()["constants"]["win_table"]
    assert header["format_version"] == str(layout["format_version"])
    assert header["policy"] == "--type=hastybot-endgame"
    assert header["generation"] == "3"


def test_read_rejects_another_format_version(tmp_path):
    path = tmp_path / "t.csv"
    wt.write(path, wt.fit(_synthetic_samples(50_000)), {})
    version = ffi.format_layout()["constants"]["win_table"]["format_version"]
    text = path.read_text().replace(f"format_version={version}", f"format_version={version + 1}")
    path.write_text(text)
    with pytest.raises(ValueError, match="format version"):
        wt.read(path)


def test_write_rejects_a_provenance_value_that_breaks_the_header(tmp_path):
    with pytest.raises(ValueError):
        wt.write(tmp_path / "t.csv", wt.fit(_synthetic_samples(50_000)), {"note": "a\nb"})


def test_read_samples_from_engine_slogs(tmp_path):
    binary = _ENGINE_DIR / "test_slog_writer"
    if not binary.is_file():
        pytest.skip("test_slog_writer not built -- run py/build.py first")
    subprocess.run([str(binary), str(tmp_path), "6", "3"], check=True, capture_output=True)
    samples = wt.read_samples(sorted(tmp_path.glob("*.slog")))
    assert samples.games == 6
    assert len(samples) > 0
    assert set(np.unique(samples.outcome)) <= {0.0, 0.5, 1.0}
    assert samples.unseen.min() > 0
    assert samples.unseen.max() <= 100
