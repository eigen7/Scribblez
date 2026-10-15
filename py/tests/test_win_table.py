"""Tests for the win-table fit (scribblez.leave_inference.win_table), with
the engine writing and evaluating the tables it fits."""

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


def _features(spread, unseen) -> np.ndarray:
    f = np.empty(len(spread), dtype=ffi.struct_dtype("WinTableFeature"))
    f["spread"], f["tiles_unseen"] = spread, unseen
    return f


def _synthetic_samples(n: int, seed: int = 0) -> wt.Samples:
    rng = np.random.default_rng(seed)
    records = np.empty(n, dtype=ffi.struct_dtype("WinTableSample"))
    spread = rng.normal(0, 60, n).round().astype(np.int32)
    unseen = rng.integers(1, 90, n).astype(np.int32)
    records["feature"] = _features(spread, unseen)
    records["outcome"] = (rng.random(n) < _true_p(spread, unseen)).astype(np.float32)
    return wt.Samples(records, games=n // 20)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory) -> tuple[np.ndarray, Path]:
    grid = wt.fit(_synthetic_samples(400_000))
    path = tmp_path_factory.mktemp("wt") / "t.csv"
    wt.write(path, grid, {"policy": "--type=hastybot-endgame", "generation": 3})
    return grid, path


def test_fit_recovers_a_known_table_and_is_monotone(fitted):
    grid, path = fitted
    assert grid.shape == (2 * wt.max_spread() + 1, wt.max_unseen() + 1)
    # Monotone in spread in every column (rows run from +max_spread down).
    assert (np.diff(grid[::-1], axis=0) >= -1e-12).all()
    spread = np.array([-60, -20, 0, 20, 60])
    for u in (10, 45, 80):
        unseen = np.full_like(spread, u)
        got = ffi.win_table_eval(path, _features(spread, unseen))
        np.testing.assert_allclose(got, _true_p(spread, unseen), atol=0.03)


def test_engine_reads_back_the_fitted_grid(fitted):
    grid, path = fitted
    ms = wt.max_spread()
    rng = np.random.default_rng(5)
    spread = rng.integers(-ms, ms + 1, 200)
    unseen = rng.integers(0, wt.max_unseen() + 1, 200)
    got = ffi.win_table_eval(path, _features(spread, unseen))
    np.testing.assert_allclose(got, grid[ms - spread, unseen], atol=5e-7)


def test_evaluation_clamps_into_the_table(fitted):
    _, path = fitted
    ms, mu = wt.max_spread(), wt.max_unseen()
    np.testing.assert_array_equal(
        ffi.win_table_eval(path, _features([10_000, -10_000, 5], [5, 5, 500])),
        ffi.win_table_eval(path, _features([ms, -ms, 5], [5, 5, mu])),
    )


def test_fitted_table_beats_a_flat_one_on_fresh_samples(fitted, tmp_path):
    _, path = fitted
    flat = tmp_path / "flat.csv"
    wt.write(flat, np.full((2 * wt.max_spread() + 1, wt.max_unseen() + 1), 0.5), {})
    test = _synthetic_samples(50_000, seed=2)
    assert wt.log_loss(path, test) < wt.log_loss(flat, test) - 0.05


def test_the_file_carries_its_version_and_provenance(fitted):
    _, path = fitted
    lines = path.read_text().splitlines()
    assert lines[0].startswith("# ")
    assert any(line.startswith("# format_version=") for line in lines[:4])
    assert "# policy=--type=hastybot-endgame" in lines
    assert "# generation=3" in lines


def test_engine_refuses_another_format_version(fitted, tmp_path):
    _, path = fitted
    text = path.read_text()
    version = next(line for line in text.splitlines() if line.startswith("# format_version="))
    stale = tmp_path / "stale.csv"
    stale.write_text(text.replace(version, f"# format_version={int(version.split('=')[1]) + 1}"))
    with pytest.raises(ValueError, match="format version"):
        ffi.win_table_eval(stale, _features([0], [50]))


def test_write_rejects_a_provenance_value_that_breaks_the_header(tmp_path):
    grid = np.full((2 * wt.max_spread() + 1, wt.max_unseen() + 1), 0.5)
    with pytest.raises(ValueError):
        wt.write(tmp_path / "t.csv", grid, {"note": "a\nb"})
    with pytest.raises(ValueError):
        wt.write(tmp_path / "t.csv", grid, {"format_version": 9})


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
