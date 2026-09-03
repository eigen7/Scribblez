"""End-to-end tests for the streaming DataLoader via the Python FFI.

These tests create .slog files using the C++ test_slog_writer binary,
then exercise:
  1. Epoch determinism: same seed → identical output
  2. Coverage: all positions appear exactly once per epoch
  3. Memory-budget stress: tiny budget, verify all data is still yielded
  4. Streaming dataset iteration via SlogDataset.iter_batches()
"""

import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest
from scribblez.dataset import SlogDataset, row_layout, slice_row_batch
from scribblez.ffi import (
    NativeDataLoader,
    format_layout,
    get_input_shapes,
    get_target_shapes,
    read_file_header,
    row_size_floats,
)

# This checkout's own binaries, not the primary checkout's -- see the note in
# test_move_set_eval_targets.py.
_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"

# The four placement heads, the two per-side legality masks, and the footprint
# class-space width, all from the same FFI source the engine targets use.
_PLACEMENT_HEADS = tuple(format_layout()["constants"]["placement_head_names"])
_PLACEMENT_MASKS = tuple(format_layout()["constants"]["placement_mask_names"])
_FOOTPRINT_CLASSES = format_layout()["constants"]["footprint"]["num_classes"]

# ---------------------------------------------------------------------------
# Fixture: generate .slog files using the test_slog_writer binary.
# ---------------------------------------------------------------------------


def generate_test_slogs(tmpdir: Path, num_games: int = 12, games_per_file: int = 4) -> list[Path]:
    """Generate .slog files for testing using the test_slog_writer binary."""
    binary = _ENGINE_DIR / "test_slog_writer"
    if not binary.is_file():
        pytest.skip("test_slog_writer not built -- run 'make test_slog_writer' first")

    result = subprocess.run(
        [str(binary), str(tmpdir), str(num_games), str(games_per_file)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"test_slog_writer failed: {result.stderr}"

    slogs = sorted(tmpdir.glob("*.slog"))
    assert len(slogs) > 0
    return slogs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEpochDeterminism:
    def test_same_seed_same_output(self, tmp_path):
        slogs = generate_test_slogs(tmp_path)

        def run_epoch(seed: int) -> np.ndarray:
            loader = NativeDataLoader(
                memory_budget=256 * 1024 * 1024, num_workers=2, num_prefetch=1
            )
            for p in slogs:
                num_pos, fsize = read_file_header(p)
                loader.add_file(p, num_pos, fsize)
            loader.epoch_start(batch_size=4, post_move=True, apply_symmetry=True, seed=seed)
            all_data = []
            while True:
                batch = loader.load_batch()
                if batch is None:
                    break
                all_data.append(batch.copy())
            return np.concatenate(all_data, axis=0)

        d1 = run_epoch(seed=12345)
        d2 = run_epoch(seed=12345)
        d3 = run_epoch(seed=99999)

        assert d1.shape == d2.shape
        assert d1.shape[0] > 0
        np.testing.assert_array_equal(d1, d2)

        # Different seed → different order.
        assert d1.shape == d3.shape
        assert not np.array_equal(d1, d3)


class TestEpochCoverage:
    def test_all_positions_appear_once(self, tmp_path):
        """Verify every position appears exactly once per epoch by running
        two epochs with different seeds and checking they contain the same
        set of rows (just in different order)."""
        slogs = generate_test_slogs(tmp_path)

        loader = NativeDataLoader(memory_budget=256 * 1024 * 1024, num_workers=2, num_prefetch=1)
        for p in slogs:
            num_games, fsize = read_file_header(p)
            loader.add_file(p, num_games, fsize)
        # read_file_header reports games; the default (turns_per_game=0) epoch
        # expands each game into all its eligible turns, so the per-epoch row
        # count is the loader's total expanded position count.
        total = loader.num_positions

        def drain_epoch(seed: int) -> np.ndarray:
            loader.epoch_start(batch_size=3, post_move=True, apply_symmetry=False, seed=seed)
            rows = []
            while True:
                batch = loader.load_batch()
                if batch is None:
                    break
                rows.append(batch.copy())
            return np.concatenate(rows, axis=0)

        epoch1 = drain_epoch(seed=7777)
        epoch2 = drain_epoch(seed=8888)

        assert epoch1.shape[0] == total
        assert epoch2.shape[0] == total

        # Different order.
        assert not np.array_equal(epoch1, epoch2)

        # Same set of rows (sort by raw bytes and compare).
        e1_sorted = np.sort(epoch1.view(np.uint8).reshape(total, -1), axis=0)
        e2_sorted = np.sort(epoch2.view(np.uint8).reshape(total, -1), axis=0)
        np.testing.assert_array_equal(e1_sorted, e2_sorted)


class TestStreamingDataset:
    def test_iter_batches(self, tmp_path):
        """Test the SlogDataset.iter_batches() method."""
        generate_test_slogs(tmp_path)

        ds = SlogDataset(
            tmp_path, post_move=True, apply_symmetry=True, memory_budget=256 * 1024 * 1024
        )
        batches = list(ds.iter_batches(batch_size=4, seed=555))
        assert len(batches) > 0

        # Each batch should have the expected tensor keys and the session's
        # own input dims (derived, so a layout change can't leave a stale magic
        # number here).
        in_shapes = {s.name: tuple(s.dims) for s in get_input_shapes()}
        for b in batches:
            assert "input_spatial" in b
            assert "input_scalar" in b
            assert "wld" in b
            assert "score_diff" in b
            for head in _PLACEMENT_HEADS:
                assert head in b  # footprint class index
            for mask in _PLACEMENT_MASKS:
                assert mask in b  # per-side legality mask
            assert tuple(b["input_spatial"].shape[1:]) == in_shapes["input_spatial"]
            assert b["input_scalar"].shape[1] == in_shapes["input_scalar"][0]
            assert b["wld"].shape[1] == 3
            assert b["score_diff"].shape[1] == 1
            # Each placement head is a single footprint class index; each side
            # carries one FOOTPRINT_CLASSES-wide legality mask (not a per-cell
            # (15,15) map).
            for head in _PLACEMENT_HEADS:
                assert b[head].shape[1] == 1
            for mask in _PLACEMENT_MASKS:
                assert b[mask].shape[1] == _FOOTPRINT_CLASSES

        # Determinism: same seed, same output.
        batches2 = list(ds.iter_batches(batch_size=4, seed=555))
        assert len(batches) == len(batches2)
        for b1, b2 in zip(batches, batches2, strict=True):
            for key in b1:
                np.testing.assert_array_equal(b1[key].numpy(), b2[key].numpy())

    def test_drop_last_yields_only_full_batches(self, tmp_path):
        """With a batch size that does not divide the epoch's row count, the
        default epoch ends in a short batch and drop_last omits exactly that
        batch, leaving every yielded batch at batch_size rows."""
        generate_test_slogs(tmp_path)  # 12 games; one row per game below
        ds = SlogDataset(
            tmp_path, post_move=True, apply_symmetry=False, memory_budget=256 * 1024 * 1024
        )
        kw = dict(batch_size=5, seed=9, turns_per_game=1)
        sizes = [b["wld"].shape[0] for b in ds.iter_batches(**kw)]
        assert sizes == [5, 5, 2]
        kept = [b["wld"].shape[0] for b in ds.iter_batches(**kw, drop_last=True)]
        assert kept == [5, 5]


class TestUnreadableFile:
    def test_deleted_file_raises_not_hangs(self, tmp_path):
        """A registered .slog that vanishes before its body is read must raise a
        clean error, not wedge forever.

        Regression guard for the deadlock where a failed body load left
        DataFile::buffer() waiting on `buffer_ != nullptr` forever, hanging the
        decode worker (and load_batch) indefinitely. On a live generational tag
        a window file can be evicted or rewritten between when SlogDataset reads
        the headers and when the loader lazily loads the bodies, so this is a
        real path, not a contrived one. The iteration runs on a watchdog thread
        so a regressed hang fails the test instead of blocking the suite.
        """
        slogs = generate_test_slogs(tmp_path)
        assert len(slogs) >= 2

        ds = SlogDataset(
            tmp_path, post_move=True, apply_symmetry=True, memory_budget=256 * 1024 * 1024
        )
        # Remove one registered file out from under the loader before iterating;
        # its header (and thus its rows) were already registered at construction.
        slogs[0].unlink()

        result: dict[str, object] = {}

        def drain():
            try:
                # turns_per_game=0 touches every game of every file, so the
                # deleted file is certainly demanded.
                for _ in ds.iter_batches(batch_size=4, seed=1, turns_per_game=0):
                    pass
                result["ok"] = True
            except Exception as e:  # noqa: BLE001 -- the test asserts on the type
                result["error"] = e

        worker = threading.Thread(target=drain, daemon=True)
        worker.start()
        worker.join(timeout=60)

        assert not worker.is_alive(), "load_batch hung on an unreadable file (deadlock regressed)"
        assert "ok" not in result, "iteration unexpectedly succeeded over a deleted file"
        assert isinstance(result.get("error"), OSError)


def test_slice_row_batch_matches_dataset():
    """slice_row_batch reproduces the named tensors with correct shapes/values
    (guards the row-slicing SlogDataset applies to every loaded batch)."""
    if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
        pytest.skip("libscribblez_ffi.so not built -- run py/build.py first")
    rf = row_size_floats()
    rng = np.random.default_rng(0)
    batch = rng.standard_normal((5, rf)).astype(np.float32)

    input_shapes, targets = row_layout()
    out = slice_row_batch(batch, input_shapes, targets)

    # Every input + target tensor is present with the advertised shape.
    expected = {s.name: (5, *s.dims) for s in get_input_shapes()}
    expected.update({s.name: (5, *s.dims) for s in get_target_shapes()})
    assert set(out) == set(expected)
    for name, shape in expected.items():
        assert tuple(out[name].shape) == shape

    # The flat concatenation of all regions reconstructs the original row.
    flat = np.concatenate(
        [out[s.name].numpy().reshape(5, -1) for s in get_input_shapes()]
        + [out[name].numpy().reshape(5, -1) for name, *_ in targets],
        axis=1,
    )
    assert np.array_equal(flat, batch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
