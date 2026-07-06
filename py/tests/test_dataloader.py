"""End-to-end tests for the streaming DataLoader via the Python FFI.

These tests create .slog files using the C++ test_slog_writer binary,
then exercise:
  1. Epoch determinism: same seed → identical output
  2. Coverage: all positions appear exactly once per epoch
  3. Memory-budget stress: tiny budget, verify all data is still yielded
  4. Streaming dataset iteration via SlogDataset.iter_batches()
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from scribblez.dataset import SlogDataset, row_layout, slice_row_batch
from scribblez.ffi import (
    NativeDataLoader,
    get_input_shapes,
    get_target_shapes,
    read_file_header,
    row_size_floats,
)

# ---------------------------------------------------------------------------
# Fixture: generate .slog files using the test_slog_writer binary.
# ---------------------------------------------------------------------------


def generate_test_slogs(tmpdir: Path, num_games: int = 12, games_per_file: int = 4) -> list[Path]:
    """Generate .slog files for testing using the test_slog_writer binary."""
    binary = Path("/workspace/repo/target/engine/test_slog_writer")
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


class TestMemoryBudgetStress:
    def test_tiny_budget(self, tmp_path):
        slogs = generate_test_slogs(tmp_path, num_games=20, games_per_file=4)

        # Find largest file size.
        max_fsize = 0
        file_info = []
        for p in slogs:
            num_games, fsize = read_file_header(p)
            file_info.append((num_games, fsize))
            max_fsize = max(max_fsize, fsize)

        # Budget: just one file. This forces eviction on every file switch.
        loader = NativeDataLoader(memory_budget=max_fsize + 1, num_workers=1, num_prefetch=1)
        for i, p in enumerate(slogs):
            loader.add_file(p, file_info[i][0], file_info[i][1])
        # file_info holds per-file game counts; the epoch yields every eligible
        # turn, so the expected row count is the loader's expanded position count.
        total = loader.num_positions

        # Run epoch with small batches.
        loader.epoch_start(batch_size=2, post_move=True, apply_symmetry=True, seed=42)
        rows_decoded = 0
        while True:
            batch = loader.load_batch()
            if batch is None:
                break
            rows_decoded += batch.shape[0]

        assert rows_decoded == total

        # Determinism still holds under memory pressure.
        loader.epoch_start(batch_size=2, post_move=True, apply_symmetry=True, seed=42)
        run1 = []
        while True:
            batch = loader.load_batch()
            if batch is None:
                break
            run1.append(batch.copy())
        run1_data = np.concatenate(run1, axis=0)

        loader.epoch_start(batch_size=2, post_move=True, apply_symmetry=True, seed=42)
        run2 = []
        while True:
            batch = loader.load_batch()
            if batch is None:
                break
            run2.append(batch.copy())
        run2_data = np.concatenate(run2, axis=0)

        np.testing.assert_array_equal(run1_data, run2_data)


class TestStreamingDataset:
    def test_iter_batches(self, tmp_path):
        """Test the SlogDataset.iter_batches() method."""
        generate_test_slogs(tmp_path)

        ds = SlogDataset(
            tmp_path, post_move=True, apply_symmetry=True, memory_budget=256 * 1024 * 1024
        )
        batches = list(ds.iter_batches(batch_size=4, seed=555))
        assert len(batches) > 0

        # Each batch should have the expected tensor keys.
        for b in batches:
            assert "input_spatial" in b
            assert "input_scalar" in b
            assert "wld" in b
            assert "score_diff" in b
            assert "opp_next_placement" in b
            assert "self_next_placement" in b
            assert "opp_win_placement" in b
            assert "self_win_placement" in b
            assert b["input_spatial"].shape[1:] == (88, 15, 15)
            assert b["input_scalar"].shape[1] == 992
            assert b["wld"].shape[1] == 3
            assert b["score_diff"].shape[1] == 1
            assert b["opp_next_placement"].shape[1:] == (15, 15)
            assert b["self_next_placement"].shape[1:] == (15, 15)
            assert b["opp_win_placement"].shape[1:] == (15, 15)
            assert b["self_win_placement"].shape[1:] == (15, 15)

        # Determinism: same seed, same output.
        batches2 = list(ds.iter_batches(batch_size=4, seed=555))
        assert len(batches) == len(batches2)
        for b1, b2 in zip(batches, batches2, strict=True):
            for key in b1:
                np.testing.assert_array_equal(b1[key].numpy(), b2[key].numpy())


def test_slice_row_batch_matches_dataset():
    """slice_row_batch reproduces the named tensors with correct shapes/values
    (guards the row-slicing SlogDataset applies to every loaded batch)."""
    if not Path("/workspace/repo/target/engine/libscribblez_ffi.so").is_file():
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
