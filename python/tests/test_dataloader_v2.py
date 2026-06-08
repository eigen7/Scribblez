"""End-to-end tests for the v2 streaming DataLoader via the Python FFI.

These tests create .slog files using the C++ test binary's infrastructure
(via a small helper that shells out to write test games), then exercise:
  1. Epoch determinism: same seed → identical output
  2. Coverage: all positions appear exactly once per epoch
  3. Memory-budget stress: tiny budget, verify all data is still yielded
  4. Streaming dataset iteration via SlogDataset.iter_batches_streaming()
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scribblez.ffi import NativeDataLoader, read_file_header, row_size_floats


# ---------------------------------------------------------------------------
# Fixture: generate .slog files by running a minimal game-writing binary.
# We use the scribblez_tests binary which writes .slog as a side-effect.
# Instead, we'll call play_game --type=hasty for a few games.
# If play_game is not available (no macondo), we'll use a synthetic approach.
# ---------------------------------------------------------------------------

def _find_play_game() -> Path | None:
    candidates = [
        Path("/workspace/repo/build/engine/play_game"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _write_test_slogs(out_dir: Path, num_games: int = 12, games_per_file: int = 4) -> list[Path]:
    """Write test .slog files. Uses a small embedded C++ helper via the existing
    test infrastructure. Since play_game requires macondo, we instead directly
    use the scribblez_tests approach: build minimal games using our dataloader_smoke
    or write a small C helper.

    For now, we'll use a subprocess approach with a small inline C++ program
    that links scribblez_core. Alternatively, we check if the test binary left
    .slogs around, or we create synthetic ones.

    Simplest: use the dataloader_smoke binary if it supports writing test data.
    """
    # Actually, the cleanest approach: write a tiny Python script that calls
    # the FFI to validate, and use the test binary to generate files.
    # But the test binary cleans up after itself.
    #
    # Let's write a small standalone binary that generates .slog files.
    # Since this is a test, we'll create it as a compile step.
    #
    # Simplest for now: call the existing dataloader_smoke with a --generate flag
    # or compile a helper. But we don't want to complicate the build.
    #
    # ALTERNATIVE: We have the play_game binary. If we can run hasty-vs-hasty
    # games without macondo (using a tiny in-memory dictionary), that would work.
    # But play_game requires --lexicon pointing to a .kwg file.
    #
    # PRAGMATIC SOLUTION: Create a standalone slog-writer binary in the build.
    # For this test file, we'll compile and run it inline.
    pass


def _create_slog_writer_binary() -> Path:
    """Compile a tiny slog-writer binary if not already present."""
    binary = Path("/workspace/repo/build/engine/test_slog_writer")
    if binary.is_file():
        return binary
    # Write the source and compile it.
    src = Path("/workspace/repo/engine/apps/test_slog_writer.cpp")
    if not src.is_file():
        # This binary will be created separately -- skip if not available.
        pytest.skip("test_slog_writer binary not available")
    # Try to compile.
    result = subprocess.run(
        ["make", "-C", "/workspace/repo/build", "test_slog_writer"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.skip(f"Cannot compile test_slog_writer: {result.stderr}")
    return binary


def generate_test_slogs(tmpdir: Path, num_games: int = 12, games_per_file: int = 4) -> list[Path]:
    """Generate .slog files for testing using the test_slog_writer binary."""
    binary = Path("/workspace/repo/build/engine/test_slog_writer")
    if not binary.is_file():
        pytest.skip("test_slog_writer not built -- run 'make test_slog_writer' first")

    result = subprocess.run(
        [str(binary), str(tmpdir), str(num_games), str(games_per_file)],
        capture_output=True, text=True
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
        row_floats = row_size_floats()

        def run_epoch(seed: int) -> np.ndarray:
            loader = NativeDataLoader(memory_budget=256 * 1024 * 1024, num_workers=2, num_prefetch=1)
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
        slogs = generate_test_slogs(tmp_path)
        row_floats = row_size_floats()

        loader = NativeDataLoader(memory_budget=256 * 1024 * 1024, num_workers=2, num_prefetch=1)
        total = 0
        for p in slogs:
            num_pos, fsize = read_file_header(p)
            loader.add_file(p, num_pos, fsize)
            total += num_pos

        # Reference: load all chronologically, no symmetry.
        ref = loader.load(0, total, post_move=True, apply_symmetry=False)

        # Epoch: no symmetry, so row content is identical, just reordered.
        loader.epoch_start(batch_size=3, post_move=True, apply_symmetry=False, seed=7777)
        epoch_rows = []
        while True:
            batch = loader.load_batch()
            if batch is None:
                break
            epoch_rows.append(batch.copy())
        epoch_data = np.concatenate(epoch_rows, axis=0)

        assert epoch_data.shape[0] == total

        # Check every reference row appears in epoch data.
        # Sort both by their bytes for comparison.
        ref_sorted = np.sort(ref.view(np.uint8).reshape(total, -1), axis=0)
        epoch_sorted = np.sort(epoch_data.view(np.uint8).reshape(total, -1), axis=0)
        np.testing.assert_array_equal(ref_sorted, epoch_sorted)


class TestMemoryBudgetStress:
    def test_tiny_budget(self, tmp_path):
        slogs = generate_test_slogs(tmp_path, num_games=20, games_per_file=4)
        row_floats = row_size_floats()

        # Find largest file size.
        max_fsize = 0
        file_info = []
        for p in slogs:
            num_pos, fsize = read_file_header(p)
            file_info.append((num_pos, fsize))
            max_fsize = max(max_fsize, fsize)

        # Budget: just one file. This forces eviction on every file switch.
        loader = NativeDataLoader(
            memory_budget=max_fsize + 1, num_workers=1, num_prefetch=1
        )
        total = 0
        for i, p in enumerate(slogs):
            loader.add_file(p, file_info[i][0], file_info[i][1])
            total += file_info[i][0]

        # Run epoch with small batches.
        loader.epoch_start(batch_size=2, post_move=True, apply_symmetry=True, seed=42)
        rows_decoded = 0
        while True:
            batch = loader.load_batch()
            if batch is None:
                break
            rows_decoded += batch.shape[0]
            # Memory should be bounded.
            assert loader.resident_bytes <= 2 * max_fsize + 200

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
    def test_iter_batches_streaming(self, tmp_path):
        """Test the SlogDataset.iter_batches_streaming() method."""
        from scribblez.dataset import SlogDataset

        slogs = generate_test_slogs(tmp_path)

        ds = SlogDataset(
            tmp_path, post_move=True, apply_symmetry=True, memory_budget=256 * 1024 * 1024
        )
        # Don't call ds.load() -- streaming doesn't need it.
        batches = list(ds.iter_batches_streaming(batch_size=4, seed=555))
        assert len(batches) > 0

        # Each batch should have the expected tensor keys.
        for b in batches:
            assert "input_spatial" in b
            assert "input_scalar" in b
            assert "wld" in b
            assert "score_diff" in b
            assert "opp_next_placement" in b
            assert b["input_spatial"].shape[1:] == (32, 15, 15)
            assert b["input_scalar"].shape[1] == 58
            assert b["wld"].shape[1] == 3
            assert b["score_diff"].shape[1] == 801
            assert b["opp_next_placement"].shape[1:] == (15, 15)

        # Determinism: same seed, same output.
        batches2 = list(ds.iter_batches_streaming(batch_size=4, seed=555))
        assert len(batches) == len(batches2)
        for b1, b2 in zip(batches, batches2):
            for key in b1:
                np.testing.assert_array_equal(b1[key].numpy(), b2[key].numpy())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
