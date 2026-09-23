"""End-to-end tests for the streaming DataLoader through the Python FFI, over .slog
files written by the C++ test_slog_writer binary."""

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


def generate_test_slogs(tmpdir: Path, num_games: int = 12, games_per_file: int = 4) -> list[Path]:
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
        """Two epochs under different seeds hold the same multiset of rows in a
        different order."""
        slogs = generate_test_slogs(tmp_path)

        loader = NativeDataLoader(memory_budget=256 * 1024 * 1024, num_workers=2, num_prefetch=1)
        for p in slogs:
            num_games, fsize = read_file_header(p)
            loader.add_file(p, num_games, fsize)
        # read_file_header counts games, but the default (turns_per_game=0) epoch
        # expands each game into all its eligible turns, so an epoch's row count
        # is the loader's expanded position count.
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

        assert not np.array_equal(epoch1, epoch2)

        # Same rows: compare after sorting by raw bytes.
        e1_sorted = np.sort(epoch1.view(np.uint8).reshape(total, -1), axis=0)
        e2_sorted = np.sort(epoch2.view(np.uint8).reshape(total, -1), axis=0)
        np.testing.assert_array_equal(e1_sorted, e2_sorted)


class TestStreamingDataset:
    def test_iter_batches(self, tmp_path):
        generate_test_slogs(tmp_path)

        ds = SlogDataset(
            tmp_path, post_move=True, apply_symmetry=True, memory_budget=256 * 1024 * 1024
        )
        batches = list(ds.iter_batches(batch_size=4, seed=555))
        assert len(batches) > 0

        # Input dims come from the session rather than literals, so a layout
        # change cannot leave a stale magic number here.
        in_shapes = {s.name: tuple(s.dims) for s in get_input_shapes()}
        for b in batches:
            assert "input_spatial" in b
            assert "input_scalar" in b
            assert "wld" in b
            assert "score_diff" in b
            for head in _PLACEMENT_HEADS:
                assert head in b
            for mask in _PLACEMENT_MASKS:
                assert mask in b
            assert tuple(b["input_spatial"].shape[1:]) == in_shapes["input_spatial"]
            assert b["input_scalar"].shape[1] == in_shapes["input_scalar"][0]
            assert b["wld"].shape[1] == 3
            assert b["score_diff"].shape[1] == 1
            # Each placement head is one footprint class index; each side's
            # legality mask spans the footprint class space.
            for head in _PLACEMENT_HEADS:
                assert b[head].shape[1] == 1
            for mask in _PLACEMENT_MASKS:
                assert b[mask].shape[1] == _FOOTPRINT_CLASSES

        # Same seed, same batches.
        batches2 = list(ds.iter_batches(batch_size=4, seed=555))
        assert len(batches) == len(batches2)
        for b1, b2 in zip(batches, batches2, strict=True):
            for key in b1:
                np.testing.assert_array_equal(b1[key].numpy(), b2[key].numpy())

    def test_drop_last_yields_only_full_batches(self, tmp_path):
        """An epoch whose row count batch_size does not divide ends in a short
        batch; drop_last omits exactly that one."""
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
        """A registered .slog that vanishes before its body is read must raise,
        not hang.

        The failure mode: a failed body load that never signals leaves
        DataFile::buffer() waiting forever, and with it the decode worker and
        load_batch. On a live generational tag this is a real path: a window file
        can be evicted between SlogDataset reading the headers and the loader
        lazily loading the bodies. Iteration runs on a watchdog thread so a hang
        fails the test instead of blocking the suite.
        """
        slogs = generate_test_slogs(tmp_path)
        assert len(slogs) >= 2

        ds = SlogDataset(
            tmp_path, post_move=True, apply_symmetry=True, memory_budget=256 * 1024 * 1024
        )
        # Its header, and so its rows, were registered at construction.
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
    """slice_row_batch, which SlogDataset applies to every loaded batch, yields every
    named tensor at its advertised shape and loses no values."""
    if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
        pytest.skip("libscribblez_ffi.so not built -- run py/build.py first")
    rf = row_size_floats()
    rng = np.random.default_rng(0)
    batch = rng.standard_normal((5, rf)).astype(np.float32)

    input_shapes, targets = row_layout()
    out = slice_row_batch(batch, input_shapes, targets)

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
