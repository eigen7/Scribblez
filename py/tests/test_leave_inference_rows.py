"""The leave inference task's rows through the FFI (engine
training/leave_inference_task.h): the DataLoader and decode_rows over .slog
files written by the C++ test_slog_writer binary."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from scribblez import ffi
from scribblez.dataset import SlogDataset, row_layout

# This checkout's own binaries, not the primary checkout's -- see the note in
# test_move_set_eval_targets.py.
_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
_LAYOUT = ffi.format_layout()["constants"]["leave_inference"]


@pytest.fixture
def slogs(tmp_path) -> list[Path]:
    binary = _ENGINE_DIR / "test_slog_writer"
    if not binary.is_file():
        pytest.skip("test_slog_writer not built -- run py/build.py first")
    subprocess.run([str(binary), str(tmp_path), "8", "4"], check=True, capture_output=True)
    return sorted(tmp_path.glob("*.slog"))


def test_shapes_cover_the_row():
    inputs = ffi.task_input_shapes("leave_inference")
    targets = ffi.task_target_shapes("leave_inference")
    assert [s.name for s in inputs] == ["input_spatial", "input_scalar"]
    assert [s.name for s in targets] == [
        "opp_leave_tiles",
        "opp_leave_length",
        "opp_leave_valid",
        "opp_leave_cause",
        "unseen_pool",
        "bag_size",
    ]
    total = sum(int(np.prod(s.dims)) for s in inputs + targets)
    assert total == ffi.task_row_floats("leave_inference")


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError):
        ffi.task_row_floats("no_such_task")


def test_dataloader_rows_carry_consistent_targets(slogs):
    ds = SlogDataset(slogs[0].parent, task="leave_inference", post_move=False)
    batch = next(ds.iter_batches(batch_size=32, seed=1))
    tiles = batch["opp_leave_tiles"].numpy()
    length = batch["opp_leave_length"].numpy()[:, 0]
    valid = batch["opp_leave_valid"].numpy()[:, 0]
    cause = batch["opp_leave_cause"].numpy()[:, 0]
    assert batch["input_spatial"].shape[1:] == ffi.task_input_shapes("leave_inference")[0].dims
    np.testing.assert_array_equal((tiles >= 0).sum(axis=1), length)
    assert ((tiles >= -1) & (tiles < 27)).all()
    # Sorted, with the padding last.
    padded = np.where(tiles < 0, 99, tiles)
    assert (np.diff(padded, axis=1) >= 0).all()
    assert set(np.unique(cause)) <= set(_LAYOUT["cause"].values())
    np.testing.assert_array_equal(valid, (cause == _LAYOUT["cause"]["valid"]).astype(np.float32))
    assert valid.sum() > 0
    assert ((length >= 1) & (length <= _LAYOUT["max_leave_length"]))[valid == 1].all()


def test_decode_rows_matches_the_task_width(slogs):
    games = np.array([0, 1, 1], dtype=np.int64)
    turns = np.array([0, 2, 5], dtype=np.int64)
    rows = ffi.decode_rows(slogs[0], games, turns, post_move=False, task="leave_inference")
    assert rows.shape == (3, ffi.task_row_floats("leave_inference"))
    _, targets = row_layout(
        ffi.task_input_shapes("leave_inference"), ffi.task_target_shapes("leave_inference")
    )
    start = {name: begin for name, begin, _, _ in targets}
    assert rows[0, start["opp_leave_cause"]] == _LAYOUT["cause"]["no_opponent_move"]
    pool = rows[:, start["unseen_pool"] : start["unseen_pool"] + 27]
    assert (pool.sum(axis=1) >= rows[:, start["bag_size"]] + 7).all()
