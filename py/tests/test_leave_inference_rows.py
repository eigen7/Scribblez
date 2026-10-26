"""The leave inference task's rows through the FFI (engine
training/leave_inference_task.h): the DataLoader and decode_rows over .slog
files written by the C++ test_slog_writer binary."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from scribblez import ffi
from scribblez import params as params_mod
from scribblez.dataset import SlogDataset, row_layout
from scribblez.workloads.leave_inference import LeaveInferenceParams

# This checkout's own binaries, not the primary checkout's -- see the note in
# test_move_set_eval_targets.py.
_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
_LAYOUT = ffi.format_layout()["constants"]["leave_inference"]


def _write_slogs(out_dir: Path) -> list[Path]:
    binary = _ENGINE_DIR / "test_slog_writer"
    if not binary.is_file():
        pytest.skip("test_slog_writer not built -- run py/build.py first")
    subprocess.run([str(binary), str(out_dir), "8", "4"], check=True, capture_output=True)
    return sorted(out_dir.glob("*.slog"))


@pytest.fixture
def slogs(tmp_path) -> list[Path]:
    """test_slog_writer games with their .forgone sidecars, which leave
    inference rows require."""
    paths = _write_slogs(tmp_path)
    tool = _ENGINE_DIR / "forgone_play_tool"
    subprocess.run(
        [str(tool), "--slog-dir", str(tmp_path), "--threads", "2"], check=True, capture_output=True
    )
    return paths


def test_shapes_cover_the_row():
    inputs = ffi.task_input_shapes("leave_inference")
    targets = ffi.task_target_shapes("leave_inference")
    assert [s.name for s in inputs] == ["input_spatial", "input_scalar"]
    assert [s.name for s in targets] == [
        "opp_leave_tiles",
        "opp_leave_length",
        "opp_leave_valid",
        "opp_leave_cause",
        "leave_pool",
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
    pool = rows[:, start["leave_pool"] : start["leave_pool"] + 27]
    assert (pool.sum(axis=1) >= rows[:, start["bag_size"]] + 7).all()


def test_a_slog_without_its_sidecar_is_refused(tmp_path):
    _write_slogs(tmp_path)
    with pytest.raises(OSError, match="forgone"):
        SlogDataset(tmp_path, task="leave_inference", post_move=False)


def test_forgone_block_is_populated(slogs):
    rows = ffi.decode_rows(
        slogs[0], np.array([2]), np.array([6]), post_move=False, task="leave_inference"
    )
    spatial, scalar = ffi.task_input_shapes("leave_inference")
    begin = int(np.prod(spatial.dims))
    forgone = rows[0, begin : begin + scalar.dims[0]][-3 * 2 * 27 :]  # the last input block
    assert forgone[0::2].sum() > 0  # some margins apply


def test_public_rows_withhold_only_the_observers_tiles(slogs):
    """A public row is the observer's row with the rack and exchange blocks
    zeroed and the unseen pool widened by the rack; the planes and the target,
    the observer's leave pool included, are the same."""
    games, turns = np.array([1, 2, 3]), np.array([3, 6, 9])
    seen = ffi.decode_rows(slogs[0], games, turns, post_move=False, task="leave_inference")
    public = ffi.decode_rows(slogs[0], games, turns, post_move=False, task="public_leave_inference")
    spatial, scalar = ffi.task_input_shapes("leave_inference")
    begin = int(np.prod(spatial.dims))
    end = begin + scalar.dims[0]
    np.testing.assert_array_equal(public[:, :begin], seen[:, :begin])
    np.testing.assert_array_equal(public[:, end:], seen[:, end:])
    rack = slice(begin, begin + 27)
    assert seen[:, rack].sum() > 0
    assert (public[:, rack] == 0).all()
    unseen_thermometer = slice(begin + 27, begin + 127)
    assert public[:, unseen_thermometer].sum() > seen[:, unseen_thermometer].sum()


# The current scalars (engine encoding/leave_inference_input_encoder.h): the
# observer's rack (27), the unseen pool (100), the score (1), the bag (1) and
# the leave's length (6), then the observer's two exchanges (2 x 27), then
# whether their private history is hidden.
_EXCHANGES = slice(27 + 100 + 1 + 1 + 6, 27 + 100 + 1 + 1 + 6 + 54)
_HIDDEN = 27 + 100 + 1 + 1 + 6 + 54


def _scalars(batch):
    return batch["input_scalar"].numpy()


def test_decode_rows_hides_the_private_history_on_request(slogs):
    games, turns = np.array([0, 1, 2], dtype=np.int64), np.array([5, 6, 7], dtype=np.int64)
    split = ffi.task_input_floats("leave_inference")
    spatial = int(np.prod(ffi.task_input_shapes("leave_inference")[0].dims))
    own = ffi.decode_rows(slogs[0], games, turns, post_move=False, task="leave_inference")
    hidden = ffi.decode_rows(
        slogs[0], games, turns, post_move=False, task="leave_inference", hide_private=True
    )
    assert (own[:, spatial + _HIDDEN] == 0).all() and (hidden[:, spatial + _HIDDEN] == 1).all()
    assert (hidden[:, spatial + _EXCHANGES.start : spatial + _EXCHANGES.stop] == 0).all()
    # The rack, unseen pool and the newest slot are the same; targets differ
    # only in the pool, by an exchange the observer just made.
    np.testing.assert_array_equal(
        own[:, : spatial + _EXCHANGES.start], hidden[:, : spatial + _EXCHANGES.start]
    )
    assert (hidden[:, split:] >= own[:, split:]).all()


def test_a_fraction_hides_that_share_of_rows_without_moving_the_transposes(slogs):
    def batch(fraction):
        ds = SlogDataset(
            slogs[0].parent, task="leave_inference", post_move=False, hide_private_fraction=fraction
        )
        return next(ds.iter_batches(batch_size=256, seed=3))

    none, some, again, every = batch(0.0), batch(0.3), batch(0.3), batch(1.0)
    hidden = _scalars(some)[:, _HIDDEN] == 1
    assert 0.15 < hidden.mean() < 0.45
    assert (_scalars(none)[:, _HIDDEN] == 0).all() and (_scalars(every)[:, _HIDDEN] == 1).all()
    np.testing.assert_array_equal(_scalars(again)[:, _HIDDEN] == 1, hidden)  # seeded
    # The rows the fraction leaves alone are the unhidden epoch's, transposed alike.
    np.testing.assert_array_equal(
        some["input_spatial"].numpy()[~hidden], none["input_spatial"].numpy()[~hidden]
    )


def test_the_fraction_must_be_a_probability():
    with pytest.raises(params_mod.ParamsError, match="hide_private_fraction"):
        params_mod.validate(LeaveInferenceParams, {"hide_private_fraction": 1.5})
