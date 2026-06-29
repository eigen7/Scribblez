"""Tests for the streaming self-play training source and row-slicing helper.

Covers:
  1. slice_row_batch matches SlogDataset's slicing (guards the DRY extraction).
  2. StreamingTrainSource smoke: pull a few slots, check shapes/finiteness,
     release, and stop cleanly.
"""

from pathlib import Path

import numpy as np
import pytest

from scribblez.dataset import row_layout, slice_row_batch
from scribblez.ffi import get_input_shapes, get_target_shapes, row_size_floats

_FFI_LIB = Path("/workspace/repo/target/engine/libscribblez_ffi.so")


def _require_engine():
    if not _FFI_LIB.is_file():
        pytest.skip("libscribblez_ffi.so not built -- run py/build.py first")


def test_slice_row_batch_matches_dataset():
    """slice_row_batch reproduces the named tensors with correct shapes/values."""
    _require_engine()
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


def test_streaming_source_smoke():
    """Pull a few full slots, verify shape/finiteness, release, and stop."""
    _require_engine()
    from scribblez.ffi import StreamingTrainSource

    bs = 32
    src = StreamingTrainSource(
        batch_size=bs, num_slots=2, num_threads=2, seed=7, handicap_max=0, pin_memory=False
    )
    src.start()
    rf = row_size_floats()
    try:
        for _ in range(4):
            res = src.next_slot()
            assert res is not None
            idx, tensor = res
            arr = tensor.numpy().copy()
            src.release(idx)
            assert arr.shape == (bs, rf)
            assert np.isfinite(arr).all()
        st = src.stats()
        assert st["rows_committed"] >= bs  # at least one full slot produced
        assert st["games_played"] >= st["rows_committed"]
    finally:
        src.stop()


def test_streaming_source_anchor_planes_populated():
    """A streamed row carries non-zero per-letter anchor planes (encoder ran)."""
    _require_engine()
    from scribblez.ffi import StreamingTrainSource

    bs = 32
    src = StreamingTrainSource(batch_size=bs, num_slots=2, num_threads=2, seed=3, pin_memory=False)
    src.start()
    try:
        res = src.next_slot()
        assert res is not None
        idx, tensor = res
        arr = tensor.numpy().copy()
        src.release(idx)
        spatial = arr[:, : 85 * 225].reshape(bs, 85, 15, 15)
        assert (spatial[:, 33:85] != 0).any()  # horizontal+vertical anchor planes
    finally:
        src.stop()


def test_dashboard_throughput_grid(tmp_path):
    """The Streaming dashboard panel builds a figure when data exists, else a notice."""
    from scribblez.dashboard import db, plots

    empty = db.connect(tmp_path / "empty.db")
    assert type(plots.throughput_grid(empty)).__name__ == "Div"  # no-data notice

    conn = db.connect(tmp_path / "dash.db")
    for i in range(4):
        db.write_throughput(
            conn,
            {
                "t": float(i),
                "positions": 1000 * (i + 1),
                "games": 1010 * (i + 1),
                "positions_per_s": 1100.0,
                "producer_blocked_ns": 10**9 * i,
                "consumer_blocked_ns": 2 * 10**9 * i,
                "bottleneck": "cpu",
            },
        )
    assert type(plots.throughput_grid(conn)).__name__ == "Column"  # rendered figures


def test_db_throughput_roundtrip(tmp_path):
    """write_throughput then read_throughput returns the samples in order."""
    from scribblez.dashboard import db

    conn = db.connect(tmp_path / "dash.db")
    samples = [
        {
            "t": 1.0 + i,
            "positions": 100 * (i + 1),
            "games": 110 * (i + 1),
            "positions_per_s": 1000.0 + i,
            "producer_blocked_ns": 5 * i,
            "consumer_blocked_ns": 7 * i,
            "bottleneck": "cpu" if i % 2 else "gpu",
        }
        for i in range(3)
    ]
    for s in samples:
        db.write_throughput(conn, s)
    got = db.read_throughput(conn)
    assert len(got) == 3
    assert [g["positions"] for g in got] == [100, 200, 300]
    assert [g["bottleneck"] for g in got] == ["gpu", "cpu", "gpu"]
    assert got[2]["consumer_blocked_ns"] == 14
    assert "games_per_s" not in got[0]  # collapsed: positions/s == games/s


def test_db_train_step_roundtrip(tmp_path):
    """write_train_steps then read_train_steps returns per-minibatch arrays in order."""
    from scribblez.dashboard import db, plots

    conn = db.connect(tmp_path / "dash.db")
    assert plots.train_step_grid(conn) is None  # no data -> caller falls back

    rows = [
        {
            "step": i,
            "positions": 8 * (i + 1),
            "loss": 1.0 - 0.01 * i,
            "loss_wld": 0.9,
            "loss_score_diff": 0.5,
            "loss_opp_next_placement": 0.3,
            "wld_acc": 0.5 + 0.001 * i,
        }
        for i in range(10)
    ]
    db.write_train_steps(conn, rows)
    db.write_train_steps(conn, [])  # empty flush is a no-op
    ts = db.read_train_steps(conn)
    assert list(ts["step"]) == list(range(10))
    assert ts["loss"][0] == 1.0 and abs(ts["loss"][9] - 0.91) < 1e-9
    assert type(plots.train_step_grid(conn)).__name__ == "Column"


def test_db_loss_weights_drive_stacked_plot(tmp_path):
    """Recorded loss weights round-trip in order and switch the loss panel to the
    weighted-stack view; without them the plot falls back to lines."""
    from scribblez.dashboard import db, plots

    conn = db.connect(tmp_path / "dash.db")
    db.write_train_steps(conn, [{"step": 1, "positions": 8, "loss": 1.0, "loss_a": 0.6, "loss_b": 0.4}])

    assert db.read_loss_weights(conn) == {}  # none yet -> overlaid lines
    assert type(plots.train_step_grid(conn)).__name__ == "Column"

    db.write_loss_weights(conn, {"loss_a": 1.0, "loss_b": 0.5})
    assert list(db.read_loss_weights(conn).items()) == [("loss_a", 1.0), ("loss_b", 0.5)]
    assert type(plots.train_step_grid(conn)).__name__ == "Column"  # absolute stack
    assert type(plots.train_step_grid(conn, normalized=True)).__name__ == "Column"

    # Normalized bands are each point's share of the weighted column total.
    import numpy as np

    ts = db.read_train_steps(conn)
    bands = plots._loss_bands(ts, slice(None), db.read_loss_weights(conn), normalized=True)
    assert [lbl for lbl, _ in bands] == ["loss_a", "0.5 x loss_b"]  # weight-1 label omits factor
    total = sum(y for _, y in bands)  # 0.6*1 + 0.4*0.5 = 0.8 -> shares 0.75, 0.25
    assert np.allclose(total, 1.0)
    assert np.allclose(bands[0][1], 0.75) and np.allclose(bands[1][1], 0.25)


def test_train_step_writer_adaptive_resolution(tmp_path):
    """One row per minibatch until fine_positions, then one mean-aggregated row
    per window. Partial final bucket is flushed by close()."""
    from scribblez.dashboard import db
    from scribblez.train_common import TrainStepWriter

    conn = db.connect(tmp_path / "dash.db")
    # Minibatches of 4 positions; fine until 16, then aggregate every 8 positions.
    w = TrainStepWriter(conn, fine_positions=16, window=8, start_positions=0)
    pos = 0
    for i in range(8):
        pos += 4
        w.record(step=i + 1, positions=pos, metrics={"loss": float(i), "x_acc": 1.0})
        w.commit()  # completed buckets flush; an open coarse bucket stays open
    w.close()

    ts = db.read_train_steps(conn)
    # fine: one row per minibatch at 4,8,12,16; coarse: {20,24}->24, {28,32}->32.
    assert list(ts["positions"]) == [4, 8, 12, 16, 24, 32]
    assert list(ts["loss"]) == [0.0, 1.0, 2.0, 3.0, 4.5, 6.5]  # coarse rows are means
    assert list(ts["x_acc"]) == [1.0] * 6


class _FakeSource:
    """Stands in for StreamingTrainSource: yields a fixed random batch each call."""

    def __init__(self, batch_size, row_floats):
        import torch

        self.bs = batch_size
        self._tensor = torch.rand(batch_size, row_floats)
        self._n = 0

    def start(self):
        pass

    def next_slot(self):
        self._n += 1
        return 0, self._tensor

    def release(self, slot_index):
        pass

    def stats(self):
        return {
            "games_played": self._n * self.bs,
            "games_dropped": 0,
            "rows_committed": self._n * self.bs,
            "slots_published": self._n,
            "producer_blocked_ns": 0,
            "consumer_blocked_ns": self._n,
        }

    def stop(self):
        pass


def test_streaming_loop_one_step(tmp_path):
    """The training loop writes a checkpoint, an ONNX export, and a throughput row."""
    _require_engine()
    import torch

    from scribblez.dashboard import db
    from scribblez.ffi import get_input_shapes, row_size_floats
    from scribblez.post_move_value_model import PostMoveValueModel
    from scribblez.paths import TagPaths
    from scripts.train_post_move_model import build_arg_parser, run_streaming_training

    in_shapes = {s.name: s.dims for s in get_input_shapes()}
    sp, sc = in_shapes["input_spatial"][0], in_shapes["input_scalar"][0]

    args = build_arg_parser().parse_args(
        [
            "-t", "looptest", "--device", "cpu", "--batch-size", "8",
            "--checkpoint-every", "8", "--log-every", "8", "--max-positions", "16",
            "--num-blocks", "1", "--trunk-channels", "8",
            "--no-dashboard", "--no-probe", "--no-calibration",
        ]
    )
    paths = TagPaths("looptest", mount_root=tmp_path)
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    model = PostMoveValueModel(spatial_planes=sp, scalar_size=sc, num_blocks=1, trunk_channels=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    conn = db.connect(paths.dashboard_db)

    source = _FakeSource(args.batch_size, row_size_floats())
    final = run_streaming_training(
        model, optimizer, source, conn, paths, device, args,
        probe_enabled=False, test_ds=None, start_ckpt=0, start_positions=0, start_step=0,
        spatial_planes=sp, scalar_size=sc,
    )

    assert final == 16
    assert paths.rolling_checkpoint.exists()
    assert paths.onnx_path(1).exists()  # first checkpoint exported
    assert len(db.read_throughput(conn)) >= 1
    # One train_step row per minibatch (16 positions / batch 8 == 2 steps).
    ts = db.read_train_steps(conn)
    assert list(ts["step"]) == [1, 2]
    # Resume state is recoverable (including the minibatch step counter).
    ckpt = torch.load(paths.rolling_checkpoint, map_location="cpu", weights_only=False)
    assert ckpt["positions"] == 16 and ckpt["ckpt_idx"] == 2 and ckpt["step"] == 2
