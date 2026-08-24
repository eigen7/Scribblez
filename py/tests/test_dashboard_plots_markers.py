"""Tests for the Loss tab's figures: the per-epoch loss view with its
control-change markers, and the value-quality grid."""

import numpy as np
from bokeh.models import Label, LinearScale, LogScale, Plot, Span
from scribblez.dashboard import db, plots


def _seed_metrics(conn):
    for epoch, (pos, loss) in enumerate([(100, 2.0), (200, 1.5), (300, 1.2)], start=1):
        db.write_metrics(conn, epoch, {"positions": pos, "loss": loss})


def test_metrics_loss_grid_none_without_loss(tmp_path):
    conn = db.connect(tmp_path / "d.db")
    assert plots.metrics_loss_grid(conn) is None


def test_metrics_loss_grid_adds_control_markers(tmp_path):
    conn = db.connect(tmp_path / "d.db")
    _seed_metrics(conn)
    db.write_control_event(conn, 150, "lr", 2.5e-4)
    db.write_control_event(conn, 250, "lr", 1e-4)

    grid = plots.metrics_loss_grid(conn)
    assert grid is not None
    # The grid stacks every knob-variant row; each row's loss figure carries its
    # own markers, so select recursively within a row.
    for axis_row in _rows_by_name(grid, LOSS_ROW_NAMES).values():
        spans = list(axis_row.select({"type": Span}))
        labels = list(axis_row.select({"type": Label}))
        assert {int(s.location) for s in spans} == {150, 250}
        assert len(labels) == 2


def test_db_loss_weights_drive_stacked_plot(tmp_path):
    """Recorded loss weights round-trip in order and switch the loss panel to the
    weighted-stack view; without them the plot falls back to lines."""
    conn = db.connect(tmp_path / "d.db")
    db.write_metrics(conn, 1, {"positions": 8, "loss": 1.0, "loss_a": 0.6, "loss_b": 0.4})

    assert db.read_loss_weights(conn) == {}  # none yet -> overlaid lines
    assert type(plots.metrics_loss_grid(conn)).__name__ == "Column"

    db.write_loss_weights(conn, {"loss_a": 1.0, "loss_b": 0.5})
    assert list(db.read_loss_weights(conn).items()) == [("loss_a", 1.0), ("loss_b", 0.5)]
    assert type(plots.metrics_loss_grid(conn)).__name__ == "Column"  # stacked variants

    # Normalized bands are each point's share of the weighted column total.
    _, series = plots._metrics_series(conn)
    bands = plots._loss_bands(series, db.read_loss_weights(conn), normalized=True)
    assert [lbl for lbl, _ in bands] == ["loss_a", "0.5 x loss_b"]  # weight-1 label omits factor
    total = sum(y for _, y in bands)  # 0.6*1 + 0.4*0.5 = 0.8 -> shares 0.75, 0.25
    assert np.allclose(total, 1.0)
    assert np.allclose(bands[0][1], 0.75) and np.allclose(bands[1][1], 0.25)


LOSS_ROW_NAMES = [
    f"{x}|{n}"
    for n in (plots.NORM_ABSOLUTE, plots.NORM_PERCENT)
    for x in (plots.X_AXIS_LINEAR, plots.X_AXIS_LOG)
]


def _rows_by_name(grid, names):
    """The grid's named variant rows, {name: row}."""
    rows = {name: list(grid.select({"name": name})) for name in names}
    assert all(len(found) == 1 for found in rows.values())
    return {name: found[0] for name, found in rows.items()}


def _assert_x_axis_variants(grid, names, n_panels, lo, hi):
    """`grid` carries the variant rows `names`, `n_panels` figures each; the log
    rows' figures have an explicit positive x-range covering the data `lo..hi`
    (so an x=0 checkpoint does not pin it), the linear rows' are auto-ranged."""
    for name, row in _rows_by_name(grid, names).items():
        figs = list(row.select({"type": Plot}))
        assert len(figs) == n_panels
        if name.startswith(plots.X_AXIS_LOG):
            assert all(isinstance(f.x_scale, LogScale) for f in figs)
            assert all(0.0 < f.x_range.start < lo and f.x_range.end > hi for f in figs)
        else:
            assert all(isinstance(f.x_scale, LinearScale) for f in figs)


def test_metrics_loss_grid_knob_variants(tmp_path):
    """All four knob variants ride along in one grid, on both the overlaid-lines
    and the weighted-stack loss panels, so the knobs flip without a refetch. The
    percent variants rescale each stack column to sum to 1."""
    conn = db.connect(tmp_path / "d.db")
    for epoch, pos in enumerate([0, 100, 1000, 10000]):
        db.write_metrics(
            conn, epoch, {"positions": pos, "loss": 1.0, "loss_a": 0.6, "top1_acc": 0.5}
        )
    _assert_x_axis_variants(plots.metrics_loss_grid(conn), LOSS_ROW_NAMES, 2, 100, 10000)  # lines
    db.write_loss_weights(conn, {"loss_a": 1.0})
    grid = plots.metrics_loss_grid(conn)
    _assert_x_axis_variants(grid, LOSS_ROW_NAMES, 2, 100, 10000)  # stack
    rows = _rows_by_name(grid, LOSS_ROW_NAMES)
    for name, row in rows.items():
        titles = {f.title.text for f in row.select({"type": Plot})}
        expected = (
            "Train loss (stacked, % of total)"
            if name.endswith(plots.NORM_PERCENT)
            else "Train loss (stacked, weighted)"
        )
        assert expected in titles, name


def test_eval_quality_grid_x_axis_variants(tmp_path):
    """The value-quality grid carries both epoch-axis variants."""
    conn = db.connect(tmp_path / "d.db")
    for epoch in [0, 1, 10, 100]:
        db.write_metrics(conn, epoch, {"eval_win_mae": 0.1, "eval_sd_mean_mae": 5.0})
    _assert_x_axis_variants(
        plots.eval_quality_grid(conn, "t"), [plots.X_AXIS_LINEAR, plots.X_AXIS_LOG], 2, 1, 100
    )
