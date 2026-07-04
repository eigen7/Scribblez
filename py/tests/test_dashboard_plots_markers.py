"""Tests for the per-epoch loss view and its control-change markers."""

from bokeh.models import Label, Span
from scribblez.dashboard import db, plots


def _seed_metrics(conn):
    for epoch, (pos, loss) in enumerate([(100, 2.0), (200, 1.5), (300, 1.2)], start=1):
        db.write_metrics(conn, epoch, {"positions": pos, "loss": loss})


def test_metrics_loss_grid_none_without_loss(tmp_path):
    conn = db.connect(tmp_path / "d.db")
    assert plots.metrics_loss_grid(conn) is None


def test_metric_vs_positions_maps_values_by_epoch(tmp_path):
    conn = db.connect(tmp_path / "d.db")
    _seed_metrics(conn)
    xs, ys = plots._metric_vs_positions(conn, "loss")
    assert [int(x) for x in xs] == [100, 200, 300]
    assert [float(y) for y in ys] == [2.0, 1.5, 1.2]


def test_metrics_loss_grid_adds_control_markers(tmp_path):
    conn = db.connect(tmp_path / "d.db")
    _seed_metrics(conn)
    db.write_control_event(conn, 150, "base_lr", 2.5e-4)
    db.write_control_event(conn, 250, "base_lr", 1e-4)

    grid = plots.metrics_loss_grid(conn)
    assert grid is not None
    # The grid is a column(row(loss_fig, ...)); the markers live on the nested loss
    # figure, so select recursively across the layout.
    spans = list(grid.select({"type": Span}))
    labels = list(grid.select({"type": Label}))
    assert {int(s.location) for s in spans} == {150, 250}
    assert len(labels) == 2
