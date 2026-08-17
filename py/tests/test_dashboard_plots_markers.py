"""Tests for the per-epoch loss view and its control-change markers."""

import numpy as np
from bokeh.models import Label, Span
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
    # The grid is a column(row(loss_fig, ...)); the markers live on the nested loss
    # figure, so select recursively across the layout.
    spans = list(grid.select({"type": Span}))
    labels = list(grid.select({"type": Label}))
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
    assert type(plots.metrics_loss_grid(conn)).__name__ == "Column"  # absolute stack
    assert type(plots.metrics_loss_grid(conn, normalized=True)).__name__ == "Column"

    # Normalized bands are each point's share of the weighted column total.
    _, series = plots._metrics_series(conn)
    bands = plots._loss_bands(series, db.read_loss_weights(conn), normalized=True)
    assert [lbl for lbl, _ in bands] == ["loss_a", "0.5 x loss_b"]  # weight-1 label omits factor
    total = sum(y for _, y in bands)  # 0.6*1 + 0.4*0.5 = 0.8 -> shares 0.75, 0.25
    assert np.allclose(total, 1.0)
    assert np.allclose(bands[0][1], 0.75) and np.allclose(bands[1][1], 0.25)
