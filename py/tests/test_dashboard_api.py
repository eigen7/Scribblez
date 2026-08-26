"""Tests for the React dashboard's Tornado data API core (no server spun up)."""

import json

from scribblez.dashboard import api, db


def _seed(path):
    conn = db.connect(path)
    db.write_loss_weights(conn, {"loss_score_pdf": 1.0, "loss_score_cdf": 2.0})
    for epoch in range(1, 6):
        db.write_metrics(
            conn,
            epoch,
            {
                "positions": 100 * epoch,
                "loss": 1.0 / epoch,
                "loss_score_pdf": 0.4 / epoch,
                "loss_score_cdf": 0.6 / epoch,
                "score_acc": 0.5 + 0.01 * epoch,
            },
        )
    return conn


def test_build_figure_item_loss_is_serializable(tmp_path):
    conn = _seed(tmp_path / "dashboard.db")
    item, structure = api.build_figure_item(conn, "loss", {}, str(tmp_path))
    assert item is not None
    assert isinstance(structure, str) and structure
    assert {"doc", "root_id", "target_id"} <= set(item)
    json.dumps(item)  # must round-trip to the client as JSON


def test_build_figure_item_empty_db_is_none(tmp_path):
    conn = db.connect(tmp_path / "empty.db")
    assert api.build_figure_item(conn, "loss", {}, str(tmp_path)) == (None, None)


def test_build_figure_item_unknown_name_is_none(tmp_path):
    conn = _seed(tmp_path / "dashboard.db")
    assert api.build_figure_item(conn, "no_such_figure", {}, str(tmp_path)) == (None, None)


def test_version_token_counts_rows(tmp_path):
    conn = _seed(tmp_path / "dashboard.db")
    token = api.version_token(conn)
    # 5 epochs x 5 metrics (positions, loss, loss_score_pdf, loss_score_cdf,
    # score_acc) = 25 rows.
    assert token["metrics"] == 25
