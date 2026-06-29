"""Tests for the React dashboard's Tornado data API core (no server spun up)."""

import json

from scribblez.dashboard import api, db


def _seed(path):
    conn = db.connect(path)
    db.write_loss_weights(conn, {"loss_score_pdf": 1.0, "loss_score_cdf": 2.0})
    db.write_train_steps(
        conn,
        [
            {"step": s, "positions": 100 * s, "n": 1,
             "loss": 1.0 / s, "loss_score_pdf": 0.4 / s, "loss_score_cdf": 0.6 / s,
             "score_acc": 0.5 + 0.01 * s}
            for s in range(1, 6)
        ],
    )
    return conn


def test_build_figure_item_train_step_is_serializable(tmp_path):
    conn = _seed(tmp_path / "dashboard.db")
    item = api.build_figure_item(conn, "train_step", {})
    assert item is not None
    assert {"doc", "root_id", "target_id"} <= set(item)
    json.dumps(item)  # must round-trip to the client as JSON


def test_build_figure_item_empty_db_is_none(tmp_path):
    conn = db.connect(tmp_path / "empty.db")
    assert api.build_figure_item(conn, "train_step", {}) is None


def test_build_figure_item_unknown_name_is_none(tmp_path):
    conn = _seed(tmp_path / "dashboard.db")
    assert api.build_figure_item(conn, "no_such_figure", {}) is None


def test_version_token_counts_rows(tmp_path):
    conn = _seed(tmp_path / "dashboard.db")
    token = api.version_token(conn)
    # 5 points x 4 series (loss, loss_score_pdf, loss_score_cdf, score_acc) = 20 rows.
    assert token["train_step"] == 20
    assert token["metrics"] == 0
