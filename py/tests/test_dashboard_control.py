"""Unit tests for the dashboard control table (live operator knobs)."""

from scribblez.dashboard import db


def _conn(tmp_path):
    return db.connect(tmp_path / "dashboard.db")


def test_read_control_default_when_unset(tmp_path):
    conn = _conn(tmp_path)
    assert db.read_control(conn, "base_lr") is None
    assert db.read_control(conn, "base_lr", default=1e-3) == 1e-3


def test_init_control_seeds_then_preserves(tmp_path):
    conn = _conn(tmp_path)
    db.init_control(conn, {"base_lr": 1e-3})
    assert db.read_control(conn, "base_lr") == 1e-3
    # A later operator write, then a re-seed (as on restart): the value is kept.
    db.write_control(conn, "base_lr", 2e-4)
    db.init_control(conn, {"base_lr": 1e-3})
    assert db.read_control(conn, "base_lr") == 2e-4


def test_write_control_upserts(tmp_path):
    conn = _conn(tmp_path)
    db.write_control(conn, "base_lr", 5e-4)
    assert db.read_control(conn, "base_lr") == 5e-4
    db.write_control(conn, "base_lr", 1e-4)
    assert db.read_control(conn, "base_lr") == 1e-4
    assert db.read_controls(conn) == {"base_lr": 1e-4}


def test_control_events_ordered_by_positions(tmp_path):
    conn = _conn(tmp_path)
    db.write_control_event(conn, 200, "base_lr", 1e-4)
    db.write_control_event(conn, 100, "base_lr", 5e-4)
    db.write_control_event(conn, 150, "other", 9.0)
    base_events = db.read_control_events(conn, "base_lr")
    assert [(e["positions"], e["value"]) for e in base_events] == [(100, 5e-4), (200, 1e-4)]
    all_events = db.read_control_events(conn)
    assert [e["positions"] for e in all_events] == [100, 150, 200]
