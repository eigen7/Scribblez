"""Test the generational trainer's live base-LR controller."""

from scribblez.dashboard import db
from scribblez.generational.controls import CONTROL_BASE_LR, LrController


def test_lr_controller_serves_base_and_logs_changes(tmp_path):
    conn = db.connect(tmp_path / "dashboard.db")
    db.init_control(conn, {CONTROL_BASE_LR: 1e-3})
    ctrl = LrController(conn, base_lr=1e-3, warmup_rows=0)

    # First epoch: the base is unchanged, so no control event is recorded.
    lr_fn = ctrl.epoch_lr_fn(0)
    assert lr_fn(0) == 1e-3
    assert db.read_control_events(conn, CONTROL_BASE_LR) == []

    # Operator steps the base down; the next epoch adopts it and logs the change
    # at the current rows-clock.
    db.write_control(conn, CONTROL_BASE_LR, 2e-4)
    lr_fn = ctrl.epoch_lr_fn(500)
    assert lr_fn(1000) == 2e-4  # warmup disabled, so the base applies verbatim
    events = db.read_control_events(conn, CONTROL_BASE_LR)
    assert len(events) == 1
    assert events[0]["positions"] == 500
    assert events[0]["value"] == 2e-4


def test_lr_controller_warmup_scales_base(tmp_path):
    conn = db.connect(tmp_path / "dashboard.db")
    db.init_control(conn, {CONTROL_BASE_LR: 1e-3})
    ctrl = LrController(conn, base_lr=1e-3, warmup_rows=1000)
    lr_fn = ctrl.epoch_lr_fn(0)
    assert lr_fn(500) == 5e-4  # half-way through warmup
    assert lr_fn(2000) == 1e-3  # past warmup
