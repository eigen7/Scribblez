"""Test the generational trainers' warmup-stable-decay LR schedule and the
controller that serves it as a per-batch lr_fn and logs its phase boundaries."""

import math

import pytest
from scribblez.dashboard import db
from scribblez.generational.controls import (
    LR_DECAY_FRAC,
    LR_EVENT,
    LR_FLOOR_FRAC,
    PHASE_DECAY,
    PHASE_REWARMUP,
    PHASE_STABLE,
    PHASE_WARMUP,
    WsdLrController,
    WsdSchedule,
)

# W=1000, C=10000, R=250, decay starts at t=8000, floor 1e-4.
LR = 1e-3
W, C = 1000, 10_000
R = W // 4
DECAY_START = int((1 - LR_DECAY_FRAC) * C)
FLOOR = LR * LR_FLOOR_FRAC


@pytest.fixture
def sched():
    return WsdSchedule(LR, W, C)


def test_warmup_ramps_linearly_from_zero(sched):
    assert sched.phase(0) == PHASE_WARMUP
    assert sched.value(0) == 0.0
    assert sched.value(W // 2) == pytest.approx(LR / 2)
    assert sched.value(W - 1) < LR


def test_first_cycle_has_no_rewarmup(sched):
    # Straight from warmup into the stable plateau: nothing to recover from.
    for rows in (W, W + 1, W + R // 2, W + DECAY_START - 1):
        assert sched.phase(rows) == PHASE_STABLE
        assert sched.value(rows) == LR


def test_decay_is_cosine_from_peak_to_floor(sched):
    start, end = W + DECAY_START, W + C
    assert sched.phase(start) == PHASE_DECAY
    assert sched.value(start) == pytest.approx(LR)
    mid = (start + end) // 2
    assert sched.value(mid) == pytest.approx((LR + FLOOR) / 2)
    assert sched.value(end - 1) == pytest.approx(FLOOR, rel=1e-3)
    # Cosine, not linear: the quarter point sits above the linear interpolant.
    q = start + (end - start) // 4
    linear_q = LR - (LR - FLOOR) * 0.25
    assert sched.value(q) > linear_q
    assert sched.value(q) == pytest.approx(FLOOR + (LR - FLOOR) * 0.5 * (1 + math.cos(math.pi / 4)))


def test_restart_rewarms_from_floor_inside_the_cycle(sched):
    restart = W + C
    assert sched.phase(restart) == PHASE_REWARMUP
    assert sched.value(restart) == pytest.approx(FLOOR)
    assert sched.value(restart + R // 2) == pytest.approx((LR + FLOOR) / 2)
    assert sched.phase(restart + R) == PHASE_STABLE
    assert sched.value(restart + R) == LR
    # Period is exactly C: the second cycle's decay starts C rows after the first.
    assert sched.phase(restart + DECAY_START) == PHASE_DECAY
    assert sched.value(restart + DECAY_START) == pytest.approx(LR)
    assert sched.value(restart + C - 1) == pytest.approx(sched.value(W + C - 1))


def test_degenerate_config_still_defines_every_row():
    # Re-warmup longer than the whole stable segment: no crash, values in range.
    s = WsdSchedule(LR, warmup_rows=4000, cycle_rows=1000)
    for rows in range(0, 20_000, 7):
        assert 0.0 <= s.value(rows) <= LR


def _controller(tmp_path, rows_trained):
    conn = db.connect(tmp_path / "dashboard.db")
    return conn, WsdLrController(conn, WsdSchedule(LR, W, C), rows_trained)


def test_controller_tracks_current_and_logs_boundaries(tmp_path):
    conn, ctrl = _controller(tmp_path, 0)
    assert ctrl.current == 0.0

    # Batches inside warmup: values follow the ramp, no events yet.
    assert ctrl.lr_fn(500) == pytest.approx(LR / 2)
    assert ctrl.current == pytest.approx(LR / 2)
    assert db.read_control_events(conn, LR_EVENT) == []

    # A generation-sized jump straight across the warmup end lands on the
    # plateau: one event, stamped at the batch's rows position.
    assert ctrl.lr_fn(W + 300) == LR
    events = db.read_control_events(conn, LR_EVENT)
    assert [(e["positions"], e["value"]) for e in events] == [(W + 300, LR)]

    # More plateau batches add nothing.
    ctrl.lr_fn(W + 5000)
    assert len(db.read_control_events(conn, LR_EVENT)) == 1

    # Decay start and restart each log once, at their crossing.
    ctrl.lr_fn(W + DECAY_START + 10)
    ctrl.lr_fn(W + DECAY_START + 500)
    ctrl.lr_fn(W + C + 5)
    events = db.read_control_events(conn, LR_EVENT)
    assert [e["positions"] for e in events] == [W + 300, W + DECAY_START + 10, W + C + 5]
    assert events[-1]["value"] == pytest.approx(ctrl.schedule.value(W + C + 5))
    assert ctrl.current == pytest.approx(ctrl.schedule.value(W + C + 5))


def test_controller_resumed_mid_phase_logs_nothing_spurious(tmp_path):
    resume_at = W + DECAY_START + 100
    conn, ctrl = _controller(tmp_path, resume_at)
    assert ctrl.current == pytest.approx(ctrl.schedule.value(resume_at))
    ctrl.lr_fn(resume_at)
    ctrl.lr_fn(resume_at + 50)
    assert db.read_control_events(conn, LR_EVENT) == []
    # The next real crossing (the restart) is still logged.
    ctrl.lr_fn(W + C)
    assert [e["positions"] for e in db.read_control_events(conn, LR_EVENT)] == [W + C]
