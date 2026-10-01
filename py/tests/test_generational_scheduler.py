"""Unit tests for the generation scheduler (staging ingest, lifecycle, pacing).

Chunk game-counting is faked (each fake chunk's text is its game count), so the
assignment/completion/gating logic is exercised without the C++ loader.
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from scribblez.generational import lifecycle, scheduler
from scribblez.paths import POSITION_EVAL, SCHEDULER_STATE_REL, TagPaths
from scribblez.workloads.base import SchedulerHooks
from scribblez.workloads.position_eval import PositionEvalParams


@pytest.fixture
def paths(tmp_path: Path) -> TagPaths:
    return TagPaths("t", POSITION_EVAL, mount_root=tmp_path)


def _chunk_games(chunk: Path) -> int:
    try:
        return int(chunk.read_text())
    except ValueError as e:
        raise OSError(str(e)) from e


def _stage(paths: TagPaths, name: str, games: int):
    paths.staging_dir.mkdir(parents=True, exist_ok=True)
    (paths.staging_dir / f"{name}.slog").write_text(str(games))


class Hooks(SchedulerHooks):
    def __init__(self, paths: TagPaths | None = None):
        self.gates: dict[str, str | None] = {}
        super().__init__(
            paths=paths,
            gate=lambda role, reason: self.gates.__setitem__(role, reason),
            finish=lambda role: None,
        )


def _tick(paths, hooks, *, games=100, ahead=1):
    cfg = scheduler.SchedulerConfig(games_per_generation=games, open_ahead=ahead)
    scheduler.tick(paths, cfg, hooks, chunk_games=_chunk_games)


def test_fills_and_completes_generation(paths):
    hooks = Hooks()
    _tick(paths, hooks)  # nothing staged: opens gen 0, stays open
    assert lifecycle.read_manifest(paths.generation_dir(0))["status"] == lifecycle.GENERATING
    assert hooks.gates["generate"] is None

    _stage(paths, "a", 60)
    _stage(paths, "b", 60)
    _tick(paths, hooks)
    gen0 = paths.generation_dir(0)
    assert lifecycle.is_complete(gen0)
    assert lifecycle.read_manifest(gen0)["committed_games"] == 120
    assert sorted(f.name for f in gen0.glob("*.slog")) == ["a.slog", "b.slog"]
    # With gen 0 complete and cursor still 0, gen 1 opens (open_ahead=1).
    assert lifecycle.read_manifest(paths.generation_dir(1))["status"] == lifecycle.GENERATING


def test_gates_when_ahead_of_trainer(paths):
    hooks = Hooks()
    _stage(paths, "a", 100)
    _stage(paths, "b", 100)
    _stage(paths, "c", 40)
    _tick(paths, hooks)
    # Gen 0 and gen 1 (cursor 0 + ahead 1) complete; gen 2 may not open.
    assert lifecycle.is_complete(paths.generation_dir(0))
    assert lifecycle.is_complete(paths.generation_dir(1))
    assert 2 not in lifecycle.list_generation_indices(paths)
    assert hooks.gates["generate"] == scheduler.GATE_REASON_AHEAD
    assert len(list(paths.staging_dir.glob("*.slog"))) == 1  # c waits in staging

    # The trainer advances; the next tick opens gen 2 and releases the gate.
    lifecycle.write_train_state(paths, {"generation_index": 1})
    _tick(paths, hooks)
    assert lifecycle.read_manifest(paths.generation_dir(2))["status"] == lifecycle.GENERATING
    assert lifecycle.read_manifest(paths.generation_dir(2))["committed_games"] == 40
    assert hooks.gates["generate"] is None


def test_ledger_deletes_resynced_duplicates(paths):
    hooks = Hooks()
    _stage(paths, "a", 100)
    _tick(paths, hooks)
    assert lifecycle.is_complete(paths.generation_dir(0))

    # The same chunk re-appears in staging (a cloud sync raced the ingest); it
    # is deleted, never assigned twice.
    _stage(paths, "a", 100)
    _tick(paths, hooks)
    assert list(paths.staging_dir.glob("*.slog")) == []
    assert lifecycle.read_manifest(paths.generation_dir(1))["committed_games"] == 0


def test_unreadable_chunk_is_quarantined(paths):
    hooks = Hooks()
    paths.staging_dir.mkdir(parents=True, exist_ok=True)
    (paths.staging_dir / "bad.slog").write_text("garbage")
    _stage(paths, "ok", 100)
    _tick(paths, hooks)
    assert lifecycle.is_complete(paths.generation_dir(0))
    assert (paths.staging_dir / "bad.bad").exists()
    assert not (paths.staging_dir / "bad.slog").exists()


def test_committed_count_self_heals(paths):
    """A chunk landing in the open generation outside the ledger/manifest path
    (a crash between rename and manifest write) is still counted."""
    hooks = Hooks()
    _tick(paths, hooks)  # opens gen 0
    gen0 = paths.generation_dir(0)
    (gen0 / "stray.slog").write_text("100")
    _tick(paths, hooks)
    assert lifecycle.is_complete(gen0)
    assert lifecycle.read_manifest(gen0)["committed_games"] == 100


class _FinishHooks(Hooks):
    def __init__(self, paths: TagPaths):
        super().__init__(paths=paths)
        self.finished: list[str] = []
        self.finish = self.finished.append


def _task_tick(
    tmp_path,
    *,
    max_rows: int,
    rows_trained: int | None,
    trainer_running: bool = True,
):
    """One tick_for_task on a position_eval task under tmp_path, with the
    trainer's cursor at `rows_trained` (None: no cursor yet)."""
    spec = SimpleNamespace(params_cls=PositionEvalParams)
    paths = TagPaths("t", POSITION_EVAL, mount_root=tmp_path)
    if rows_trained is not None:
        paths.train_state_path.parent.mkdir(parents=True, exist_ok=True)
        paths.train_state_path.write_text(json.dumps({"rows_trained": rows_trained}))
    hooks = _FinishHooks(paths)
    hooks.role_running = lambda role: role == "train" and trainer_running
    task = SimpleNamespace(tag="t", params={"max_rows": max_rows})
    scheduler.tick_for_task(spec, task, hooks)
    return hooks


def test_finishes_the_generators_once_the_trainer_reaches_max_rows(tmp_path):
    hooks = _task_tick(tmp_path, max_rows=1000, rows_trained=1000)
    assert hooks.finished == ["generate"]
    assert "generate" not in hooks.gates  # no scheduling after the end


# ---- the controller only gates: the data home schedules ----------------------


def _write_state(paths: TagPaths, gate: str | None, heartbeat: float):
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / SCHEDULER_STATE_REL).write_text(
        json.dumps({"gate": gate, "heartbeat": heartbeat})
    )


@pytest.mark.parametrize(("max_rows", "rows_trained"), [(1000, 999), (1000, None), (0, 10**9)])
def test_the_tag_is_gated_from_its_record_and_never_ticked(tmp_path, max_rows, rows_trained):
    """Short of max_rows (or without one), the data home runs the scheduler;
    the controller must not touch the tree (a second scheduler), only carry
    the home's gate to the generators."""
    paths = TagPaths("t", POSITION_EVAL, mount_root=tmp_path)
    _write_state(paths, scheduler.GATE_REASON_AHEAD, time.time())
    hooks = _task_tick(tmp_path, max_rows=max_rows, rows_trained=rows_trained)
    assert hooks.finished == []
    assert hooks.gates["generate"] == scheduler.GATE_REASON_AHEAD
    assert lifecycle.list_generation_indices(paths) == []  # no generation opened here


@pytest.mark.parametrize(
    ("trainer_running", "record_age", "expected"),
    [
        (True, 1.0, None),  # fresh and ungated: generate
        (False, 1.0, scheduler.GATE_REASON_NO_TRAINER),  # nothing would take the chunks
        (True, scheduler.HEARTBEAT_STALE_SECONDS + 1, scheduler.GATE_REASON_NO_HEARTBEAT),
        (True, None, scheduler.GATE_REASON_NO_HEARTBEAT),  # no record yet (trainer starting)
    ],
)
def test_home_gate(paths, trainer_running, record_age, expected):
    """`record_age`: how long ago this controller saw the heartbeat change."""
    start = 1_000_000.0
    if record_age is not None:
        _write_state(paths, None, 12345.0)  # the home's clock: irrelevant
        scheduler.home_gate(paths, True, start)  # first seen now
    assert scheduler.home_gate(paths, trainer_running, start + (record_age or 0)) == expected


def test_the_heartbeat_is_judged_on_the_controllers_clock(paths):
    """A home whose clock is far off (or simply another machine's) neither
    parks its generators nor hides a dead heartbeat: what counts is when the
    controller saw the value change."""
    t = 2_000_000.0
    _write_state(paths, None, 1.0)  # a heartbeat from 1970, by the home's clock
    assert scheduler.home_gate(paths, True, t) is None
    later = t + scheduler.HEARTBEAT_STALE_SECONDS + 1
    assert scheduler.home_gate(paths, True, later) == scheduler.GATE_REASON_NO_HEARTBEAT
    _write_state(paths, None, 2.0)  # it beat again
    assert scheduler.home_gate(paths, True, later) is None
