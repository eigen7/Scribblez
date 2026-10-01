"""Tests for a tag's data home (scribblez.generational.data_home): the
scheduler run beside the trainer, its published state, and the eviction guard
of a remote home whose generations the controller pulls.

Chunk game counts are faked (each chunk's text is its count), as in
test_generational_scheduler.py.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from cloud.sinks import LocalSink
from scribblez.generational import data_home, lifecycle, scheduler
from scribblez.paths import POSITION_EVAL, SCHEDULER_STATE_REL, TagPaths
from scribblez.workloads.position_eval import PositionEvalParams


@pytest.fixture
def paths(tmp_path: Path) -> TagPaths:
    return TagPaths("t", POSITION_EVAL, mount_root=tmp_path)


def _home(paths, games=100, ahead=1):
    cfg = scheduler.SchedulerConfig(games_per_generation=games, open_ahead=ahead)
    return data_home.DataHome(
        paths, cfg, LocalSink(paths.root), chunk_games=lambda chunk: int(chunk.read_text())
    )


def _state(paths) -> dict:
    return json.loads((paths.root / SCHEDULER_STATE_REL).read_text())


def test_a_staged_chunk_is_assigned_to_a_generation(paths):
    paths.staging_dir.mkdir(parents=True)
    (paths.staging_dir / "a.slog").write_text("100")
    _home(paths).step()
    assert lifecycle.is_complete(paths.generation_dir(0))


def test_the_state_carries_the_schedulers_gate_and_a_heartbeat(paths):
    paths.staging_dir.mkdir(parents=True)
    # Gens 0 and 1 fill; c waits, since gen 2 would be too far ahead.
    for name, games in (("a", 100), ("b", 100), ("c", 40)):
        (paths.staging_dir / f"{name}.slog").write_text(str(games))
    home = _home(paths)
    home.step()
    assert _state(paths)["gate"] == scheduler.GATE_REASON_AHEAD
    first = _state(paths)["heartbeat"]

    lifecycle.write_train_state(paths, {"generation_index": 1, "rows_trained": 10})
    home.step()  # the trainer advanced: gen 2 opens, takes c, and stays open
    assert _state(paths)["gate"] is None
    assert _state(paths)["heartbeat"] >= first


def test_any_failure_stops_the_thread_and_reaches_the_trainer(paths, monkeypatch):
    def broken_tick(*a, **k):
        raise RuntimeError("scheduler bug")

    monkeypatch.setattr(data_home.scheduler, "tick", broken_tick)
    monkeypatch.setattr(data_home, "POLL_SECONDS", 0)
    home = _home(paths)
    home.start()
    home._thread.join(timeout=5)
    assert not home._thread.is_alive()
    with pytest.raises(RuntimeError, match="scheduler bug"):
        home.check()


def test_start_for_is_remote_only_when_the_controller_says_so(paths, monkeypatch):
    monkeypatch.setattr(data_home.DataHome, "start", lambda self: None)
    ctx = SimpleNamespace(records_sink=LocalSink(paths.root))
    monkeypatch.delenv("SCZ_REMOTE_HOME", raising=False)
    assert not data_home.start_for(ctx, paths, PositionEvalParams()).remote
    monkeypatch.setenv("SCZ_REMOTE_HOME", "1")
    assert data_home.start_for(ctx, paths, PositionEvalParams()).remote


def test_a_remote_homes_trainer_keeps_each_generation_until_it_is_pulled(paths):
    """The controller's copy is the durable one: a generation the controller
    has not acknowledged (lifecycle.ACK_NAME) outlives the window."""
    for i in range(4):
        lifecycle.open_generation(paths, i, target_games=1)
        lifecycle.mark_complete(paths.generation_dir(i), 1)
    (paths.generation_dir(0) / lifecycle.ACK_NAME).touch()
    evicted = lifecycle.evict_beyond_window(paths, 3, 2, keep_unpulled=True)
    assert evicted == [0]  # gen 1 is outside the window too, but not yet pulled
    assert lifecycle.list_generation_indices(paths) == [1, 2, 3]
    assert lifecycle.evict_beyond_window(paths, 3, 2) == [1]  # a local home just evicts
