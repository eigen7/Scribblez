"""Unit tests for the generation scheduler (staging ingest, lifecycle, pacing).

Chunk game-counting is faked (each fake chunk's text is its game count), so the
assignment/completion/gating logic is exercised without the C++ loader.
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from scribblez.dashboard.workers import SYNC_INTERVAL_SECONDS
from scribblez.generational import lifecycle, scheduler
from scribblez.generational.data_home import POLL_SECONDS
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
    def __init__(self, publish: bool = False, paths: TagPaths | None = None):
        self.gates: dict[str, str | None] = {}
        self.mirrored: list[tuple[str, str]] = []
        self.published: list[str] = []
        self.publish_fails = False
        self.uploading = False  # the upload is still running
        super().__init__(
            paths=paths,
            gate=lambda role, reason: self.gates.__setitem__(role, reason),
            finish=lambda role: None,
            mirror=lambda name, dest: self.mirrored.append((name, dest)),
            publish=self._publish if publish else None,
        )

    def _publish(self, dest_rel: str) -> bool:
        if self.publish_fails:
            raise RuntimeError("bucket unreachable")
        if self.uploading:
            return False
        self.published.append(dest_rel)
        return True


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


def test_mirror_replays_ingest(paths):
    hooks = Hooks()
    _stage(paths, "a", 100)
    _tick(paths, hooks)
    assert hooks.mirrored == [("a.slog", "generations/gen_000000")]


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


def test_a_complete_generation_is_published_once(paths):
    hooks = Hooks(publish=True)
    _stage(paths, "a", 100)
    _tick(paths, hooks)
    gen0 = paths.generation_dir(0)
    assert lifecycle.is_complete(gen0) and lifecycle.is_published(gen0)
    assert hooks.published == ["generations/gen_000000"]
    _tick(paths, hooks)  # the open gen 1 is not complete: nothing more to publish
    assert hooks.published == ["generations/gen_000000"]
    assert not lifecycle.is_published(paths.generation_dir(1))


def test_a_failed_publish_is_retried_next_tick(paths):
    """The manifest records publication only once the hook returned, so an
    upload that failed (or a controller that died mid-way) is redone."""
    hooks = Hooks(publish=True)
    hooks.publish_fails = True
    _stage(paths, "a", 100)
    with pytest.raises(RuntimeError):
        _tick(paths, hooks)
    gen0 = paths.generation_dir(0)
    assert lifecycle.is_complete(gen0) and not lifecycle.is_published(gen0)
    hooks.publish_fails = False
    _tick(paths, hooks)
    assert lifecycle.is_published(gen0)
    assert hooks.published == ["generations/gen_000000"]


def test_a_generation_still_uploading_is_marked_once_it_is_there(paths):
    """The dashboard uploads in the background; a generation is recorded as
    published only once the hook says it is in the bucket."""
    hooks = Hooks(publish=True)
    hooks.uploading = True
    _stage(paths, "a", 100)
    _tick(paths, hooks)
    gen0 = paths.generation_dir(0)
    assert lifecycle.is_complete(gen0) and not lifecycle.is_published(gen0)
    hooks.uploading = False
    _tick(paths, hooks)
    assert lifecycle.is_published(gen0)


def test_without_a_publish_hook_nothing_is_marked(paths):
    hooks = Hooks()
    _stage(paths, "a", 100)
    _tick(paths, hooks)
    assert not lifecycle.is_published(paths.generation_dir(0))


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
    data_plane: str = scheduler.DATA_PLANE_LEGACY,
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
    task = SimpleNamespace(tag="t", params={"max_rows": max_rows}, data_plane=data_plane)
    scheduler.tick_for_task(spec, task, hooks)
    return hooks


def test_finishes_the_generators_once_the_trainer_reaches_max_rows(tmp_path):
    hooks = _task_tick(tmp_path, max_rows=1000, rows_trained=1000)
    assert hooks.finished == ["generate"]
    assert "generate" not in hooks.gates  # no scheduling after the end


@pytest.mark.parametrize(("max_rows", "rows_trained"), [(1000, 999), (1000, None), (0, 10**9)])
def test_keeps_scheduling_short_of_max_rows_or_without_one(tmp_path, max_rows, rows_trained):
    hooks = _task_tick(tmp_path, max_rows=max_rows, rows_trained=rows_trained)
    assert hooks.finished == []
    assert hooks.gates["generate"] is None  # the ordinary tick ran and opened a generation


# ---- a data home's tag: the controller only gates ------------------------------


def _write_state(paths: TagPaths, gate: str | None, heartbeat: float):
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / SCHEDULER_STATE_REL).write_text(
        json.dumps({"gate": gate, "heartbeat": heartbeat})
    )


def test_a_data_home_tag_is_gated_from_its_record_and_never_ticked(tmp_path):
    """The data home runs the scheduler; the controller must not touch the tree
    (a second scheduler), only carry the home's gate to the generators."""
    paths = TagPaths("t", POSITION_EVAL, mount_root=tmp_path)
    _write_state(paths, scheduler.GATE_REASON_AHEAD, time.time())
    hooks = _task_tick(tmp_path, max_rows=0, rows_trained=5, data_plane="home")
    assert hooks.gates["generate"] == scheduler.GATE_REASON_AHEAD
    assert lifecycle.list_generation_indices(paths) == []  # no generation opened here


def test_a_data_home_tag_still_finishes_at_max_rows(tmp_path):
    hooks = _task_tick(tmp_path, max_rows=1000, rows_trained=1000, data_plane="home")
    assert hooks.finished == ["generate"]


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


def test_the_heartbeat_outlasts_a_bucket_synced_record():
    """A remote data home's record reaches the controller through the sync
    watcher; a heartbeat judged stale inside a few sync rounds would park its
    generators while it is healthy."""
    assert scheduler.HEARTBEAT_STALE_SECONDS >= 3 * (SYNC_INTERVAL_SECONDS + POLL_SECONDS)
