"""Tests for a tag's data home (scribblez.generational.data_home): bucket
ingest, the scheduler run beside the trainer, and its published state.

The bucket is a dict behind a fake rclone, and chunk game counts are faked
(each chunk's text is its count), as in test_generational_scheduler.py.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from cloud.credentials import R2Credentials
from cloud.sinks import LocalSink
from scribblez.generational import data_home, lifecycle, scheduler
from scribblez.paths import POSITION_EVAL, SCHEDULER_STATE_REL, TagPaths
from scribblez.workloads.position_eval import PositionEvalParams

R2 = R2Credentials(account_id="a", access_key_id="k", secret_access_key="s", bucket="b")
STAGING = "r2:b/position_eval/t/staging"


class FakeBucket:
    """The staging prefix as {name: text}, served through rclone's lsf,
    copyto and deletefile. `fail` names a verb that fails from now on."""

    def __init__(self, objects: dict[str, str]):
        self.objects = dict(objects)
        self.fail: str | None = None

    def rclone(self, r2, verb, *args, capture=False, input_text=None):
        if verb == self.fail:
            return subprocess.CompletedProcess([verb], 1, "", "simulated failure")
        if verb == "lsf":
            assert args == ("--files-only", STAGING), args
            return subprocess.CompletedProcess(
                [verb], 0, "".join(f"{n}\n" for n in self.objects), ""
            )
        name = args[0].removeprefix(f"{STAGING}/")
        if verb == "copyto":
            Path(args[1]).write_text(self.objects[name])
        else:
            assert verb == "deletefile", verb
            del self.objects[name]
        return subprocess.CompletedProcess([verb], 0, "", "")


@pytest.fixture
def paths(tmp_path: Path) -> TagPaths:
    return TagPaths("t", POSITION_EVAL, mount_root=tmp_path)


def _home(paths, bucket=None, monkeypatch=None, games=100, ahead=1):
    if bucket is not None:
        monkeypatch.setattr(data_home, "rclone", bucket.rclone)
    cfg = scheduler.SchedulerConfig(games_per_generation=games, open_ahead=ahead)
    return data_home.DataHome(
        paths,
        cfg,
        LocalSink(paths.root),
        r2=R2 if bucket is not None else None,
        chunk_games=lambda chunk: int(chunk.read_text()),
    )


def _state(paths) -> dict:
    return json.loads((paths.root / SCHEDULER_STATE_REL).read_text())


def test_remote_chunks_are_moved_out_of_the_bucket_into_a_generation(paths, monkeypatch):
    bucket = FakeBucket({"a.slog": "60", "b.slog": "60", "notes.txt": "x"})
    _home(paths, bucket, monkeypatch).step()

    gen0 = paths.generation_dir(0)
    assert lifecycle.is_complete(gen0)
    assert sorted(f.name for f in gen0.glob("*.slog")) == ["a.slog", "b.slog"]
    assert bucket.objects == {"notes.txt": "x"}  # ingested chunks leave the bucket
    assert not any((paths.data_dir / "work" / data_home.INGRESS_WORK_DIR).iterdir())


def test_a_colocated_generators_chunk_needs_no_bucket(paths):
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


@pytest.mark.parametrize("verb", ["lsf", "copyto", "deletefile"])
def test_a_failing_bucket_is_retried_not_raised(paths, monkeypatch, verb):
    bucket = FakeBucket({"a.slog": "100"})
    bucket.fail = verb
    home = _home(paths, bucket, monkeypatch)
    home.step()  # does not raise; the state is still published
    assert _state(paths)["gate"] is None

    bucket.fail = None
    home.step()
    assert lifecycle.is_complete(paths.generation_dir(0))
    assert bucket.objects == {}


def test_any_other_failure_stops_the_thread_and_reaches_the_trainer(paths, monkeypatch):
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


def test_start_for_runs_only_for_a_home_tag(paths, monkeypatch):
    monkeypatch.delenv("R2_BUCKET", raising=False)
    started = []
    monkeypatch.setattr(data_home.DataHome, "start", lambda self: started.append(self))
    ctx = SimpleNamespace(
        data_plane=scheduler.DATA_PLANE_LEGACY, records_sink=LocalSink(paths.root)
    )
    assert data_home.start_for(ctx, paths, PositionEvalParams()) is None
    ctx.data_plane = scheduler.DATA_PLANE_HOME
    home = data_home.start_for(ctx, paths, PositionEvalParams())
    assert started == [home] and home._r2 is None  # no credentials: colocated generators only
