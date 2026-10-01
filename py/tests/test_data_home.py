"""Tests for a tag's data home (scribblez.generational.data_home): bucket
ingest, the scheduler run beside the trainer, and its published state.

The bucket is a dict behind a fake rclone, and chunk game counts are faked
(each chunk's text is its count), as in test_generational_scheduler.py.
"""

import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from cloud.credentials import R2Credentials
from cloud.sinks import LocalSink
from scribblez.generational import data_home, lifecycle, scheduler
from scribblez.paths import POSITION_EVAL, SCHEDULER_STATE_REL, TagPaths
from scribblez.workloads.position_eval import PositionEvalParams

R2 = R2Credentials(account_id="a", access_key_id="k", secret_access_key="s", bucket="b")
PREFIX = "r2:b/position_eval/t"


class FakeBucket:
    """The tag's bucket prefix as {key under the prefix: text}, served through
    the rclone verbs the data home uses. `fail` names a verb that fails from
    now on."""

    def __init__(self, objects: dict[str, str]):
        self.objects = dict(objects)
        self.fail: str | None = None

    def _key(self, path: str) -> str:
        assert path.startswith(PREFIX), path
        return path[len(PREFIX) :].lstrip("/")

    def rclone(self, r2, verb, *args, capture=False, input_text=None):
        if verb == self.fail:
            return subprocess.CompletedProcess([verb], 1, "", "simulated failure")
        flags = [a for a in args if a.startswith("--")]
        args = [a for a in args if not a.startswith("--") and a != lifecycle.MANIFEST_NAME]
        out = getattr(self, "_" + verb)(flags, *args)
        return subprocess.CompletedProcess([verb], 0, out or "", "")

    def _lsf(self, flags, path):
        key = self._key(path)
        if key in self.objects:
            return key.rsplit("/", 1)[-1] + "\n"
        under = [k[len(key) + 1 :] for k in self.objects if k.startswith(key + "/")]
        if "--dirs-only" in flags:
            return "".join(sorted({u.split("/")[0] + "/\n" for u in under if "/" in u}))
        return "".join(f"{u}\n" for u in sorted(under) if "/" not in u)

    def _copyto(self, flags, src, dst):
        if src.startswith(PREFIX):
            Path(dst).write_text(self.objects[self._key(src)])
        else:
            self.objects[self._key(dst)] = Path(src).read_text()

    def _copy(self, flags, src, dst):
        if src.startswith(PREFIX):  # bucket dir -> local dir
            key = self._key(src)
            for k, text in self.objects.items():
                if k.startswith(key + "/"):
                    out = Path(dst) / k[len(key) + 1 :]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(text)
        else:  # local dir -> bucket dir, the manifest excluded (see rclone())
            for f in Path(src).iterdir():
                if f.name != lifecycle.MANIFEST_NAME:
                    self.objects[f"{self._key(dst)}/{f.name}"] = f.read_text()

    def _deletefile(self, flags, path):
        del self.objects[self._key(path)]

    def _purge(self, flags, path):
        key = self._key(path)
        self.objects = {k: v for k, v in self.objects.items() if not k.startswith(key + "/")}


@pytest.fixture
def paths(tmp_path: Path) -> TagPaths:
    return TagPaths("t", POSITION_EVAL, mount_root=tmp_path)


def _home(paths, bucket=None, monkeypatch=None, games=100, ahead=1, window=0, uploads=False):
    if bucket is not None:
        monkeypatch.setattr(data_home, "rclone", bucket.rclone)
    cfg = scheduler.SchedulerConfig(games_per_generation=games, open_ahead=ahead)
    return data_home.DataHome(
        paths,
        cfg,
        LocalSink(paths.root),
        r2=R2 if bucket is not None else None,
        chunk_games=lambda chunk: int(chunk.read_text()),
        window=window,
        uploads=uploads,
    )


def _state(paths) -> dict:
    return json.loads((paths.root / SCHEDULER_STATE_REL).read_text())


def test_remote_chunks_are_moved_out_of_the_bucket_into_a_generation(paths, monkeypatch):
    bucket = FakeBucket({"staging/a.slog": "60", "staging/b.slog": "60", "staging/notes.txt": "x"})
    _home(paths, bucket, monkeypatch).step()

    gen0 = paths.generation_dir(0)
    assert lifecycle.is_complete(gen0)
    assert sorted(f.name for f in gen0.glob("*.slog")) == ["a.slog", "b.slog"]
    assert bucket.objects == {"staging/notes.txt": "x"}  # ingested chunks leave the bucket
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
    bucket = FakeBucket({"staging/a.slog": "100"})
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
    home._threads[0].join(timeout=5)  # the scheduler's
    assert not home._threads[0].is_alive()
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


# ---- a data home that can vanish: the bucket keeps it resumable -----------------


def _stage(paths, *games):
    paths.staging_dir.mkdir(parents=True, exist_ok=True)
    for i, g in enumerate(games):
        (paths.staging_dir / f"c{i}.slog").write_text(str(g))


def test_complete_generations_are_uploaded_manifest_last_and_marked(paths, monkeypatch):
    bucket = FakeBucket({})
    order = []
    copyto = bucket._copyto
    monkeypatch.setattr(
        bucket, "_copyto", lambda f, src, dst: order.append(dst) or copyto(f, src, dst)
    )
    monkeypatch.setattr(
        bucket, "_copy", lambda f, src, dst, _c=bucket._copy: order.append(dst) or _c(f, src, dst)
    )
    _stage(paths, 100, 40)  # gen 0 completes; gen 1 stays open
    _home(paths, bucket, monkeypatch, uploads=True).step()

    assert bucket.objects["generations/gen_000000/c0.slog"] == "100"
    assert (
        json.loads(bucket.objects["generations/gen_000000/manifest.json"])["status"] == "complete"
    )
    assert order == [
        f"{PREFIX}/generations/gen_000000",
        f"{PREFIX}/generations/gen_000000/manifest.json",
    ]
    assert lifecycle.is_published(paths.generation_dir(0))
    assert not any(k.startswith("generations/gen_000001") for k in bucket.objects)  # still open


def test_a_failed_upload_is_retried_and_its_generation_kept(paths, monkeypatch):
    bucket = FakeBucket({})
    bucket.fail = "copy"
    _stage(paths, 100, 100, 100)
    home = _home(paths, bucket, monkeypatch, ahead=5, window=1, uploads=True)
    home.step()
    assert not lifecycle.is_published(paths.generation_dir(0))
    # The trainer is past gen 0, but gen 0 is not in the bucket yet: it stays.
    lifecycle.write_train_state(paths, {"generation_index": 3})
    assert lifecycle.evict_beyond_window(paths, 2, 1, keep_unpublished=True) == []
    bucket.fail = None
    home.step()
    assert all(lifecycle.is_published(paths.generation_dir(i)) for i in range(3))
    assert lifecycle.evict_beyond_window(paths, 2, 1, keep_unpublished=True) == [0, 1]


def test_generations_behind_the_window_leave_the_bucket(paths, monkeypatch):
    bucket = FakeBucket({f"generations/gen_00000{i}/manifest.json": "{}" for i in range(5)})
    lifecycle.write_train_state(paths, {"generation_index": 4})
    _home(paths, bucket, monkeypatch, window=2, uploads=True).step()
    kept = sorted({k.split("/")[1] for k in bucket.objects if k.startswith("generations/")})
    assert kept == ["gen_000002", "gen_000003", "gen_000004"]


def test_a_fresh_home_restores_the_window_and_numbers_after_the_bucket(paths, monkeypatch):
    """A new machine resumes from the bucket: the window behind the cursor and
    the uploaded generations ahead of it, and new generations take the next
    free index, never one the bucket already holds."""
    manifest = json.dumps({"status": "complete", "committed_games": 100, "target_games": 100})
    bucket = FakeBucket({})
    for i in range(1, 6):
        bucket.objects[f"generations/gen_00000{i}/c.slog"] = "100"
        bucket.objects[f"generations/gen_00000{i}/manifest.json"] = manifest
    bucket.objects["generations/gen_000006/c.slog"] = "100"  # an upload that died mid-way
    lifecycle.write_train_state(paths, {"generation_index": 4})
    home = _home(paths, bucket, monkeypatch, ahead=5, window=2, uploads=True)
    home.restore()

    assert lifecycle.list_generation_indices(paths) == [2, 3, 4, 5]  # 1 is behind the window
    assert all(lifecycle.is_published(paths.generation_dir(i)) for i in (2, 3, 4, 5))
    home.step()
    assert lifecycle.read_manifest(paths.generation_dir(6))["status"] == lifecycle.GENERATING


def test_a_failing_listing_at_restore_stops_the_trainer(paths, monkeypatch):
    """Resuming on a partial window would silently diverge."""
    bucket = FakeBucket({})
    bucket.fail = "lsf"
    with pytest.raises(AssertionError, match="listing"):
        _home(paths, bucket, monkeypatch, uploads=True).restore()


def test_start_for_restores_and_uploads_only_for_a_remote_home(paths, monkeypatch):
    """The controller says so (SCZ_HOME_UPLOADS): a trainer's records are
    local wherever it runs, so the sink no longer tells."""
    for var in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.setenv(var, "x")
    monkeypatch.setattr(data_home.DataHome, "start", lambda self: None)
    restored = []
    monkeypatch.setattr(
        data_home.DataHome, "restore", lambda self, required: restored.append(required)
    )
    ctx = SimpleNamespace(data_plane=scheduler.DATA_PLANE_HOME, records_sink=LocalSink(paths.root))
    monkeypatch.delenv("SCZ_HOME_UPLOADS", raising=False)
    assert not data_home.start_for(ctx, paths, PositionEvalParams()).uploads
    monkeypatch.setenv("SCZ_HOME_UPLOADS", "1")
    assert data_home.start_for(ctx, paths, PositionEvalParams()).uploads
    # A localhost home restores what a previous home uploaded, but does not
    # need the bucket; a bucket trainer's home does.
    assert restored == [False, True]


def test_an_optional_restore_survives_an_unreachable_bucket(paths, monkeypatch):
    bucket = FakeBucket({})
    bucket.fail = "lsf"
    _home(paths, bucket, monkeypatch).restore(required=False)  # no raise
    assert lifecycle.list_generation_indices(paths) == []


def test_a_hung_upload_holds_up_neither_scheduling_nor_the_heartbeat(paths, monkeypatch):
    """Seen live on a laptop data home: an upload stuck on a dead connection
    stalled the one-pass loop, so chunks sat in staging and the heartbeat went
    stale until the controller parked the generators."""
    bucket = FakeBucket({})
    release = threading.Event()
    copy = bucket._copy

    def hung_copy(flags, src, dst):
        release.wait(timeout=10)
        copy(flags, src, dst)

    monkeypatch.setattr(bucket, "_copy", hung_copy)
    monkeypatch.setattr(data_home, "POLL_SECONDS", 0.01)
    _stage(paths, 100)
    home = _home(paths, bucket, monkeypatch, uploads=True)
    home.start()
    try:
        _wait_for(lambda: lifecycle.is_complete(paths.generation_dir(0)))  # upload now hangs
        first = _state(paths)["heartbeat"]
        (paths.staging_dir / "late.slog").write_text("100")
        _wait_for(lambda: lifecycle.is_complete(paths.generation_dir(1)))
        _wait_for(lambda: _state(paths)["heartbeat"] > first)
        assert not lifecycle.is_published(paths.generation_dir(0))  # still uploading
    finally:
        release.set()
    _wait_for(lambda: lifecycle.is_published(paths.generation_dir(0)))
    home.check()


def _wait_for(condition, timeout=5.0):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, "timed out"
        time.sleep(0.01)
