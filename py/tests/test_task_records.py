"""The task-record store (dashboard/tasks.py): one object per task in the
store, re-read only when the file changes under it, written atomically."""

import json
import os

import pytest
from scribblez import workloads
from scribblez.dashboard import tasks


@pytest.fixture
def spec():
    return workloads.get("kill_test")


@pytest.fixture
def store(tmp_path) -> tasks.TaskStore:
    return tasks.TaskStore(tmp_path)


def _save(store, spec, **fields) -> tasks.TaskRecord:
    task = tasks.TaskRecord(workload=spec.name, tag="t", params={}, created_at=0.0, **fields)
    store.save(spec, task)
    return task


def test_load_returns_the_saved_object_itself(store, spec):
    """Every reader of a task gets the same record, so there is no stale copy
    for a later save to write back over a newer one."""
    task = _save(store, spec)
    assert store.load(spec, "t") is task
    assert store.load(spec, "t") is store.load(spec, "t")


def test_a_file_written_by_someone_else_is_read_afresh(store, spec):
    """A CLI tool editing task.json while the dashboard runs (a params
    migration) is picked up on the next load rather than overwritten."""
    task = _save(store, spec, retired_spend=1.0)
    path = store.task_path(spec, "t")
    raw = json.loads(path.read_text())
    raw["retired_spend"] = 2.0
    path.write_text(json.dumps(raw))
    stamp = path.stat().st_mtime_ns + 1_000_000
    os.utime(path, ns=(stamp, stamp))
    fresh = store.load(spec, "t")
    assert fresh is not task
    assert fresh.retired_spend == 2.0
    assert store.load(spec, "t") is fresh


def test_a_save_replaces_the_file_whole(store, spec):
    """Two threads saving at once each land a complete file, never a mix, and
    leave no temporary behind."""
    task = _save(store, spec)
    task.retired_spend = 3.5
    store.save(spec, task)
    tag_dir = store.task_path(spec, "t").parent
    assert [p.name for p in tag_dir.iterdir()] == ["task.json"]
    assert json.loads(store.task_path(spec, "t").read_text())["retired_spend"] == 3.5


def test_a_deleted_tag_is_forgotten(store, spec):
    task = _save(store, spec)
    store.delete(spec, "t")
    assert store.load(spec, "t") is None
    assert _save(store, spec) is not task


def test_machines_round_trip(store, spec):
    task = _save(store, spec)
    task.machines.append(
        tasks.MachineRecord(name="m1", provider="manual", host="ubuntu@1.2.3.4", gpu_count=1)
    )
    store.save(spec, task)
    loaded = tasks.TaskStore(store.mount_root).load(spec, "t")  # a fresh read
    assert loaded.machine("m1").host == "ubuntu@1.2.3.4"
    assert loaded.machine("m1").gpu_count == 1
    assert loaded.slots_on("m1") == []


def test_fields_a_record_no_longer_declares_are_dropped_on_read(store, spec):
    """A task.json written before a record field was removed still loads:
    the stale keys are dropped, and the next save writes the file without
    them."""
    task = _save(store, spec)
    task.workers.append(
        tasks.WorkerRecord(worker_id="w0", role="generate", kind="local", desired_state="paused")
    )
    store.save(spec, task)
    path = store.task_path(spec, "t")
    raw = json.loads(path.read_text())
    raw["workers"][0]["pod_id"] = "abc"
    raw["workers"][0]["spend"] = 1.5
    raw["gone"] = True
    path.write_text(json.dumps(raw))
    stamp = path.stat().st_mtime_ns + 1_000_000
    os.utime(path, ns=(stamp, stamp))
    loaded = store.load(spec, "t")
    assert loaded.workers[0].worker_id == "w0"
    store.save(spec, loaded)
    raw = json.loads(path.read_text())
    assert "gone" not in raw and "pod_id" not in raw["workers"][0]
