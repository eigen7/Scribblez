"""The task-record registry (dashboard/tasks.py): one object per task in the
process, re-read only when the file changes under it, written atomically."""

import json
import os

import pytest
from scribblez import workloads
from scribblez.dashboard import tasks


@pytest.fixture
def spec(tmp_path, monkeypatch):
    monkeypatch.setattr(workloads.WorkloadSpec, "data_dir", lambda self, tag: tmp_path / tag)
    return workloads.get("kill_test")


def _save(spec, **fields) -> tasks.TaskRecord:
    task = tasks.TaskRecord(workload=spec.name, tag="t", params={}, created_at=0.0, **fields)
    tasks.save_task(spec, task)
    return task


def test_load_returns_the_saved_object_itself(spec):
    """Every reader of a task gets the same record, so there is no stale copy
    for a later save to write back over a newer one."""
    task = _save(spec)
    assert tasks.load_task(spec, "t") is task
    assert tasks.load_task(spec, "t") is tasks.load_task(spec, "t")


def test_a_file_written_by_someone_else_is_read_afresh(spec):
    """A CLI tool editing task.json while the dashboard runs (a params
    migration) is picked up on the next load rather than overwritten."""
    task = _save(spec, retired_spend=1.0)
    path = tasks.task_path(spec, "t")
    raw = json.loads(path.read_text())
    raw["retired_spend"] = 2.0
    path.write_text(json.dumps(raw))
    stamp = path.stat().st_mtime_ns + 1_000_000
    os.utime(path, ns=(stamp, stamp))
    fresh = tasks.load_task(spec, "t")
    assert fresh is not task
    assert fresh.retired_spend == 2.0
    assert tasks.load_task(spec, "t") is fresh


def test_a_save_replaces_the_file_whole(spec):
    """Two threads saving at once each land a complete file, never a mix, and
    leave no temporary behind."""
    task = _save(spec)
    task.retired_spend = 3.5
    tasks.save_task(spec, task)
    tag_dir = tasks.task_path(spec, "t").parent
    assert [p.name for p in tag_dir.iterdir()] == ["task.json"]
    assert json.loads(tasks.task_path(spec, "t").read_text())["retired_spend"] == 3.5


def test_a_deleted_tag_is_forgotten(spec):
    task = _save(spec)
    tasks.delete_tag(spec, "t")
    assert tasks.load_task(spec, "t") is None
    assert _save(spec) is not task
