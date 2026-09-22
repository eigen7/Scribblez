"""Tests for the Stats tab payload (scribblez.dashboard.master_api._stats_by_role)."""

import time

from scribblez import workloads
from scribblez.dashboard import master_api, tasks
from scribblez.dashboard.tasks import TaskRecord, WorkerRecord


def _worker(worker_id: str) -> WorkerRecord:
    return WorkerRecord(worker_id=worker_id, role="generate", kind="ssh", desired_state="running")


def test_workers_without_a_record_yet_are_listed_pending(tmp_path, monkeypatch):
    """A worker in its first cycle has published nothing; the tab still counts
    and lists it, with a null updated_at rather than a stale one."""
    spec = workloads.get("kill_test")
    monkeypatch.setattr(tasks, "task_path", lambda spec, tag: tmp_path / "task.json")
    task = TaskRecord(workload=spec.name, tag="t", params={}, created_at=time.time())
    task.workers = [_worker("ssh-0"), _worker("ssh-1")]
    tasks.save_task(spec, task)
    reported = {
        "worker_id": "ssh-0", "kind": "ssh", "role": "generate", "units_total": 4,
        "cycles_total": 1, "updated_at": 5.0, "recent": [],
    }  # fmt: skip
    monkeypatch.setattr(master_api.worker_stats_figures, "read_stats", lambda d: [reported])

    payload = master_api._stats_by_role(spec, "t")
    rows = {w["worker_id"]: w for w in payload["workers"]}
    assert rows["ssh-0"]["units_total"] == 4 and rows["ssh-0"]["updated_at"] == 5.0
    assert rows["ssh-1"]["updated_at"] is None and rows["ssh-1"]["units_per_hour"] is None
    assert payload["updated_at"] == 5.0
