"""Unit tests for the WorkerManager's slot lifecycle policy: slots are added
paused and nothing launches at add time, a cloud slot's pod is created on its
first start, and the cloud display-state mapping covers the no-pod-yet case.

Every path that would launch compute or touch cloud credentials is patched to
fail, so a regression back toward launch-on-add breaks loudly.
"""

import pytest
from scribblez import workloads
from scribblez.dashboard import tasks
from scribblez.dashboard.workers import WorkerManager, _cloud_state
from scripts.cloud_fleet import CpuResources


def _fail(*args, **kwargs):
    raise AssertionError("launched compute (or loaded cloud credentials) at the wrong time")


@pytest.fixture
def spec():
    return workloads.get("kill_test")


@pytest.fixture
def task() -> tasks.TaskRecord:
    return tasks.TaskRecord(workload="kill_test", tag="t", params={}, created_at=0.0)


@pytest.fixture
def manager(tmp_path, monkeypatch) -> WorkerManager:
    monkeypatch.setattr(tasks, "task_path", lambda spec, tag: tmp_path / f"{tag}.task.json")
    monkeypatch.setattr(WorkerManager, "_ensure_sync", lambda self, spec, task: None)
    for name in ("_spawn_local", "_run_ssh_container", "_create_pod", "_cloud"):
        monkeypatch.setattr(WorkerManager, name, _fail)
    return WorkerManager()


def _add_cloud(manager, spec, task, count=1):
    return manager.add_cloud(
        spec, task, "generate", count=count, resources=CpuResources(vcpus=8, flavor="cpu3c")
    )


def test_add_local_is_paused_and_not_spawned(manager, spec, task):
    w = manager.add_local(spec, task, "generate", threads=2)
    assert w.desired_state == "paused"
    assert w.pid is None
    assert tasks.load_task(spec, "t").workers[0].desired_state == "paused"


def test_add_ssh_is_paused_and_runs_no_container(manager, spec, task):
    w = manager.add_ssh(spec, task, "generate", host="user@h", threads=None)
    assert w.desired_state == "paused"


def test_add_cloud_is_paused_with_no_pod(manager, spec, task):
    added = _add_cloud(manager, spec, task, count=2)
    assert [w.desired_state for w in added] == ["paused", "paused"]
    assert all(w.pod_id is None for w in added)
    assert all(w.worker_id.startswith("scz-t-") for w in added)
    assert {(w.vcpus, w.flavor) for w in added} == {(8, "cpu3c")}


def test_first_start_creates_the_pod(manager, spec, task, monkeypatch):
    (w,) = _add_cloud(manager, spec, task)
    created = []
    monkeypatch.setattr(
        WorkerManager, "_create_pod", lambda self, spec, task, w: created.append(w.worker_id)
    )
    manager.set_worker_state(spec, task, w.worker_id, run=True)
    assert created == [w.worker_id]
    assert w.desired_state == "running"


def test_pause_and_remove_before_first_start_need_no_cloud(manager, spec, task):
    (w,) = _add_cloud(manager, spec, task)
    manager.set_worker_state(spec, task, w.worker_id, run=False)
    manager.remove_worker(spec, task, w.worker_id)
    assert task.workers == []


def test_status_of_slot_without_a_pod(manager, spec, task):
    (w,) = _add_cloud(manager, spec, task)
    (info,) = manager.worker_status(spec, task)
    assert info["state"] == "paused"
    w.desired_state = "running"  # e.g. pod creation failed after Start
    (info,) = manager.worker_status(spec, task)
    assert info["state"] == "starting"


def test_cloud_state_mapping():
    assert _cloud_state("running", True, gated=True, desired_status="RUNNING") == "waiting"
    assert _cloud_state("running", True, gated=False, desired_status="RUNNING") == "running"
    assert _cloud_state("running", False, gated=False, desired_status="RUNNING") == "starting"
    assert _cloud_state("running", False, gated=False, desired_status="EXITED") == "interrupted"
    assert _cloud_state("paused", True, gated=False, desired_status="RUNNING") == "stopping"
    assert _cloud_state("paused", False, gated=False, desired_status="EXITED") == "paused"
    # No pod yet: pods are created on first start.
    assert _cloud_state("paused", False, gated=False, desired_status=None) == "paused"
    assert _cloud_state("running", False, gated=False, desired_status=None) == "starting"
    assert _cloud_state("running", False, gated=True, desired_status=None) == "waiting"
