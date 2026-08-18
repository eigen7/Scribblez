"""Unit tests for the WorkerManager's slot lifecycle policy: slots are added
paused and nothing launches at add time, a cloud slot's pod is created on its
first start, and the cloud display-state mapping covers the no-pod-yet case.

Every path that would launch compute or touch cloud credentials is patched to
fail, so a regression back toward launch-on-add breaks loudly.
"""

import asyncio
import time

import pytest
from scribblez import workloads
from scribblez.dashboard import tasks
from scribblez.dashboard import workers as workers_mod
from scribblez.dashboard.workers import (
    WorkerManager,
    _cloud_state,
    _resource_record_fields,
    _worker_resources,
)
from scripts.cloud_fleet import CpuResources, GpuResources


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
    assert not w.launched


class _FakeSshMachine:
    """SshMachine stand-in whose container probe always returns `state`."""

    state = "unreachable"

    def __init__(self, host):
        self.host = host

    def container_state(self, name: str) -> str:
        return self.state


def test_unlaunched_ssh_slot_stays_manageable_when_unreachable(manager, spec, task, monkeypatch):
    """A slot whose container was never confirmed created reads `missing` on
    an unreachable probe (the host may be bogus -- it is unvalidated until
    first start): status shows it paused, and remove works instead of
    refusing until the host comes online."""
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    w = manager.add_ssh(spec, task, "generate", host="user@no-such-host", threads=None)
    (info,) = manager.worker_status(spec, task, observe=True)
    assert info["state"] == "paused"
    manager.remove_worker(spec, task, w.worker_id)
    assert task.workers == []


def test_probe_heals_launched_after_in_doubt_start(manager, spec, task, monkeypatch):
    """If the first start's ssh link died after `docker run` was dispatched,
    the container exists while `launched` is still False; the next successful
    probe re-observes it and flips the marker."""
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    monkeypatch.setattr(_FakeSshMachine, "state", "running")
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    w.desired_state = "running"
    (info,) = manager.worker_status(spec, task, observe=True)
    assert info["state"] == "running"
    assert w.launched


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


def test_reconcile_creates_the_missing_pod(manager, spec, task, monkeypatch):
    """A should-run slot with no pod (its creation failed on start, or a gate
    released it before its first start) gets its pod created by reconcile."""
    (w,) = _add_cloud(manager, spec, task)
    w.desired_state = "running"
    created = []
    monkeypatch.setattr(
        WorkerManager, "_create_pod", lambda self, spec, task, w: created.append(w.worker_id)
    )
    (info,) = manager.worker_status(spec, task, observe=True)
    manager._reconcile_worker(spec, task, w, workers_mod.RUN, info)
    assert created == [w.worker_id]


def test_reconcile_contains_per_slot_failures(manager, spec, task, monkeypatch):
    """One slot's persistently failing enforcement (e.g. pod creation on an
    out-of-stock flavor) must not abort the pass: later slots still get
    their tick."""
    added = _add_cloud(manager, spec, task, count=2)
    for w in added:
        w.desired_state = "running"
    attempted = []

    def boom(self, spec, task, w):
        attempted.append(w.worker_id)
        raise RuntimeError("no capacity for flavor cpu3c")

    monkeypatch.setattr(WorkerManager, "_create_pod", boom)
    monkeypatch.setattr(manager, "_all_tasks", lambda: iter([(spec, task)]))
    asyncio.run(manager.reconcile())
    assert attempted == [w.worker_id for w in added]


def test_status_of_slot_without_a_pod(manager, spec, task):
    (w,) = _add_cloud(manager, spec, task)
    (info,) = manager.worker_status(spec, task)
    assert info["state"] == "paused"
    w.desired_state = "running"  # e.g. pod creation failed after Start
    (info,) = manager.worker_status(spec, task)
    assert info["state"] == "starting"


def test_worker_resources_roundtrip():
    for res in (
        CpuResources(vcpus=8, flavor="cpu3c"),
        GpuResources(gpu_type_id="A100", gpu_count=2),
    ):
        w = tasks.WorkerRecord(
            worker_id="x", role="generate", kind="cloud", desired_state="paused",
            **_resource_record_fields(res),
        )  # fmt: skip
        assert _worker_resources(w) == res


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


def test_first_remote_worker_deploys_and_pins_the_task(manager, spec, task, monkeypatch):
    """Deployment is not an operator step: the task pins a bundle the first
    time a bucket-delivering worker needs one, and every later worker joins
    that same bundle rather than whatever the tree has become."""
    deploys = []
    monkeypatch.setattr(WorkerManager, "deploy", lambda self, spec, task: deploys.append(1) or "b1")
    assert manager.task_bundle_id(spec, task) == "b1"
    task.bundle_id = "b1"  # what deploy() itself records
    assert manager.task_bundle_id(spec, task) == "b1"
    assert len(deploys) == 1


def test_bundle_drift_compares_tree_against_pinned_bundle(manager, spec, task, monkeypatch):
    monkeypatch.setattr(workers_mod, "source_hash", lambda cache=None: "now")
    assert not manager.bundle_drift(task)  # nothing pinned yet

    task.bundle_source_hash = "now"
    assert not manager.bundle_drift(task)

    task.bundle_source_hash = "then"
    assert manager.bundle_drift(task)

    # An unbuilt arch is not evidence of drift, only absence of evidence.
    monkeypatch.setattr(workers_mod, "source_hash", lambda cache=None: None)
    assert not manager.bundle_drift(task)


class _RecordingSshMachine(_FakeSshMachine):
    """Records the container operations reconcile performs."""

    state = "stopped"
    ops: list = []

    def start_container(self, name):
        self.ops.append(("start", name))

    def pause_container(self, name):
        self.ops.append(("pause", name))

    def unpause_container(self, name):
        self.ops.append(("unpause", name))

    def stop_container(self, name):
        self.ops.append(("stop", name))

    def remove_container(self, name):
        self.ops.append(("remove", name))


def _stopped_ssh_slot(manager, spec, task, monkeypatch, *, slot_bundle, task_bundle):
    monkeypatch.setattr(workers_mod, "SshMachine", _RecordingSshMachine)
    _RecordingSshMachine.ops = []
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    w.desired_state, w.launched, w.bundle_id = "running", True, slot_bundle
    task.bundle_id = task_bundle
    return w


def test_a_stopped_ssh_slot_on_the_task_bundle_is_restarted(manager, spec, task, monkeypatch):
    w = _stopped_ssh_slot(manager, spec, task, monkeypatch, slot_bundle="b1", task_bundle="b1")
    stopped = {"observed_running": False, "ssh_probe": "stopped"}
    manager._reconcile_worker(spec, task, w, workers_mod.RUN, stopped)
    assert [op for op, _ in _RecordingSshMachine.ops] == ["start"]


def test_a_redeployed_task_replaces_its_ssh_container(manager, spec, task, monkeypatch):
    """A container's bundle is fixed in the environment it was created with,
    so joining a new one means replacement, not a restart."""
    recreated = []
    monkeypatch.setattr(
        WorkerManager,
        "_run_ssh_container",
        lambda self, spec, task, w: recreated.append(w.worker_id),
    )
    w = _stopped_ssh_slot(manager, spec, task, monkeypatch, slot_bundle="b1", task_bundle="b2")
    stopped = {"observed_running": False, "ssh_probe": "stopped"}
    manager._reconcile_worker(spec, task, w, workers_mod.RUN, stopped)
    assert [op for op, _ in _RecordingSshMachine.ops] == ["remove"]
    assert recreated == [w.worker_id]


def test_intent_separates_a_gate_from_an_operator_pause():
    """Both mean not-working, but a gate flips many times an hour and an
    operator pause is a deliberate stop -- ssh slots treat them differently."""
    task = tasks.TaskRecord(workload="kill_test", tag="t", params={}, created_at=0.0)
    w = tasks.WorkerRecord(worker_id="ssh-0", role="generate", kind="ssh", desired_state="running")
    assert workers_mod._intent(w, task) == workers_mod.RUN
    task.gates["generate"] = "ahead of trainer"
    assert workers_mod._intent(w, task) == workers_mod.PARK
    w.desired_state = "paused"
    assert workers_mod._intent(w, task) == workers_mod.STOP


def _ssh_slot(manager, spec, task, monkeypatch, probe: str):
    monkeypatch.setattr(workers_mod, "SshMachine", _RecordingSshMachine)
    _RecordingSshMachine.ops = []
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    w.desired_state, w.launched, w.bundle_id = "running", True, "b1"
    task.bundle_id = "b1"
    return w, {"observed_running": probe == "running", "ssh_probe": probe}


def test_a_gate_parks_an_ssh_container_by_pausing_it(manager, spec, task, monkeypatch):
    """Stopping would discard the chunk in flight and make the next start
    refetch and unpack the bundle before its first game."""
    w, info = _ssh_slot(manager, spec, task, monkeypatch, probe="running")
    manager._reconcile_worker(spec, task, w, workers_mod.PARK, info)
    assert [op for op, _ in _RecordingSshMachine.ops] == ["pause"]


def test_a_released_gate_unpauses_rather_than_starting(manager, spec, task, monkeypatch):
    w, info = _ssh_slot(manager, spec, task, monkeypatch, probe="paused")
    manager._reconcile_worker(spec, task, w, workers_mod.RUN, info)
    assert [op for op, _ in _RecordingSshMachine.ops] == ["unpause"]


def test_an_operator_pause_stops_a_parked_container(manager, spec, task, monkeypatch):
    """docker stop cannot signal a frozen process, so a paused container is
    thawed before it is asked to exit cleanly."""
    w, info = _ssh_slot(manager, spec, task, monkeypatch, probe="paused")
    manager._reconcile_worker(spec, task, w, workers_mod.STOP, info)
    assert [op for op, _ in _RecordingSshMachine.ops] == ["unpause", "stop"]


def test_a_parked_local_worker_is_simply_stopped(manager, spec, task, monkeypatch):
    """A local worker restarts in about a second; there is nothing to save."""
    stopped = []
    monkeypatch.setattr(WorkerManager, "_stop_local", lambda self, spec, task, w: stopped.append(1))
    w = manager.add_local(spec, task, "generate", threads=1)
    w.desired_state = "running"
    manager._reconcile_worker(spec, task, w, workers_mod.PARK, {"observed_running": True})
    assert stopped == [1]


def test_status_requests_read_observations_not_make_them(manager, spec, task, monkeypatch):
    """The browser polls every few seconds; if that reached the machine, one
    slow host would stall every request the dashboard serves."""
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    monkeypatch.setattr(_FakeSshMachine, "state", "running")
    manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)

    (info,) = manager.worker_status(spec, task)
    assert info["ssh_probe"] == "unknown"  # nothing has observed it yet
    assert info["state"] == "checking"

    (info,) = manager.worker_status(spec, task, observe=True)
    assert info["ssh_probe"] == "running"

    # The machine changes under us; a plain status request keeps reporting the
    # last observation rather than going and looking.
    monkeypatch.setattr(_FakeSshMachine, "state", "stopped")
    (info,) = manager.worker_status(spec, task)
    assert info["ssh_probe"] == "running"


def test_observations_expire(manager, spec, task, monkeypatch):
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    monkeypatch.setattr(_FakeSshMachine, "state", "running")
    manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    manager.worker_status(spec, task, observe=True)

    monkeypatch.setattr(_FakeSshMachine, "state", "stopped")
    now = time.time() + workers_mod.OBSERVATION_TTL_SECONDS + 1
    monkeypatch.setattr(workers_mod.time, "time", lambda: now)
    (info,) = manager.worker_status(spec, task, observe=True)
    assert info["ssh_probe"] == "stopped"


def test_deploy_refuses_a_worker_image_that_cannot_load_this_tree(manager, spec, task, monkeypatch):
    """Deploying a bundle onto an image whose libraries are older than the
    ones it was compiled against just crash-loops every worker."""
    monkeypatch.setattr(
        workers_mod.runtime_abi, "read_record",
        lambda root: {"image": "w", "versions": {"libstdc++.so.6": "libstdc++.so.6.0.33"}},
    )  # fmt: skip
    monkeypatch.setattr(
        workers_mod.runtime_abi, "local_versions",
        lambda: {"libstdc++.so.6": "libstdc++.so.6.0.35"},
    )  # fmt: skip
    with pytest.raises(AssertionError, match="build_and_push_worker_image"):
        manager.deploy(spec, task)


def test_deploy_says_nothing_about_an_image_no_push_has_described(manager, spec, task, monkeypatch):
    """The record only exists once a push has written one; its absence is not
    evidence of a stale image."""
    monkeypatch.setattr(workers_mod.runtime_abi, "read_record", lambda root: None)
    # _cloud is the fixture's tripwire: reaching it means the check passed.
    with pytest.raises(AssertionError, match="launched compute"):
        manager.deploy(spec, task)
