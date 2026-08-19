"""Unit tests for the WorkerManager's slot lifecycle policy: slots are added
paused and nothing launches at add time, a cloud slot's pod is created on its
first start, and the cloud display-state mapping covers the no-pod-yet case.

Every path that would launch compute or touch cloud credentials is patched to
fail, so a regression back toward launch-on-add breaks loudly.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest
from cloud.ssh_machine import SshMachineError
from scribblez import workloads
from scribblez.dashboard import db, tasks
from scribblez.dashboard import workers as workers_mod
from scribblez.dashboard.workers import (
    WorkerManager,
    _cloud_state,
    _container_name,
    _resource_record_fields,
    _worker_resources,
)
from scribblez.paths import TagPaths
from scribblez.workloads.position_eval import SPEC as POSITION_EVAL_SPEC
from scribblez.workloads.position_eval import PositionEvalParams
from scripts.cloud_fleet import CpuResources, GpuResources

# The fixture below replaces the launch paths with _fail; a test that wants to
# exercise one for real puts this back.
_REAL_RUN_SSH_CONTAINER = WorkerManager._run_ssh_container


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
    exit_reason = "exit 1: something went wrong"

    def __init__(self, host):
        self.host = host

    def container_state(self, name: str) -> str:
        return self.state

    def container_exit(self, name: str) -> str:
        return self.exit_reason


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


# Enough of a credentials object for the container-creation path.
_CREDS = SimpleNamespace(registry=SimpleNamespace(worker_image="repo/worker"), r2=None)


class _RecordingSshMachine(_FakeSshMachine):
    """Records the container operations reconcile performs."""

    state = "stopped"
    ops: list = []

    def start_container(self, name):
        self.ops.append(("start", name))

    def pull_image(self, image):
        self.ops.append(("pull", image))

    def run_container(self, name, image, env, *, gpus=False):
        self.ops.append(("run", "gpu" if gpus else name))

    def copy_from_container(self, name, path, dest):
        self.ops.append(("copy", path))
        return False

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
    w.undelivered = 0
    stopped = {"observed_running": False, "ssh_probe": "stopped"}
    manager._reconcile_worker(spec, task, w, workers_mod.RUN, stopped)
    assert [op for op, _ in _RecordingSshMachine.ops] == ["copy", "remove"]  # swept first
    assert recreated == [w.worker_id]


def test_a_container_still_holding_output_is_drained_before_it_is_replaced(
    manager, spec, task, monkeypatch
):
    """Replacing a container discards whatever it never handed over. Starting
    it is what lets the next passes collect from it -- a pull needs it
    running -- and the replacement waits for a collection to report zero."""
    monkeypatch.setattr(WorkerManager, "_run_ssh_container", _fail)
    w = _stopped_ssh_slot(manager, spec, task, monkeypatch, slot_bundle="b1", task_bundle="b2")
    w.undelivered = 900
    stopped = {"observed_running": False, "ssh_probe": "stopped"}
    manager._reconcile_worker(spec, task, w, workers_mod.RUN, stopped)
    assert [op for op, _ in _RecordingSshMachine.ops] == ["start"]


def test_a_container_of_unknown_backlog_is_not_replaced_either(manager, spec, task, monkeypatch):
    """No collection has ever reported on it, and its record predates the
    count -- which is not the same as knowing it is empty."""
    monkeypatch.setattr(WorkerManager, "_run_ssh_container", _fail)
    w = _stopped_ssh_slot(manager, spec, task, monkeypatch, slot_bundle="b1", task_bundle="b2")
    w.undelivered = None
    stopped = {"observed_running": False, "ssh_probe": "stopped"}
    manager._reconcile_worker(spec, task, w, workers_mod.RUN, stopped)
    assert [op for op, _ in _RecordingSshMachine.ops] == ["start"]


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


def test_a_stopped_container_reports_why(manager, spec, task, monkeypatch):
    """A worker that cannot start (a bundle its image cannot load, say) used
    to read as a bare "exited" flickering back to "running" -- the reason was
    only in `docker logs` on the machine."""
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    monkeypatch.setattr(_FakeSshMachine, "state", "stopped")
    monkeypatch.setattr(_FakeSshMachine, "exit_reason", "exit 1: GLIBCXX_3.4.35 not found")
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    w.desired_state, w.launched = "running", True

    (info,) = manager.worker_status(spec, task, observe=True)
    assert info["state"] == "exited"
    assert info["exit_reason"] == "exit 1: GLIBCXX_3.4.35 not found"

    # It comes up: the reason is stale and goes away.
    monkeypatch.setattr(_FakeSshMachine, "state", "running")
    manager._probes.clear()
    (info,) = manager.worker_status(spec, task, observe=True)
    assert "exit_reason" not in info


def test_restarts_back_off_while_a_container_keeps_dying(manager, spec, task, monkeypatch):
    """Restarting a container that dies instantly does not fix it; the pass
    runs every few seconds and should not spend an ssh round trip each time."""
    monkeypatch.setattr(workers_mod, "SshMachine", _RecordingSshMachine)
    monkeypatch.setattr(_RecordingSshMachine, "state", "stopped")
    _RecordingSshMachine.ops = []
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    w.desired_state, w.launched, w.bundle_id = "running", True, "b1"
    task.bundle_id = "b1"
    stopped = {"observed_running": False, "ssh_probe": "stopped"}

    for _ in range(3):
        manager._reconcile_worker(spec, task, w, workers_mod.RUN, stopped)
    assert len(_RecordingSshMachine.ops) == 1  # the later passes backed off

    # Time enough for the second attempt, which is allowed.
    key = workers_mod._key(spec, task.tag, w.worker_id)
    attempts, next_at = manager._restarts[key]
    monkeypatch.setattr(workers_mod.time, "time", lambda: next_at + 1)
    manager._reconcile_worker(spec, task, w, workers_mod.RUN, stopped)
    assert len(_RecordingSshMachine.ops) == 2
    assert manager._restarts[key][0] == attempts + 1


def test_a_worker_that_comes_up_clears_its_backoff(manager, spec, task, monkeypatch):
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    monkeypatch.setattr(_FakeSshMachine, "state", "running")
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    w.launched = True
    key = workers_mod._key(spec, task.tag, w.worker_id)
    manager._restarts[key] = (5, time.time() + 300)
    manager.worker_status(spec, task, observe=True)
    assert key not in manager._restarts


def test_resuming_a_parked_worker_is_not_a_restart(manager, spec, task, monkeypatch):
    """Unpausing must not count toward the crash-loop backoff -- a gate parks
    and releases a generator many times an hour."""
    monkeypatch.setattr(workers_mod, "SshMachine", _RecordingSshMachine)
    _RecordingSshMachine.ops = []
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    w.desired_state, w.launched, w.bundle_id = "running", True, "b1"
    task.bundle_id = "b1"
    paused = {"observed_running": False, "ssh_probe": "paused"}
    for _ in range(4):
        manager._reconcile_worker(spec, task, w, workers_mod.RUN, paused)
    assert [op for op, _ in _RecordingSshMachine.ops] == ["unpause"] * 4
    assert manager._restarts == {}


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


def test_collecting_records_what_the_container_still_holds(manager, spec, task, monkeypatch):
    """This wiring is what stands between a redeployed container and having
    its undelivered output thrown away, so it is worth pinning down."""
    from cloud.ssh_transfer import PullResult

    monkeypatch.setattr(
        workers_mod,
        "pull_results",
        lambda *a, **k: PullResult(pulled=["data/staging/c1.slog"], remaining=87),
    )
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    manager._collect_ssh(spec, task, w)
    assert w.undelivered == 87
    assert tasks.load_task(spec, "t").worker(w.worker_id).undelivered == 87  # survives a restart


def test_status_reports_the_backlog_including_none_left(manager, spec, task, monkeypatch):
    """Zero and "never collected from" are different answers to "is it safe to
    remove this?", so the status distinguishes them."""
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)

    (info,) = manager.worker_status(spec, task)
    assert info["undelivered"] is None  # nothing has collected from it yet

    w.undelivered = 340
    (info,) = manager.worker_status(spec, task)
    assert info["undelivered"] == 340

    w.undelivered = 0
    (info,) = manager.worker_status(spec, task)
    assert info["undelivered"] == 0


def test_a_drained_container_on_an_old_bundle_is_stopped_so_it_can_be_replaced(
    manager, spec, task, monkeypatch
):
    """Replacement only acts on a container that is down, and draining one
    leaves it running -- so without this the slot ran on the old bundle
    forever, which is what pinning a task to a bundle exists to prevent."""
    w = _stopped_ssh_slot(manager, spec, task, monkeypatch, slot_bundle="b1", task_bundle="b2")
    w.undelivered = 0
    running = {"observed_running": True, "ssh_probe": "running"}
    manager._reconcile_worker(spec, task, w, workers_mod.RUN, running)
    assert [op for op, _ in _RecordingSshMachine.ops] == ["stop"]


def test_a_drained_container_on_the_task_bundle_is_left_alone(manager, spec, task, monkeypatch):
    """Only a bundle it has moved past justifies stopping a working worker."""
    w = _stopped_ssh_slot(manager, spec, task, monkeypatch, slot_bundle="b1", task_bundle="b1")
    w.undelivered = 0
    running = {"observed_running": True, "ssh_probe": "running"}
    manager._reconcile_worker(spec, task, w, workers_mod.RUN, running)
    assert _RecordingSshMachine.ops == []


def test_a_container_that_never_came_up_is_replaced_by_a_redeploy(manager, spec, task, monkeypatch):
    """A crash-looping container has nothing to hand over -- it is created
    holding nothing -- and a redeploy is often exactly the fix for whatever it
    is crashing on, so it must not be restarted forever instead."""
    recreated = []
    monkeypatch.setattr(
        WorkerManager,
        "_run_ssh_container",
        lambda self, spec, task, w: recreated.append(w.worker_id),
    )
    w = _stopped_ssh_slot(manager, spec, task, monkeypatch, slot_bundle="b1", task_bundle="b2")
    w.undelivered = 0  # what _run_ssh_container records when it creates one
    stopped = {"observed_running": False, "ssh_probe": "stopped"}
    manager._reconcile_worker(spec, task, w, workers_mod.RUN, stopped)
    assert [op for op, _ in _RecordingSshMachine.ops] == ["copy", "remove"]  # swept first
    assert recreated == [w.worker_id]


def test_an_unreachable_machine_is_not_acted_on(manager, spec, task, monkeypatch):
    """Its container may well be running; nothing here can tell."""
    monkeypatch.setattr(WorkerManager, "_run_ssh_container", _fail)
    w = _stopped_ssh_slot(manager, spec, task, monkeypatch, slot_bundle="b1", task_bundle="b1")
    for probe in ("unreachable", "unknown"):
        manager._reconcile_worker(
            spec, task, w, workers_mod.RUN, {"observed_running": False, "ssh_probe": probe}
        )
    assert _RecordingSshMachine.ops == []


def test_a_reused_worker_id_does_not_inherit_a_backlog(manager, spec, task, monkeypatch):
    """Slot ids are reused once freed, and a count that outlived its slot
    would block the new container's replacement and misreport what removing it
    would discard."""
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    monkeypatch.setattr(_FakeSshMachine, "state", "missing")
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    w.undelivered = 900
    manager.remove_worker(spec, task, w.worker_id)

    fresh = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    assert fresh.worker_id == w.worker_id  # the id came back
    assert fresh.undelivered is None


def test_creating_a_container_records_that_it_holds_nothing(manager, spec, task, monkeypatch):
    """What makes a container that never came up replaceable rather than
    restarted forever: it is known empty from the moment it exists."""
    monkeypatch.setattr(workers_mod, "SshMachine", _RecordingSshMachine)
    monkeypatch.setattr(WorkerManager, "_run_ssh_container", _REAL_RUN_SSH_CONTAINER)
    monkeypatch.setattr(WorkerManager, "_cloud", lambda self: (_CREDS, None))
    monkeypatch.setattr(WorkerManager, "task_bundle_id", lambda self, spec, task: "b1")
    monkeypatch.setattr(workers_mod, "bundle_worker_env", lambda *a, **k: {})
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    manager._run_ssh_container(spec, task, w)
    assert (w.launched, w.undelivered, w.bundle_id) == (True, 0, "b1")


class _DispatchSpec:
    """position_eval's match_eval role with its tag tree in the test's tmp dir:
    what reconcile and the dispatch tick ask of a spec."""

    name = "position_eval"
    scheduler = ""
    params_cls = PositionEvalParams
    roles = (POSITION_EVAL_SPEC.role("match_eval"),)

    def __init__(self, mount_root):
        self._mount_root = mount_root

    def paths(self, tag: str) -> TagPaths:
        return TagPaths(tag, "position_eval", mount_root=self._mount_root)

    def role(self, name: str):
        return POSITION_EVAL_SPEC.role(name)


def test_reconcile_dispatches_to_running_slots_only(manager, tmp_path, monkeypatch):
    """The controller's half of a dispatch-driven role runs from the reconcile
    pass, against the slots that are really running: a model pushed to a slot
    whose worker is down would sit there unplayed while the ledger read as a
    match in flight."""
    spec = _DispatchSpec(tmp_path)
    task = tasks.TaskRecord(workload=spec.name, tag="t", params={}, created_at=0.0)
    paths = spec.paths("t")
    paths.onnx_dir.mkdir(parents=True)
    paths.onnx_path(10).write_bytes(b"onnx")
    db.connect(paths.dashboard_db).close()

    running = manager.add_local(spec, task, "match_eval", threads=1)
    task.workers.append(
        tasks.WorkerRecord(
            worker_id="local-1", role="match_eval", kind="local", desired_state="paused"
        )
    )
    monkeypatch.setattr(manager, "_all_tasks", lambda: iter([(spec, task)]))
    monkeypatch.setattr(WorkerManager, "_local_alive", lambda self, spec, task, w: w is running)
    monkeypatch.setattr(WorkerManager, "_reconcile_worker", lambda *a, **k: None)
    asyncio.run(manager.reconcile())

    assert [p.name for p in paths.match_inbox_dir(running.worker_id).iterdir()] == [
        paths.onnx_path(10).name
    ]
    assert not paths.match_inbox_dir("local-1").exists()


def test_a_slot_being_created_reads_as_starting_not_exited(manager, spec, task, monkeypatch):
    """What the operator sees between pressing Start and the container
    existing -- on a machine taking a new image, minutes of it."""
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    _FakeSshMachine.state = "missing"
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    w.desired_state, w.launched = "running", False
    (info,) = manager.worker_status(spec, task, observe=True)
    assert info["state"] == "starting"


def test_a_creation_that_failed_says_why(manager, spec, task, monkeypatch):
    """A slot whose container cannot be created reads `starting` forever,
    since nothing of it exists to have exited. The reason is the only account
    of that, so a probe finding no container must not wipe it."""
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    monkeypatch.setattr(WorkerManager, "_run_ssh_container", _REAL_RUN_SSH_CONTAINER)
    monkeypatch.setattr(WorkerManager, "_cloud", lambda self: (_CREDS, None))
    monkeypatch.setattr(WorkerManager, "task_bundle_id", lambda self, spec, task: "b1")
    monkeypatch.setattr(workers_mod, "bundle_worker_env", lambda *a, **k: {})
    _FakeSshMachine.state = "missing"

    def refuse(self, image):
        raise SshMachineError("user@laptop: pulling scribblez failed: no basic auth credentials")

    monkeypatch.setattr(_FakeSshMachine, "pull_image", refuse, raising=False)
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    w.desired_state = "running"
    with pytest.raises(SshMachineError):
        manager._run_ssh_container(spec, task, w)

    (info,) = manager.worker_status(spec, task, observe=True)
    assert info["state"] == "starting"
    assert "no basic auth credentials" in info["exit_reason"]


def test_a_gpu_role_gets_the_machines_gpus(manager, monkeypatch):
    """A container for a GPU role is run with --gpus: the match-eval worker
    plays a neural agent, which needs the machine's GPU (and the worker image
    carries the TensorRT builder for it)."""
    spec = workloads.get("position_eval")
    task = tasks.TaskRecord(workload=spec.name, tag="t", params={}, created_at=0.0)
    monkeypatch.setattr(workers_mod, "SshMachine", _RecordingSshMachine)
    monkeypatch.setattr(WorkerManager, "_run_ssh_container", _REAL_RUN_SSH_CONTAINER)
    monkeypatch.setattr(WorkerManager, "_cloud", lambda self: (_CREDS, None))
    monkeypatch.setattr(WorkerManager, "task_bundle_id", lambda self, spec, task: "b1")
    monkeypatch.setattr(workers_mod, "bundle_worker_env", lambda *a, **k: {})
    _RecordingSshMachine.ops = []

    generator = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    manager._run_ssh_container(spec, task, generator)
    matcher = manager.add_ssh(spec, task, "match_eval", host="user@laptop", threads=None)
    manager._run_ssh_container(spec, task, matcher)

    assert [arg for op, arg in _RecordingSshMachine.ops if op == "run"] == [
        _container_name(spec, "t", generator.worker_id),
        "gpu",
    ]


def test_a_running_container_still_holding_output_is_not_stopped_by_a_redeploy(
    manager, spec, task, monkeypatch
):
    """Stopping it costs the cycle in flight and strands the rest, and the
    redeploy has all the time in the world: it waits for the drain."""
    running = {"observed_running": True, "ssh_probe": "running"}
    for holding in (900, None):
        w = _stopped_ssh_slot(manager, spec, task, monkeypatch, slot_bundle="b1", task_bundle="b2")
        w.undelivered = holding
        manager._reconcile_worker(spec, task, w, workers_mod.RUN, running)
        assert _RecordingSshMachine.ops == [], f"stopped a container holding {holding}"
        manager.remove_worker(spec, task, w.worker_id)


def test_a_failed_collection_gives_up_the_count_rather_than_keeping_a_stale_one(
    manager, spec, task, monkeypatch
):
    """A count only means something while collection is working. Left
    standing, a container whose pulls keep failing would report the last
    number it managed -- or the zero it was created with -- and be replaced or
    removed as drained while it filled up."""

    def boom(*args, **kwargs):
        raise SshMachineError("read timed out")

    monkeypatch.setattr(workers_mod, "pull_results", boom)
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    w.undelivered = 0  # what creation recorded
    with pytest.raises(SshMachineError):
        manager._collect_ssh(spec, task, w)
    assert w.undelivered is None


def test_a_container_is_not_destroyed_when_its_final_sweep_fails(manager, spec, task, monkeypatch):
    """The sweep is the last chance at what the worker flushed while stopping.
    Leaving the slot on the old bundle is recoverable; the output is not."""
    monkeypatch.setattr(WorkerManager, "_run_ssh_container", _fail)

    def boom(self, name, path, dest):
        raise SshMachineError("connection reset")

    monkeypatch.setattr(_RecordingSshMachine, "copy_from_container", boom)
    w = _stopped_ssh_slot(manager, spec, task, monkeypatch, slot_bundle="b1", task_bundle="b2")
    w.undelivered = 0
    stopped = {"observed_running": False, "ssh_probe": "stopped"}
    with pytest.raises(SshMachineError):
        manager._reconcile_worker(spec, task, w, workers_mod.RUN, stopped)
    assert _RecordingSshMachine.ops == []  # nothing removed


def test_a_crashlooping_container_is_restarted_not_destroyed(manager, spec, task, monkeypatch):
    """It ran long enough to hold a backlog and then stopped staying up, so
    nothing can collect from it and nothing can measure what it holds.
    Recovering the slot means discarding that, which is the operator's call --
    the workers table shows the reason it is down and Remove says what would
    go. An automatic replacement here would be this PR's own bug, wearing the
    disguise of a repair."""
    monkeypatch.setattr(WorkerManager, "_run_ssh_container", _fail)
    for bundle in ("b1", "b2"):  # its own bundle, and one the task moved past
        w = _stopped_ssh_slot(
            manager, spec, task, monkeypatch, slot_bundle="b1", task_bundle=bundle
        )
        w.undelivered = 40
        key = workers_mod._key(spec, task.tag, w.worker_id)
        stopped = {"observed_running": False, "ssh_probe": "stopped"}
        for attempt in range(6):
            manager._restarts[key] = (attempt, 0.0)  # backoff elapsed
            manager._reconcile_worker(spec, task, w, workers_mod.RUN, stopped)
        assert {op for op, _ in _RecordingSshMachine.ops} == {"start"}
        manager.remove_worker(spec, task, w.worker_id)


def test_an_unreachable_machine_gives_up_a_recorded_zero(manager, spec, task, monkeypatch):
    """The count says the container was empty when someone last looked. A
    machine off the network for hours has a worker that went on filling it the
    whole time, and believing the old zero would authorise sweeping -- in one
    unbounded copy -- exactly the backlog this PR exists to bound."""
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    monkeypatch.setattr(_FakeSshMachine, "state", "unreachable")
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    w.launched, w.undelivered = True, 0

    manager.worker_status(spec, task, observe=True)
    assert w.undelivered is None

    # A positive count only ever refuses, so it is kept.
    w.undelivered = 40
    manager.worker_status(spec, task, observe=True)
    assert w.undelivered == 40


def test_a_restart_does_not_inherit_a_zero_it_cannot_vouch_for(manager, spec, task, monkeypatch):
    """Whatever the workers did while no dashboard was watching is precisely
    what a zero from before the restart does not cover. A positive count
    survives: forgetting that is how a backlog gets thrown away."""
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    drained = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    drained.undelivered = 0
    holding = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    holding.undelivered = 900
    tasks.save_task(spec, task)

    # The walk is pinned to this task: _all_tasks otherwise lists the real
    # mount, and a test that reads it passes or fails on what happens to be
    # there.
    monkeypatch.setattr(tasks, "list_tags", lambda spec: [{"tag": "t", "has_task": True}])
    fresh = WorkerManager()  # the dashboard comes back up
    monkeypatch.setattr(fresh, "_cloud", _fail)
    reloaded = next(t for _, t in fresh._all_tasks() if t.tag == "t")
    assert reloaded.worker(drained.worker_id).undelivered is None
    assert reloaded.worker(holding.worker_id).undelivered == 900
    # Vetted once, then left alone: a count this process recorded stands.
    reloaded.worker(drained.worker_id).undelivered = 0
    tasks.save_task(spec, reloaded)
    again = next(t for _, t in fresh._all_tasks() if t.tag == "t")
    assert again.worker(drained.worker_id).undelivered == 0
