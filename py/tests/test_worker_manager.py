"""Unit tests for the dashboard's WorkerManager: the slot lifecycle and its reconcile
pass, registered and rented machines, bundle deployment and container replacement,
output collection and backlog accounting, and the bucket legs of a task whose
trainer runs off the controller.

Every path that would launch compute or touch cloud credentials is patched to
fail, so anything that launches where it should not breaks loudly.
"""

import asyncio
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cloud.credentials import R2Credentials, RegistryConfig
from cloud.providers.base import Instance, MachineType, ProviderError
from cloud.ssh_machine import SshMachineError
from scribblez import workloads
from scribblez.dashboard import db, tasks
from scribblez.dashboard import workers as workers_mod
from scribblez.dashboard.workers import (
    WorkerManager,
    _container_name,
    _key,
)
from scribblez.paths import DEFAULT_MOUNT_ROOT, TagPaths
from scribblez.workloads.position_eval import SPEC as POSITION_EVAL_SPEC
from scribblez.workloads.position_eval import PositionEvalParams
from sim_world import SyncExecutor

# The fixture below replaces the launch paths with _fail; a test that wants to
# exercise one for real puts this back.
_REAL_RUN_SSH_CONTAINER = WorkerManager._run_ssh_container
_REAL_SPAWN_LOCAL = WorkerManager._spawn_local
_REAL_ENSURE_SYNC = WorkerManager._ensure_sync


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
    for name in ("_spawn_local", "_run_ssh_container", "_creds"):
        monkeypatch.setattr(WorkerManager, name, _fail)
    return WorkerManager(tmp_path)


@pytest.fixture
def tags_root(manager, spec) -> Path:
    """The workload's tag dirs under the manager's root: deleting a tag
    removes that tree for real."""
    return spec.tags_root(manager.mount_root)


def test_add_local_is_paused_and_not_spawned(manager, spec, task):
    w = manager.add_local(spec, task, "generate", threads=2)
    assert w.desired_state == "paused"
    assert w.pid is None
    assert manager.tasks.load(spec, "t").workers[0].desired_state == "paused"


def test_add_ssh_is_paused_and_runs_no_container(manager, spec, task):
    w = manager.add_ssh(spec, task, "generate", host="user@h", threads=None)
    assert w.desired_state == "paused"
    assert not w.launched


class _FakeSshMachine:
    """SshMachine stand-in whose container probe always returns `state`."""

    state = "unreachable"
    exit_reason = "exit 1: something went wrong"
    machine_state = "up"
    built: list[tuple] = []  # every (host, identity_file, known_hosts_file) constructed

    def __init__(self, host, identity_file=None, known_hosts_file=None):
        self.host = host
        _FakeSshMachine.built.append((host, identity_file, known_hosts_file))

    def probe(self, ready_file=None) -> str:
        return self.machine_state

    def container_state(self, name: str) -> str:
        return self.state

    def container_exit(self, name: str) -> str:
        return self.exit_reason

    # The arch survey a bare host or registered machine gets at its first
    # slot start; a test that cares records the pull and asks something else.
    def pull_image(self, image):
        pass

    def detect_arch(self, image) -> str:
        return "znver3"


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


def test_reconcile_contains_per_slot_failures(manager, spec, task, monkeypatch):
    """One slot's persistently failing enforcement (a machine that keeps
    refusing the container) must not abort the pass: later slots still get
    their tick."""
    _fake_ssh(monkeypatch, state="missing")
    added = [manager.add_ssh(spec, task, "generate", host="u@h", threads=None) for _ in range(2)]
    for w in added:
        w.desired_state = "running"
    attempted = []

    def boom(self, spec, task, w, intent, probe):
        attempted.append(w.worker_id)
        raise RuntimeError("docker: no such image")

    monkeypatch.setattr(WorkerManager, "_reconcile_ssh", boom)
    monkeypatch.setattr(manager, "all_tasks", lambda: iter([(spec, task)]))
    asyncio.run(manager.reconcile())
    assert attempted == [w.worker_id for w in added]


def test_a_failed_listing_does_not_stop_the_pass(rented, manager, spec, task, monkeypatch):
    """A failed provider listing (expired credentials, say) leaves a rented
    tag's machines unobserved for the pass, but the pass goes on: that tag's
    other slots and every later tag's still get their tick. A slot on one of
    the unobserved machines gets none, since nothing says the machine is up."""
    provider, _ = rented

    def expired():
        raise ProviderError("expired")

    monkeypatch.setattr(provider, "describe", expired)
    manager._instances = ({}, 0.0)  # the next observation must list afresh
    later = tasks.TaskRecord(workload=spec.name, tag="later", params={}, created_at=0.0)
    for t in (task, later):
        manager.add_local(spec, t, "generate", threads=1)
    manager.add_ssh(spec, task, "generate", machine="m1", threads=1, check_gpu=False)
    ticked = []
    monkeypatch.setattr(
        WorkerManager,
        "_reconcile_worker",
        lambda self, spec, t, w, intent, info: ticked.append((t.tag, w.worker_id)),
    )
    monkeypatch.setattr(manager, "all_tasks", lambda: iter([(spec, task), (spec, later)]))
    asyncio.run(manager.reconcile())
    assert ticked == [("t", "local-0"), ("later", "local-0")]


def test_a_pause_survives_a_pass_that_looked_at_the_task_before_it(manager, spec, task):
    """The failure mode: "Pause all" lands in the handler's copy of the task,
    while the reconcile pass, having loaded its own copy before the click, saves
    "running" back over it; the next pass then starts a fresh process. With one
    shared record per task, the pass saves the pause it did not know about."""
    w = manager.add_local(spec, task, "generate", threads=1)
    w.desired_state = "running"
    manager.tasks.save(spec, task)

    in_pass = manager.tasks.load(spec, "t")
    clicked = manager.tasks.load(spec, "t")
    manager.set_worker_state(spec, clicked, w.worker_id, run=False)
    manager.worker_status(spec, in_pass, observe=True)  # the pass's later save

    assert manager.tasks.load(spec, "t").worker(w.worker_id).desired_state == "paused"
    stored = tasks.TaskStore(manager.mount_root).load(spec, "t")  # as a restart reads it
    assert stored.worker(w.worker_id).desired_state == "paused"


def test_a_status_poll_writes_nothing(manager, spec, task):
    """The browser polls every second; those reads must not race the pass's
    saves. Only the observing pass saves."""
    manager.add_local(spec, task, "generate", threads=1)
    before = manager.control.version
    manager.worker_status(spec, task)
    assert manager.control.version == before
    manager.worker_status(spec, task, observe=True)
    assert manager.control.version != before


def test_reconcile_skips_a_slot_removed_between_its_steps(manager, spec, task, monkeypatch):
    """A Remove clicked while the pass was observing runs between its steps
    (same serialized thread). The slot it removed is neither enforced nor a
    reason for the pass to fall over."""
    kept, gone = (manager.add_local(spec, task, "generate", threads=1) for _ in range(2))
    real_status = WorkerManager.worker_status

    def status_then_remove(self, spec, task, *, observe=False):
        out = real_status(self, spec, task, observe=observe)
        task.workers.remove(gone)  # the handler, between this step and the next
        return out

    enforced = []
    monkeypatch.setattr(WorkerManager, "worker_status", status_then_remove)
    monkeypatch.setattr(
        WorkerManager, "_reconcile_worker", lambda self, spec, task, w, *a: enforced.append(w)
    )
    monkeypatch.setattr(manager, "all_tasks", lambda: iter([(spec, task)]))
    asyncio.run(manager.reconcile())
    assert enforced == [kept]


def test_redeploy_builds_off_the_blocking_thread_then_pins(manager, spec, task, monkeypatch):
    """A build takes minutes, and on the blocking thread it would hold up every
    Pause and Remove clicked meanwhile. It runs on its own thread; only the
    repin is serialized."""
    seen = {}

    def build(self, archs):
        seen["thread"] = threading.current_thread().name
        seen["archs"] = archs
        seen["blocking_free"] = manager._blocking.submit(lambda: True).result(timeout=5)
        return SimpleNamespace(bundle_id="b2", source_hash="h2", archs=archs)

    monkeypatch.setattr(WorkerManager, "_build_bundle", build)
    monkeypatch.setattr(WorkerManager, "_creds", lambda self: _CREDS)
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    assert asyncio.run(manager.redeploy(spec, task)) == "b2"
    assert seen["archs"] == ["znver3"]  # what the laptop reported
    assert seen["thread"].startswith("scz-build")
    assert seen["blocking_free"]
    assert manager.tasks.load(spec, "t").bundle_id == "b2"


# ---- machines ----------------------------------------------------------------


def _fake_ssh(monkeypatch, state="unreachable", machine_state="up"):
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    monkeypatch.setattr(_FakeSshMachine, "state", state)
    monkeypatch.setattr(_FakeSshMachine, "machine_state", machine_state)
    monkeypatch.setattr(_FakeSshMachine, "built", [])


def test_a_slot_on_a_machine_dials_with_the_machines_key(manager, spec, task, monkeypatch):
    """The slot names the machine; the machine record carries the address
    and the key material every link is built from."""
    _fake_ssh(monkeypatch, state="running")
    manager.add_machine(spec, task, "m1", "ubuntu@1.2.3.4", identity_file="/k/m1.pem")
    w = manager.add_ssh(spec, task, "generate", machine="m1", threads=4)
    assert w.host is None and w.machine == "m1"
    (info,) = manager.worker_status(spec, task, observe=True)
    assert info["host"] == "ubuntu@1.2.3.4"
    assert info["ssh"] == "ssh ubuntu@1.2.3.4"
    assert ("ubuntu@1.2.3.4", "/k/m1.pem", None) in _FakeSshMachine.built
    assert manager.tasks.load(spec, "t").worker(w.worker_id).machine == "m1"


def test_an_ssh_slot_names_exactly_one_of_host_and_machine(manager, spec, task):
    with pytest.raises(AssertionError, match="host or a machine"):
        manager.add_ssh(spec, task, "generate", threads=None)
    with pytest.raises(KeyError, match="no machine"):
        manager.add_ssh(spec, task, "generate", machine="nope", threads=None)


class _GpuRoles:
    """Two non-singleton GPU roles that allow ssh slots: what the machine fit
    check sees, without the singleton rule speaking first."""

    name = "position_eval"
    gpu_need = ""  # no measured figures: the GPU-fit check stays out of these tests
    _roles = {
        r.name: r
        for r in (
            replace(POSITION_EVAL_SPEC.role("match_eval"), singleton=False),
            replace(POSITION_EVAL_SPEC.role("match_eval"), name="match_b", singleton=False),
        )
    }

    def role(self, name: str):
        return self._roles[name]

    def paths(self, tag: str, mount_root: Path) -> TagPaths:
        return TagPaths(tag, self.name, mount_root)


def test_a_gpu_role_is_refused_only_on_a_machine_without_a_gpu(manager, monkeypatch):
    """Refused at add time by the machine's known shape, not at `docker run`
    on the remote after a deploy. GPU slots share a machine's GPUs (every
    container runs under --gpus all); an unknown count is not checked."""
    spec = _GpuRoles()
    task = tasks.TaskRecord(workload=spec.name, tag="t", params={}, created_at=0.0)
    manager.add_machine(spec, task, "cpu", "u@h", gpu_count=0)
    manager.add_machine(spec, task, "gpu1", "u@g", gpu_count=1)
    manager.add_machine(spec, task, "unknown", "u@x")
    with pytest.raises(AssertionError, match="has no GPU for role 'match_eval'"):
        manager.add_ssh(spec, task, "match_eval", machine="cpu", threads=None)
    manager.add_ssh(spec, task, "match_eval", machine="gpu1", threads=None)
    manager.add_ssh(spec, task, "match_b", machine="gpu1", threads=None)  # shares the one GPU
    manager.add_ssh(spec, task, "match_b", machine="unknown", threads=None)


def test_machine_status_is_observed_by_the_pass_and_read_by_everyone_else(
    manager, spec, task, monkeypatch
):
    _fake_ssh(monkeypatch, machine_state="no docker")
    manager.add_machine(spec, task, "m1", "u@h")
    (m,) = manager.machine_status(spec, task)
    assert m["state"] == "checking"  # no pass has reached it
    (m,) = manager.machine_status(spec, task, observe=True)
    assert m["state"] == "no docker"
    monkeypatch.setattr(_FakeSshMachine, "machine_state", "up")
    (m,) = manager.machine_status(spec, task)
    assert m["state"] == "no docker"  # a read, not a probe


def test_removing_a_machine_removes_its_slots_under_the_slot_rule(manager, spec, task, monkeypatch):
    _fake_ssh(monkeypatch, state="running")
    manager.add_machine(spec, task, "m1", "u@h")
    w = manager.add_ssh(spec, task, "generate", machine="m1", threads=None)
    with pytest.raises(AssertionError, match="running; pause it first"):
        manager.remove_machine(spec, task, "m1")
    assert task.machines and task.workers  # nothing half-done
    monkeypatch.setattr(_FakeSshMachine, "state", "missing")
    manager._probes.clear()
    manager.remove_machine(spec, task, "m1")
    assert task.machines == [] and task.workers == []
    assert manager.tasks.load(spec, "t").find(w.worker_id) is None


def test_reconcile_leaves_slots_on_a_machine_that_is_not_up_alone(manager, spec, task, monkeypatch):
    _fake_ssh(monkeypatch, state="missing", machine_state="unreachable")
    manager.add_machine(spec, task, "m1", "u@h")
    w = manager.add_ssh(spec, task, "generate", machine="m1", threads=None)
    w.desired_state = "running"
    enforced = []
    monkeypatch.setattr(
        WorkerManager, "_reconcile_worker", lambda self, spec, task, w, *a: enforced.append(w)
    )
    monkeypatch.setattr(manager, "all_tasks", lambda: iter([(spec, task)]))
    asyncio.run(manager.reconcile())
    assert enforced == []
    monkeypatch.setattr(_FakeSshMachine, "machine_state", "up")
    manager._machine_probes.clear()
    manager._ssh_down.clear()  # past the negative cache
    asyncio.run(manager.reconcile())
    assert enforced == [w]


# ---- rented machines ---------------------------------------------------------


class _FakeProvider:
    """A provider whose instances the test moves through their states."""

    name = "aws"
    ssh_user = "ubuntu"
    identity_file = "/k/scribblez.pem"
    ready_file = "/var/lib/scribblez/ready"
    region = "us-east-1"

    def __init__(self):
        self.instances: dict[str, Instance] = {}
        self.calls: list[tuple] = []
        self.refuse: ProviderError | None = None

    def account(self):
        return "AWS account 1 as user scribblez, us-east-1"

    def catalog(self):
        return [
            MachineType("g6.2xlarge", 8, 1, "L4", "znver3", 1.0),
            MachineType("c7a.4xlarge", 16, 0, "", "znver4", 0.5),
        ]

    def prepare(self):
        pass

    def spot_prices(self):
        return {"g6.2xlarge": 0.4}

    def launch(self, request):
        if self.refuse is not None:
            raise self.refuse
        inst = Instance(
            id=f"i-{len(self.instances) + 1}", state="pending", type_id=request.type_id,
            owner=request.owner, address=None, launched_at=time.time(),
            spot=request.spot, cost_per_hr=0.4 if request.spot else None,
        )  # fmt: skip
        self.instances[inst.id] = inst
        self.calls.append(("launch", request.type_id))
        return inst

    def describe(self):
        return dict(self.instances)

    def stop(self, instance_id):
        self.calls.append(("stop", instance_id))
        self.instances[instance_id].state = "stopping"

    def start(self, instance_id):
        if self.refuse is not None:
            raise self.refuse
        self.calls.append(("start", instance_id))
        self.instances[instance_id].state = "pending"

    def terminate(self, instance_id):
        self.calls.append(("terminate", instance_id))
        self.instances[instance_id].state = "terminated"

    def refusal(self, error, type_id):
        return f"refused {type_id}: {error}"


@pytest.fixture
def rented(manager, spec, task, monkeypatch, tmp_path):
    """A task with one rented machine, its instance pending, the provider
    faked, ssh faked `up`."""
    provider = _FakeProvider()
    monkeypatch.setattr(manager, "_provider", lambda: provider)
    monkeypatch.setattr(workers_mod, "MACHINES_DIR", tmp_path / "machines")
    _fake_ssh(monkeypatch, state="missing", machine_state="up")
    m = manager.rent_machine(spec, task, "m1", "g6.2xlarge")
    return provider, m


def _observe(manager, spec, task):
    """A pass's machine step: fresh listing and probes. The slot probes the
    machine step reads are the previous pass's; observing the slots first
    stands in for that pass."""
    manager._probes.clear()
    manager.worker_status(spec, task, observe=True)
    manager._instances = ({}, 0.0)
    manager._machine_probes.clear()
    manager._ssh_down.clear()
    status = manager.machine_status(spec, task, observe=True)
    manager._reconcile_machines(spec, task, status)
    return status


def test_renting_records_the_instance_and_its_key_material(rented, manager, spec, task, tmp_path):
    provider, m = rented
    assert provider.calls == [("launch", "g6.2xlarge")]
    assert m.instance_id == "i-1" and m.instance_type == "g6.2xlarge"
    assert m.host == "ubuntu@pending-i-1" and m.identity_file == "/k/scribblez.pem"
    known_hosts = tmp_path / "machines" / spec.name / "t" / "m1" / "known_hosts"
    assert m.known_hosts_file == str(known_hosts) and known_hosts.read_text() == ""
    assert m.gpu_count == 1 and m.arch == "znver3" and m.cost_per_hr == 1.0
    assert manager.tasks.load(spec, "t").machine("m1").instance_id == "i-1"
    assert provider.instances["i-1"].owner == f"{spec.name}/t/m1"


def test_a_dispatch_role_on_a_rented_machine_keeps_its_data_local(rented, manager, task):
    """Dispatch reads results only from the slot's filesystem, so a rented
    match-eval slot keeps its data there for the ssh pull while the machine's
    generator delivers its chunks through the bucket. Records are local for
    both: every ssh slot's are collected."""
    match = manager.add_ssh(_GpuRoles(), task, "match_eval", machine="m1", threads=None)
    assert manager._slot_data_sink(_GpuRoles(), task, match) == "local"
    gen = tasks.WorkerRecord(
        worker_id="g", role="generate", kind="ssh", desired_state="paused", machine="m1"
    )
    assert manager._slot_data_sink(POSITION_EVAL_SPEC, task, gen) == "r2"
    assert manager._slot_records_sink(POSITION_EVAL_SPEC, task, gen) == "local"


def _dispatch_task(train_finished: bool) -> tasks.TaskRecord:
    """A position_eval task with a trainer slot and a running match slot."""
    task = tasks.TaskRecord(workload="position_eval", tag="t", params={}, created_at=0.0)
    train = tasks.WorkerRecord(worker_id="tr", role="train", kind="local", desired_state="paused")
    train.finished = train_finished
    match = tasks.WorkerRecord(
        worker_id="me", role="match_eval", kind="local", desired_state="running"
    )
    task.workers = [train, match]
    return task


@pytest.mark.parametrize(
    ("outstanding", "train_finished", "finishes"),
    [(False, True, True), (True, True, False), (False, False, False)],
)
def test_dispatch_finishes_its_role_once_the_trainer_is_done_and_nothing_is_owed(
    manager, monkeypatch, outstanding, train_finished, finishes
):
    """An idle match slot would otherwise keep its rented machine up forever."""
    task = _dispatch_task(train_finished)
    monkeypatch.setattr(workers_mod.workloads, "resolve", lambda path: lambda *a: outstanding)
    role = POSITION_EVAL_SPEC.role("match_eval")
    manager._dispatch_role(POSITION_EVAL_SPEC, task, role, [])
    match = task.worker("me")
    assert match.finished == finishes
    assert match.desired_state == ("paused" if finishes else "running")


def test_a_rented_machine_without_a_name_gets_one(rented, manager, spec, task):
    provider, m = rented
    second = manager.rent_machine(spec, task, "", "c7a.4xlarge")
    third = manager.rent_machine(spec, task, "", "c7a.4xlarge")
    assert (second.name, third.name) == ("aws-1", "aws-2")
    assert manager.rental_offer()["account"].startswith("AWS account 1")
    assert [t["id"] for t in manager.rental_offer()["types"]] == ["g6.2xlarge", "c7a.4xlarge"]


def test_a_spot_machine_records_its_market_rate(rented, manager, spec, task):
    provider, m = rented
    s = manager.rent_machine(spec, task, "", "g6.2xlarge", spot=True)
    assert s.spot and s.cost_per_hr == 0.4 and provider.instances[s.instance_id].spot
    assert not m.spot and m.cost_per_hr == 1.0
    offer = manager.rental_offer()
    assert offer["spot_prices"] == {"g6.2xlarge": 0.4}
    (_, info) = manager.machine_status(spec, task)
    assert info["spot"] and info["cost_per_hr"] == 0.4


def test_a_refused_launch_reaches_the_form_and_records_nothing(
    manager, spec, task, monkeypatch, tmp_path
):
    provider = _FakeProvider()
    provider.refuse = ProviderError("VcpuLimitExceeded", "quota")
    monkeypatch.setattr(manager, "_provider", lambda: provider)
    monkeypatch.setattr(workers_mod, "MACHINES_DIR", tmp_path / "machines")
    with pytest.raises(AssertionError, match="refused g6.2xlarge: VcpuLimitExceeded"):
        manager.rent_machine(spec, task, "m1", "g6.2xlarge")
    assert task.machines == []
    assert not (tmp_path / "machines" / "m1").exists()  # nothing left behind


def test_a_rented_machine_reads_launching_then_preparing_then_up(
    rented, manager, spec, task, monkeypatch
):
    provider, m = rented
    (info,) = _observe(manager, spec, task)
    assert info["state"] == "launching"  # pending: not probed
    provider.instances["i-1"].state = "running"
    provider.instances["i-1"].address = "1.2.3.4"
    monkeypatch.setattr(_FakeSshMachine, "machine_state", "preparing")
    (info,) = _observe(manager, spec, task)
    assert info["state"] == "preparing" and m.host == "ubuntu@1.2.3.4"
    monkeypatch.setattr(_FakeSshMachine, "machine_state", "up")
    (info,) = _observe(manager, spec, task)
    assert info["state"] == "up"
    # Running but not answering: still booting inside the grace, unreachable after.
    monkeypatch.setattr(_FakeSshMachine, "machine_state", "unreachable")
    (info,) = _observe(manager, spec, task)
    assert info["state"] == "launching"
    m.launched_at = time.time() - workers_mod.BOOT_GRACE_SECONDS - 1
    (info,) = _observe(manager, spec, task)
    assert info["state"] == "unreachable"


def test_the_rented_probe_asks_for_the_ready_marker(rented, manager, spec, task, monkeypatch):
    provider, m = rented
    provider.instances["i-1"].state = "running"
    asked = []
    monkeypatch.setattr(
        _FakeSshMachine, "probe", lambda self, ready_file=None: asked.append(ready_file) or "up"
    )
    _observe(manager, spec, task)
    assert asked == ["/var/lib/scribblez/ready"]


def test_spend_accrues_while_the_instance_bills(rented, manager, spec, task, monkeypatch):
    provider, m = rented
    m.observed_at = time.time() - 3600
    _observe(manager, spec, task)  # pending: billing
    assert 0.99 < m.spend < 1.01
    provider.instances["i-1"].state = "stopped"
    _observe(manager, spec, task)
    m.observed_at = time.time() - 3600
    _observe(manager, spec, task)  # stopped: not billing
    assert m.spend < 1.02
    assert manager.tasks.load(spec, "t").machine("m1").spend == m.spend


def test_stopping_task_rentals_pauses_their_slots_and_skips_the_idle_wait(
    rented, manager, spec, task, monkeypatch
):
    """Stop all cloud spending: a task's own rented machine has its slots
    paused and stops as soon as they are down, not IDLE_STOP_SECONDS later."""
    provider, m = rented
    provider.instances["i-1"].state = "running"
    w = manager.add_ssh(spec, task, "generate", machine="m1", threads=None)
    w.desired_state = "running"
    monkeypatch.setattr(manager, "all_tasks", lambda: [(spec, task)])
    manager._instance_index(True)
    assert manager.stop_task_rentals() == [f"{spec.name}/t/m1"]
    assert w.desired_state == "paused"
    monkeypatch.setattr(_FakeSshMachine, "state", "stopped")
    _observe(manager, spec, task)
    assert ("stop", "i-1") in provider.calls


def test_an_idle_machine_is_stopped_after_the_timeout(rented, manager, spec, task, monkeypatch):
    """Nothing running on it: an operator-paused or finished slot with an
    exited container. A slot that wants running and has no container is a
    pending start, so the machine stays."""
    provider, m = rented
    provider.instances["i-1"].state = "running"
    w = manager.add_ssh(spec, task, "generate", machine="m1", threads=None)
    w.desired_state = "running"
    _observe(manager, spec, task)
    assert manager._idle_since == {}  # a pending start
    w.desired_state = "paused"
    monkeypatch.setattr(_FakeSshMachine, "state", "stopped")
    _observe(manager, spec, task)
    key = workers_mod._machine_key(spec, "t", "m1")
    assert key in manager._idle_since and ("stop", "i-1") not in provider.calls
    manager._idle_since[key] -= workers_mod.IDLE_STOP_SECONDS + 1
    _observe(manager, spec, task)
    assert ("stop", "i-1") in provider.calls
    (info,) = _observe(manager, spec, task)
    assert info["state"] == "stopping"


def test_a_gated_slot_keeps_the_machine_up(rented, manager, spec, task, monkeypatch):
    """A gate is expected to lift; a role that is done is finished instead."""
    provider, m = rented
    provider.instances["i-1"].state = "running"
    w = manager.add_ssh(spec, task, "generate", machine="m1", threads=None)
    w.desired_state = "running"
    task.gates["generate"] = "ahead of trainer"
    monkeypatch.setattr(_FakeSshMachine, "state", "paused")
    manager.worker_status(spec, task, observe=True)
    _observe(manager, spec, task)
    assert manager._idle_since == {}


def test_finishing_a_role_releases_its_machine(rented, manager, spec, task, monkeypatch):
    """The scheduler's finish hook: the gated slot becomes finished, its gate
    goes, and once its container has stopped the machine idles toward a stop."""
    provider, m = rented
    provider.instances["i-1"].state = "running"
    w = manager.add_ssh(spec, task, "generate", machine="m1", threads=None)
    w.desired_state = "running"
    task.gates["generate"] = "target reached"
    # The bucket hooks load cloud credentials; this test needs only finish.
    monkeypatch.setattr(manager, "_make_mirror", lambda spec, task: None)
    monkeypatch.setattr(manager, "_make_publish", lambda spec, task: None)
    manager._scheduler_hooks(spec, task).finish("generate")
    assert (w.desired_state, w.finished, task.gates) == ("paused", True, {})
    assert manager.tasks.load(spec, "t").worker(w.worker_id).finished  # saved
    monkeypatch.setattr(_FakeSshMachine, "state", "stopped")
    manager.worker_status(spec, task, observe=True)
    _observe(manager, spec, task)
    assert workers_mod._machine_key(spec, "t", "m1") in manager._idle_since


def test_finish_role_leaves_other_roles_and_paused_slots():
    task = tasks.TaskRecord(workload="kill_test", tag="t", params={}, created_at=0.0)
    gen = tasks.WorkerRecord(worker_id="a", role="generate", kind="ssh", desired_state="running")
    paused = tasks.WorkerRecord(worker_id="b", role="generate", kind="ssh", desired_state="paused")
    train = tasks.WorkerRecord(worker_id="c", role="train", kind="ssh", desired_state="running")
    task.workers = [gen, paused, train]
    assert workers_mod._finish_role(task, "generate")
    assert gen.finished and not paused.finished and not train.finished
    assert train.desired_state == "running"
    assert not workers_mod._finish_role(task, "generate")  # nothing left to change


def test_a_running_container_keeps_the_machine_up(rented, manager, spec, task, monkeypatch):
    provider, m = rented
    provider.instances["i-1"].state = "running"
    w = manager.add_ssh(spec, task, "generate", machine="m1", threads=None)
    w.desired_state = "paused"
    monkeypatch.setattr(_FakeSshMachine, "state", "running")
    manager.worker_status(spec, task, observe=True)  # remembers the container running
    _observe(manager, spec, task)
    assert manager._idle_since == {}


def test_a_stopped_machine_is_started_when_a_slot_wants_running(
    rented, manager, spec, task, monkeypatch
):
    provider, m = rented
    provider.instances["i-1"].state = "stopped"
    w = manager.add_ssh(spec, task, "generate", machine="m1", threads=None)
    (info,) = _observe(manager, spec, task)
    assert info["state"] == "stopped" and ("start", "i-1") not in provider.calls
    w.desired_state = "running"
    _observe(manager, spec, task)
    assert ("start", "i-1") in provider.calls
    (info,) = _observe(manager, spec, task)
    assert info["state"] == "launching"


def test_a_refused_start_is_shown_and_backed_off(rented, manager, spec, task, monkeypatch):
    provider, m = rented
    provider.instances["i-1"].state = "stopped"
    w = manager.add_ssh(spec, task, "generate", machine="m1", threads=None)
    w.desired_state = "running"
    provider.refuse = ProviderError("InsufficientInstanceCapacity", "none")
    with pytest.raises(ProviderError):
        _observe(manager, spec, task)
    (info,) = manager.machine_status(spec, task)
    assert info["exit_reason"] == "refused g6.2xlarge: InsufficientInstanceCapacity"
    assert info["retry_in_s"] > 0
    _observe(manager, spec, task)  # inside the backoff: not tried again
    assert provider.calls.count(("start", "i-1")) == 0


def test_removing_a_rented_machine_terminates_it_and_retires_its_spend(rented, manager, spec, task):
    provider, m = rented
    m.spend = 2.5
    manager.remove_machine(spec, task, "m1")
    assert ("terminate", "i-1") in provider.calls
    assert task.machines == [] and task.retired_spend == pytest.approx(2.5, abs=1e-3)


def test_the_listing_follows_a_rent_and_a_remove_without_waiting_for_a_pass(
    rented, manager, spec, task, monkeypatch
):
    """Between an action and the next pass's listing, a status poll reads
    the cached listing: a just-rented machine must not read `gone` for
    want of its instance there, and a just-removed one's instance must not
    show on the burn strip as a running orphan."""
    provider, m = rented
    monkeypatch.setattr(manager, "all_tasks", lambda: iter([(spec, task)]))
    (info,) = manager.machine_status(spec, task)  # no observation: the listing as cached
    assert info["state"] == "launching"
    provider.instances["i-1"].state = "running"
    manager.remove_machine(spec, task, "m1")
    assert manager.fleet()["instances"] == []


def test_a_refused_terminate_keeps_the_machine_and_says_why(
    rented, manager, spec, task, monkeypatch
):
    """A Remove the provider refuses (a policy missing an action) reaches
    the operator as the refusal sentence, and the record stays: the instance
    is still there, still billing, still the task's to remove."""
    provider, m = rented

    def refuse(instance_id):
        raise ProviderError(
            "UnauthorizedOperation", "not authorized: ec2:CancelSpotInstanceRequests"
        )

    monkeypatch.setattr(provider, "terminate", refuse)
    with pytest.raises(AssertionError, match="refused g6.2xlarge: UnauthorizedOperation"):
        manager.remove_machine(spec, task, "m1")
    assert task.machines == [m]


def test_a_gone_machines_slots_are_removable_outright(rented, manager, spec, task, monkeypatch):
    """The instance is terminated (by a spot interruption, or in the console):
    its containers went with its disk. The unreachable rule would refuse
    forever; instead the slots go, and the provider is not asked to
    terminate again."""
    provider, m = rented
    w = manager.add_ssh(spec, task, "generate", machine="m1", threads=None)
    w.launched = True
    provider.instances["i-1"].state = "terminated"
    monkeypatch.setattr(_FakeSshMachine, "state", "unreachable")
    (info,) = _observe(manager, spec, task)
    assert info["state"] == "gone"
    manager.remove_machine(spec, task, "m1")
    assert task.workers == [] and task.machines == []
    assert ("terminate", "i-1") not in provider.calls


def test_orphans_are_our_instances_no_task_names(rented, manager, spec, task, monkeypatch):
    provider, m = rented
    provider.instances["i-7"] = Instance(
        id="i-7", state="running", type_id="c7a.4xlarge", owner="position_eval/old/g",
        address=None, launched_at=time.time() - 120,
    )  # fmt: skip
    provider.instances["i-8"] = Instance(
        id="i-8",
        state="terminated",
        type_id="c7a.4xlarge",
        owner=None,
        address=None,
        launched_at=None,
    )
    monkeypatch.setattr(manager, "all_tasks", lambda: iter([(spec, task)]))
    manager._instances = ({}, 0.0)
    orphans = manager.orphans(observe=True)
    assert [o["instance_id"] for o in orphans] == ["i-7"]
    assert orphans[0]["owner"] == "position_eval/old/g" and orphans[0]["uptime_s"] >= 120
    manager.terminate_orphan("i-7")
    assert ("terminate", "i-7") in provider.calls


def test_fleet_adds_up_what_bills_whoever_tracks_it(rented, manager, spec, task, monkeypatch):
    """The burn strip's view: every instance tagged ours, the task's own and
    an orphan alike, each at its rate -- the catalog's for on-demand, its own
    for spot -- with only pending/running ones in the sum."""
    provider, m = rented
    provider.instances["i-7"] = Instance(
        id="i-7", state="running", type_id="c7a.4xlarge", owner="position_eval/old/g",
        address=None, launched_at=time.time() - 120,
    )  # fmt: skip
    provider.instances["i-8"] = Instance(
        id="i-8", state="stopped", type_id="g6.2xlarge", owner="position_eval/old/s",
        address=None, launched_at=None, spot=True, cost_per_hr=0.4,
    )  # fmt: skip
    m.cost_per_hr = 0.37  # the record's rate, as a spot launch leaves it, wins over the catalog's
    provider.instances["i-9"] = Instance(
        id="i-9", state="terminated", type_id="c7a.4xlarge", owner=None, address=None,
        launched_at=None,
    )  # fmt: skip
    monkeypatch.setattr(manager, "all_tasks", lambda: iter([(spec, task)]))
    manager._instances = ({}, 0.0)
    manager._list_fleet()
    fleet = manager.fleet()
    assert fleet["error"] is None and fleet["observed_at"] is not None
    by_id = {r["instance_id"]: r for r in fleet["instances"]}
    assert set(by_id) == {"i-1", "i-7", "i-8"}
    assert by_id["i-1"]["tracked"] and by_id["i-1"]["cost_per_hr"] == 0.37
    assert not by_id["i-7"]["tracked"] and by_id["i-7"]["cost_per_hr"] == 0.5
    assert by_id["i-7"]["uptime_s"] >= 120
    assert by_id["i-8"]["spot"] and by_id["i-8"]["cost_per_hr"] == 0.4
    assert fleet["burn_per_hr"] == pytest.approx(0.87)


def test_fleet_step_lists_without_rented_machines_and_keeps_a_failure(manager, monkeypatch):
    """The step runs whether or not any task names a machine (a task.json
    that lost its machines must not hide their instances), and a listing
    that fails leaves its reason for the strip rather than a stale zero."""
    provider = _FakeProvider()
    provider.instances["i-3"] = Instance(
        id="i-3", state="running", type_id="c7a.4xlarge", owner="position_eval/lost/g",
        address=None, launched_at=time.time(),
    )  # fmt: skip
    monkeypatch.setattr(manager, "_provider", lambda: provider)
    monkeypatch.setattr(manager, "all_tasks", lambda: iter([]))
    manager._list_fleet()
    fleet = manager.fleet()
    assert [r["instance_id"] for r in fleet["instances"]] == ["i-3"]
    assert fleet["burn_per_hr"] == 0.5 and not fleet["instances"][0]["tracked"]

    def broken():
        raise ProviderError("RequestExpired", "the clock is off")

    monkeypatch.setattr(manager, "_provider", broken)
    manager._instances = ({}, 0.0)
    manager._list_fleet()
    fleet = manager.fleet()
    assert fleet["error"] == "RequestExpired" and fleet["observed_at"] is None


# ---- finished slots ----------------------------------------------------------


def test_an_ssh_worker_that_exits_zero_is_finished_not_restarted(manager, spec, task, monkeypatch):
    """The trainer at max_rows exits 0; restarting it every backoff period
    forever would keep its machine from ever idling."""
    _fake_ssh(monkeypatch, state="stopped")
    monkeypatch.setattr(_FakeSshMachine, "exit_reason", "exit 0: Stopped at 1000 rows")
    w = manager.add_ssh(spec, task, "generate", host="u@h", threads=None)
    w.desired_state, w.launched = "running", True
    (info,) = manager.worker_status(spec, task, observe=True)
    assert info["state"] == "finished"
    assert w.desired_state == "paused" and w.finished
    assert manager.tasks.load(spec, "t").worker(w.worker_id).finished
    # Start is the way back: it clears the mark and runs the slot again.
    monkeypatch.setattr(WorkerManager, "_run_ssh_container", lambda *a: None)
    manager._probes.clear()
    monkeypatch.setattr(_FakeSshMachine, "state", "missing")
    manager.set_worker_state(spec, task, w.worker_id, run=True)
    assert not w.finished and w.desired_state == "running"


def test_an_ssh_worker_that_died_is_exited_and_restarted(manager, spec, task, monkeypatch):
    _fake_ssh(monkeypatch, state="stopped")
    monkeypatch.setattr(_FakeSshMachine, "exit_reason", "exit 143: SIGTERM: drained")
    w = manager.add_ssh(spec, task, "generate", host="u@h", threads=None)
    w.desired_state, w.launched = "running", True
    (info,) = manager.worker_status(spec, task, observe=True)
    assert info["state"] == "exited"
    assert w.desired_state == "running" and not w.finished


def test_a_local_child_that_exits_zero_is_finished(manager, spec, task, monkeypatch):
    w = manager.add_local(spec, task, "generate", threads=1)
    w.desired_state = "running"
    manager._local[_key(spec, "t", w.worker_id)] = SimpleNamespace(poll=lambda: 0, returncode=0)
    (info,) = manager.worker_status(spec, task, observe=True)
    assert info["state"] == "finished"
    assert w.desired_state == "paused"


def test_the_pass_acts_on_the_liveness_it_observed(manager, spec, task, monkeypatch):
    """A local worker seen alive, then gone before the pass acts, is left for
    the next pass: judged gone with its exit unread, a worker that finished
    would be respawned."""
    w = manager.add_local(spec, task, "generate", threads=1)
    w.desired_state = "running"
    manager._local[_key(spec, "t", w.worker_id)] = SimpleNamespace(
        poll=lambda: None, returncode=None
    )
    checks = iter([True])  # alive when observed, gone on any later look
    monkeypatch.setattr(workers_mod, "worker_pid_alive", lambda *a: next(checks, False))
    (info,) = manager.worker_status(spec, task, observe=True)
    assert info["observed_running"] is True


def _starting_ssh_slot(manager, spec, task, monkeypatch):
    monkeypatch.setattr(workers_mod, "SshMachine", _RecordingSshMachine)
    _RecordingSshMachine.ops = []
    monkeypatch.setattr(WorkerManager, "_run_ssh_container", _REAL_RUN_SSH_CONTAINER)
    monkeypatch.setattr(WorkerManager, "_creds", lambda self: _CREDS)
    monkeypatch.setattr(workers_mod, "bundle_worker_env", lambda *a, **k: {})
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    w.desired_state = "running"
    return w


def test_first_remote_worker_builds_the_bundle_off_the_blocking_thread(
    manager, spec, task, monkeypatch
):
    """Deployment is not an operator step: the task pins a bundle the first
    time a remote worker starts, and later workers join it. The build takes
    minutes, so the slot start that needs it kicks it off on the build thread
    and waits (`starting`, with the reason on its row); a later pass pins and
    starts. Waiting is not a failed attempt, so no backoff accrues."""
    release = threading.Event()
    seen = {}

    def build(self, archs):
        seen["thread"] = threading.current_thread().name
        seen["archs"] = archs
        release.wait(timeout=5)
        return SimpleNamespace(bundle_id="b1", source_hash="h1", archs=archs)

    monkeypatch.setattr(WorkerManager, "_build_bundle", build)
    w = _starting_ssh_slot(manager, spec, task, monkeypatch)
    missing = {"observed_running": False, "ssh_probe": "missing"}
    key = _key(spec, "t", w.worker_id)

    manager._reconcile_worker(spec, task, w, workers_mod.RUN, missing)
    # The arch survey pulled the image to ask its compiler; nothing ran.
    assert [op for op, _ in _RecordingSshMachine.ops] == ["pull"]
    assert manager._exits[key] == "building the worker bundle for znver3"
    assert manager._blocking.submit(lambda: True).result(timeout=5)  # answers during the build
    assert key not in manager._restarts
    manager._reconcile_worker(spec, task, w, workers_mod.RUN, missing)  # still building: same
    assert [op for op, _ in _RecordingSshMachine.ops] == ["pull"] and task.bundle_id is None

    release.set()
    manager._pending_builds[f"{spec.name}/t"].result(timeout=5)
    manager._reconcile_worker(spec, task, w, workers_mod.RUN, missing)
    assert seen["thread"].startswith("scz-build")
    assert seen["archs"] == ["znver3"]  # the laptop's, asked once and kept
    assert (w.arch, task.bundle_archs) == ("znver3", ["znver3"])
    assert [op for op, _ in _RecordingSshMachine.ops] == ["pull", "pull", "run"]
    assert (task.bundle_id, w.bundle_id, w.launched) == ("b1", "b1", True)
    assert key not in manager._exits
    # A second slot joins the pinned bundle without another build.
    monkeypatch.setattr(WorkerManager, "_build_bundle", _fail)
    w2 = _starting_ssh_slot(manager, spec, task, monkeypatch)
    manager._reconcile_worker(spec, task, w2, workers_mod.RUN, missing)
    assert w2.bundle_id == "b1"


def test_a_slot_whose_arch_the_bundle_lacks_rebuilds_with_it_added(
    manager, spec, task, monkeypatch
):
    """A task pinned to a znver3 bundle gains a znver4 machine: its slot does
    not fall back to a generic build -- the same tree is built again for both
    archs and the task repinned. A slot on an arch the bundle has joins it
    with no build at all."""
    builds = []
    release = threading.Event()

    def build(self, archs):
        builds.append(archs)
        release.wait(timeout=5)
        return SimpleNamespace(bundle_id=f"b-{'+'.join(archs)}", source_hash="h", archs=archs)

    monkeypatch.setattr(WorkerManager, "_build_bundle", build)
    task.bundle_id, task.bundle_source_hash, task.bundle_archs = "b-znver3", "h", ["znver3"]
    task.machines.append(
        tasks.MachineRecord(name="m4", provider="aws", host="ubuntu@x", arch="znver4")
    )
    w4 = _starting_ssh_slot(manager, spec, task, monkeypatch)
    w4.host, w4.machine = None, "m4"
    w3 = _starting_ssh_slot(manager, spec, task, monkeypatch)  # a laptop: znver3
    missing = {"observed_running": False, "ssh_probe": "missing"}

    manager._reconcile_worker(spec, task, w3, workers_mod.RUN, missing)
    assert builds == [] and w3.bundle_id == "b-znver3"

    manager._reconcile_worker(spec, task, w4, workers_mod.RUN, missing)  # kicks off the build
    release.set()
    manager._pending_builds[f"{spec.name}/t"].result(timeout=5)
    manager._reconcile_worker(spec, task, w4, workers_mod.RUN, missing)
    assert builds == [["znver3", "znver4"]]
    assert (task.bundle_id, task.bundle_archs) == ("b-znver3+znver4", ["znver3", "znver4"])
    assert w4.bundle_id == "b-znver3+znver4"


def test_a_failed_bundle_build_is_the_slots_exit_reason(manager, spec, task, monkeypatch):
    release = threading.Event()

    def build(self, archs):
        release.wait(timeout=5)
        raise RuntimeError("make exited 2")

    monkeypatch.setattr(WorkerManager, "_build_bundle", build)
    w = _starting_ssh_slot(manager, spec, task, monkeypatch)
    missing = {"observed_running": False, "ssh_probe": "missing"}
    key = _key(spec, "t", w.worker_id)
    manager._reconcile_worker(spec, task, w, workers_mod.RUN, missing)  # kicks the build off
    release.set()
    manager._pending_builds[f"{spec.name}/t"].exception(timeout=5)
    with pytest.raises(RuntimeError, match="make exited 2"):
        manager._reconcile_worker(spec, task, w, workers_mod.RUN, missing)
    assert manager._exits[key] == "bundle build failed: make exited 2"
    assert task.bundle_id is None  # the next allowed attempt builds again


def test_bundle_drift_compares_tree_against_pinned_bundle(manager, spec, task, monkeypatch):
    monkeypatch.setattr(workers_mod, "source_hash", lambda archs, cache=None: "now")
    assert not manager.bundle_drift(task)  # nothing pinned yet

    task.bundle_source_hash, task.bundle_archs = "now", ["x86-64"]
    assert not manager.bundle_drift(task)

    task.bundle_source_hash = "then"
    assert manager.bundle_drift(task)

    # An unbuilt arch is not evidence of drift, only absence of evidence.
    monkeypatch.setattr(workers_mod, "source_hash", lambda archs, cache=None: None)
    assert not manager.bundle_drift(task)


# Enough of a credentials object for the container-creation path.
_CREDS = SimpleNamespace(registry=RegistryConfig(worker_image="repo/worker"), r2=None)


class _RecordingSshMachine(_FakeSshMachine):
    """Records the container operations reconcile performs."""

    state = "stopped"
    ops: list = []

    def start_container(self, name):
        self.ops.append(("start", name))

    def pull_image(self, image):
        self.ops.append(("pull", image))

    def run_container(self, name, image, env, *, gpus=False, volume=None):
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


def test_a_pass_after_a_pause_observes_the_container_afresh(manager, spec, task, monkeypatch):
    """The pass cadence matches the probe TTL, so the pass after a pause can
    fall inside it; reading the probe from before the pause, it would pause
    again, and docker refuses with "already paused"."""
    w, _ = _ssh_slot(manager, spec, task, monkeypatch, probe="running")
    monkeypatch.setattr(_RecordingSshMachine, "state", "running")
    probe = manager._refresh_probe(spec, task, w)
    manager._reconcile_worker(
        spec, task, w, workers_mod.PARK, {"observed_running": True, "ssh_probe": probe}
    )
    monkeypatch.setattr(_RecordingSshMachine, "state", "paused")
    assert manager._refresh_probe(spec, task, w) == "paused"


def test_an_operator_pause_stops_a_parked_container(manager, spec, task, monkeypatch):
    """docker stop cannot signal a frozen process, so a paused container is
    thawed before it is asked to exit cleanly."""
    w, info = _ssh_slot(manager, spec, task, monkeypatch, probe="paused")
    manager._reconcile_worker(spec, task, w, workers_mod.STOP, info)
    assert [op for op, _ in _RecordingSshMachine.ops] == ["unpause", "stop"]


def _collectable_ssh_slot(manager, spec, task, monkeypatch, *, probe: str):
    """A running ssh slot wired to a machine whose re-probe returns `probe`,
    for exercising a collection's response to a pull that raced a stop."""
    monkeypatch.setattr(workers_mod, "SshMachine", _RecordingSshMachine)
    monkeypatch.setattr(_RecordingSshMachine, "state", probe)
    _RecordingSshMachine.ops = []
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    w.desired_state, w.launched, w.undelivered = "running", True, 7
    return w


def _raise_stopped(*args, **kwargs):
    raise SshMachineError("user@laptop: Error response from daemon: container c is not running")


def test_a_collect_that_raced_a_stop_is_not_an_error(manager, spec, task, monkeypatch):
    """An operator pausing a slot stops its container synchronously, so a
    pull can reach a container that stopped since the pass probed it running.
    A re-probe that no longer finds it running makes that the benign race it
    is: the pull swallows it, leaving the count unknown for the flushed output
    the next start (or a replacement's sweep) will take."""
    monkeypatch.setattr(workers_mod, "pull_results", _raise_stopped)
    w = _collectable_ssh_slot(manager, spec, task, monkeypatch, probe="stopped")
    assert manager._pull_ssh(spec, task, w) is None  # does not raise
    _collect(manager, spec, task, w)
    assert w.undelivered is None


def test_a_collect_failure_on_a_running_container_propagates(manager, spec, task, monkeypatch):
    """A pull that failed while the container is still up (a slow link hitting
    the transfer timeout, say) is a real failure, not the stop race -- the
    caller must see it."""
    monkeypatch.setattr(workers_mod, "pull_results", _raise_stopped)
    w = _collectable_ssh_slot(manager, spec, task, monkeypatch, probe="running")
    with pytest.raises(SshMachineError):
        manager._pull_ssh(spec, task, w)


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
    """A worker that cannot start (say, a bundle its image cannot load) shows
    why on its row. Otherwise it reads as a bare "exited" flickering back to
    "running", with the reason only in `docker logs` on the machine."""
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
    stale = {"image": "w", "versions": {"libstdc++.so.6": "libstdc++.so.6.0.33"}}
    monkeypatch.setattr(workers_mod.runtime_abi, "read_records", lambda root: {"engine": stale})
    monkeypatch.setattr(
        workers_mod.runtime_abi, "local_versions",
        lambda: {"libstdc++.so.6": "libstdc++.so.6.0.35"},
    )  # fmt: skip
    with pytest.raises(AssertionError, match="build_and_push_worker_image"):
        manager.deploy(spec, task)


def test_deploy_says_nothing_about_an_image_no_push_has_described(manager, spec, task, monkeypatch):
    """The record only exists once a push has written one; its absence is not
    evidence of a stale image."""
    monkeypatch.setattr(workers_mod.runtime_abi, "read_records", lambda root: None)
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
    _collect(manager, spec, task, w)
    assert w.undelivered == 87
    assert manager.tasks.load(spec, "t").worker(w.worker_id).undelivered == 87  # survives a restart


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
    leaves it running -- so without this the slot would run on the old bundle
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


def test_deleting_a_tag_takes_its_idle_slots_with_it(manager, spec, task, tags_root):
    """The task record is what tracks a slot; releasing the slots is part of
    deleting the tag, not a chore to be done first."""
    for _ in range(2):
        manager.add_local(spec, task, "generate", threads=1)
    assert (tags_root / "t" / "task.json").is_file()

    manager.delete_task(spec, "t")
    assert not (tags_root / "t").exists()


def test_deleting_a_tag_releases_its_ssh_container(manager, spec, task, tags_root, monkeypatch):
    monkeypatch.setattr(workers_mod, "SshMachine", _RecordingSshMachine)
    _RecordingSshMachine.ops = []
    manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)

    name = _container_name(spec, "t", task.workers[0].worker_id)
    manager.delete_task(spec, "t")
    assert _RecordingSshMachine.ops == [("remove", name)]


def test_deleting_a_tag_refuses_while_a_worker_is_meant_to_run(manager, spec, task, tags_root):
    """Including a gated one: the scheduler resumes it on its own, so it is
    the operator's intent that decides, not whether it happens to be parked."""
    w = manager.add_local(spec, task, "generate", threads=1)
    w.desired_state = "running"
    task.gates = {"generate": "waiting for data"}
    manager.tasks.save(spec, task)

    with pytest.raises(AssertionError, match=f"pause {w.worker_id} first"):
        manager.delete_task(spec, "t")
    assert manager.tasks.load(spec, "t").workers  # the tag survives intact


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


def test_a_reused_worker_id_does_not_inherit_a_stale_exit_reason(manager, spec, task, monkeypatch):
    """A stale exit reason or restart backoff pinned to a freed id -- or to a
    tag deleted and recreated under the same name, which reproduces the same
    key -- would narrate a fresh container's death before it has ever run."""
    monkeypatch.setattr(workers_mod, "SshMachine", _RecordingSshMachine)
    monkeypatch.setattr(_RecordingSshMachine, "state", "stopped")
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    manager.worker_status(spec, task, observe=True)
    key = workers_mod._key(spec, task.tag, w.worker_id)
    assert key in manager._exits
    manager._restarts[key] = (5, time.time() + 300)

    manager.remove_worker(spec, task, w.worker_id)
    assert key not in manager._exits
    assert key not in manager._restarts

    fresh = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    assert fresh.worker_id == w.worker_id  # the id came back
    monkeypatch.setattr(_RecordingSshMachine, "state", "missing")
    (info,) = manager.worker_status(spec, task, observe=True)
    assert "exit_reason" not in info


def test_creating_a_container_records_that_it_holds_nothing(manager, spec, task, monkeypatch):
    """What makes a container that never came up replaceable rather than
    restarted forever: it is known empty from the moment it exists."""
    monkeypatch.setattr(workers_mod, "SshMachine", _RecordingSshMachine)
    monkeypatch.setattr(WorkerManager, "_run_ssh_container", _REAL_RUN_SSH_CONTAINER)
    monkeypatch.setattr(WorkerManager, "_creds", lambda self: _CREDS)
    monkeypatch.setattr(WorkerManager, "_bundle_for_start", lambda self, spec, task, w, key: "b1")
    monkeypatch.setattr(workers_mod, "bundle_worker_env", lambda *a, **k: {})
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    manager._run_ssh_container(spec, task, w)
    assert (w.launched, w.undelivered, w.bundle_id) == (True, 0, "b1")


class _DispatchSpec:
    """position_eval's match_eval role with its tag tree in the test's tmp dir:
    what reconcile and the dispatch tick ask of a spec."""

    name = "position_eval"
    scheduler = ""
    gpu_need = ""
    params_cls = PositionEvalParams
    roles = (POSITION_EVAL_SPEC.role("match_eval"),)

    def __init__(self, mount_root):
        self._mount_root = mount_root

    def paths(self, tag: str, mount_root: Path) -> TagPaths:
        return TagPaths(tag, "position_eval", mount_root)

    def role(self, name: str):
        return POSITION_EVAL_SPEC.role(name)


def test_reconcile_dispatches_to_running_slots_only(manager, tmp_path, monkeypatch):
    """The controller's half of a dispatch-driven role runs from the reconcile
    pass, against the slots that are really running: a model pushed to a slot
    whose worker is down would sit there unplayed while the ledger read as a
    match in flight."""
    spec = _DispatchSpec(tmp_path)
    task = tasks.TaskRecord(workload=spec.name, tag="t", params={}, created_at=0.0)
    paths = manager.tasks.paths(spec, "t")
    paths.onnx_dir.mkdir(parents=True)
    paths.onnx_path(10).write_bytes(b"onnx")
    db.connect(paths.dashboard_db).close()

    running = manager.add_local(spec, task, "match_eval", threads=1)
    task.workers.append(
        tasks.WorkerRecord(
            worker_id="local-1", role="match_eval", kind="local", desired_state="paused"
        )
    )
    monkeypatch.setattr(manager, "all_tasks", lambda: iter([(spec, task)]))
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
    monkeypatch.setattr(WorkerManager, "_creds", lambda self: _CREDS)
    monkeypatch.setattr(WorkerManager, "_bundle_for_start", lambda self, spec, task, w, key: "b1")
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
    monkeypatch.setattr(WorkerManager, "_creds", lambda self: _CREDS)
    monkeypatch.setattr(WorkerManager, "_bundle_for_start", lambda self, spec, task, w, key: "b1")
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
    _collect(manager, spec, task, w)  # reported, not raised: the next pass pulls again
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
    go. An automatic replacement would silently throw that backlog away."""
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
    whole time. Believing the stale zero would authorise sweeping that whole
    backlog in one unbounded copy."""
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
    manager.tasks.save(spec, task)

    fresh = WorkerManager(manager.mount_root)  # the dashboard comes back up
    monkeypatch.setattr(fresh, "_creds", _fail)
    reloaded = fresh.tasks.load(spec, "t")
    fresh._forget_stale_counts(spec, reloaded)  # its first pass
    assert reloaded.worker(drained.worker_id).undelivered is None
    assert reloaded.worker(holding.worker_id).undelivered == 900
    # Vetted once, then left alone: a count this process recorded stands.
    reloaded.worker(drained.worker_id).undelivered = 0
    fresh._forget_stale_counts(spec, reloaded)  # its next pass
    assert reloaded.worker(drained.worker_id).undelivered == 0


# --- the bucket legs for a trainer running elsewhere --------------------------

_R2 = SimpleNamespace(bucket="b")
_BUCKET_CREDS = SimpleNamespace(registry=RegistryConfig(worker_image="repo/worker"), r2=_R2)


class _Rclone:
    """Records rclone invocations; every call succeeds."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, r2, *args, capture=False, input_text=None):
        self.calls.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def _train_task(tag="t", kinds=("ssh",)):
    """A position_eval task with an ssh generator and a train slot of the
    given kind."""
    task = tasks.TaskRecord(workload="position_eval", tag=tag, params={}, created_at=0.0)
    task.workers.append(
        tasks.WorkerRecord(
            worker_id="g", role="generate", kind="ssh", desired_state="running", host="u@h"
        )
    )
    for kind in kinds:
        task.workers.append(
            tasks.WorkerRecord(
                worker_id=f"tr-{kind}", role="train", kind=kind, desired_state="running",
                host="u@h" if kind == "ssh" else None,
            )
        )  # fmt: skip
    return task


def test_publish_copies_the_chunks_by_size_and_then_the_manifest(manager, tmp_path, monkeypatch):
    """On the upload thread, never the blocking one: a tag moving onto a
    rented trainer uploads every generation it has, and the dashboard must
    keep serving meanwhile. The hook says 'not yet' until the copy is done."""
    spec = workloads.get("position_eval")
    monkeypatch.setattr(WorkerManager, "_creds", lambda self: _BUCKET_CREDS)
    rc = _Rclone()
    monkeypatch.setattr(workers_mod, "rclone", rc)
    task = _train_task()
    publish = manager._make_publish(spec, task)
    gen_dir = manager.tasks.paths(spec, "t").generation_dir(3)
    assert publish("generations/gen_000003") is False  # started
    manager._publishing[(_key(spec, task.tag), "generations/gen_000003")].result(timeout=10)
    assert publish("generations/gen_000003") is True  # collected
    assert rc.calls == [
        ("copy", "--size-only", "--exclude", "manifest.json", str(gen_dir),
         "r2:b/position_eval/t/generations/gen_000003"),
        ("copyto", str(gen_dir / "manifest.json"),
         "r2:b/position_eval/t/generations/gen_000003/manifest.json"),
    ]  # fmt: skip
    # No bucket-delivering slot: no hook at all, as for the mirror.
    local = tasks.TaskRecord(workload="position_eval", tag="l", params={}, created_at=0.0)
    assert manager._make_publish(spec, local) is None


class _FakeWatcher:
    def __init__(self, argv):
        self.argv = argv
        self.terminated = False

    def poll(self):
        return 1 if self.terminated else None

    def terminate(self):
        self.terminated = True


def _all_ssh_task(tag="t"):
    """A position_eval task with an ssh generator and an ssh trainer and no
    cloud slot: the shape a rented machine hosts (docs/plans/cloud_machines.md).
    The bucket legs must not read it as having nothing to do."""
    task = tasks.TaskRecord(workload="position_eval", tag=tag, params={}, created_at=0.0)
    for wid, role in (("g", "generate"), ("tr", "train")):
        task.workers.append(
            tasks.WorkerRecord(
                worker_id=wid, role=role, kind="ssh", desired_state="running", host="u@h"
            )
        )
    return task


def test_an_ssh_trainer_reads_generations_from_the_bucket_and_sends_records_home(manager):
    """A legacy-plane trainer's generations come through the bucket wherever
    it runs; its records, exports and state pairs are collected over ssh,
    like the generator's chunks on the same machine."""
    spec = workloads.get("position_eval")
    task = _all_ssh_task()
    assert manager._slot_data_sink(spec, task, task.worker("g")) == "local"
    assert manager._slot_data_sink(spec, task, task.worker("tr")) == "r2"
    for w in task.workers:
        assert manager._slot_records_sink(spec, task, w) == "local"
        assert manager._collected(spec, task, w)


def test_a_generator_on_a_rented_machine_delivers_through_the_bucket(rented, manager, spec, task):
    """A rented machine's chunks go through the bucket until data moves over
    ssh too; a registered machine keeps the control link. Records come home
    over ssh from both."""
    provider, m = rented
    manager.add_machine(spec, task, "laptop", "u@h")
    on_rented = manager.add_ssh(spec, task, "generate", machine="m1", threads=None)
    on_laptop = manager.add_ssh(spec, task, "generate", machine="laptop", threads=None)
    assert manager._slot_data_sink(spec, task, on_rented) == "r2"
    assert manager._slot_data_sink(spec, task, on_laptop) == "local"
    assert manager._slot_records_sink(spec, task, on_rented) == "local"
    assert manager._has_bucket_slots(spec, task)


def test_local_workers_run_niced_in_their_own_session(manager, spec, task, monkeypatch, tmp_path):
    """A local worker yields the machine to whatever short job competes with it
    -- a bundle build for a billing fleet above all -- by running at
    LOCAL_WORKER_NICE, applied in the child before exec so its threads inherit
    it. It runs in a session of its own, so only the dashboard's SIGTERM, which
    it answers by flushing, stops it."""
    monkeypatch.setattr(WorkerManager, "_spawn_local", _REAL_SPAWN_LOCAL)
    monkeypatch.setattr(
        WorkerManager, "_log_file", lambda self, spec, tag, name: open(tmp_path / "log", "ab")
    )
    niced = []
    monkeypatch.setattr(workers_mod.os, "nice", lambda n: niced.append(n) or n)
    spawned = {}

    def popen(argv, **kwargs):
        kwargs["preexec_fn"]()  # what the child would run
        spawned["argv"] = argv
        spawned["new_session"] = kwargs.get("start_new_session")
        return SimpleNamespace(pid=4242)

    monkeypatch.setattr(workers_mod.subprocess, "Popen", popen)
    w = manager.add_local(spec, task, "generate", threads=4)
    manager._spawn_local(spec, task, w)
    assert niced == [workers_mod.LOCAL_WORKER_NICE] and w.pid == 4242
    assert spawned["argv"][-1] == "cloud.worker_entrypoint"
    assert spawned["new_session"]  # out of reach of a Ctrl-C in the dashboard's terminal


def test_an_all_ssh_task_with_a_trainer_gets_its_bucket_legs(manager, tmp_path, monkeypatch):
    """No cloud slot anywhere: the data watcher and the scheduler's publish and
    mirror hooks exist for the ssh trainer, whose generations come through the
    bucket; its controls go over ssh into its container -- and none of this
    exists for a task whose slots are local."""
    spec = workloads.get("position_eval")
    monkeypatch.setattr(WorkerManager, "_ensure_sync", _REAL_ENSURE_SYNC)
    monkeypatch.setattr(WorkerManager, "_creds", lambda self: _BUCKET_CREDS)
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    spawned = []
    monkeypatch.setattr(
        workers_mod.subprocess,
        "Popen",
        lambda argv, **k: spawned.append(_FakeWatcher(argv)) or spawned[-1],
    )
    pushed = []
    monkeypatch.setattr(
        workers_mod,
        "push_file",
        lambda machine, container, **kw: pushed.append((container, kw["rel_dest"])),
    )

    task = _all_ssh_task()
    manager._ensure_sync(spec, task)
    assert len(spawned) == 1
    assert manager._make_publish(spec, task) is not None
    assert manager._make_mirror(spec, task) is not None
    path = manager.tasks.paths(spec, "t").controls_path
    path.parent.mkdir(parents=True, exist_ok=True)  # the watcher's log dir made it
    path.write_text("{}")
    status = [{"worker_id": w.worker_id, "ssh_probe": "running"} for w in task.workers]
    manager._push_controls(spec, task, status)
    manager._push_controls(spec, task, status)  # unchanged: not pushed again
    assert pushed == [("scz-position_eval-t-tr", "controls.json")]

    local = tasks.TaskRecord(workload="position_eval", tag="u", params={}, created_at=0.0)
    local.workers.append(
        tasks.WorkerRecord(worker_id="tr", role="train", kind="local", desired_state="running")
    )
    manager._ensure_sync(spec, local)
    assert len(spawned) == 1
    assert manager._make_publish(spec, local) is None
    assert manager._make_mirror(spec, local) is None


def test_an_ssh_trainers_container_runs_the_torch_image_with_local_records(
    manager, tmp_path, monkeypatch
):
    spec = workloads.get("position_eval")
    task = _all_ssh_task()
    task.bundle_id, task.bundle_archs = "b1", ["znver3"]  # the fake machines' arch
    envs = {}

    class _Recording(_FakeSshMachine):
        def pull_image(self, image):
            pass

        def run_container(self, name, image, env, *, gpus=False, volume=None):
            envs[name] = (image, env["SCZ_SINK"], gpus, env.get("SCZ_DATA_SINK"))

    monkeypatch.setattr(workers_mod, "SshMachine", _Recording)
    monkeypatch.setattr(WorkerManager, "_run_ssh_container", _REAL_RUN_SSH_CONTAINER)
    monkeypatch.setattr(WorkerManager, "_creds", lambda self: _BUCKET_CREDS)
    monkeypatch.setattr(workers_mod, "bundle_worker_env", lambda *a, **k: {})
    for w in task.workers:
        manager._run_ssh_container(spec, task, w)
    by_role = {k.rsplit("-", 1)[-1]: v for k, v in envs.items()}
    # Records are local for both; the trainer's generations come through the
    # bucket. A data sink matching the records sink is left unset: a bundle
    # predating SCZ_DATA_SINK would refuse to start on it.
    assert by_role["tr"] == ("repo/worker:latest-torch", "local", True, "r2")
    assert by_role["g"] == ("repo/worker", "local", False, None)


def test_a_data_sink_differing_from_the_records_sink_reaches_the_container(
    manager, tmp_path, monkeypatch
):
    spec = workloads.get("position_eval")
    task = _all_ssh_task()
    task.bundle_id, task.bundle_archs = "b1", ["znver3"]
    envs = {}

    class _Recording(_FakeSshMachine):
        def pull_image(self, image):
            pass

        def run_container(self, name, image, env, *, gpus=False, volume=None):
            envs[name] = (env["SCZ_SINK"], env.get("SCZ_DATA_SINK"))

    monkeypatch.setattr(workers_mod, "SshMachine", _Recording)
    monkeypatch.setattr(WorkerManager, "_run_ssh_container", _REAL_RUN_SSH_CONTAINER)
    monkeypatch.setattr(WorkerManager, "_creds", lambda self: _BUCKET_CREDS)
    monkeypatch.setattr(workers_mod, "bundle_worker_env", lambda *a, **k: {})
    monkeypatch.setattr(WorkerManager, "_slot_data_sink", lambda self, spec, task, w: "r2")
    for w in task.workers:
        manager._run_ssh_container(spec, task, w)
    by_role = {k.rsplit("-", 1)[-1]: v for k, v in envs.items()}
    assert by_role["tr"] == ("local", "r2")
    assert by_role["g"] == ("local", "r2")


def test_reconcile_collects_from_every_ssh_slot(manager, tmp_path, monkeypatch):
    """A trainer's outputs are in its container now, like a generator's
    chunks, and both are collected over ssh."""
    spec = workloads.get("position_eval")
    task = _all_ssh_task()
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    monkeypatch.setattr(_FakeSshMachine, "state", "running")
    collected = []
    monkeypatch.setattr(
        WorkerManager, "_collect_step", lambda self, spec, task, w: collected.append(w.worker_id)
    )
    monkeypatch.setattr(WorkerManager, "_reconcile_worker", lambda *a, **k: None)
    monkeypatch.setattr(WorkerManager, "_ensure_sync", lambda *a: None)
    monkeypatch.setattr(WorkerManager, "_push_controls", lambda *a: None)
    monkeypatch.setattr(manager, "all_tasks", lambda: iter([(spec, task)]))
    asyncio.run(manager.reconcile())
    assert collected == ["g", "tr"]


def _slot_with_inputs(manager, spec, task, monkeypatch, tmp_path, sink: str):
    """A starting ssh generator whose role reads one out-of-tag file, on a
    task whose bundle is pinned; `sink` is where the slot delivers."""
    src = tmp_path / "teacher.onnx"
    src.write_bytes(b"onnx")
    monkeypatch.setattr(
        workers_mod,
        "_role_inputs",
        lambda spec, role, params, mount_root: {"inputs/teacher.onnx": src},
    )
    monkeypatch.setattr(manager, "_slot_records_sink", lambda spec, task, w: sink)
    # The bundle is someone else's concern here: pinned, never built.
    monkeypatch.setattr(WorkerManager, "_bundle_for_start", lambda self, *a, **k: "b1")
    w = _starting_ssh_slot(manager, spec, task, monkeypatch)
    return w, src


def test_an_own_machine_slots_inputs_are_pushed_into_its_container(
    manager, spec, task, monkeypatch, tmp_path
):
    rc = _Rclone()
    monkeypatch.setattr(workers_mod, "rclone", rc)
    pushed = []

    def push(machine, container, *, remote_root, rel_dest, src):
        _RecordingSshMachine.ops.append(("push", container))
        pushed.append((remote_root, rel_dest, src))

    monkeypatch.setattr(workers_mod, "push_file", push)
    w, src = _slot_with_inputs(manager, spec, task, monkeypatch, tmp_path, sink="local")

    manager._run_ssh_container(spec, task, w)
    assert rc.calls == []
    # Into the container, so after it exists; under the tag root there.
    assert [op for op, _ in _RecordingSshMachine.ops] == ["pull", "run", "push"]
    assert pushed == [(str(manager.tasks.paths(spec, "t").root), "inputs/teacher.onnx", src)]


def test_a_role_without_inputs_stages_nothing(manager, spec, task, monkeypatch):
    assert workers_mod._role_inputs(spec, spec.role("generate"), None, manager.mount_root) == {}


def test_a_missing_input_is_the_slots_reason_not_a_reconcile_exception(
    manager, spec, task, monkeypatch, tmp_path
):
    """The teacher tag's export is gone: the slot's row says so, the way a
    machine that cannot serve the role says so, instead of an assertion in
    the reconcile log and a slot reading `starting` forever."""
    rc = _Rclone()
    monkeypatch.setattr(workers_mod, "rclone", rc)
    w, src = _slot_with_inputs(manager, spec, task, monkeypatch, tmp_path, sink="r2")
    monkeypatch.setattr(WorkerManager, "_creds", lambda self: _BUCKET_CREDS)
    src.unlink()

    with pytest.raises(workers_mod.SshMachineError, match="inputs/teacher.onnx is missing"):
        manager._run_ssh_container(spec, task, w)
    assert "inputs/teacher.onnx is missing" in manager._exits[_key(spec, "t", w.worker_id)]
    assert rc.calls == [] and _RecordingSshMachine.ops == []  # nothing started


def test_two_tasks_machines_of_one_name_keep_separate_known_hosts(
    manager, spec, monkeypatch, tmp_path
):
    """Names are unique per task only: renting another task's `aws-1` must not
    truncate the first one's host keys."""
    provider = _FakeProvider()
    monkeypatch.setattr(manager, "_provider", lambda: provider)
    monkeypatch.setattr(workers_mod, "MACHINES_DIR", tmp_path / "machines")
    _fake_ssh(monkeypatch, state="missing", machine_state="up")
    a = tasks.TaskRecord(workload=spec.name, tag="a", params={}, created_at=0.0)
    b = tasks.TaskRecord(workload=spec.name, tag="b", params={}, created_at=0.0)
    first = manager.rent_machine(spec, a, None, "g6.2xlarge")
    Path(first.known_hosts_file).write_text("host-a ssh-ed25519 AAAA\n")
    second = manager.rent_machine(spec, b, None, "g6.2xlarge")
    assert first.name == second.name
    assert first.known_hosts_file != second.known_hosts_file
    assert Path(first.known_hosts_file).read_text() == "host-a ssh-ed25519 AAAA\n"


def test_a_slot_on_a_down_machine_reports_the_machine_not_its_stale_reason(
    manager, spec, task, monkeypatch
):
    """Reconcile leaves a slot alone while its machine is not up, so the
    slot's own last reason (a bundle build finished minutes ago) goes stale;
    its row says what it is actually waiting on."""
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    monkeypatch.setattr(_FakeSshMachine, "state", "missing")
    manager.add_machine(spec, task, "aws-1", "u@h", gpu_count=1)
    w = manager.add_ssh(spec, task, "generate", machine="aws-1", threads=None)
    w.desired_state = "running"
    manager._exits[workers_mod._key(spec, "t", w.worker_id)] = "building the worker bundle"
    mkey = workers_mod._machine_key(spec, "t", "aws-1")

    manager._machine_states[mkey] = "stopped"
    manager._exits[mkey] = "No g6.2xlarge capacity in the zone right now."
    (info,) = manager.worker_status(spec, task)
    assert info["exit_reason"] == (
        "waiting for machine aws-1 (stopped: No g6.2xlarge capacity in the zone right now.)"
    )

    del manager._exits[mkey]  # a machine down without a refusal on record
    (info,) = manager.worker_status(spec, task)
    assert info["exit_reason"] == "waiting for machine aws-1 (stopped)"

    manager._machine_states[mkey] = "up"  # its own reason again, once it can be acted on
    (info,) = manager.worker_status(spec, task)
    assert info["exit_reason"] == "building the worker bundle"


def test_restarts_after_a_crash_are_recorded_and_clean_exits_are_not(
    manager, spec, task, monkeypatch
):
    """The tag queue fails a crash-looping slot from these records, so a
    restart after a non-zero exit must be recorded, a restart after exit 0
    must not, and records fall out of the window."""
    local = manager.add_local(spec, task, "generate", 4)
    ssh = manager.add_ssh(spec, task, "generate", host="h", threads=None)
    monkeypatch.setattr(manager, "_spawn_local", lambda *a: None)
    monkeypatch.setattr(manager, "_start_or_replace", lambda *a: None)
    down = {"observed_running": False}

    monkeypatch.setattr(manager, "_local_exit_code", lambda *a: 1)
    manager._reconcile_worker(spec, task, local, workers_mod.RUN, down)
    monkeypatch.setattr(manager, "_local_exit_code", lambda *a: 0)
    manager._reconcile_worker(spec, task, local, workers_mod.RUN, down)
    assert manager.recent_crashes(spec, "t", local.worker_id, 60) == ["exit 1"]

    manager._exits[_key(spec, "t", ssh.worker_id)] = "exit 1: CUDA out of memory"
    manager._reconcile_ssh(spec, task, ssh, workers_mod.RUN, "stopped")
    manager._restarts.clear()
    manager._exits[_key(spec, "t", ssh.worker_id)] = "exit 0: done"
    manager._reconcile_ssh(spec, task, ssh, workers_mod.RUN, "stopped")
    assert manager.recent_crashes(spec, "t", ssh.worker_id, 60) == ["exit 1: CUDA out of memory"]

    later = time.time() + 120
    monkeypatch.setattr(workers_mod.time, "time", lambda: later)
    assert manager.recent_crashes(spec, "t", ssh.worker_id, 60) == []


def test_an_interrupted_worker_is_not_a_crash(manager, spec, task, monkeypatch):
    """A worker that exits 143 was SIGTERMed from outside: a gate or pause
    from the dashboard (three gates in half an hour once failed every tag on
    localhost), a docker stop, or a spot interruption stopping its host.
    Neither counts toward failing its tag; an exit the worker makes itself
    does."""
    w = manager.add_local(spec, task, "generate", 4)
    monkeypatch.setattr(manager, "_spawn_local", lambda *a: None)
    exit_code = 143
    monkeypatch.setattr(manager, "_local_exit_code", lambda *a: exit_code)
    down = {"observed_running": False}
    for _ in range(3):  # gated, then released; or interrupted with its host
        manager._reconcile_worker(spec, task, w, workers_mod.RUN, down)
    assert manager.recent_crashes(spec, "t", w.worker_id, 60) == []
    exit_code = 1  # its own failure
    manager._reconcile_worker(spec, task, w, workers_mod.RUN, down)
    assert manager.recent_crashes(spec, "t", w.worker_id, 60) == ["exit 1"]


def test_a_container_interrupted_with_its_host_is_not_a_crash(manager, spec, task, monkeypatch):
    """The ssh side: a container stopped with exit 143 (its spot host was
    interrupted) is restarted without a crash; one that failed on its own is
    counted, and so is one whose exit reason could not be read."""
    ssh = manager.add_ssh(spec, task, "generate", host="h", threads=None)
    monkeypatch.setattr(manager, "_start_or_replace", lambda *a: None)
    key = _key(spec, "t", ssh.worker_id)
    manager._exits[key] = "exit 143: host stopped"
    manager._reconcile_ssh(spec, task, ssh, workers_mod.RUN, "stopped")
    assert manager.recent_crashes(spec, "t", ssh.worker_id, 60) == []
    manager._restarts.clear()
    manager._exits[key] = "exit 2: out of disk"
    manager._reconcile_ssh(spec, task, ssh, workers_mod.RUN, "stopped")
    assert manager.recent_crashes(spec, "t", ssh.worker_id, 60) == ["exit 2: out of disk"]
    for unreadable in ("", "exit : no such container"):
        manager._restarts.clear()
        manager._exits[key] = unreadable
        manager._reconcile_ssh(spec, task, ssh, workers_mod.RUN, "stopped")
    assert len(manager.recent_crashes(spec, "t", ssh.worker_id, 60)) == 3


def test_a_removed_slots_memory_does_not_pass_to_the_next_slot_with_its_id(manager, spec, task):
    """A requeued tag gets its layout's worker ids back. The old container's
    last probe, read as the new slot's, was a phantom crash and a start of a
    container that did not exist."""
    w = manager.add_local(spec, task, "generate", 1)
    key = _key(spec, "t", w.worker_id)
    manager._probes[key] = ("stopped", 0.0)
    manager._exits[key] = "exit 143: stopped"
    manager._crashes[key] = [(0.0, "exit 1")]
    manager._down_since[key] = 0.0
    manager.remove_worker(spec, task, w.worker_id)
    for memory in (manager._probes, manager._exits, manager._crashes, manager._down_since):
        assert key not in memory


def test_cloud_sync_is_told_the_tag_dirs_mount_root(manager, spec, task):
    """cloud_sync resolves the tag dir itself; naming the root keeps its pull
    where this process puts the tag (a test's redirected root, here), never
    the live tags under the default one."""
    argv = manager.cloud_sync_argv(spec, task)
    root = argv[argv.index("--mount-root") + 1]
    assert root == str(manager.mount_root)
    assert not root.startswith(str(DEFAULT_MOUNT_ROOT))


# ---- the generation data plane (generational/data_home.py) -------------------------


def _local_training_task(desired: str = "paused") -> tasks.TaskRecord:
    """A position_eval task with a local generator and a local trainer."""
    task = tasks.TaskRecord(workload="position_eval", tag="t", params={}, created_at=0.0)
    for wid, role in (("g", "generate"), ("tr", "train")):
        task.workers.append(
            tasks.WorkerRecord(worker_id=wid, role=role, kind="local", desired_state=desired)
        )
    return task


def test_the_data_plane_moves_only_while_every_slot_is_stopped(manager, monkeypatch):
    """Two schedulers must never run on one tag: the controller's until the
    switch, the data home's after it."""
    spec = workloads.get("position_eval")
    task = _local_training_task(desired="running")
    with pytest.raises(AssertionError, match="pause every slot first"):
        manager.set_data_plane(spec, task, "home")

    task = _local_training_task()
    monkeypatch.setattr(WorkerManager, "_seen_alive", lambda self, spec, task, w: w.role == "train")
    with pytest.raises(AssertionError, match="still alive"):
        manager.set_data_plane(spec, task, "home")

    monkeypatch.setattr(WorkerManager, "_seen_alive", lambda self, spec, task, w: False)
    task.gates["generate"] = "ahead of trainer"
    manager.set_data_plane(spec, task, "home")
    assert task.data_plane == "home" and task.gates == {}  # the new scheduler decides
    manager.set_data_plane(spec, task, "legacy")
    assert manager.tasks.load(spec, "t").data_plane == "legacy"


def _rented_home_task(manager, monkeypatch) -> tasks.TaskRecord:
    """A data-home position_eval task whose trainer and a generator share rented
    machine m1, with a second generator on rented m2 and one on localhost; a
    match slot on m1 too."""
    monkeypatch.setattr(WorkerManager, "_rented", lambda self, task, w: w.kind == "ssh")
    task = tasks.TaskRecord(workload="position_eval", tag="t", params={}, created_at=0.0)
    task.data_plane = "home"
    for wid, role, kind, machine in (
        ("tr", "train", "ssh", "m1"),
        ("g1", "generate", "ssh", "m1"),
        ("g2", "generate", "ssh", "m2"),
        ("gl", "generate", "local", None),
        ("me", "match_eval", "ssh", "m1"),
    ):
        task.workers.append(
            tasks.WorkerRecord(
                worker_id=wid, role=role, kind=kind, desired_state="paused", machine=machine
            )
        )
    return task


def test_a_rented_data_homes_slots_deliver_by_machine(manager, monkeypatch):
    """On the home machine: the shared volume for data. Elsewhere: bucket
    staging, which the home ingests. Match eval keeps its own filesystem, where
    dispatch reads it. Every slot's records are collected over ssh."""
    spec = workloads.get("position_eval")
    task = _rented_home_task(manager, monkeypatch)
    sinks = {
        w.worker_id: (
            manager._slot_data_sink(spec, task, w),
            manager._slot_records_sink(spec, task, w),
            manager._collected(spec, task, w) if w.kind == "ssh" else None,
        )
        for w in task.workers
    }
    assert sinks == {
        "tr": ("home", "local", True),
        "g1": ("home", "local", True),
        "g2": ("r2", "local", True),
        "gl": ("r2", "local", None),
        "me": ("local", "local", True),
    }
    # The home takes bucket staging itself, so cloud_sync (the watcher, or a
    # drain's last pull) never copies it here.
    assert manager._has_bucket_data(spec, task) and not manager._has_bucket_slots(spec, task)
    task.data_plane = "legacy"  # the routing is unchanged for a legacy tag
    assert manager._slot_data_sink(spec, task, task.worker("g1")) == "r2"
    assert manager._has_bucket_slots(spec, task)


def test_a_remote_homes_scheduler_record_is_copied_here(manager, monkeypatch):
    """Generators are gated by the home's record (its gate and heartbeat), read
    in a small call of its own; an empty read (no record written yet) leaves
    the last copy in place."""
    spec = workloads.get("position_eval")
    task = _rented_home_task(manager, monkeypatch)
    reads = [b'{"gate": "full", "heartbeat": 5}', b""]
    commands = []

    class _Home:
        def read_from_container(self, container, command, timeout=None):
            commands.append((container, command[-1]))
            return reads.pop(0)

    monkeypatch.setattr(WorkerManager, "_ssh_machine", lambda self, task, w: _Home())
    record = manager.tasks.paths(spec, "t").root / "scheduler_state.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    manager._pull_scheduler_state(spec, task, task.worker("tr"))
    assert json.loads(record.read_text()) == {"gate": "full", "heartbeat": 5}
    assert commands[0][0] == "scz-position_eval-t-tr"
    assert str(record) in commands[0][1]
    manager._pull_scheduler_state(spec, task, task.worker("tr"))
    assert json.loads(record.read_text())["heartbeat"] == 5
    assert not list(record.parent.glob(".scheduler_state.json.tmp"))


def test_a_home_machine_container_mounts_the_tag_volume(manager, monkeypatch):
    spec = workloads.get("position_eval")
    task = _rented_home_task(manager, monkeypatch)
    task.bundle_id, task.bundle_archs = "b1", ["znver3"]
    runs, volumes = {}, []

    class _Recording(_FakeSshMachine):
        def pull_image(self, image):
            pass

        def create_volume(self, name):
            volumes.append(name)

        def run_container(self, name, image, env, *, gpus=False, volume=None):
            runs[name.rsplit("-", 1)[-1]] = (
                volume,
                env.get("SCZ_DATA_SINK"),
                env.get("SCZ_DATA_PLANE"),
            )

    monkeypatch.setattr(workers_mod, "SshMachine", _Recording)
    monkeypatch.setattr(WorkerManager, "_run_ssh_container", _REAL_RUN_SSH_CONTAINER)
    monkeypatch.setattr(WorkerManager, "_creds", lambda self: _BUCKET_CREDS)
    monkeypatch.setattr(
        WorkerManager,
        "_machine_record",
        lambda self, task, name: SimpleNamespace(
            host=f"u@{name}",
            identity_file=None,
            known_hosts_file=None,
            arch="znver3",
            instance_id="i",
        ),
    )
    monkeypatch.setattr(workers_mod, "bundle_worker_env", lambda *a, **k: {})
    for w in task.workers:
        if w.kind == "ssh":
            manager._run_ssh_container(spec, task, w)
    root = str(spec.paths("t", workers_mod.DEFAULT_MOUNT_ROOT).root)
    vol = "scz-position_eval-t-data"
    # SCZ_DATA_SINK=local matches the records sink, so it is left unset.
    assert runs["tr"] == ((vol, root), None, "home")
    assert runs["g1"] == ((vol, root), None, None)
    assert runs["g2"] == (None, "r2", None)  # away from the home: bucket staging
    assert runs["me"] == (None, None, None)
    assert set(volumes) == {vol}


def test_a_data_home_tags_trainer_joins_while_the_others_are_stopped(manager, monkeypatch):
    spec = workloads.get("position_eval")
    task = tasks.TaskRecord(workload="position_eval", tag="t", params={}, created_at=0.0)
    task.data_plane = "home"
    monkeypatch.setattr(WorkerManager, "_seen_alive", lambda self, spec, task, w: False)
    with pytest.raises(AssertionError, match="add this tag's trainer first"):
        manager._check_role(spec, task, "generate", "ssh", check_gpu=False)
    manager._check_role(spec, task, "match_eval", "ssh", check_gpu=False)  # not a data-plane slot
    manager._check_role(spec, task, "train", "ssh", check_gpu=False)
    g = tasks.WorkerRecord(worker_id="g", role="generate", kind="local", desired_state="running")
    task.workers.append(g)
    with pytest.raises(AssertionError, match="pause this tag's other slots"):
        manager._check_role(spec, task, "train", "ssh", check_gpu=False)
    g.desired_state = "paused"
    manager._check_role(spec, task, "train", "local", check_gpu=False)


def test_moving_the_trainer_rehomes_the_other_slots(manager, monkeypatch):
    """Your scenario: the trainer leaves m1 for localhost. The generators' stopped
    containers are recreated at their next start (m1's now delivers to the
    bucket), and m1's volume, which nothing works in any more, goes."""
    spec = workloads.get("position_eval")
    task = _rented_home_task(manager, monkeypatch)
    task.workers = [w for w in task.workers if w.worker_id != "tr"]  # removed from m1
    monkeypatch.setattr(WorkerManager, "_seen_alive", lambda self, spec, task, w: False)
    monkeypatch.setattr(WorkerManager, "_machine_gone", lambda self, spec, task, w: False)
    discarded, volumes_gone = [], []
    monkeypatch.setattr(
        WorkerManager,
        "_discard_container",
        lambda self, spec, task, w: discarded.append(w.worker_id),
    )
    monkeypatch.setattr(
        WorkerManager,
        "_ssh_machine",
        lambda self, task, w: SimpleNamespace(
            remove_volume=lambda name: volumes_gone.append(w.machine)
        ),
    )
    manager.add_local(spec, task, "train", threads=4, check_gpu=False)
    assert sorted(discarded) == ["g1", "g2", "me"]
    assert sorted(volumes_gone) == ["m1", "m2"]  # once per machine
    g1 = task.worker("g1")
    assert manager._slot_data_sink(spec, task, g1) == "r2"  # away from the new home
    assert manager._slot_data_sink(spec, task, task.worker("gl")) == "local"  # now at home


def test_removing_a_stopped_trainer_takes_its_final_state_pair(manager, monkeypatch):
    """A trainer's last flush, after the last collection, holds its final state
    pair; removal sweeps it, and installs it under the cursor rule. A
    generator's backlog is the Remove dialog's to warn about."""
    spec = workloads.get("position_eval")
    task = _rented_home_task(manager, monkeypatch)
    manager.tasks.save(spec, task)
    monkeypatch.setattr(WorkerManager, "_refresh_probe", lambda self, spec, task, w: "stopped")
    monkeypatch.setattr(WorkerManager, "_machine_gone", lambda self, spec, task, w: False)
    machine = SimpleNamespace(remove_container=lambda name: None, remove_volume=lambda name: None)
    monkeypatch.setattr(WorkerManager, "_ssh_machine", lambda self, task, w: machine)
    swept = []
    monkeypatch.setattr(
        WorkerManager, "_sweep_ssh", lambda self, m, spec, task, w: swept.append(w.worker_id)
    )
    manager.remove_worker(spec, task, "g2")
    manager.remove_worker(spec, task, "tr")
    assert swept == ["tr"]


def test_a_rented_data_home_cannot_go_back_to_legacy(manager, monkeypatch):
    spec = workloads.get("position_eval")
    task = _rented_home_task(manager, monkeypatch)
    monkeypatch.setattr(WorkerManager, "_seen_alive", lambda self, spec, task, w: False)
    with pytest.raises(AssertionError, match="cannot go back to legacy"):
        manager.set_data_plane(spec, task, "legacy")


def test_switching_recreates_the_ssh_containers_on_a_fresh_bundle(manager, monkeypatch):
    """A stopped container keeps the sinks and mount it was created with, so the
    switch removes it (collecting what it holds first) for the next start to
    recreate on a bundle that has the data home."""
    spec = workloads.get("position_eval")
    task = _rented_home_task(manager, monkeypatch)
    task.data_plane = "legacy"
    task.bundle_id = "old"
    monkeypatch.setattr(WorkerManager, "_seen_alive", lambda self, spec, task, w: False)
    monkeypatch.setattr(
        WorkerManager, "_machine_gone", lambda self, spec, task, w: w.machine == "m2"
    )
    monkeypatch.setattr(WorkerManager, "_refresh_probe", lambda self, spec, task, w: "stopped")
    monkeypatch.setattr(
        WorkerManager,
        "_ssh_machine",
        lambda self, task, w: SimpleNamespace(remove_container=lambda name: removed.append(name)),
    )
    monkeypatch.setattr(workers_mod, "sweep_stopped", lambda machine, **k: swept.append(k) or [])
    monkeypatch.setattr(
        WorkerManager, "_transfer_target", lambda self, spec, task, w: {"w": w.worker_id}
    )
    removed, swept = [], []
    manager.set_data_plane(spec, task, "home")
    assert sorted(n.rsplit("-", 1)[-1] for n in removed) == [
        "g1",
        "me",
        "tr",
    ]  # m2's machine is gone
    assert [s["w"] for s in swept] == ["tr", "g1", "me"]  # every collected container
    assert task.bundle_id is None and task.data_plane == "home"
    assert not any(w.launched for w in task.workers if w.machine == "m1")


def test_the_tag_volume_goes_with_the_last_slot_on_its_machine(manager, monkeypatch):
    spec = workloads.get("position_eval")
    task = _rented_home_task(manager, monkeypatch)
    manager.tasks.save(spec, task)
    monkeypatch.setattr(WorkerManager, "_refresh_probe", lambda self, spec, task, w: "missing")
    monkeypatch.setattr(WorkerManager, "_machine_gone", lambda self, spec, task, w: False)
    gone = []
    monkeypatch.setattr(
        WorkerManager,
        "_ssh_machine",
        lambda self, task, w: SimpleNamespace(
            remove_volume=lambda name: gone.append((w.machine, name))
        ),
    )
    for wid in ("tr", "g1", "g2"):  # the trainer first, as the tag queue's release goes
        manager.remove_worker(spec, task, wid)
    assert gone == [("m2", "scz-position_eval-t-data")]  # m2 had only g2; me stays on m1
    manager.remove_worker(spec, task, "me")
    assert gone[-1] == ("m1", "scz-position_eval-t-data")


def test_only_a_generational_workload_has_a_data_plane_to_move(manager, spec, task):
    with pytest.raises(AssertionError, match="has no data home"):
        manager.set_data_plane(spec, task, "home")


def test_a_data_home_gets_no_bucket_mirror_publish_or_sync_watcher(manager, monkeypatch):
    """The data home moves bucket chunks in itself; the controller's copy of
    them, and its re-upload for a remote trainer, would be a second data plane."""
    spec = workloads.get("position_eval")
    monkeypatch.setattr(WorkerManager, "_creds", lambda self: _BUCKET_CREDS)
    monkeypatch.setattr(WorkerManager, "_ensure_sync", _REAL_ENSURE_SYNC)
    spawned = []
    monkeypatch.setattr(
        workers_mod.subprocess,
        "Popen",
        lambda argv, **k: spawned.append(argv) or _FakeWatcher(argv),
    )
    task = _all_ssh_task()
    task.data_plane = "home"
    assert manager._make_mirror(spec, task) is None
    assert manager._make_publish(spec, task) is None
    manager._ensure_sync(spec, task)
    assert spawned == []


def test_a_data_homes_trainer_is_told_so_and_given_the_bucket(manager, monkeypatch, tmp_path):
    spec = workloads.get("position_eval")
    monkeypatch.setattr(WorkerManager, "_spawn_local", _REAL_SPAWN_LOCAL)
    r2 = R2Credentials(account_id="a", access_key_id="k", secret_access_key="s", bucket="b")
    monkeypatch.setattr(WorkerManager, "_creds", lambda self: SimpleNamespace(r2=r2))
    monkeypatch.setattr(
        WorkerManager, "_log_file", lambda self, spec, tag, name: open(tmp_path / "log", "ab")
    )
    monkeypatch.setattr(workers_mod.os, "nice", lambda n: n)
    envs = {}
    monkeypatch.setattr(
        workers_mod.subprocess,
        "Popen",
        lambda argv, env, **k: (
            envs.__setitem__(env["SCZ_WORKER_ID"], env) or SimpleNamespace(pid=1)
        ),
    )
    task = _local_training_task()
    task.data_plane = "home"
    for w in task.workers:
        manager._spawn_local(spec, task, w)
    assert envs["tr"]["SCZ_DATA_PLANE"] == "home" and envs["tr"]["R2_BUCKET"] == "b"
    assert "SCZ_DATA_PLANE" not in envs["g"]  # a generator just delivers locally


def test_the_scheduler_hooks_say_whether_a_role_is_running(manager, monkeypatch):
    spec = workloads.get("position_eval")
    task = _local_training_task()
    monkeypatch.setattr(WorkerManager, "_seen_alive", lambda self, spec, task, w: w.role == "train")
    hooks = manager._scheduler_hooks(spec, task)
    assert hooks.role_running("train") and not hooks.role_running("generate")


def test_a_paused_slot_reads_paused_even_while_its_role_is_gated(manager):
    """Pausing a data home's trainer gates its generators ("trainer not
    running"); a generator the operator paused too must not read `waiting`."""
    spec = workloads.get("position_eval")
    task = _local_training_task()
    task.gates["generate"] = "trainer not running"
    task.worker("g").desired_state = "running"
    rows = {r["worker_id"]: r for r in manager.worker_status(spec, task, observe=False)}
    assert rows["g"]["state"] == "waiting" and rows["g"]["gate_reason"] == "trainer not running"
    task.worker("g").desired_state = "paused"
    rows = {r["worker_id"]: r for r in manager.worker_status(spec, task, observe=False)}
    assert rows["g"]["state"] == "paused" and "gate_reason" not in rows["g"]


# ---- records and state over ssh ------------------------------------------------------


def test_a_new_trainer_container_is_seeded_with_the_controllers_state(manager, monkeypatch):
    """The controller's checkpoint and cursor are the state a trainer anywhere
    resumes from; they go in right after the container is created, the cursor
    last as the pair's commit marker."""
    spec = workloads.get("position_eval")
    task = _all_ssh_task()
    task.bundle_id, task.bundle_archs = "b1", ["znver3"]
    paths = manager.tasks.paths(spec, "t")
    paths.rolling_checkpoint.parent.mkdir(parents=True)
    paths.rolling_checkpoint.write_text("w")
    paths.train_state_path.write_text('{"rows_trained": 5}')
    envs, pushed = {}, []

    class _Recording(_FakeSshMachine):
        def pull_image(self, image):
            pass

        def run_container(self, name, image, env, *, gpus=False, volume=None):
            envs[name.rsplit("-", 1)[-1]] = env.get("SCZ_STATE_SEED")

    monkeypatch.setattr(workers_mod, "SshMachine", _Recording)
    monkeypatch.setattr(WorkerManager, "_run_ssh_container", _REAL_RUN_SSH_CONTAINER)
    monkeypatch.setattr(WorkerManager, "_creds", lambda self: _BUCKET_CREDS)
    monkeypatch.setattr(workers_mod, "bundle_worker_env", lambda *a, **k: {})
    monkeypatch.setattr(
        workers_mod, "push_file", lambda m, c, **kw: pushed.append((c, kw["rel_dest"]))
    )
    for w in task.workers:
        manager._run_ssh_container(spec, task, w)
    assert envs == {"tr": "1", "g": None}
    assert pushed == [
        ("scz-position_eval-t-tr", ".seed/model.pt"),
        ("scz-position_eval-t-tr", ".seed/train_state.json"),
    ]


def test_a_trainers_collection_takes_its_outputs_and_installs_pairs(manager, monkeypatch):
    spec = workloads.get("position_eval")
    task = _all_ssh_task()
    target = manager._transfer_target(spec, task, task.worker("tr"))
    assert target["data_dirs"][-2:] == ["models", "records"]
    assert target["pair_dirs"] == {"state": "train_state.json"}
    assert "pair_dirs" not in manager._transfer_target(spec, task, task.worker("g"))

    paths = manager.tasks.paths(spec, "t")

    def pull(machine, **target):  # a pull that brought a pair home
        pair = paths.root / "state" / "gen_000007"
        pair.mkdir(parents=True)
        (pair / "model.pt").write_text("w7")
        (pair / "train_state.json").write_text('{"rows_trained": 700}')
        return SimpleNamespace(pulled=["state/gen_000007/model.pt"], remaining=0)

    monkeypatch.setattr(workers_mod, "pull_results", pull)
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    _collect(manager, spec, task, task.worker("tr"))
    assert paths.rolling_checkpoint.read_text() == "w7"
    assert json.loads(paths.train_state_path.read_text())["rows_trained"] == 700


def test_a_data_homes_slots_leave_its_staging_alone(manager, monkeypatch):
    """The home's scheduler takes from its staging; a collection must never
    pull those chunks away, from the trainer or from a colocated generator."""
    spec = workloads.get("position_eval")
    task = _rented_home_task(manager, monkeypatch)
    assert manager._transfer_target(spec, task, task.worker("tr"))["data_dirs"] == [
        "models",
        "records",
    ]
    assert manager._transfer_target(spec, task, task.worker("g1"))["data_dirs"] == []
    assert manager._transfer_target(spec, task, task.worker("g2"))["data_dirs"] != []


def test_a_slow_pull_runs_off_the_blocking_thread_and_is_recorded_when_done(manager, monkeypatch):
    """A trainer's checkpoint can take minutes over a home link; the pass starts
    it on the tag's transfer thread, skips the slot while it runs, and records
    its count on a later pass."""
    spec = workloads.get("position_eval")
    task = _all_ssh_task()
    w = task.worker("tr")
    release, threads = threading.Event(), []

    def slow_pull(self, spec, task, w):
        threads.append(threading.current_thread().name)
        release.wait(timeout=5)
        return SimpleNamespace(remaining=3, pulled=[])

    monkeypatch.setattr(WorkerManager, "_pull_ssh", slow_pull)
    saved = []
    monkeypatch.setattr(manager.tasks, "save", lambda spec, task: saved.append(w.undelivered))
    key = _key_of(spec, task, w)
    manager._collect_step(spec, task, w)  # starts it and returns at once
    first = manager._collecting[key]
    manager._collect_step(spec, task, w)  # still running: no second pull
    assert manager._collecting[key] is first and manager._pulling(key)
    release.set()
    first.result(timeout=5)
    manager._collect_step(spec, task, w)  # records it, then starts the next
    assert saved == [3]
    manager._collecting[key].result(timeout=5)
    manager._collect_step(spec, task, w)
    assert w.undelivered == 3  # the last count stands while the next pull runs
    manager._collecting[key].result(timeout=5)
    assert len(threads) >= 2 and all(n.startswith("scz-transfer") for n in threads)
    assert threading.current_thread().name not in threads


def test_a_slot_is_parked_or_replaced_only_once_its_pull_is_done(manager, monkeypatch):
    """A pause would freeze a pull in flight until it timed out, and a
    replacement needs the count of a pull that has finished. So enforcement
    waits for the pull, and no further pull is started for a slot on its way
    to being parked or replaced."""
    spec = workloads.get("position_eval")
    task = _all_ssh_task()
    w = task.worker("g")
    release = threading.Event()

    def slow_pull(self, spec, task, w):
        release.wait(timeout=5)
        return SimpleNamespace(remaining=0, pulled=[])

    monkeypatch.setattr(WorkerManager, "_pull_ssh", slow_pull)
    monkeypatch.setattr(workers_mod, "SshMachine", _RecordingSshMachine)
    _RecordingSshMachine.ops = []
    key = _key_of(spec, task, w)
    manager._collect_step(spec, task, w)
    manager._reconcile_ssh(spec, task, w, workers_mod.PARK, "running")
    assert _RecordingSshMachine.ops == []  # the pull is still running
    release.set()
    manager._collecting[key].result(timeout=5)
    task.gates = [w.role]
    manager._collect_step(spec, task, w)  # records it; parking, so no next pull
    assert not manager._pulling(key) and key not in manager._collecting
    manager._reconcile_ssh(spec, task, w, workers_mod.PARK, "running")
    assert [op for op, _ in _RecordingSshMachine.ops] == ["pause"]


def _collect(manager, spec, task, w):
    """One collection by the pass's path, run on the calling thread."""
    manager._new_transfer_pool = SyncExecutor
    manager._collect_step(spec, task, w)


def _key_of(spec, task, w):
    return workers_mod._key(spec, task.tag, w.worker_id)
