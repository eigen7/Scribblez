"""The pool renting machines for the tag queue (dashboard/pool_rentals.py,
tag_queue._rent_for) against a fake provider: renting only within the cap and
only for a tag that fits the type, a refusal backing off, adoption of an
instance a crash left unrecorded, the lease-driven lifecycle (held -> stopped,
idle -> terminated), orphan accounting, and the lease's spend reaching its
tag."""

import time

import pytest
from cloud.providers.base import Instance, MachineType, ProviderError
from scribblez.dashboard import pool as pool_mod
from scribblez.dashboard import pool_rentals as rentals_mod
from scribblez.dashboard import queue as queue_mod
from scribblez.dashboard import tag_queue as tq_mod
from scribblez.dashboard import tasks
from scribblez.dashboard import workers as workers_mod
from scribblez.dashboard.pool import Lease
from scribblez.dashboard.tag_queue import HELD, RUNNING, TagQueue
from scribblez.dashboard.workers import WorkerManager
from scribblez.workloads.position_eval import SPEC

CAMPAIGN = {
    "trunk": "transformer",
    "activation_checkpointing": False,
    "max_rows": 1000,
    "match_every_generations": 0,
}


class _Provider:
    """Instances the test moves through their states; every call recorded."""

    name = "aws"
    ssh_user = "ubuntu"
    identity_file = "/k/scribblez.pem"
    region = "us-east-1"

    def __init__(self):
        self.instances: dict[str, Instance] = {}
        self.calls: list[tuple] = []
        self.refuse: ProviderError | None = None

    def catalog(self):
        return [MachineType("g6.2xlarge", 8, 1, "L4", "znver3", 1.0, 23034 / 1024)]

    def launch(self, request):
        if self.refuse is not None:
            raise self.refuse
        inst = Instance(
            id=f"i-{len(self.instances) + 1}", state="pending", type_id=request.type_id,
            owner=request.owner, address="1.2.3.4", launched_at=time.time(), spot=request.spot,
        )  # fmt: skip
        self.instances[inst.id] = inst
        self.calls.append(("launch", request.owner))
        return inst

    def describe(self):
        return dict(self.instances)

    def stop(self, instance_id):
        self.calls.append(("stop", instance_id))
        self.instances[instance_id].state = "stopped"

    def start(self, instance_id):
        self.calls.append(("start", instance_id))
        self.instances[instance_id].state = "pending"

    def terminate(self, instance_id):
        self.calls.append(("terminate", instance_id))
        self.instances[instance_id].state = "terminated"

    def refusal(self, error, type_id):
        return f"refused {type_id}: {error}"


class _Link:
    """The ssh link to a rented machine: its containers never created."""

    def __init__(self, host, identity_file=None, known_hosts_file=None):
        self.host = host

    def container_state(self, name: str) -> str:
        return "missing"

    def remove_volume(self, name: str):
        pass  # the tag's data-home volume, released as its slots leave


@pytest.fixture
def renting(tmp_path, monkeypatch):
    monkeypatch.setattr(pool_mod, "canonical_host", lambda h: h.split("@", 1)[-1].lower())
    monkeypatch.setattr(workers_mod, "MACHINES_DIR", tmp_path / "machines")
    monkeypatch.setattr(rentals_mod, "MACHINES_DIR", tmp_path / "machines")
    monkeypatch.setattr(workers_mod, "SshMachine", _Link)
    provider = _Provider()
    manager = WorkerManager(tmp_path)
    monkeypatch.setattr(manager, "_provider", lambda: provider)
    created: list = []
    monkeypatch.setattr(manager, "all_tasks", lambda: [(SPEC, t) for t in created])
    manager.add_capacity("g6", "g6.2xlarge", spot=True, cap=1)
    q = TagQueue(manager)
    # The enqueue-time build: done at once, as if the bundle were pinned.
    monkeypatch.setattr(q, "_submit_build", lambda spec, task, e, pool: None)

    def make(tag: str) -> tasks.TaskRecord:
        manager.tasks.save(
            SPEC,
            tasks.TaskRecord(workload="position_eval", tag=tag, params=CAMPAIGN, created_at=0.0),
        )
        created.append(manager.tasks.load(SPEC, tag))
        return created[-1]

    def enqueue(tag: str):
        make(tag)
        q.enqueue("position_eval", tag, confirm=True)
        queue = manager.queue_store.load()
        queue.entry("position_eval", tag).bundle = queue_mod.BUNDLE_READY
        manager.queue_store.save(queue)

    yield q, manager, provider, enqueue
    q.shutdown()


def _pool_machine(manager: WorkerManager, name):
    return manager.pool_store.load().find(name)


def test_a_tag_no_owned_machine_takes_gets_a_rental(renting):
    q, manager, provider, enqueue = renting
    enqueue("a")
    q.tick()
    assert provider.calls == [("launch", "pool/g6-1")]
    m = _pool_machine(manager, "g6-1")
    assert m.capacity == "g6" and m.machine.instance_id == "i-1" and m.machine.spot
    assert m.lease.tag == "a" and m.lease.phase == RUNNING
    assert m.machine.host == "ubuntu@1.2.3.4" and m.machine.arch == "znver3"
    a = manager.tasks.load(SPEC, "a")
    assert {(w.role, w.machine) for w in a.workers} == {("train", "g6-1"), ("generate", "g6-1")}
    assert manager.queue_store.load().entries == []


def test_the_cap_holds_and_a_second_tag_waits(renting):
    q, manager, provider, enqueue = renting
    enqueue("a")
    enqueue("b")
    q.tick()
    assert [c for c in provider.calls if c[0] == "launch"] == [("launch", "pool/g6-1")]
    (row,) = q.status()["entries"]
    assert row["tag"] == "b" and "at its cap (1 of 1" in row["refusals"]["rent g6"]


def test_a_refused_rental_backs_off_and_leaves_nothing(renting):
    q, manager, provider, enqueue = renting
    provider.refuse = ProviderError("VcpuLimitExceeded")
    enqueue("a")
    q.tick()
    assert manager.pool_store.load().machines == []
    provider.refuse = None
    q.tick()  # still within the retry backoff: no second ask
    assert provider.calls == []
    assert "refused: refused g6.2xlarge" in q.status()["entries"][0]["refusals"]["rent g6"]


def test_an_instance_a_crash_left_unrecorded_is_adopted(renting):
    """The machine and lease were saved, the launch happened, the dashboard
    died before recording the instance: the next pass adopts it by its owner
    tag instead of renting a second one."""
    q, manager, provider, enqueue = renting
    enqueue("a")
    pool = manager.pool_store.load()
    m = q._rentals.prospect(pool.capacity[0])
    m.name = m.machine.name = "g6-1"
    m.lease = Lease("position_eval", "a", "reserved", 0.0)
    pool.machines.append(m)
    manager.pool_store.save(pool)
    provider.instances["i-9"] = Instance(
        "i-9", "running", "g6.2xlarge", "pool/g6-1", "5.6.7.8", 0.0
    )
    manager._instances = ({}, 0.0)
    q.tick()
    assert not any(c[0] == "launch" for c in provider.calls)
    assert _pool_machine(manager, "g6-1").machine.instance_id == "i-9"
    assert _pool_machine(manager, "g6-1").lease.phase == RUNNING


def test_an_unrecorded_stray_counts_against_the_cap(renting):
    q, manager, provider, enqueue = renting
    provider.instances["i-9"] = Instance("i-9", "running", "g6.2xlarge", "pool/g6-7", None, 0.0)
    manager._instances = ({}, 0.0)
    manager._instance_index(True)
    enqueue("a")
    q.tick()
    assert provider.calls == []


def test_a_held_rental_is_stopped_and_an_idle_one_terminated(renting, monkeypatch):
    q, manager, provider, enqueue = renting
    enqueue("a")
    q.tick()
    provider.instances["i-1"].state = "running"
    for _ in range(tq_mod.FAIL_AFTER):
        manager._note_crash("position_eval/a/ssh-0", "exit 1")
    q.tick()  # failed with nothing queued: held
    assert _pool_machine(manager, "g6-1").lease.phase == HELD
    manager._instances = ({}, 0.0)
    q.tick()
    assert ("stop", "i-1") in provider.calls

    pool = manager.pool_store.load()
    pool.machine("g6-1").lease = None  # say the operator dealt with it
    manager.pool_store.save(pool)
    monkeypatch.setattr(rentals_mod, "IDLE_TERMINATE_SECONDS", 0.0)
    manager._instances = ({}, 0.0)
    q.tick()
    assert ("terminate", "i-1") in provider.calls
    assert _pool_machine(manager, "g6-1") is None


def test_pool_rentals_are_not_orphans(renting):
    q, manager, provider, enqueue = renting
    enqueue("a")
    q.tick()
    assert manager.orphans(observe=True) == []
    provider.instances["i-99"] = Instance("i-99", "running", "g6.2xlarge", "pool/nope", None, 0.0)
    manager._instances = ({}, 0.0)
    assert [o["owner"] for o in manager.orphans(observe=True)] == ["pool/nope"]


def test_the_leases_spend_reaches_its_tag(renting, monkeypatch):
    q, manager, provider, enqueue = renting
    enqueue("a")
    q.tick()
    pool = manager.pool_store.load()
    pool.machine("g6-1").machine.spend = 3.0  # accrued while leased from 0
    manager.pool_store.save(pool)
    a = manager.tasks.load(SPEC, "a")
    assert manager.lease_spend(a) == pytest.approx(3.0, abs=1e-3)
    for w in a.workers:
        w.finished, w.desired_state = True, "paused"
    monkeypatch.setattr(q, "_drain", lambda spec, task: None)  # nothing on the machine
    q.tick()  # completed: releasing
    for f in list(q._drains.values()):
        f.result(timeout=10)
    q.tick()
    # The ticks themselves accrue a few microseconds of the machine's rate.
    assert manager.tasks.load(SPEC, "a").retired_spend == pytest.approx(3.0, abs=1e-3)
    assert _pool_machine(manager, "g6-1").lease is None


def test_capacity_names_and_caps_are_validated(renting):
    _, manager, _, _ = renting
    with pytest.raises(AssertionError, match="letters, digits"):
        manager.add_capacity("g6-big", "g6.2xlarge", spot=False, cap=1)
    with pytest.raises(AssertionError, match="no machine type"):
        manager.add_capacity("x", "p5.48xlarge", spot=False, cap=1)
    with pytest.raises(AssertionError, match="exists"):
        manager.add_capacity("g6", "g6.2xlarge", spot=False, cap=1)
    manager.set_capacity_cap("g6", 3)
    assert manager.pool_store.load().capacity[0].cap == 3
    manager.remove_capacity("g6")
    assert manager.pool_store.load().capacity == []


def test_a_listing_failure_does_not_stall_the_queue(renting):
    """AWS throttling or expired credentials: the rentals step skips this pass,
    and the queue's leases still advance."""
    q, manager, provider, enqueue = renting
    enqueue("a")
    q.tick()
    a = manager.tasks.load(SPEC, "a")
    for w in a.workers:
        w.finished, w.desired_state = True, "paused"

    def throttled():
        raise ProviderError("RequestLimitExceeded")

    provider.describe = throttled
    manager._instances = ({}, 0.0)
    q.tick()
    assert _pool_machine(manager, "g6-1").lease.phase == "releasing"


def test_a_rental_whose_instance_vanished_requeues_its_tag(renting):
    """Terminated outside the dashboard (the EC2 console): the slots go without
    a drain (their containers went with the instance), the spend is retired,
    the machine is dropped, and the tag is back at the head of the queue with
    its eligibility."""
    q, manager, provider, enqueue = renting
    enqueue("a")
    queue = manager.queue_store.load()
    queue.entry("position_eval", "a").memory_override_gb = 18.0
    manager.queue_store.save(queue)
    q.tick()
    pool = manager.pool_store.load()
    record = pool.machine("g6-1").machine
    record.launched_at, record.spend = 0.0, 2.0  # long past its boot grace
    manager.pool_store.save(pool)
    provider.instances["i-1"].state = "terminated"
    manager._instances = ({}, 0.0)
    q.tick()
    assert _pool_machine(manager, "g6-1") is None
    a = manager.tasks.load(SPEC, "a")
    assert a.workers == [] and a.retired_spend == pytest.approx(2.0, abs=1e-3)
    (entry,) = manager.queue_store.load().entries
    assert entry.tag == "a" and entry.memory_override_gb == 18.0


def test_a_just_launched_instance_missing_from_the_listing_is_not_gone(renting):
    q, manager, provider, enqueue = renting
    enqueue("a")
    q.tick()
    del provider.instances["i-1"]  # an eventually consistent listing
    manager._instances = ({}, 0.0)
    q.tick()
    assert _pool_machine(manager, "g6-1").lease.tag == "a"


def test_removing_an_unleased_rental_terminates_it(renting):
    q, manager, provider, enqueue = renting
    enqueue("a")
    q.tick()
    pool = manager.pool_store.load()
    pool.machine("g6-1").lease = None
    manager.pool_store.save(pool)
    manager.tasks.load(SPEC, "a").workers.clear()  # say its slots were removed
    manager.remove_pool_machine("g6-1")
    assert ("terminate", "i-1") in provider.calls
    assert _pool_machine(manager, "g6-1") is None


def _record_without_instance(q, capacity: str | None):
    pool = q._m.pool_store.load()
    m = q._rentals.prospect(pool.capacity[0])
    m.name = m.machine.name = "g6-1"
    m.capacity = capacity
    m.lease = Lease("position_eval", "a", "reserved", 0.0)
    pool.machines.append(m)
    q._m.pool_store.save(pool)


def test_a_recorded_rental_with_no_instance_is_launched(renting):
    """The dashboard died between recording the machine and launching it."""
    q, manager, provider, enqueue = renting
    enqueue("a")
    _record_without_instance(q, "g6")
    q.tick()
    assert provider.calls[0] == ("launch", "pool/g6-1")
    assert _pool_machine(manager, "g6-1").machine.instance_id == "i-1"


def test_a_recorded_rental_whose_capacity_is_gone_is_dropped(renting):
    q, manager, provider, enqueue = renting
    enqueue("a")
    _record_without_instance(q, "retired")
    q._rentals.reconcile(manager.pool_store.load())
    assert _pool_machine(manager, "g6-1") is None
    assert not any(c[0] == "launch" for c in provider.calls)


def test_the_task_view_does_not_accrue_a_pool_rentals_spend(renting, monkeypatch):
    """Only the pool's own step accrues a rental; the leasing task's machine
    status must not accrue it a second time."""
    q, manager, provider, enqueue = renting
    enqueue("a")
    q.tick()
    accrued = []
    monkeypatch.setattr(workers_mod, "_accrue_machine", lambda m, billing: accrued.append(m.name))
    (info,) = manager.machine_status(SPEC, manager.tasks.load(SPEC, "a"), observe=True)
    assert info["pool"] is True and accrued == []


def test_a_failed_listing_never_reads_as_every_rental_gone(renting):
    """After a restart the cached listing is empty; if the pass's own listing
    then fails, an aged rental must not be taken for gone and dropped while its
    instance keeps billing."""
    q, manager, provider, enqueue = renting
    enqueue("a")
    q.tick()
    pool = manager.pool_store.load()
    pool.machine("g6-1").machine.launched_at = 0.0
    manager.pool_store.save(pool)

    def throttled():
        raise ProviderError("RequestLimitExceeded")

    provider.describe = throttled
    manager._instances = ({}, 0.0)  # a fresh process
    q.tick()
    assert _pool_machine(manager, "g6-1").lease.tag == "a"
    assert len(manager.tasks.load(SPEC, "a").workers) == 2


def test_a_vanished_instance_does_not_rerun_a_completed_tag(renting, monkeypatch):
    q, manager, provider, enqueue = renting
    enqueue("a")
    q.tick()
    a = manager.tasks.load(SPEC, "a")
    for w in a.workers:
        w.finished, w.desired_state = True, "paused"
    monkeypatch.setattr(q, "_drain", lambda spec, task: (_ for _ in ()).throw(OSError("slow")))
    q.tick()  # completed: releasing, its drain failing
    pool = manager.pool_store.load()
    pool.machine("g6-1").machine.launched_at = 0.0
    manager.pool_store.save(pool)
    provider.instances["i-1"].state = "terminated"
    manager._instances = ({}, 0.0)
    q.tick()
    assert _pool_machine(manager, "g6-1") is None
    assert manager.queue_store.load().entries == []


def test_stop_all_cloud_spending_requeues_the_tag_and_terminates_the_rental(renting, monkeypatch):
    """The burn strip's one button. Release alone handed the rental to the next
    queued tag; stopping must requeue the tag, never place on the machine
    again, terminate it once free, and rent nothing more."""
    q, manager, provider, enqueue = renting
    enqueue("a")
    enqueue("b")  # waits: the cap is 1
    q.tick()
    provider.instances["i-1"].state = "running"
    manager._instances = ({}, 0.0)
    manager._instance_index(True)
    report = q.stop_cloud(dry_run=True)
    assert report == {
        "caps": ["g6"], "requeue": ["position_eval/a"], "terminate": ["g6-1"],
        "stop": [], "orphans": [],
    }  # fmt: skip
    assert manager.pool_store.load().capacity[0].cap == 1  # a dry run changes nothing

    monkeypatch.setattr(q, "_drain", lambda spec, task: None)
    q.stop_cloud()
    assert manager.pool_store.load().capacity[0].cap == 0
    assert _pool_machine(manager, "g6-1").retiring
    for _ in range(4):
        for future in list(q._drains.values()):
            future.result(timeout=10)
        manager._instances = ({}, 0.0)
        q.tick()
    assert ("terminate", "i-1") in provider.calls
    assert _pool_machine(manager, "g6-1") is None
    assert [e.tag for e in manager.queue_store.load().entries] == ["a", "b"]
    assert [c for c in provider.calls if c[0] == "launch"] == [("launch", "pool/g6-1")]
