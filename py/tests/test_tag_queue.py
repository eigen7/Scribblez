"""The tag queue's lifecycle (dashboard/tag_queue.py) on a real WorkerManager
with its launch paths faked: enqueue and its warnings, placement, completion
and release, failure (hand-over or hold), requeue, and recovery of a placement
a restart interrupted. The pass is driven by calling tick() directly; the
slots it creates are never actually started."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from scribblez.dashboard import placement, tasks
from scribblez.dashboard import pool as pool_mod
from scribblez.dashboard import queue as queue_mod
from scribblez.dashboard import tag_queue as tq_mod
from scribblez.dashboard import workers as workers_mod
from scribblez.dashboard.control_store import ControlStore
from scribblez.dashboard.pool import Hardware, Lease, PoolMachine
from scribblez.dashboard.tag_queue import (
    EMPTY_POOL,
    HELD,
    RELEASING,
    RESERVED,
    RUNNING,
    TagQueue,
)
from scribblez.dashboard.workers import WorkerManager
from scribblez.workloads.position_eval import SPEC

# A tuning-campaign tag: the transformer profile's trainer, a row budget, and
# match eval off.
ENDS = {
    "trunk": "transformer",
    "activation_checkpointing": False,
    "max_rows": 1000,
    "match_every_generations": 0,
}


@pytest.fixture
def queued(tmp_path, monkeypatch):
    """A TagQueue over a WorkerManager rooted at tmp_path (pool, queue and task
    records under it), localhost in the pool (16 GiB), and tags listed only as
    the test creates them."""
    monkeypatch.setattr(pool_mod, "local_hardware", lambda: Hardware(28, 1, 16.0))
    monkeypatch.setattr(pool_mod, "canonical_host", lambda h: h.split("@", 1)[-1].lower())
    manager = WorkerManager(tmp_path)
    created: list = []
    monkeypatch.setattr(manager, "all_tasks", lambda: [(SPEC, t) for t in created])
    manager.add_pool_machine("localhost")

    def make(tag: str, **params) -> tasks.TaskRecord:
        task = tasks.TaskRecord(
            workload="position_eval", tag=tag, params={**ENDS, **params}, created_at=0.0
        )
        manager.tasks.save(SPEC, task)
        created.append(manager.tasks.load(SPEC, tag))
        return created[-1]

    q = TagQueue(manager)
    yield q, manager, make
    q.shutdown()


def _lease(manager: WorkerManager, name="localhost") -> Lease | None:
    return manager.pool_store.load().machine(name).lease


def _drain_then_tick(q: TagQueue):
    """Let the in-flight drains finish, then run the pass that acts on them."""
    for future in list(q._drains.values()):
        future.result(timeout=10)
    q.tick()


def test_enqueue_warns_before_queueing(queued):
    q, manager, make = queued
    make("endless", max_rows=-1)
    out = q.enqueue("position_eval", "endless")
    assert out["queued"] is False
    assert any("no end condition: position_eval/endless" in w for w in out["warnings"])
    assert any(tq_mod.LOCAL_CODE_WARNING in w for w in out["warnings"])
    assert manager.queue_store.load().entries == []
    assert q.enqueue("position_eval", "endless", confirm=True)["queued"] is True
    assert [e.tag for e in manager.queue_store.load().entries] == ["endless"]


def test_enqueue_refuses_a_tag_with_slots(queued):
    q, manager, make = queued
    task = make("busy")
    manager.add_local(SPEC, task, "train", 4, check_gpu=False)
    with pytest.raises(AssertionError, match="already has slots"):
        q.enqueue("position_eval", "busy", confirm=True)


def test_a_placed_tag_gets_its_layout_and_its_machine(queued):
    q, manager, make = queued
    make("a")
    q.enqueue("position_eval", "a", confirm=True)
    q.tick()
    assert _lease(manager) == Lease("position_eval", "a", RUNNING, _lease(manager).since, "")
    task = manager.tasks.load(SPEC, "a")
    assert [(w.role, w.kind, w.threads, w.desired_state) for w in task.workers] == [
        ("train", "local", 28, "running"),  # add_local's default is all cores
        ("generate", "local", 28, "running"),
    ]
    assert manager.queue_store.load().entries == []


def test_a_placement_is_committed_whole(queued, monkeypatch):
    """The lease, the queue entry's removal and the slots commit together: no
    reader, and no restart, finds the tag both queued and placed, or placed
    with no slots."""
    q, manager, make = queued
    make("a")
    q.enqueue("position_eval", "a", confirm=True)
    manager.claim_writer()  # the test's own thread now reads committed copies
    seen = []
    start_slots = q._start_slots

    def start_then_look(m, pool):
        start_slots(m, pool)
        seen.append(_elsewhere(_placement, manager))

    monkeypatch.setattr(q, "_start_slots", start_then_look)
    manager._blocking.submit(q.tick).result()
    assert seen == [(None, ["a"], 0)]  # mid-placement, nothing of it
    assert _placement(manager) == ("a", [], 2)


def _placement(manager) -> tuple:
    """(the tag leasing localhost, the queued tags, tag a's slot count), as a
    reader sees them."""
    lease = manager.pool_store.load().machine("localhost").lease
    queued = [e.tag for e in manager.queue_store.load().entries]
    return lease and lease.tag, queued, len(manager.tasks.load(SPEC, "a").workers)


def test_a_tag_that_does_not_fit_waits_with_its_reason(queued):
    q, manager, make = queued
    make("big", match_every_generations=5)  # trainer + match eval: 16.6 GiB > 14.5
    q.enqueue("position_eval", "big", confirm=True)
    q.tick()
    assert _lease(manager) is None
    (row,) = q.status()["entries"]
    assert "needs 16.6 GiB" in row["refusals"]["localhost"]


def test_a_busy_machine_is_not_placed_on(queued):
    q, manager, make = queued
    hand = make("hand")
    w = manager.add_local(SPEC, hand, "train", 4, check_gpu=False)
    w.desired_state = "running"
    make("a")
    q.enqueue("position_eval", "a", confirm=True)
    q.tick()
    assert _lease(manager) is None
    assert "busy: position_eval/hand/local-0" in q.status()["entries"][0]["refusals"]["localhost"]


def test_completion_releases_the_machine_to_the_next_tag(queued):
    q, manager, make = queued
    make("a")
    make("b")
    q.enqueue("position_eval", "a", confirm=True)
    q.enqueue("position_eval", "b", confirm=True)
    q.tick()
    assert _lease(manager).tag == "a"
    a = manager.tasks.load(SPEC, "a")
    for w in a.workers:
        w.finished, w.desired_state = True, "paused"
    q.tick()
    assert _lease(manager).phase == RELEASING
    _drain_then_tick(q)  # the drain succeeds: a's slots go, the machine is free
    assert manager.tasks.load(SPEC, "a").workers == []
    q.tick()
    assert _lease(manager).tag == "b"


def test_a_crash_looping_tag_fails_and_holds_its_machine(queued):
    q, manager, make = queued
    make("a")
    q.enqueue("position_eval", "a", confirm=True)
    q.tick()
    for _ in range(tq_mod.FAIL_AFTER):
        manager._note_crash("position_eval/a/local-0", "exit 1: CUDA out of memory")
    q.tick()
    lease = _lease(manager)
    assert lease.phase == HELD and "CUDA out of memory" in lease.reason
    a = manager.tasks.load(SPEC, "a")
    assert all(w.desired_state == "paused" for w in a.workers)
    assert a.worker("local-0").failed.startswith("local-0: exit 1")
    status = {i["worker_id"]: i for i in manager.worker_status(SPEC, a)}
    assert status["local-0"]["state"] == "failed"
    assert status["local-0"]["exit_reason"].startswith("local-0: exit 1")
    # Held means kept: nothing is drained or removed while nothing needs it.
    q.tick()
    assert _lease(manager).phase == HELD and len(manager.tasks.load(SPEC, "a").workers) == 2

    # A queued tag that can use the machine gets it, after the drain.
    make("b")
    q.enqueue("position_eval", "b", confirm=True)
    q.tick()
    assert _lease(manager).phase == RELEASING
    _drain_then_tick(q)
    q.tick()
    assert _lease(manager).tag == "b"


def test_a_failed_tag_hands_over_at_once_when_a_tag_is_waiting(queued):
    q, manager, make = queued
    make("a")
    make("b")
    q.enqueue("position_eval", "a", confirm=True)
    q.enqueue("position_eval", "b", confirm=True)
    q.tick()
    for _ in range(tq_mod.FAIL_AFTER):
        manager._note_crash("position_eval/a/local-0", "exit 1")
    q.tick()
    assert _lease(manager).phase == RELEASING


def test_requeue_puts_the_tag_back_at_the_head(queued):
    """Its release done, the requeued tag is back at the head of the queue, and
    the same pass places it again on the machine it just freed."""
    q, manager, make = queued
    make("a")
    make("b")
    q.enqueue("position_eval", "a", confirm=True)
    q.tick()
    q.enqueue("position_eval", "b", confirm=True)
    q.requeue("position_eval", "a")
    assert _lease(manager).phase == RELEASING and _lease(manager).reason == tq_mod.REQUEUED
    _drain_then_tick(q)
    assert _lease(manager).tag == "a" and _lease(manager).phase == RUNNING
    assert [e.tag for e in manager.queue_store.load().entries] == ["b"]


def test_a_requeue_is_committed_whole(queued, monkeypatch):
    """Pausing the slots and turning the lease to a requeue commit together:
    no reader or restart finds the slots paused on a lease still running."""
    q, manager, make = queued
    make("a")
    q.enqueue("position_eval", "a", confirm=True)
    q.tick()
    manager.claim_writer()
    seen = []
    submit_drain = q._submit_drain

    def look_then_drain(m):
        seen.append(_elsewhere(_release_state, manager))
        submit_drain(m)

    monkeypatch.setattr(q, "_submit_drain", look_then_drain)
    manager._blocking.submit(q.requeue, "position_eval", "a").result()
    assert seen == [(RUNNING, {"running"}, [])]
    assert _release_state(manager) == (RELEASING, {"paused"}, [])


def test_a_finished_release_is_committed_whole(queued, monkeypatch):
    """Ending a requeued tag's lease and putting it back in the queue commit
    together: a crash between them would lose the tag from both."""
    q, manager, make = queued
    make("a")
    q.enqueue("position_eval", "a", confirm=True)
    q.tick()
    q.requeue("position_eval", "a")
    manager.claim_writer()
    seen = []
    save_queue = manager.queue_store.save

    def look_then_save(queue):
        if not seen:
            seen.append(_elsewhere(_release_state, manager))
        save_queue(queue)

    monkeypatch.setattr(manager.queue_store, "save", look_then_save)
    manager._blocking.submit(_drain_then_tick, q).result()
    assert seen == [(RELEASING, {"paused"}, [])]  # the lease not yet ended


def _release_state(manager) -> tuple:
    """(localhost's lease phase, tag a's slots' desired states, the queued
    tags), as a reader sees them."""
    lease = manager.pool_store.load().machine("localhost").lease
    a = manager.tasks.load(SPEC, "a")
    queued = [e.tag for e in manager.queue_store.load().entries]
    return lease and lease.phase, {w.desired_state for w in a.workers}, queued


def _elsewhere(fn, *args):
    """`fn(*args)` on a thread of its own: off the writer, as a reader."""
    with ThreadPoolExecutor(max_workers=1) as reader:
        return reader.submit(fn, *args).result()


def test_a_reserved_lease_is_completed_after_a_restart(queued):
    """A placement interrupted between writing the lease and creating the
    slots: the next process finishes it rather than placing twice."""
    q, manager, make = queued
    make("a")
    q.enqueue("position_eval", "a", confirm=True)
    pool = manager.pool_store.load()
    pool.machine("localhost").lease = Lease("position_eval", "a", RESERVED, 0.0)
    manager.pool_store.save(pool)
    fresh = TagQueue(q._m)
    fresh.tick()
    assert _lease(manager).phase == RUNNING
    assert manager.queue_store.load().entries == []  # the stale entry is dropped
    assert len(manager.tasks.load(SPEC, "a").workers) == 2
    fresh.shutdown()


def test_release_finishes_an_endless_tag(queued):
    q, manager, make = queued
    make("endless", max_rows=-1)
    q.enqueue("position_eval", "endless", confirm=True)
    q.tick()
    q.release("position_eval", "endless")
    q.tick()
    assert _lease(manager).phase == RELEASING


def test_reordering(queued):
    q, manager, make = queued
    for tag in "abc":
        make(tag, max_rows=-1 if tag == "c" else 1000)
        q.enqueue("position_eval", tag, machines=["nowhere"], confirm=True)
    q.move("position_eval", "c", -2)
    assert [e.tag for e in manager.queue_store.load().entries] == ["c", "a", "b"]
    q.dequeue("position_eval", "a")
    assert [e.tag for e in manager.queue_store.load().entries] == ["c", "b"]
    rows = q.status()["entries"]
    assert [r["end_condition"] for r in rows] == [False, True]


def test_has_end_condition_uses_the_tags_params(queued):
    q, manager, make = queued
    assert placement.has_end_condition(SPEC, tq_mod._params(SPEC, make("x")))


class _Link:
    """The ssh link to a registered pool machine: an L4-sized hardware report."""

    def __init__(self, host, identity_file=None, known_hosts_file=None):
        self.host = host

    def hardware_report(self) -> str:
        return "8\n23034\n"

    def remove_volume(self, name: str):
        pass  # the tag's data-home volume, released as its slots leave


def test_an_ssh_machine_takes_the_tag_once_its_bundle_is_pinned(queued, monkeypatch):
    """A tag that may run on a registered machine has its bundle built at
    enqueue; placement waits for the pin, then puts ssh slots on the machine
    by its pool name, which resolves through the lease."""
    from concurrent.futures import Future

    q, manager, make = queued
    monkeypatch.setattr(workers_mod, "SshMachine", _Link)
    manager.remove_pool_machine("localhost")
    manager.add_pool_machine("gpu-box", "me@gpu-box")
    pinned = []
    monkeypatch.setattr(manager, "_pin_bundle", lambda spec, task, m: pinned.append(m))
    submitted = []
    monkeypatch.setattr(q, "_submit_build", lambda spec, task, e, pool: submitted.append(e.key))
    make("a")
    q.enqueue("position_eval", "a", confirm=True)
    assert submitted == [("position_eval", "a")]
    assert manager.queue_store.load().entry("position_eval", "a").bundle == "building"
    q.tick()  # the build is not done: nothing placed
    assert _lease(manager, "gpu-box") is None

    done: Future = Future()
    # The build thread only reports the arch it detected; the pass records it.
    done.set_result(({"gpu-box": "znver3"}, "manifest"))
    q._builds[("position_eval", "a")] = done
    q.tick()
    assert pinned == ["manifest"]
    fresh = pool_mod.pool_store(ControlStore(manager.mount_root)).load()  # as saved
    assert fresh.machine("gpu-box").machine.arch == "znver3"
    lease = _lease(manager, "gpu-box")
    assert lease.tag == "a" and lease.phase == RUNNING
    a = manager.tasks.load(SPEC, "a")
    assert {(w.role, w.kind, w.machine, w.threads) for w in a.workers} == {
        ("train", "ssh", "gpu-box", None),
        ("generate", "ssh", "gpu-box", 8),
    }
    assert manager._machine_record(a, "gpu-box").host == "me@gpu-box"


def _local_gpu(manager: WorkerManager, gb: float):
    """Give the pooled localhost a GPU of `gb` GiB."""
    pool = manager.pool_store.load()
    pool.machine("localhost").hardware = Hardware(28, 1, gb)
    manager.pool_store.save(pool)


def test_a_hand_placed_trainer_is_checked_against_the_pool_machine(queued):
    """The asus-laptop OOM by hand: the same measured need refuses a slot the
    machine cannot fit. An unmeasured configuration is not refused."""
    q, manager, make = queued
    _local_gpu(manager, 13.0)  # < 14.03
    task = make("hand")
    with pytest.raises(AssertionError, match="would need 14.0 GiB"):
        manager.add_local(SPEC, task, "train", None)
    _local_gpu(manager, 16.0)
    manager.add_local(SPEC, task, "train", None)
    unmeasured = make("unmeasured", batch_size=512)
    _local_gpu(manager, 1.0)
    manager.add_local(SPEC, unmeasured, "train", None)


def test_one_lease_failing_does_not_stall_placement(queued, monkeypatch):
    q, manager, make = queued
    make("a")
    q.enqueue("position_eval", "a", confirm=True)
    monkeypatch.setattr(q, "_advance_lease", lambda *a: (_ for _ in ()).throw(OSError("down")))
    pool = manager.pool_store.load()
    pool.machine("localhost").lease = Lease("position_eval", "gone", RUNNING, 0.0)
    manager.pool_store.save(pool)
    q.tick()  # the lease raises; the tick still reaches placement and returns
    assert q.status()["entries"][0]["refusals"]["localhost"].startswith("leased by")


def test_a_requeue_survives_a_failed_first_drain_with_its_eligibility(queued, monkeypatch):
    """Requeue drains at once, before reconcile has stopped the workers, so
    the first drain fails. The drain error must not erase the requeue, and the
    tag must come back with the entry's machine list and memory override."""
    q, manager, make = queued
    make("a")
    q.enqueue("position_eval", "a", machines=["localhost"], memory_override_gb=9.0, confirm=True)
    q.tick()
    alive = iter([True, False])
    monkeypatch.setattr(tq_mod, "worker_pid_alive", lambda *a: next(alive, False))
    q.requeue("position_eval", "a")
    _drain_then_tick_allowing_failure(q)
    assert _lease(manager).phase == RELEASING and _lease(manager).reason.startswith("draining")
    for _ in range(3):  # the retry is submitted one pass, acted on the next
        _drain_then_tick_allowing_failure(q)
        if _lease(manager).phase != RELEASING:
            break
    lease = _lease(manager)
    assert lease.tag == "a" and lease.phase == RUNNING  # placed again at once
    assert lease.machines == ["localhost"] and lease.memory_override_gb == 9.0


def _drain_then_tick_allowing_failure(q: TagQueue):
    for future in list(q._drains.values()):
        future.exception(timeout=10)
    q.tick()


def test_an_override_below_the_measured_need_is_placed(queued):
    """The operator's override decides the fit; slot creation must not re-check
    against the measured figure and leave the lease reserved forever."""
    q, manager, make = queued
    _local_gpu(manager, 10.0)  # < 14.03 measured
    make("a")
    q.enqueue("position_eval", "a", memory_override_gb=9.0, confirm=True)
    q.tick()
    assert _lease(manager).phase == RUNNING
    assert len(manager.tasks.load(SPEC, "a").workers) == 2


class _StoppedLink(_Link):
    """A registered machine whose containers have all stopped, each with a log."""

    def container_state(self, name: str) -> str:
        return "stopped"

    def container_logs(self, name: str) -> str:
        return f"log of {name}\n"

    def container_exit(self, name: str) -> str:
        return "exit 0: done"

    def remove_container(self, name: str):
        pass


def test_the_drain_saves_logs_and_sweeps_before_removing(queued, monkeypatch):
    """The ssh half of a release: every container's full log saved into the
    tag and every container swept (each delivers locally, a trainer's state
    pairs included), and the trainer's machine, the tag's data home, swept of
    its generations, all before the slots are removed."""
    q, manager, make = queued
    monkeypatch.setattr(workers_mod, "SshMachine", _StoppedLink)
    swept, homes = [], []
    monkeypatch.setattr(
        workers_mod, "sweep_stopped", lambda machine, **target: swept.append(target) or []
    )
    monkeypatch.setattr(
        workers_mod, "sweep_dirs", lambda machine, **target: homes.append(target) or []
    )
    manager.remove_pool_machine("localhost")
    manager.add_pool_machine("gpu-box", "me@gpu-box")
    make("a")
    q.enqueue("position_eval", "a", confirm=True)
    queue = manager.queue_store.load()
    queue.entry("position_eval", "a").bundle = queue_mod.BUNDLE_READY
    manager.queue_store.save(queue)
    q._builds.clear()
    q.tick()
    a = manager.tasks.load(SPEC, "a")
    for w in a.workers:
        w.finished, w.desired_state = True, "paused"
    q.tick()
    _drain_then_tick(q)

    logs = manager.tasks.paths(SPEC, "a").logs_dir
    assert sorted(p.name for p in logs.glob("*.container.log")) == [
        "ssh-0.container.log",
        "ssh-1.container.log",
    ]
    # The drain sweeps both; removing the trainer sweeps it once more for a
    # final state pair, and then its home's staging and ledger
    # (WorkerManager.remove_worker).
    assert [t["container"] for t in swept] == [
        "scz-position_eval-a-ssh-0",
        "scz-position_eval-a-ssh-1",
        "scz-position_eval-a-ssh-0",
        "scz-position_eval-a-ssh-0",
    ]
    assert [(t["container"], t["rel"]) for t in homes] == [
        ("scz-position_eval-a-ssh-0", "data/generations")
    ]
    assert manager.tasks.load(SPEC, "a").workers == []
    assert _lease(manager, "gpu-box") is None


def test_a_queued_or_placed_tag_refuses_slots_added_by_hand(queued):
    """Hand-added slots on a queued tag would be skipped at placement, leaving
    the machine leased while the tag runs elsewhere."""
    q, manager, make = queued
    make("a")
    q.refuse_hand_placement("position_eval", "a")  # neither queued nor placed
    q.enqueue("position_eval", "a", confirm=True)
    with pytest.raises(AssertionError, match="is queued"):
        q.refuse_hand_placement("position_eval", "a")
    q.tick()
    assert _lease(manager).tag == "a"
    with pytest.raises(AssertionError, match="placed on pool machine localhost"):
        q.refuse_hand_placement("position_eval", "a")


def test_an_empty_pool_says_so(queued):
    q, manager, make = queued
    manager.remove_pool_machine("localhost")
    make("a")
    q.enqueue("position_eval", "a", confirm=True)
    q.tick()
    assert q.status()["entries"][0]["refusals"] == {"pool": EMPTY_POOL}


def test_a_hand_placed_trainer_counts_other_tags_on_the_gpu(queued):
    """Two tags' trainers by hand on one GPU: the second is refused while the
    first is meant to run, and allowed once it is paused."""
    q, manager, make = queued
    first = make("first")
    w = manager.add_local(SPEC, first, "train", None)
    w.desired_state = "running"
    second = make("second")
    with pytest.raises(AssertionError, match="would need 28.1 GiB"):
        manager.add_local(SPEC, second, "train", None)
    w.desired_state = "paused"
    manager.add_local(SPEC, second, "train", None)


def test_the_plan_shows_the_roles_and_each_machines_slots(queued):
    """What the queue would start for a tag, before and after enqueueing: the
    roles come from its params (match eval only with a cadence), and each
    machine gets its own thread count and fit."""
    q, manager, make = queued
    make("a", match_every_generations=0)
    plan = q.plan("position_eval", "a")
    assert plan["roles"] == ["train", "generate"]
    [local] = plan["machines"]
    assert local["machine"] == "localhost" and local["refusal"] is None
    assert local["slots"] == [
        {"role": "train", "threads": None},
        {"role": "generate", "threads": 28},
    ]
    make("b", match_every_generations=5)
    plan = q.plan("position_eval", "b")
    assert plan["roles"] == ["train", "generate", "match_eval"]
    assert "needs 16.6 GiB" in plan["machines"][0]["refusal"]
    q.enqueue("position_eval", "a", confirm=True)
    manager.remove_pool_machine("localhost")
    assert q.plan("position_eval", "a")["machines"] == []  # still answers once queued


def test_an_ssh_machine_added_after_enqueueing_gets_the_tag_built(queued, monkeypatch):
    """Tags enqueued while the pool had only localhost need no bundle; one
    added later (a registered machine, or rental capacity) must start the
    build, or the tag waits on 'its bundle is none' forever."""
    q, manager, make = queued
    monkeypatch.setattr(q, "_submit_build", lambda spec, task, e, pool: None)
    make("a", match_every_generations=5)  # 16.6 GiB: not localhost's 16
    q.enqueue("position_eval", "a", confirm=True)
    assert manager.queue_store.load().entry("position_eval", "a").bundle == queue_mod.BUNDLE_NONE
    q.tick()
    assert manager.queue_store.load().entry("position_eval", "a").bundle == queue_mod.BUNDLE_NONE
    monkeypatch.setattr(workers_mod, "SshMachine", _Link)
    manager.add_pool_machine("gpu-box", "me@gpu-box")
    q.tick()
    assert (
        manager.queue_store.load().entry("position_eval", "a").bundle == queue_mod.BUNDLE_BUILDING
    )


def test_a_retiring_machine_says_so_on_the_queue_row():
    """Stop all cloud spending retires a rental; until it is terminated the
    queue must not show it as a machine that would take a waiting tag."""
    m = PoolMachine(name="cap-1", kind="ssh", retiring=True)
    assert "retiring" in TagQueue._why_not(m, [])
