"""The control plane under simulation (docs/plans/dashboard_state_model.md §7).

Seeded runs drive the real WorkerManager, TagQueue and PoolRentals through
the fake world in sim_world.py: operator actions, worker and container
crashes, spot interruptions, vanishing instances, failed listings and
dashboard restarts, a dashboard pass after each. After every step the plan's
invariants are checked. A failing seed replays exactly.

Invariants checked here, as numbered in the plan:
  I1  the control database imports the stores cleanly: every constraint
      holds and no row of the decision table is refused
  I2  a retiring machine never gains a lease
  I3  every billing instance is a pool or task machine; after Stop all cloud
      spending the burn reaches zero within a few passes
  I4  a slot is counted crashed only for an exit the world caused
  I5  a tag's training cursor never goes backwards in its tag dir
  I6  after a pass, no queued tag waits while a free pool machine fits it
  I7  status reads change no record (test_status_reads_change_nothing)
  I8  a restart changes no outcome (test_a_restart_changes_no_outcome)
"""

import asyncio
import dataclasses
import json
import random

import pytest
import sim_world
from scribblez.dashboard import control_db
from scribblez.dashboard.tag_queue import TagQueue
from scribblez.dashboard.workers import WorkerManager
from sim_world import SPEC, World

# The real bucket-sync launchers, which conftest stubs for every test: here
# they launch fake watchers inside the world.
_REAL_ENSURE_SYNC = WorkerManager._ensure_sync
_REAL_SYNC_ONCE = WorkerManager.sync_once

TAGS = {"t0": 600, "t1": 800, "t2": 1000, "t3": 700}  # tag -> max_rows
STOP_ALL_GRACE = 8  # passes after Stop all cloud spending for the burn to reach zero


class Violation(AssertionError):
    pass


class Sim:
    """One seeded run: the world, the dashboard it drives, and the history
    the invariants read."""

    def __init__(self, tmp_path, monkeypatch, seed: int):
        self.rng = random.Random(seed)
        self.seed = seed
        self.monkeypatch = monkeypatch
        self.world = World(tmp_path)
        sim_world.install(monkeypatch, self.world)
        monkeypatch.setattr(WorkerManager, "_ensure_sync", _REAL_ENSURE_SYNC)
        monkeypatch.setattr(WorkerManager, "sync_once", _REAL_SYNC_ONCE)
        self.world.add_host("box")
        self._boot()
        self.command(self.manager.add_pool_machine, "localhost")
        self.command(self.manager.add_pool_machine, "box", "me@box")
        self.command(self.manager.add_capacity, "cap", "g6.2xlarge", spot=True, cap=1)
        for tag, max_rows in TAGS.items():
            self.command(self.manager.tasks.create, SPEC, tag, {"max_rows": max_rows})
        self.step_no = 0
        self.log: list[str] = []
        self.rows_seen: dict[str, int] = {}
        # (machine name, instance id) -> when it was retired: a later rental
        # may reuse the name once the retired one is terminated.
        self.retired_at: dict[tuple[str, str | None], float] = {}
        self.stopped_all_at: int | None = None

    def _boot(self):
        """Start (or restart) the dashboard: a manager and queue with nothing
        in memory, over the stores on disk."""
        self.manager = WorkerManager(self.world.mount_root)
        sim_world.wire_manager(self.monkeypatch, self.manager, self.world)
        self.manager.claim_writer()
        self.queue = TagQueue(self.manager)
        self.queue._drain_thread = sim_world.SyncExecutor()

    def restart(self):
        self.manager.shutdown()
        self.queue.shutdown()
        self._boot()

    def command(self, fn, *args, **kwargs):
        """Run `fn` as the dashboard runs a handler's change: a command on its
        writer thread. The sim's own reads stay on this thread."""
        return asyncio.run(self.manager.offload(fn, *args, **kwargs))

    def dashboard_pass(self):
        async def run():
            await self.manager.offload(self.queue.tick)
            await self.manager.reconcile()

        asyncio.run(run())

    # -- one step

    def step(self, event: str | None = None):
        self.step_no += 1
        event = event or self._draw_event()
        self.log.append(f"{self.step_no}: {event}")
        getattr(self, f"_do_{event}")()
        self.world.step()
        self.dashboard_pass()
        self.check()

    def _draw_event(self) -> str:
        events = [
            ("pass", 40), ("enqueue", 12), ("dequeue", 2), ("requeue", 3), ("release", 1),
            ("stop_all", 1), ("rent_again", 2), ("crash_local", 3), ("crash_container", 3),
            ("interrupt_spot", 2), ("vanish", 1), ("listing_fails", 2), ("restart", 3),
        ]  # fmt: skip
        return self.rng.choices([e for e, _ in events], weights=[w for _, w in events])[0]

    def _pick(self, items):
        items = sorted(items)
        return self.rng.choice(items) if items else None

    def _operator(self, fn, *args, **kwargs):
        """An operator action; one the dashboard refuses is just refused."""
        try:
            self.command(fn, *args, **kwargs)
        except (AssertionError, KeyError) as e:
            self.log.append(f"   refused: {e}")

    def _do_pass(self):
        self.world.provider.listing_fails = False

    def _do_enqueue(self):
        tag = self._pick(TAGS)
        self._operator(self.queue.enqueue, SPEC.name, tag, confirm=True)

    def _do_dequeue(self):
        entries = [e.tag for e in self.manager.queue_store.load().entries]
        tag = self._pick(entries)
        if tag:
            self._operator(self.queue.dequeue, SPEC.name, tag)

    def _leased_tags(self) -> list[str]:
        pool = self.manager.pool_store.load()
        return [m.lease.tag for m in pool.machines if m.lease and m.lease.phase == "running"]

    def _do_requeue(self):
        tag = self._pick(self._leased_tags())
        if tag:
            self._operator(self.queue.requeue, SPEC.name, tag)

    def _do_release(self):
        tag = self._pick(self._leased_tags())
        if tag:
            self._operator(self.queue.release, SPEC.name, tag)

    def _do_stop_all(self):
        self.command(self.queue.stop_cloud)
        now = self.world.clock.now
        for m in self.manager.pool_store.load().machines:
            if m.retiring:
                self.retired_at.setdefault(_identity(m), now)
        self.stopped_all_at = self.step_no

    def _do_rent_again(self):
        self.command(self.manager.set_capacity_cap, "cap", 1)
        self.stopped_all_at = None

    def _do_crash_local(self):
        alive = [
            (p.env["SCZ_TAG"], p.env["SCZ_WORKER_ID"])
            for p in self.world.procs.values()
            if p.alive and p.worker is not None
        ]
        victim = self._pick(alive)
        if victim:
            self.world.crash_local(*victim)

    def _do_crash_container(self):
        running = [
            (h.name, name)
            for h in self.world.hosts.values()
            for name, c in h.containers.items()
            if c.state == "running"
        ]
        victim = self._pick(running)
        if victim:
            self.world.crash_container(self.world.hosts[victim[0]], victim[1])

    def _do_interrupt_spot(self):
        victim = self._pick(i.id for i in self.world.billing())
        if victim:
            self.world.interrupt_spot(victim)

    def _do_vanish(self):
        victim = self._pick(i.id for i in self.world.billing())
        if victim:
            self.world.vanish(victim)

    def _do_listing_fails(self):
        self.world.provider.listing_fails = True

    def _do_restart(self):
        self.restart()

    # -- the invariants

    def fail(self, invariant: str, detail: str):
        tail = "\n".join(self.log[-25:])
        raise Violation(f"{invariant} (seed {self.seed}, step {self.step_no}): {detail}\n{tail}")

    def check(self):
        self._check_i1()
        self._check_i2()
        self._check_i3()
        self._check_i4()
        self._check_i5()
        self._check_i6()

    def _check_i1(self):
        conn = control_db.connect(self.world.mount_root / "sim-check.db")
        try:
            findings = control_db.import_stores(conn, self.manager)
        finally:
            conn.close()
        if findings:
            self.fail("I1", "; ".join(str(f) for f in findings))

    def _check_i2(self):
        for m in self.manager.pool_store.load().machines:
            retired = self.retired_at.get(_identity(m))
            if retired is not None and m.lease is not None and m.lease.since > retired:
                self.fail("I2", f"retiring {m.name} gained a lease for {m.lease.tag}")

    def _check_i3(self):
        pool = self.manager.pool_store.load()
        pool_owners = {f"pool/{m.name}" for m in pool.machines}
        task_owners = {
            f"{spec.name}/{task.tag}/{m.name}"
            for spec, task in self.manager.tasks.load_all()
            for m in task.machines
        }
        for inst in self.world.billing():
            if inst.owner not in pool_owners | task_owners:
                self.fail("I3", f"{inst.id} ({inst.owner}) bills and no machine tracks it")
        if (
            self.stopped_all_at is not None
            and self.step_no - self.stopped_all_at > STOP_ALL_GRACE
            and self.world.billing()
        ):
            self.fail("I3", f"still billing {STOP_ALL_GRACE} passes after Stop all cloud spending")

    def _check_i4(self):
        for key, crashes in self.manager._crashes.items():
            _, tag, worker_id = key.split("/", 2)
            if crashes and (tag, worker_id) not in self.world.crashed:
                self.fail("I4", f"{key} counted a crash the world never caused: {crashes}")

    def _check_i5(self):
        for tag in TAGS:
            rows = sim_world.read_rows(self.world.mount_root / "tags" / SPEC.name / tag)
            if rows < self.rows_seen.get(tag, 0):
                self.fail("I5", f"{tag}'s cursor went from {self.rows_seen[tag]} to {rows}")
            self.rows_seen[tag] = max(rows, self.rows_seen.get(tag, 0))

    def _check_i6(self):
        for entry in self.queue.status()["entries"]:
            free = [m for m, why in entry["refusals"].items() if why is None]
            if free:
                self.fail("I6", f"{entry['tag']} waits while {free} would take it")

    # -- outcomes

    def run_to_completion(self, max_steps: int = 400):
        """Passes only, until every tag has trained its budget and holds
        nothing; for the restart test's before/after comparison."""
        for _ in range(max_steps):
            if self.quiescent():
                return
            self.step("pass")
        raise Violation(f"not quiescent after {max_steps} passes\n" + "\n".join(self.log[-25:]))

    def quiescent(self) -> bool:
        pool = self.manager.pool_store.load()
        return (
            not self.manager.queue_store.load().entries
            and all(m.lease is None for m in pool.machines)
            and all(not t.workers for _, t in self.manager.tasks.load_all())
        )

    def outcome(self) -> dict:
        """What a run achieved: each tag's trained rows and projected state."""
        conn = control_db.connect(self.world.mount_root / "sim-outcome.db")
        try:
            control_db.import_stores(conn, self.manager)
            states = control_db.project(conn)
        finally:
            conn.close()
        root = self.world.mount_root / "tags" / SPEC.name
        return {tag: (sim_world.read_rows(root / tag), states[(SPEC.name, tag)]) for tag in TAGS}


def _identity(m) -> tuple[str, str | None]:
    return m.name, m.machine.instance_id if m.machine is not None else None


@pytest.fixture
def sim(tmp_path, monkeypatch):
    def make(seed: int) -> Sim:
        return Sim(tmp_path, monkeypatch, seed)

    return make


@pytest.mark.parametrize("seed", range(16))
def test_random_runs_keep_the_invariants(sim, seed):
    run = sim(seed)
    for _ in range(200):
        run.step()


def test_a_seed_replays_exactly(tmp_path_factory, monkeypatch):
    """What makes a failing seed worth reporting: the same seed takes the same
    steps to the same place."""
    runs = []
    for _ in range(2):
        run = Sim(tmp_path_factory.mktemp("run"), monkeypatch, seed=5)
        for _ in range(80):
            run.step()
        runs.append((run.log, run.outcome()))
    assert runs[0] == runs[1]


def test_status_reads_change_nothing(sim):
    """I7: serving the dashboard's pages and polls changes no record."""
    run = sim(0)
    for tag in TAGS:
        run.command(run.queue.enqueue, SPEC.name, tag, confirm=True)
    for _ in range(40):
        run.step("pass")
        before = _snapshot(run)
        run.manager.pool_status()
        run.queue.status()
        for spec, task in list(run.manager.tasks.load_all()):
            run.manager.worker_status(spec, task)
            run.manager.machine_status(spec, task)
            run.queue.plan(spec.name, task.tag)
        after = _snapshot(run)
        assert after == before, f"step {run.step_no}: a status read changed a record"


def _snapshot(run: Sim) -> str:
    records = {
        "tasks": [dataclasses.asdict(t) for _, t in run.manager.tasks.load_all()],
        "pool": dataclasses.asdict(run.manager.pool_store.load()),
        "queue": dataclasses.asdict(run.manager.queue_store.load()),
    }
    return json.dumps(records, sort_keys=True, default=str)


@pytest.mark.parametrize("restart_at", [3, 9, 20])
def test_a_restart_changes_no_outcome(sim, restart_at, tmp_path_factory, monkeypatch):
    """I8: the same run with a dashboard restart in the middle ends the same:
    every tag trains exactly its budget and ends idle."""
    outcomes = []
    for restart in (False, True):
        run = Sim(tmp_path_factory.mktemp("run"), monkeypatch, seed=0)
        for tag in TAGS:
            run.command(run.queue.enqueue, SPEC.name, tag, confirm=True)
        for _ in range(restart_at):
            run.step("pass")
        if restart:
            run.step("restart")
        run.run_to_completion()
        outcomes.append(run.outcome())
    assert outcomes[0] == outcomes[1]
    assert outcomes[0] == {tag: (rows, "idle") for tag, rows in TAGS.items()}
