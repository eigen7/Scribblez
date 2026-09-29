"""The tag queue (docs/plans/tag_queue.md §4-5): places queued tags on free
pool machines, and releases each machine when its tag completes or fails.

Each reconcile pass runs `tick` on the worker manager's blocking thread,
ahead of the per-task steps, so slots a placement creates are started the
same pass. A tag's time on a machine is a lease (dashboard/pool.py) that
moves through these phases:

  reserved   written before anything else, so a crash mid-placement is
             resumed rather than repeated
  running    the tag's slots exist and want to run
  releasing  the tag completed, failed, or was requeued; its output is being
             drained off the machine on the drain thread
  held       the tag failed and no queued tag could use the machine: the
             failed slots' containers are kept for investigation until a
             queued tag can use it or the operator removes the machine

Draining before removal is what keeps a hand-over from losing anything:
every remote slot's full container log is saved into the tag's logs/, a
stopped container's last output is swept, and a tag whose output travels
through the bucket gets one final sync before its slots go.
"""

import copy
import time
from concurrent.futures import Future, ThreadPoolExecutor

from cloud.bundles import BundleManifest
from cloud.ssh_transfer import sweep_stopped

from scribblez import params as params_mod
from scribblez import workloads
from scribblez.dashboard import placement, tasks
from scribblez.dashboard import pool as pool_mod
from scribblez.dashboard import queue as queue_mod
from scribblez.dashboard.pool import Capacity, Lease, Pool, PoolMachine
from scribblez.dashboard.pool_rentals import PoolRentals
from scribblez.dashboard.queue import Queue, QueueEntry
from scribblez.dashboard.workers import (
    WorkerManager,
    _container_name,
    _machine_key,
    _machine_link,
    worker_pid_alive,
)

RESERVED, RUNNING, RELEASING, HELD = "reserved", "running", "releasing", "held"
EMPTY_POOL = "no machines or rental capacity; add one above"
# The reason shown on a lease released by Requeue. What sends the tag back to
# the head of the queue is Lease.requeue, which survives a restart mid-release
# and cannot be overwritten by a drain error's reason.
REQUEUED = "requeued"

# A queue-placed slot that crashes this many times within the window is
# failed rather than restarted forever.
FAIL_AFTER = 3
FAIL_WINDOW_SECONDS = 1800.0

# After the provider refuses a rental (quota, capacity), how long before the
# queue asks again under that capacity entry.
RENT_RETRY_SECONDS = 300.0

LOCAL_CODE_WARNING = (
    "local slots on this machine run the checkout in /workspace/repo as it is when they "
    "start, not a pinned bundle"
)


def _params(spec, task: tasks.TaskRecord):
    return params_mod.validate(spec.params_cls, task.params)


def _requeued(lease: Lease, task: tasks.TaskRecord) -> QueueEntry:
    """The queue entry a released lease's tag goes back in with: the lease's
    eligibility, and the task's pinned bundle if it has one."""
    bundle = queue_mod.BUNDLE_READY if task.bundle_id else queue_mod.BUNDLE_NONE
    return QueueEntry(
        lease.workload,
        lease.tag,
        time.time(),
        list(lease.machines),
        lease.memory_override_gb,
        bundle,
    )


class TagQueue:
    def __init__(self, manager: WorkerManager):
        self._m = manager
        # Entry key -> its enqueue-time bundle build, on the manager's build
        # thread (see _submit_build).
        self._builds: dict[tuple[str, str], Future] = {}
        # Pool machine name -> the drain of its releasing lease.
        self._drains: dict[str, Future] = {}
        self._drain_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scz-drain")
        # Entry key -> {pool machine: why it cannot take the entry}, from the
        # last pass, for the queue view.
        self._refusals: dict[tuple[str, str], dict[str, str]] = {}
        self._rentals = PoolRentals(manager)
        # Capacity entry -> (why its last rental was refused, retry after).
        self._rent_refused: dict[str, tuple[str, float]] = {}

    def _lookup(self, workload: str, tag: str):
        """(spec, task record or None) for a queue entry or lease."""
        spec = workloads.get(workload)
        return spec, self._m.tasks.load(spec, tag)

    def _fit_args(self, spec, task: tasks.TaskRecord) -> tuple:
        """The tag-side arguments placement.refusal takes: spec, params, and
        where its training state lives."""
        paths = self._m.tasks.paths(spec, task.tag)
        return spec, _params(spec, task), placement.state_home(paths, task)

    # ---- operator actions ----------------------------------------------------

    def enqueue(
        self,
        workload: str,
        tag: str,
        *,
        machines: list[str] | None = None,
        memory_override_gb: float | None = None,
        confirm: bool = False,
    ) -> dict:
        """Queue a tag. Returns {"queued", "warnings"}: with warnings and no
        `confirm`, nothing is queued and the caller asks the operator first
        (docs/plans/tag_queue.md §1 and §4)."""
        spec, task = self._lookup(workload, tag)
        assert spec.layout, f"{workload} tags are not queueable yet (no layout)"
        assert task is not None, f"tag '{tag}' has no task record"
        assert not task.workers, "the tag already has slots; remove them to queue it"
        queue, pool = self._m.queue_store.load(), self._m.pool_store.load()
        assert queue.find(workload, tag) is None, f"{workload}/{tag} is already queued"
        assert self._leased(pool, workload, tag) is None, f"{workload}/{tag} holds a machine"
        entry = QueueEntry(workload, tag, time.time(), list(machines or []), memory_override_gb)
        warnings = self._warnings(spec, task, entry, queue, pool)
        if warnings and not confirm:
            return {"queued": False, "warnings": warnings}
        if self._ssh_targets(spec, task, entry, pool):
            entry.bundle = queue_mod.BUNDLE_BUILDING
            self._submit_build(spec, task, entry, pool)
        queue.entries.append(entry)
        self._m.queue_store.save(queue)
        return {"queued": True, "warnings": warnings}

    def dequeue(self, workload: str, tag: str):
        queue = self._m.queue_store.load()
        queue.entries.remove(queue.entry(workload, tag))
        self._builds.pop((workload, tag), None)
        self._m.queue_store.save(queue)

    def move(self, workload: str, tag: str, delta: int):
        """Move a queued tag `delta` places toward the tail (negative: head)."""
        queue = self._m.queue_store.load()
        entry = queue.entry(workload, tag)
        i = queue.entries.index(entry)
        j = max(0, min(len(queue.entries) - 1, i + delta))
        queue.entries.insert(j, queue.entries.pop(i))
        self._m.queue_store.save(queue)

    def release(self, workload: str, tag: str):
        """The operator's Release, for a tag with no end condition: finish all
        its slots, which completes it, and the next pass releases its machine."""
        pool = self._m.pool_store.load()
        m = self._leased(pool, workload, tag)
        assert m is not None and m.lease.phase == RUNNING, f"{workload}/{tag} is not placed"
        spec, task = self._lookup(workload, tag)
        for w in task.workers:
            w.desired_state = "paused"
            w.finished = True
        self._m.tasks.save(spec, task)

    def requeue(self, workload: str, tag: str):
        """Take a placed or held tag off its machine and put it back at the
        head of the queue, its data kept. Its slots are paused now; the release
        drains them once they have stopped."""
        pool = self._m.pool_store.load()
        m = self._leased(pool, workload, tag)
        assert m is not None and m.lease.phase != RELEASING, f"{workload}/{tag} is not placed"
        spec, task = self._lookup(workload, tag)
        for w in task.workers:
            w.desired_state = "paused"
        self._m.tasks.save(spec, task)
        m.lease.requeue = True
        self._start_release(m, pool, REQUEUED)

    def stop_cloud(self, dry_run: bool = False) -> dict:
        """Get the cloud burn to zero (the burn strip's Stop all cloud
        spending), or with `dry_run` only say what that would do, for the
        confirmation. Every capacity cap goes to 0, so nothing more is
        rented. Every pool rental retires: its tag, if one runs there, goes
        back to the head of the queue with its data (requeue), and the machine
        is terminated once free, never placed on again. Every task-owned
        rented machine has its slots paused and stops once they are down.
        Every orphan instance is terminated. Returns what was (or would be)
        done, by kind."""
        pool = self._m.pool_store.load()
        rentals = [m for m in pool.machines if m.capacity is not None]
        report = {
            "caps": [c.name for c in pool.capacity if c.cap > 0],
            "requeue": [
                f"{m.lease.workload}/{m.lease.tag}"
                for m in rentals
                if m.lease is not None and m.lease.phase in (RESERVED, RUNNING)
            ],
            "terminate": [m.name for m in rentals],
            "stop": [f"{s.name}/{t.tag}/{m.name}" for s, t, m in self._m.task_rentals()],
            "orphans": [o["instance_id"] for o in self._m.orphans()],
        }
        if dry_run:
            return report
        for c in pool.capacity:
            c.cap = 0
        for m in rentals:
            m.retiring = True
        self._m.pool_store.save(pool)
        for m in rentals:
            if m.lease is None:
                self._m.remove_pool_machine(m.name)  # terminates it
            elif m.lease.phase in (RESERVED, RUNNING):
                self.requeue(m.lease.workload, m.lease.tag)
            elif m.lease.phase == HELD:
                self._start_release(m, pool, "released: stopping all cloud spending")
            # A releasing one is terminated once its release completes.
        self._m.stop_task_rentals()
        for instance_id in report["orphans"]:
            self._m.terminate_orphan(instance_id)
        return report

    def plan(self, workload: str, tag: str) -> dict:
        """What the queue would start for a tag, shown before and after it is
        enqueued: the roles its layout asks for (they follow from its params),
        and for each pool machine and capacity entry the slots it would get
        there, their summed GPU need, and why it could not go there, if so."""
        spec, task = self._lookup(workload, tag)
        assert spec.layout, f"{workload} tags are not queueable yet (no layout)"
        assert task is not None, f"tag '{tag}' has no task record"
        fit = self._fit_args(spec, task)
        params = fit[1]
        pool = self._m.pool_store.load()
        entry = self._m.queue_store.load().find(workload, tag) or QueueEntry(workload, tag, 0.0)
        targets = [(m.name, m) for m in pool.machines] + [
            (f"rent {c.name}", self._rentals.prospect(c)) for c in pool.capacity
        ]
        machines = []
        for name, m in targets:
            slots = placement.plan_for(spec, params, m)
            machines.append(
                {
                    "machine": name,
                    "slots": [{"role": p.role, "threads": p.threads} for p in slots],
                    "gpu_gb": placement.gpu_total(spec, slots, entry),
                    "refusal": placement.refusal(*fit, entry, m, need_bundle=False),
                }
            )
        roles = [p.role for p in workloads.resolve(spec.layout)(params, 1, None)]
        return {"roles": roles, "machines": machines}

    def refuse_hand_placement(self, workload: str, tag: str):
        """Refuse a slot added by hand to a queued or placed tag. Its slots are
        the queue's to create: a queued tag with slots elsewhere would be
        placed with those roles skipped, leaving the machine leased and idle."""
        assert self._m.queue_store.load().find(workload, tag) is None, (
            f"{workload}/{tag} is queued; dequeue it to add slots by hand"
        )
        m = self._leased(self._m.pool_store.load(), workload, tag)
        assert m is None, f"{workload}/{tag} is placed on pool machine {m.name}; requeue it first"

    def status(self) -> dict:
        """The queue in order, each entry with whether it ends on its own, its
        bundle, and why each pool machine did not take it on the last pass."""
        rows = []
        for e in self._m.queue_store.load().entries:
            spec, task = self._lookup(e.workload, e.tag)
            ends = task is not None and placement.has_end_condition(spec, _params(spec, task))
            rows.append(
                {
                    "workload": e.workload,
                    "tag": e.tag,
                    "machines": e.machines,
                    "memory_override_gb": e.memory_override_gb,
                    "bundle": e.bundle,
                    "end_condition": ends,
                    "refusals": self._refusals.get(e.key, {}),
                }
            )
        return {"entries": rows}

    # ---- the pass ------------------------------------------------------------

    def tick(self):
        """Advance every lease, then place what fits on the free machines."""
        pool, queue = self._m.pool_store.load(), self._m.queue_store.load()
        self._drop_stale_entries(queue, pool)
        self._advance_builds(queue, pool)
        self._rentals.reconcile(pool)
        # Contained per lease, as reconcile contains per slot: one machine
        # that cannot be reached must not stall every other lease, nor the
        # placements behind them.
        for m in pool.machines:
            if m.lease is not None:
                try:
                    self._advance_lease(m, pool, queue)
                except Exception as e:  # noqa: BLE001 -- retried next pass
                    print(f"tag queue: lease on {m.name}: {e}")
        self._place(pool, queue)

    def shutdown(self):
        self._drain_thread.shutdown(wait=False, cancel_futures=True)

    # ---- enqueue helpers -----------------------------------------------------

    @staticmethod
    def _leased(pool: Pool, workload: str, tag: str) -> PoolMachine | None:
        return next(
            (
                m
                for m in pool.machines
                if m.lease is not None and (m.lease.workload, m.lease.tag) == (workload, tag)
            ),
            None,
        )

    def _warnings(self, spec, task, entry: QueueEntry, queue: Queue, pool: Pool) -> list[str]:
        """What the operator should confirm before `entry` is queued: tags in
        the queue or on machines with no end condition, which hold a machine
        until released by hand, and local slots that run the live checkout."""
        out = []
        keys = [entry.key, *(e.key for e in queue.entries)]
        keys += [(m.lease.workload, m.lease.tag) for m in pool.machines if m.lease is not None]
        endless = []
        for workload, tag in keys:
            s, t = self._lookup(workload, tag)
            if t is not None and not placement.has_end_condition(s, _params(s, t)):
                endless.append(f"{workload}/{tag}")
        if endless:
            out.append(
                f"no end condition: {', '.join(endless)}; each holds its machine until "
                "released by hand"
            )
        fit = self._fit_args(spec, task)
        for m in pool.machines:
            if m.kind == "local" and placement.refusal(*fit, entry, m, need_bundle=False) is None:
                out.append(f"{m.name}: {LOCAL_CODE_WARNING}")
        return out

    def _ssh_targets(self, spec, task, entry: QueueEntry, pool: Pool) -> list[PoolMachine]:
        """The ssh machines the tag could run on, bundle aside: the pool's own,
        and what each capacity entry would rent."""
        fit = self._fit_args(spec, task)
        candidates = [
            *pool.machines,
            *(self._rentals.prospect(c) for c in pool.capacity),
        ]
        return [
            m
            for m in candidates
            if m.kind == "ssh" and placement.refusal(*fit, entry, m, need_bundle=False) is None
        ]

    def _submit_build(self, spec, task, entry: QueueEntry, pool: Pool):
        """Build the tag's bundle for its ssh machines' archs on the build
        thread; _advance_builds records the archs it detected and pins the
        bundle when done. Detecting a machine's arch pulls the worker image
        there, which is why this is not done inline. The thread gets its own
        copies of what it reads, and changes no record."""
        targets = [
            (m.name, _machine_link(m.machine), m.machine.arch)
            for m in self._ssh_targets(spec, task, entry, pool)
        ]
        known = set(task.bundle_archs) | {arch for _, _, arch in targets if arch}
        # Any worker image carries the compiler detect_arch asks.
        image = self._m._creds().registry.image_for(spec.roles[0].runtime)

        def build() -> tuple[dict[str, str], BundleManifest]:
            detected = {}
            for name, link, arch in targets:
                if arch is None:
                    link.pull_image(image)
                    detected[name] = link.detect_arch(image)
            return detected, self._m._build_bundle(sorted(known | set(detected.values())))

        self._builds[entry.key] = self._m._builds.submit(build)

    def _advance_builds(self, queue: Queue, pool: Pool):
        """Start a build for every entry with no bundle that an ssh machine or
        capacity entry could now take (one added after the tag was enqueued),
        and pin every finished build to its task. A build that was in flight
        when the dashboard restarted is submitted again."""
        changed = False
        for e in queue.entries:
            spec, task = self._lookup(e.workload, e.tag)
            if e.bundle == queue_mod.BUNDLE_NONE and self._ssh_targets(spec, task, e, pool):
                e.bundle = queue_mod.BUNDLE_BUILDING
                changed = True
            if e.bundle != queue_mod.BUNDLE_BUILDING:
                continue
            future = self._builds.get(e.key)
            if future is None:
                try:
                    self._submit_build(spec, task, e, pool)
                except Exception as ex:  # noqa: BLE001 -- e.g. no cloud credentials
                    e.bundle = f"failed: {ex}"
                    changed = True
                continue
            if not future.done():
                continue
            del self._builds[e.key]
            try:
                detected, manifest = future.result()
                for name, arch in detected.items():
                    m = pool.find(name)
                    if m is not None and m.machine is not None:
                        m.machine.arch = arch
                self._m._pin_bundle(spec, task, manifest)
                e.bundle = queue_mod.BUNDLE_READY
            except Exception as ex:  # noqa: BLE001 -- shown on the entry, retried on re-enqueue
                e.bundle = f"failed: {ex}"
            changed = True
        if changed:
            self._m.pool_store.save(pool)  # the archs the builds detected
            self._m.queue_store.save(queue)

    def _drop_stale_entries(self, queue: Queue, pool: Pool):
        """Drop entries whose task is gone, and entries a placement interrupted
        by a restart left behind after writing the lease."""
        stale = [
            e
            for e in queue.entries
            if self._lookup(e.workload, e.tag)[1] is None or self._leased(pool, *e.key) is not None
        ]
        for e in stale:
            queue.entries.remove(e)
        if stale:
            self._m.queue_store.save(queue)

    # ---- placement -----------------------------------------------------------

    def _place(self, pool: Pool, queue: Queue):
        """Match queued tags to free machines (placement.match) and place each
        match: lease first, then the queue entry goes, then the slots."""
        tasks_now = list(self._m.all_tasks())
        busy = {m.name: self._m.occupants(m, tasks_now) for m in pool.machines}
        free = [m for m in pool.machines if m.lease is None and not m.retiring and not busy[m.name]]
        args = {}  # entry key -> placement.refusal's tag-side arguments
        for e in queue.entries:
            spec, task = self._lookup(e.workload, e.tag)
            args[e.key] = self._fit_args(spec, task)
        # Whether each free machine can take each entry: once per pair. The
        # queue view's reasons are computed after the pass acts (_note_refusals).
        fits = {
            (e.key, m.name): placement.refusal(*args[e.key], e, m) is None
            for e in queue.entries
            for m in free
        }

        def eligible(e: QueueEntry, m: PoolMachine) -> bool:
            return fits[(e.key, m.name)]

        matches = placement.match(queue.entries, free, eligible)
        for e in list(queue.entries):
            if e.key not in matches:
                continue
            m = pool.machine(matches[e.key])
            spend = m.machine.spend if m.capacity is not None else 0.0
            m.lease = Lease(
                e.workload,
                e.tag,
                RESERVED,
                time.time(),
                spend_start=spend,
                machines=list(e.machines),
                memory_override_gb=e.memory_override_gb,
            )
            self._m.pool_store.save(pool)
            queue.entries.remove(e)
            self._m.queue_store.save(queue)
            self._start_slots(m, pool)
        self._rent_for(pool, queue, args)
        self._note_refusals(pool, queue, args, busy)

    def _note_refusals(self, pool: Pool, queue: Queue, args: dict, busy: dict):
        """Why each still-queued tag is still queued, per machine and capacity
        entry, after this pass's placements and rentals (the queue view)."""
        if not pool.machines and not pool.capacity:
            self._refusals = {e.key: {"pool": EMPTY_POOL} for e in queue.entries}
            return
        self._refusals = {
            e.key: {
                **{
                    m.name: self._why_not(m, busy.get(m.name, []))
                    or placement.refusal(*args[e.key], e, m)
                    for m in pool.machines
                },
                **{
                    f"rent {c.name}": self._why_not_rent(c, pool)
                    or placement.refusal(*args[e.key], e, self._rentals.prospect(c))
                    for c in pool.capacity
                },
            }
            for e in queue.entries
        }

    def _rent_for(self, pool: Pool, queue: Queue, args: dict):
        """Rent for queued tags no owned machine took, in queue order: each gets
        the first capacity entry below its cap whose type it fits. The machine
        is leased from the start, so placement proceeds as on any machine; its
        slots start once it is up."""
        for e in list(queue.entries):
            for c in pool.capacity:
                if self._why_not_rent(c, pool) is not None:
                    continue
                if placement.refusal(*args[e.key], e, self._rentals.prospect(c)) is not None:
                    continue
                try:
                    m = self._rentals.rent(c, pool, e)
                except AssertionError as ex:  # the provider refused: quota, capacity
                    self._rent_refused[c.name] = (str(ex), time.time() + RENT_RETRY_SECONDS)
                    continue
                self._rent_refused.pop(c.name, None)
                queue.entries.remove(e)
                self._m.queue_store.save(queue)
                self._start_slots(m, pool)
                break

    def _why_not_rent(self, c: Capacity, pool: Pool) -> str | None:
        """Why capacity entry `c` may not rent now, or None."""
        in_use = self._rentals.in_use(c, pool)
        if in_use >= c.cap:
            return f"at its cap ({in_use} of {c.cap} rented)"
        why, retry_at = self._rent_refused.get(c.name, ("", 0.0))
        if time.time() < retry_at:
            return f"refused: {why}"
        return None

    @staticmethod
    def _why_not(m: PoolMachine, occupants: list[str]) -> str | None:
        if m.lease is not None:
            return f"leased by {m.lease.workload}/{m.lease.tag} ({m.lease.phase})"
        if m.retiring:
            return "retiring: terminated once free (Stop all cloud spending)"
        if occupants:
            return f"busy: {', '.join(occupants)}"
        return None

    def _start_slots(self, m: PoolMachine, pool: Pool):
        """Create the leased tag's slots from its layout, set them running, and
        mark the lease running. Idempotent, so a reserved lease a restart
        interrupted is completed rather than doubled."""
        spec, task = self._lookup(m.lease.workload, m.lease.tag)
        for p in placement.plan_for(spec, _params(spec, task), m):
            if any(w.role == p.role for w in task.workers):
                continue
            # Placement already checked the fit, with the entry's override.
            if m.kind == "local":
                w = self._m.add_local(spec, task, p.role, p.threads, check_gpu=False)
            else:
                w = self._m.add_ssh(
                    spec, task, p.role, machine=m.name, threads=p.threads, check_gpu=False
                )
            w.desired_state = "running"
            # A requeued tag's new slots reuse its old worker ids.
            self._m.forget_crashes(spec, task.tag, w.worker_id)
        self._m.tasks.save(spec, task)
        m.lease.phase = RUNNING
        self._m.pool_store.save(pool)

    # ---- leases --------------------------------------------------------------

    def _advance_lease(self, m: PoolMachine, pool: Pool, queue: Queue):
        spec, task = self._lookup(m.lease.workload, m.lease.tag)
        if task is None:  # the tag was deleted under its lease
            m.lease = None
            self._m.pool_store.save(pool)
            return
        if self._rentals.is_gone(m):
            self._lose_rental(m, pool, queue, spec, task)
            return
        phase = m.lease.phase
        if phase == RESERVED:
            self._start_slots(m, pool)
        elif phase == RUNNING:
            failure = self._failure(spec, task)
            if failure is not None:
                self._fail(m, pool, queue, spec, task, failure)
            elif task.workers and all(w.finished for w in task.workers):
                self._start_release(m, pool, "completed")
        elif phase == RELEASING:
            self._finish_release(m, pool, queue)
        elif phase == HELD and self._wanted(m, queue):
            self._start_release(m, pool, m.lease.reason)

    def _lose_rental(self, m: PoolMachine, pool: Pool, queue: Queue, spec, task):
        """End the lease of a rental whose instance is gone: its containers
        went with the instance, so the slots are removed outright rather than
        drained, and the lease's spend is retired. A tag that still wanted to
        run (reserved, running, or being requeued) goes back to the head of
        the queue with its eligibility, its trainer resuming from its
        checkpoint wherever it lands next; a completed or failed one does not
        run again."""
        key = _machine_key(spec, task.tag, m.name)
        self._m._machine_states[key] = "gone"  # remove_worker skips a gone machine's probe
        for w in list(task.workers):
            self._m.remove_worker(spec, task, w.worker_id)
        task.retired_spend += pool_mod.lease_spend(m)
        self._m.tasks.save(spec, task)
        lease = m.lease
        pool.machines.remove(m)
        self._m.pool_store.save(pool)
        if lease.requeue or lease.phase in (RESERVED, RUNNING):
            queue.entries.insert(0, _requeued(lease, task))
            self._m.queue_store.save(queue)
            print(f"tag queue: {m.name}'s instance is gone; {lease.workload}/{lease.tag} requeued")
        else:
            who = f"{lease.workload}/{lease.tag}, {lease.phase}"
            print(f"tag queue: {m.name}'s instance is gone ({who}); not requeued")

    def _failure(self, spec, task) -> str | None:
        """The first slot that crashed FAIL_AFTER times within the window, as
        "<worker_id>: <last exit reason>", or None."""
        for w in task.workers:
            crashes = self._m.recent_crashes(spec, task.tag, w.worker_id, FAIL_WINDOW_SECONDS)
            if len(crashes) >= FAIL_AFTER:
                return f"{w.worker_id}: {crashes[-1]}"
        return None

    def _fail(self, m: PoolMachine, pool: Pool, queue: Queue, spec, task, failure: str):
        """Fail the tag: mark the crashing slot, pause every slot, then hand the
        machine to a queued tag that can use it, or hold it for investigation
        (docs/plans/tag_queue.md §5)."""
        crashing = failure.split(":", 1)[0]
        for w in task.workers:
            w.desired_state = "paused"
            if w.worker_id == crashing:
                w.failed = failure
        self._m.tasks.save(spec, task)
        m.lease.reason = f"failed: {failure}"
        if self._wanted(m, queue):
            self._start_release(m, pool, m.lease.reason)
        else:
            m.lease.phase = HELD
            self._m.pool_store.save(pool)

    def _wanted(self, m: PoolMachine, queue: Queue) -> bool:
        """Whether some queued tag could run on `m` once it is released."""
        if m.retiring:
            return False
        for e in queue.entries:
            spec, task = self._lookup(e.workload, e.tag)
            if placement.refusal(*self._fit_args(spec, task), e, m) is None:
                return True
        return False

    def _start_release(self, m: PoolMachine, pool: Pool, reason: str):
        m.lease.phase = RELEASING
        m.lease.reason = reason
        self._m.pool_store.save(pool)
        self._submit_drain(m)

    def _finish_release(self, m: PoolMachine, pool: Pool, queue: Queue):
        """Once the drain has succeeded, remove the tag's slots and end the
        lease; a requeued tag goes back to the head. A failed drain is retried
        next pass, with its reason on the lease."""
        spec, task = self._lookup(m.lease.workload, m.lease.tag)
        future = self._drains.get(m.name)
        if future is None:  # in flight when the dashboard restarted
            self._submit_drain(m)
            return
        if not future.done():
            return
        del self._drains[m.name]
        try:
            future.result()
        except Exception as e:  # noqa: BLE001 -- shown on the lease, retried next pass
            m.lease.reason = f"draining: {e}"
            self._m.pool_store.save(pool)
            return
        for w in list(task.workers):
            self._m.remove_worker(spec, task, w.worker_id)
        task.retired_spend += pool_mod.lease_spend(m)
        self._m.tasks.save(spec, task)
        lease = m.lease
        m.lease = None
        self._m.pool_store.save(pool)
        if lease.requeue:
            queue.entries.insert(0, _requeued(lease, task))
            self._m.queue_store.save(queue)

    def _submit_drain(self, m: PoolMachine):
        """Drain the tag leasing `m` on the drain thread, from a copy of its
        task: the thread only reads it, while the writer goes on changing the
        record."""
        spec, task = self._lookup(m.lease.workload, m.lease.tag)
        self._drains[m.name] = self._drain_thread.submit(self._drain, spec, copy.deepcopy(task))

    def _drain(self, spec, task: tasks.TaskRecord):
        """Everything the tag's slots hold, off the machine (drain thread). Raises
        while a slot is still alive; the next pass retries."""
        logs = self._m.tasks.paths(spec, task.tag).logs_dir
        logs.mkdir(parents=True, exist_ok=True)
        for w in task.workers:
            if w.kind == "local":
                assert not worker_pid_alive(w.pid, w.worker_id, task.tag), f"{w.worker_id} is alive"
                continue
            machine = self._m._ssh_machine(task, w)
            name = _container_name(spec, task.tag, w.worker_id)
            state = machine.container_state(name)
            assert state in ("stopped", "missing"), f"{w.worker_id} is {state}"
            if state == "missing":
                continue
            (logs / f"{w.worker_id}.container.log").write_text(machine.container_logs(name))
            if self._m._slot_sink(spec, task, w) == "local":
                sweep_stopped(machine, **self._m._transfer_target(spec, task, w))
        if any(w.kind == "ssh" and self._m._slot_sink(spec, task, w) == "r2" for w in task.workers):
            self._m.sync_once(spec, task)
