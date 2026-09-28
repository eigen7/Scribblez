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

import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor

from cloud.ssh_transfer import sweep_stopped

from scribblez import params as params_mod
from scribblez import workloads
from scribblez.dashboard import placement, tasks
from scribblez.dashboard import pool as pool_mod
from scribblez.dashboard import queue as queue_mod
from scribblez.dashboard.pool import Lease, Pool, PoolMachine
from scribblez.dashboard.queue import Queue, QueueEntry
from scribblez.dashboard.workers import (
    CLOUD_SYNC,
    WorkerManager,
    _bucket_trainer,
    _container_name,
    _machine_link,
    _slot_sink,
    _ssh_machine,
    _transfer_target,
    worker_pid_alive,
)

RESERVED, RUNNING, RELEASING, HELD = "reserved", "running", "releasing", "held"
# The reason on a lease released by Requeue: its tag goes back to the head of
# the queue once the release is done. Kept on the lease, so it survives a
# dashboard restart mid-release.
REQUEUED = "requeued"

# A queue-placed slot that crashes this many times within the window is
# failed rather than restarted forever.
FAIL_AFTER = 3
FAIL_WINDOW_SECONDS = 1800.0

LOCAL_CODE_WARNING = (
    "local slots on this machine run the checkout in /workspace/repo as it is when they "
    "start, not a pinned bundle"
)


def _lookup(workload: str, tag: str):
    """(spec, task record or None) for a queue entry or lease."""
    spec = workloads.get(workload)
    return spec, tasks.load_task(spec, tag)


def _params(spec, task: tasks.TaskRecord):
    return params_mod.validate(spec.params_cls, task.params)


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
        spec, task = _lookup(workload, tag)
        assert spec.layout, f"{workload} tags are not queueable yet (no layout)"
        assert task is not None, f"tag '{tag}' has no task record"
        assert not task.workers, "the tag already has slots; remove them to queue it"
        queue, pool = queue_mod.load_queue(), pool_mod.load_pool()
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
        queue_mod.save_queue(queue)
        return {"queued": True, "warnings": warnings}

    def dequeue(self, workload: str, tag: str):
        queue = queue_mod.load_queue()
        queue.entries.remove(queue.entry(workload, tag))
        self._builds.pop((workload, tag), None)
        queue_mod.save_queue(queue)

    def move(self, workload: str, tag: str, delta: int):
        """Move a queued tag `delta` places toward the tail (negative: head)."""
        queue = queue_mod.load_queue()
        entry = queue.entry(workload, tag)
        i = queue.entries.index(entry)
        j = max(0, min(len(queue.entries) - 1, i + delta))
        queue.entries.insert(j, queue.entries.pop(i))
        queue_mod.save_queue(queue)

    def release(self, workload: str, tag: str):
        """The operator's Release, for a tag with no end condition: finish all
        its slots, which completes it, and the next pass releases its machine."""
        pool = pool_mod.load_pool()
        m = self._leased(pool, workload, tag)
        assert m is not None and m.lease.phase == RUNNING, f"{workload}/{tag} is not placed"
        spec, task = _lookup(workload, tag)
        for w in task.workers:
            w.desired_state = "paused"
            w.finished = True
        tasks.save_task(spec, task)

    def requeue(self, workload: str, tag: str):
        """Take a placed or held tag off its machine and put it back at the
        head of the queue, its data kept. Its slots are paused now; the release
        drains them once they have stopped."""
        pool = pool_mod.load_pool()
        m = self._leased(pool, workload, tag)
        assert m is not None and m.lease.phase != RELEASING, f"{workload}/{tag} is not placed"
        spec, task = _lookup(workload, tag)
        for w in task.workers:
            w.desired_state = "paused"
        tasks.save_task(spec, task)
        self._start_release(m, pool, REQUEUED)

    def status(self) -> dict:
        """The queue in order, each entry with whether it ends on its own, its
        bundle, and why each pool machine did not take it on the last pass."""
        rows = []
        for e in queue_mod.load_queue().entries:
            spec, task = _lookup(e.workload, e.tag)
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
        pool, queue = pool_mod.load_pool(), queue_mod.load_queue()
        self._drop_stale_entries(queue, pool)
        self._advance_builds(queue, pool)
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
            s, t = _lookup(workload, tag)
            if t is not None and not placement.has_end_condition(s, _params(s, t)):
                endless.append(f"{workload}/{tag}")
        if endless:
            out.append(
                f"no end condition: {', '.join(endless)}; each holds its machine until "
                "released by hand"
            )
        params = _params(spec, task)
        for m in pool.machines:
            if (
                m.kind == "local"
                and placement.refusal(spec, params, entry, m, need_bundle=False) is None
            ):
                out.append(f"{m.name}: {LOCAL_CODE_WARNING}")
        return out

    def _ssh_targets(self, spec, task, entry: QueueEntry, pool: Pool) -> list[PoolMachine]:
        """The ssh pool machines the tag could run on, bundle aside."""
        params = _params(spec, task)
        return [
            m
            for m in pool.machines
            if m.kind == "ssh"
            and placement.refusal(spec, params, entry, m, need_bundle=False) is None
        ]

    def _submit_build(self, spec, task, entry: QueueEntry, pool: Pool):
        """Build the tag's bundle for its ssh machines' archs on the build
        thread; _advance_builds pins it when done. Detecting a machine's arch
        pulls the worker image there, which is why this is not done inline."""
        targets = self._ssh_targets(spec, task, entry, pool)
        # Any worker image carries the compiler detect_arch asks.
        image = self._m._creds().registry.image_for(spec.roles[0].runtime)

        def build():
            for m in targets:
                if m.machine.arch is None:
                    link = _machine_link(m.machine)
                    link.pull_image(image)
                    m.machine.arch = link.detect_arch(image)
            archs = sorted(set(task.bundle_archs) | {m.machine.arch for m in targets})
            return self._m._build_bundle(archs)

        self._builds[entry.key] = self._m._builds.submit(build)

    def _advance_builds(self, queue: Queue, pool: Pool):
        """Pin every finished enqueue-time build to its task. A build that was
        in flight when the dashboard restarted is submitted again."""
        changed = False
        for e in queue.entries:
            if e.bundle != queue_mod.BUNDLE_BUILDING:
                continue
            spec, task = _lookup(e.workload, e.tag)
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
                self._m._pin_bundle(spec, task, future.result())
                e.bundle = queue_mod.BUNDLE_READY
            except Exception as ex:  # noqa: BLE001 -- shown on the entry, retried on re-enqueue
                e.bundle = f"failed: {ex}"
            changed = True
        if changed:
            pool_mod.save_pool(pool)  # the archs the builds detected
            queue_mod.save_queue(queue)

    def _drop_stale_entries(self, queue: Queue, pool: Pool):
        """Drop entries whose task is gone, and entries a placement interrupted
        by a restart left behind after writing the lease."""
        stale = [
            e
            for e in queue.entries
            if _lookup(e.workload, e.tag)[1] is None or self._leased(pool, *e.key) is not None
        ]
        for e in stale:
            queue.entries.remove(e)
        if stale:
            queue_mod.save_queue(queue)

    # ---- placement -----------------------------------------------------------

    def _place(self, pool: Pool, queue: Queue):
        """Match queued tags to free machines (placement.match) and place each
        match: lease first, then the queue entry goes, then the slots."""
        tasks_now = list(self._m.all_tasks())
        busy = {m.name: self._m.occupants(m, tasks_now) for m in pool.machines}
        free = [m for m in pool.machines if m.lease is None and not busy[m.name]]
        params = {}
        for e in queue.entries:
            spec, task = _lookup(e.workload, e.tag)
            params[e.key] = (spec, _params(spec, task))
        self._refusals = {
            e.key: {
                m.name: self._why_not(m, busy[m.name]) or placement.refusal(*params[e.key], e, m)
                for m in pool.machines
            }
            for e in queue.entries
        }

        def eligible(e: QueueEntry, m: PoolMachine) -> bool:
            return placement.refusal(*params[e.key], e, m) is None

        matches = placement.match(queue.entries, free, eligible)
        for e in list(queue.entries):
            if e.key not in matches:
                continue
            m = pool.machine(matches[e.key])
            m.lease = Lease(e.workload, e.tag, RESERVED, time.time())
            pool_mod.save_pool(pool)
            queue.entries.remove(e)
            queue_mod.save_queue(queue)
            self._start_slots(m, pool)

    @staticmethod
    def _why_not(m: PoolMachine, occupants: list[str]) -> str | None:
        if m.lease is not None:
            return f"leased by {m.lease.workload}/{m.lease.tag} ({m.lease.phase})"
        if occupants:
            return f"busy: {', '.join(occupants)}"
        return None

    def _start_slots(self, m: PoolMachine, pool: Pool):
        """Create the leased tag's slots from its layout, set them running, and
        mark the lease running. Idempotent, so a reserved lease a restart
        interrupted is completed rather than doubled."""
        spec, task = _lookup(m.lease.workload, m.lease.tag)
        for p in placement.plan_for(spec, _params(spec, task), m):
            if any(w.role == p.role for w in task.workers):
                continue
            if m.kind == "local":
                w = self._m.add_local(spec, task, p.role, p.threads)
            else:
                w = self._m.add_ssh(spec, task, p.role, machine=m.name, threads=p.threads)
            w.desired_state = "running"
            # A requeued tag's new slots reuse its old worker ids.
            self._m.forget_crashes(spec, task.tag, w.worker_id)
        tasks.save_task(spec, task)
        m.lease.phase = RUNNING
        pool_mod.save_pool(pool)

    # ---- leases --------------------------------------------------------------

    def _advance_lease(self, m: PoolMachine, pool: Pool, queue: Queue):
        spec, task = _lookup(m.lease.workload, m.lease.tag)
        if task is None:  # the tag was deleted under its lease
            m.lease = None
            pool_mod.save_pool(pool)
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
        tasks.save_task(spec, task)
        m.lease.reason = f"failed: {failure}"
        if self._wanted(m, queue):
            self._start_release(m, pool, m.lease.reason)
        else:
            m.lease.phase = HELD
            pool_mod.save_pool(pool)

    def _wanted(self, m: PoolMachine, queue: Queue) -> bool:
        """Whether some queued tag could run on `m` once it is released."""
        for e in queue.entries:
            spec, task = _lookup(e.workload, e.tag)
            if placement.refusal(spec, _params(spec, task), e, m) is None:
                return True
        return False

    def _start_release(self, m: PoolMachine, pool: Pool, reason: str):
        m.lease.phase = RELEASING
        m.lease.reason = reason
        pool_mod.save_pool(pool)
        spec, task = _lookup(m.lease.workload, m.lease.tag)
        self._drains[m.name] = self._drain_thread.submit(self._drain, spec, task)

    def _finish_release(self, m: PoolMachine, pool: Pool, queue: Queue):
        """Once the drain has succeeded, remove the tag's slots and end the
        lease; a requeued tag goes back to the head. A failed drain is retried
        next pass, with its reason on the lease."""
        spec, task = _lookup(m.lease.workload, m.lease.tag)
        future = self._drains.get(m.name)
        if future is None:  # in flight when the dashboard restarted
            self._drains[m.name] = self._drain_thread.submit(self._drain, spec, task)
            return
        if not future.done():
            return
        del self._drains[m.name]
        try:
            future.result()
        except Exception as e:  # noqa: BLE001 -- shown on the lease, retried next pass
            m.lease.reason = f"draining: {e}"
            pool_mod.save_pool(pool)
            return
        for w in list(task.workers):
            self._m.remove_worker(spec, task, w.worker_id)
        key = (m.lease.workload, m.lease.tag)
        requeued = m.lease.reason == REQUEUED
        m.lease = None
        pool_mod.save_pool(pool)
        if requeued:
            bundle = queue_mod.BUNDLE_READY if task.bundle_id else queue_mod.BUNDLE_NONE
            queue.entries.insert(0, QueueEntry(*key, time.time(), bundle=bundle))
            queue_mod.save_queue(queue)

    def _drain(self, spec, task: tasks.TaskRecord):
        """Everything the tag's slots hold, off the machine (drain thread). Raises
        while a slot is still alive; the next pass retries."""
        logs = spec.paths(task.tag).logs_dir
        logs.mkdir(parents=True, exist_ok=True)
        for w in task.workers:
            if w.kind == "local":
                assert not worker_pid_alive(w.pid, w.worker_id, task.tag), f"{w.worker_id} is alive"
                continue
            machine = _ssh_machine(task, w)
            name = _container_name(spec, task.tag, w.worker_id)
            state = machine.container_state(name)
            assert state in ("stopped", "missing"), f"{w.worker_id} is {state}"
            if state == "missing":
                continue
            (logs / f"{w.worker_id}.container.log").write_text(machine.container_logs(name))
            if _slot_sink(spec, task, w) == "local":
                sweep_stopped(machine, **_transfer_target(spec, task, w))
        if any(w.kind == "ssh" and _slot_sink(spec, task, w) == "r2" for w in task.workers):
            argv = [sys.executable, str(CLOUD_SYNC), "--workload", spec.name, "-t", task.tag]
            if _bucket_trainer(spec, task):
                argv.append("--trainer-outputs")
            subprocess.run(argv, check=True, capture_output=True, text=True)
