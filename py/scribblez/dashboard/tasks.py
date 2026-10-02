"""Task records for the master dashboard.

A task is one (workload, tag) pair with frozen params, worker slots and
machines, persisted as task.json in the tag's root. A tag directory without a
task.json still appears in listings, read-only.

A task's record lives in two places. Its frozen fields (params, profile,
bundle pin) stay in task.json, which tools outside the dashboard read and
migrate_tag_params edits. Its control state (slots, machines, gates, spend)
is a row of the control store (control_store.py), which the dashboard alone
writes. A TaskStore, one per dashboard process and mount root, joins the two
into one TaskRecord: the writer thread gets the same live object on every
load until task.json changes under it, and saves that object; every other
thread reads the last committed copy. With a copy per caller, the last save
would win: an operator's pause, saved by its command, would be overwritten
seconds later by the pass's copy, loaded as "running" before the click, and
the worker started again. With one live object there is nothing stale to
save.
"""

import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from scribblez import params as params_mod
from scribblez.dashboard import worker_stats_figures
from scribblez.dashboard.control_store import ControlStore
from scribblez.dashboard.queue import BUNDLE_FAILED_PREFIX, Queue, QueueEntry
from scribblez.paths import TagPaths
from scribblez.workloads import WORKLOADS, WorkloadSpec, resolve

# A tag's states (TaskStore.state).
COMPLETE, FAILED, RUNNING, QUEUED, PAUSED, IDLE = (
    "complete",
    "failed",
    "running",
    "queued",
    "paused",
    "idle",
)


@dataclass
class WorkerRecord:
    """One worker slot: durable identity and desired state. The actual state of
    its process or container is observed live by the WorkerManager."""

    worker_id: str
    role: str  # which of the workload's roles this slot runs
    kind: str  # "local" | "ssh"
    desired_state: str  # "running" | "paused"
    threads: int | None = None  # local/ssh: engine thread count (None: all cores)
    host: str | None = None  # ssh: SSH destination ("user@host" or an ssh-config alias)
    # ssh: the name of the task's machine the slot runs on, whose record carries
    # the address and key material. Exactly one of `host` and `machine` is set.
    machine: str | None = None
    # ssh on a bare host: its CPU microarchitecture (a GCC -march value), which
    # the task's bundle must be built for. Asked of the host at the slot's first
    # start. A machine-backed slot uses its machine record's instead.
    arch: str | None = None
    # The worker exited on reaching its role's terminal condition (a trainer's
    # max_rows, a generator's cycle cap). Its desired state is then paused, so
    # reconcile does not restart it forever, and it displays as `finished`
    # rather than `paused`. Cleared by a Start.
    finished: bool = False
    # Why the tag queue gave up on the slot: it crashed repeatedly
    # (dashboard/tag_queue.py). Its desired state is then paused and it
    # displays as `failed`. Cleared by a Start.
    failed: str | None = None
    # ssh: whether the slot's container is known to exist. False from add until a
    # start confirms it, or a probe finds the container an in-doubt start (ssh
    # lost mid-command) did create. While False, an unreachable probe reads as
    # "missing", so the slot stays removable even when its host is bogus or
    # offline (a host string is not validated until the first start). add_ssh
    # sets it False; the True default errs toward the stricter unreachable
    # handling for any record that lacks the field.
    launched: bool = True
    pid: int | None = None  # local: OS pid of the backing subprocess, if spawned
    # ssh: the bundle the container was created with. The bundle is fixed in the
    # container's environment, so a slot whose bundle differs from its task's
    # is replaced rather than restarted.
    bundle_id: str | None = None
    # ssh: finished output the container holds that the controller has not
    # collected. Zero when the container is created (which lets one that never
    # came up be replaced), then whatever each collection finds. None when not
    # known: before a container exists, and after a failed collection, since a
    # stale zero would keep claiming "drained" while the container fills up.
    # Durable because it decides whether replacing the container is safe, and a
    # dashboard restart must not turn "holding six hours of work" into
    # "nothing known, go ahead".
    undelivered: int | None = None


@dataclass
class MachineRecord:
    """A machine the task's ssh slots run on, either registered by the operator
    (`manual`) or rented for the task from a cloud provider (`aws`). It belongs
    to the task, living and dying with it like a slot, so nothing outside
    the task's record has to agree with it."""

    name: str
    provider: str  # "manual" | "aws"
    host: str  # SSH destination ("user@address"), refreshed by the provider if it moves
    identity_file: str | None = None  # private key for ssh; None: the container's own identity
    # Per-machine known_hosts with StrictHostKeyChecking=accept-new: a rented
    # machine's key is unknown at launch, and providers reuse addresses.
    known_hosts_file: str | None = None
    gpu_count: int | None = None  # GPUs on the machine; None: unknown (unchecked at add time)
    # The CPU microarchitecture (a GCC -march value) the task's bundle must
    # cover for this machine: from the catalog for a rented machine, asked of a
    # registered one at its first slot start.
    arch: str | None = None
    instance_id: str | None = None  # rented: the provider's instance
    instance_type: str | None = None  # rented: the catalog type
    spot: bool = (
        False  # rented: spare capacity at market rate, interruptible (stopped) by the provider
    )
    region: str | None = None
    cost_per_hr: float | None = None  # rented: the catalog rate
    launched_at: float | None = None
    spend: float = 0.0  # estimated dollars this machine has cost so far
    # Spend accrual: when the machine was last observed, and whether it was
    # billing then.
    observed_at: float | None = None
    observed_up: bool = False

    def spend_now(self, now: float) -> float:
        """`spend` advanced to `now`: the interval since the last observation
        is charged if the machine was billing then. Recording it is the
        observing pass's (workers._accrue_machine); a status read shows it."""
        if not self.observed_up or self.observed_at is None:
            return self.spend
        return self.spend + (now - self.observed_at) / 3600 * (self.cost_per_hr or 0.0)


@dataclass
class TaskRecord:
    workload: str
    tag: str
    params: dict  # raw param values (validated against the workload's schema)
    created_at: float
    workers: list[WorkerRecord] = field(default_factory=list)
    machines: list[MachineRecord] = field(default_factory=list)
    # Roles the workload's scheduler has parked (role -> reason). Distinct from
    # operator pause: a gated worker keeps desired_state="running" and resumes
    # automatically when the scheduler releases the gate.
    gates: dict = field(default_factory=dict)
    # Estimated spend of machines that have since been removed, so the
    # task's cumulative total survives slot removal.
    retired_spend: float = 0.0
    # Why the tag queue failed the tag (dashboard/tag_queue.py), kept after its
    # slots are released to another tag so the listing still shows it failed.
    # Cleared when the tag is queued again or a slot of it is started.
    failure: str | None = None
    # The bundle every ssh worker of this task runs, pinned when the first one
    # launches, so the fleet stays homogeneous and code edited mid-run does not
    # silently reach it. The source digest it was built from is kept so drift
    # is a comparison of digests (WorkerManager.bundle_drift), not a rebuild.
    bundle_id: str | None = None
    bundle_source_hash: str = ""
    bundle_archs: list[str] = field(default_factory=list)  # the archs the bundle was built for
    # The parameter profile the params were resolved from (WorkloadSpec
    # .profiles). Provenance only: the params are the frozen truth, and the task
    # view shows how they depart from the profile. "" for a workload without
    # profiles.
    profile: str = ""

    def worker(self, worker_id: str) -> WorkerRecord:
        w = self.find(worker_id)
        if w is None:
            raise KeyError(f"no worker '{worker_id}'")
        return w

    def find(self, worker_id: str) -> WorkerRecord | None:
        """The slot, or None once it has been removed. For steps that planned
        their work from an earlier look at the slot list."""
        for w in self.workers:
            if w.worker_id == worker_id:
                return w
        return None

    def machine(self, name: str) -> MachineRecord:
        m = self.find_machine(name)
        if m is None:
            raise KeyError(f"no machine '{name}'")
        return m

    def find_machine(self, name: str) -> MachineRecord | None:
        """The task's own machine `name`, or None."""
        return next((m for m in self.machines if m.name == name), None)

    def slots_on(self, machine: str) -> list[WorkerRecord]:
        return [w for w in self.workers if w.machine == machine]


def _declared(cls, raw: dict) -> dict:
    """`raw` restricted to `cls`'s fields, so a stored field the dataclass no
    longer declares is dropped (and gone after the next save) instead of
    failing the load."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in names}


def _from_stored(cls, raw: dict):
    return cls(**_declared(cls, raw))


# The TaskRecord fields kept in task.json: fixed once the tag is created (the
# bundle pin aside), and read by tools outside the dashboard. The rest is
# control state, kept in the control store.
FROZEN_FIELDS = (
    "workload", "tag", "params", "created_at", "bundle_id", "bundle_source_hash", "bundle_archs",
    "profile",
)  # fmt: skip


def read_params(spec: WorkloadSpec, tag: str, mount_root: Path) -> dict | None:
    """The tag's frozen params, from its task.json alone; None when it has
    none. For readers that need nothing else, such as a data-plane request."""
    path = spec.paths(tag, mount_root).root / "task.json"
    return json.loads(path.read_text())["params"] if path.is_file() else None


def _decode_task(stored: dict) -> TaskRecord:
    raw = _declared(TaskRecord, stored)
    raw["workers"] = [_from_stored(WorkerRecord, w) for w in raw.get("workers", [])]
    raw["machines"] = [_from_stored(MachineRecord, m) for m in raw.get("machines", [])]
    return TaskRecord(**raw)


def _last_active(tag_dir: Path) -> float:
    """When the tag last saw real work: the newest worker stats record or data
    subdirectory mtime, or 0 if nothing has happened yet.

    Stats are dated by their own `updated_at`, not the file's mtime: an ssh
    slot's records are copied back on every collection whether or not a cycle
    completed, so the mtime would make a stalled but reachable worker look
    active forever.

    The data scan goes two levels deep because a generational workload nests
    its files one level further (`data/generations/gen_NNNNNN/`), and a file
    landing there bumps only its own directory's mtime.
    """
    stamps = [r["updated_at"] for r in worker_stats_figures.read_stats(tag_dir / "stats")]
    data = tag_dir / "data"
    if data.is_dir():
        for p in data.iterdir():
            stamps.append(p.stat().st_mtime)
            if p.is_dir():
                stamps += [c.stat().st_mtime for c in p.iterdir()]
    return max(stamps, default=0)


def _pace(spec: WorkloadSpec, tag_dir: Path) -> dict | None:
    """The tag listing's pace: the pace role's fleet rate, or None when the
    workload names no pace role or its workers show no live rate."""
    if not spec.pace_role:
        return None
    stats = spec.role(spec.pace_role).stats
    per_hour = worker_stats_figures.pace(
        worker_stats_figures.read_stats(tag_dir / "stats"), spec.pace_role, stats, time.time()
    )
    return None if per_hour is None else {"unit": stats.unit, "per_hour": per_hour}


def _disk_bytes(tag_dir: Path) -> int:
    """The bytes the tag's tree occupies on the controller's disk, counting
    each inode once as du does (a trainer's snapshot pair is hard links). A
    remote data home's window and output not yet collected are not counted:
    the controller's tree is the durable copy, and the rest is transient."""
    seen: set[tuple[int, int]] = set()
    total = 0
    for root, _, files in os.walk(tag_dir):
        for name in files:
            try:
                st = os.lstat(os.path.join(root, name))
            except FileNotFoundError:  # renamed or evicted mid-walk
                continue
            if st.st_nlink > 1:
                if (st.st_dev, st.st_ino) in seen:
                    continue
                seen.add((st.st_dev, st.st_ino))
            total += st.st_blocks * 512
    return total


def _failed(task: TaskRecord, entry: QueueEntry | None) -> bool:
    """Whether the tag is stuck on a failure the operator has to act on."""
    return (
        task.failure is not None
        or any(w.failed is not None for w in task.workers)
        or (entry is not None and entry.bundle.startswith(BUNDLE_FAILED_PREFIX))
    )


class _TaskEntry:
    """One tag's record: its frozen fields in task.json, its control state in
    the control store's row. The writer holds one live TaskRecord, reread
    when task.json changes under it (migrate_tag_params editing its params,
    say); readers get the committed copy, as a SharedRecord gives."""

    def __init__(self, path: Path, control: ControlStore, key: str):
        self.path = path
        self._control = control
        self._key = key
        self._held: tuple[TaskRecord | None, int] | None = None  # (live, task.json mtime)
        # (readers' copy, task.json mtime, control version it reflects)
        self._copy: tuple[TaskRecord | None, int, int] | None = None
        self._written: str | None = None  # task.json as this process last wrote it
        # The live record of the tag as it was when deleted (forget), whose
        # late saves are dropped.
        self._deleted: TaskRecord | None = None
        self._lock = threading.Lock()

    def load(self) -> TaskRecord | None:
        with self._lock:
            stamp = _mtime(self.path)
            if not self._control.writer.here():
                # The version before the read: a commit landing during it
                # makes the next load read again, never keeps a stale copy.
                version = self._control.version
                c = self._copy
                if c is None or c[1] != stamp or c[2] != version:
                    c = self._copy = (self._read(stamp), stamp, version)
                return c[0]
            if self._held is None or self._held[1] != stamp:
                self._held = (self._read(stamp), stamp)
            return self._held[0]

    def save(self, task: TaskRecord):
        """Write the control row, then task.json if its frozen fields changed.
        A crash between the two leaves a row with no task.json for a new tag
        (no tag, as before), or an older bundle pin (redeployed again)."""
        assert self._copy is None or task is not self._copy[0], (
            f"{self.path}: saving a reader's copy; load the record on the writer thread"
        )
        if task is self._deleted:
            # A step that loaded the tag before a Delete ran between the steps
            # of its pass, on the same thread: writing would bring the
            # deleted tag back.
            return
        stored = asdict(task)
        self._control.put("task", self._key, json.dumps(_control_part(stored)))
        frozen = json.dumps({f: stored[f] for f in FROZEN_FIELDS}, indent=2) + "\n"
        if frozen != self._written:
            _write_atomic(self.path, frozen)
            self._written = frozen
        with self._lock:
            self._held = (task, _mtime(self.path))

    def import_json(self) -> bool:
        """Adopt the control state of a task.json written before the control
        store existed, unless the store already has a row for the tag. The
        file itself is left as it is: its next save drops those fields, once
        the row is committed. Whether it adopted one."""
        if self._control.get("task", self._key) is not None or not self.path.is_file():
            return False
        stored = asdict(_decode_task(json.loads(self.path.read_text())))
        self._control.put("task", self._key, json.dumps(_control_part(stored)))
        return True

    def forget(self):
        """Delete the control row and both copies, the tag dir being gone."""
        self._control.delete("task", self._key)
        with self._lock:
            if self._held is not None:
                self._deleted = self._held[0]
            self._held = self._copy = None
            self._written = None

    def _read(self, stamp: int) -> TaskRecord | None:
        if not stamp:
            return None
        try:
            frozen = json.loads(self.path.read_text())
        except FileNotFoundError:
            return None  # deleted since the stat: a status read racing a Delete
        body = self._control.get("task", self._key)
        control = json.loads(body) if body is not None else {}
        return _decode_task({**control, **{f: frozen[f] for f in FROZEN_FIELDS if f in frozen}})


def _control_part(stored: dict) -> dict:
    return {k: v for k, v in stored.items() if k not in FROZEN_FIELDS}


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return 0


def _write_atomic(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".json")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


class TaskStore:
    """The task records under one mount root (see the module docstring). A
    store is the only way in: nothing in the dashboard resolves a tag's
    directory without it, so a test or simulation gives it a scratch root and
    never reaches the live trees."""

    def __init__(self, mount_root: Path, control: ControlStore | None = None):
        self.mount_root = Path(mount_root)
        self._control = control or ControlStore(self.mount_root)
        self._entries: dict[tuple[str, str], _TaskEntry] = {}
        self._lock = threading.Lock()

    def paths(self, spec: WorkloadSpec, tag: str) -> TagPaths:
        return spec.paths(tag, self.mount_root)

    def task_path(self, spec: WorkloadSpec, tag: str) -> Path:
        return self.paths(spec, tag).root / "task.json"

    def load(self, spec: WorkloadSpec, tag: str) -> TaskRecord | None:
        """The task, or None when the tag has no task.json."""
        return self._entry(spec, tag).load()

    def save(self, spec: WorkloadSpec, task: TaskRecord):
        self._entry(spec, task.tag).save(task)

    def export_json(self, spec: WorkloadSpec, task: TaskRecord):
        """Write the whole record to task.json, control state included, as it
        was kept before the control store (WorkerManager.export_json_stores)."""
        _write_atomic(self.task_path(spec, task.tag), json.dumps(asdict(task), indent=2) + "\n")

    def import_json(self) -> list[str]:
        """Adopt every task.json's control state the control store lacks (the
        migration, docs/plans/dashboard_state_model.md §10). The tags adopted,
        as workload/tag."""
        adopted = []
        for spec in WORKLOADS.values():
            tags_root = spec.tags_root(self.mount_root)
            for tag_dir in sorted(tags_root.iterdir()) if tags_root.is_dir() else []:
                if self._entry(spec, tag_dir.name).import_json():
                    adopted.append(f"{spec.name}/{tag_dir.name}")
        return adopted

    def create(
        self, spec: WorkloadSpec, tag: str, raw_params: dict, profile: str | None = None
    ) -> TaskRecord:
        """Create and persist a task. Params resolve as `raw_params` over
        `profile` (the workload's default profile when None) over the
        workload's defaults. Raises params.ParamsError on bad values,
        AssertionError on a taken tag or unknown profile.

        The workload's `finalize` hook runs on the validated params: its last
        chance to resolve derived fields, since workers read task.json
        verbatim."""
        assert tag and all(c.isalnum() or c in "._-" for c in tag), f"invalid tag name '{tag}'"
        assert self.load(spec, tag) is None, f"tag '{tag}' already has a task"
        profile_name, validated = spec.resolve_params(profile, raw_params)
        if spec.finalize:
            validated = resolve(spec.finalize)(spec, self.paths(spec, tag), validated)
        task = TaskRecord(
            workload=spec.name,
            tag=tag,
            params=asdict(validated),
            created_at=time.time(),
            profile=profile_name,
        )
        self.save(spec, task)
        return task

    def delete(self, spec: WorkloadSpec, tag: str):
        """Delete a tag's dir (task record, data, stats, logs).

        The tag must have no worker slots left, since the task record is what
        tracks their containers and machines. Callers go through
        WorkerManager.delete_task, which removes the slots first.

        The control row goes before the directory: a removal that fails
        partway leaves a tag with its frozen params and no control state,
        nothing that could run.
        """
        task = self.load(spec, tag)
        assert task is None or not task.workers, "remove the tag's workers first"
        tag_dir = self.paths(spec, tag).root
        assert tag_dir.is_dir(), f"no such tag '{tag}'"
        self._control.writer.check(tag_dir)
        self._entry(spec, tag).forget()
        shutil.rmtree(tag_dir)

    def _entry(self, spec: WorkloadSpec, tag: str) -> _TaskEntry:
        with self._lock:
            e = self._entries.get((spec.name, tag))
            if e is None:
                e = self._entries[(spec.name, tag)] = _TaskEntry(
                    self.task_path(spec, tag), self._control, f"{spec.name}/{tag}"
                )
            return e

    def state(self, spec: WorkloadSpec, task: TaskRecord | None, entry: QueueEntry | None) -> str:
        """The tag's state, common to every workload; `entry` is its place in
        the tag queue, or None. The first that holds:

        complete  the workload's end condition holds (WorkloadSpec.complete),
                  or, for a workload without one, every slot is finished
        failed    the tag queue failed it, a slot of it is failed, or its
                  queue entry's bundle build failed (retried on re-enqueue)
        running   a slot wants to run (a gated one included: the scheduler
                  resumes it itself)
        queued    it waits in the tag queue for a machine
        paused    it has slots, none of which wants to run
        idle      it has no slots

        Desired rather than observed state, so it costs no ssh or cloud round
        trips."""
        if task is None:
            return IDLE
        if self._complete(spec, task):
            return COMPLETE
        if _failed(task, entry):
            return FAILED
        if any(w.desired_state == "running" for w in task.workers):
            return RUNNING
        if entry is not None:
            return QUEUED
        return PAUSED if task.workers else IDLE

    def _complete(self, spec: WorkloadSpec, task: TaskRecord) -> bool:
        if not spec.complete:
            return bool(task.workers) and all(w.finished for w in task.workers)
        params = params_mod.validate(spec.params_cls, task.params)
        return resolve(spec.complete)(spec, self.paths(spec, task.tag), params)

    def progress(self, spec: WorkloadSpec, task: TaskRecord) -> list:
        """The workload's [label, value] progress counters for the task."""
        if not spec.progress:
            return []
        params = params_mod.validate(spec.params_cls, task.params)
        paths = self.paths(spec, task.tag)
        return [list(pair) for pair in resolve(spec.progress)(spec, paths, params)]

    def _tag_dirs(self, spec: WorkloadSpec):
        """(tag dir, its task or None) for every tag dir of the workload, in
        name order."""
        tags_root = spec.tags_root(self.mount_root)
        if not tags_root.is_dir():
            return
        for tag_dir in sorted(tags_root.iterdir()):
            if tag_dir.is_dir():
                yield tag_dir, self.load(spec, tag_dir.name)

    def load_all(self):
        """(spec, task) for every task of every workload, read and nothing else."""
        for spec in WORKLOADS.values():
            for _, task in self._tag_dirs(spec):
                if task is not None:
                    yield spec, task

    def list_tags(self, spec: WorkloadSpec, queue: Queue) -> list[dict]:
        """Every tag under the workload's tags root, with listing metadata."""
        out = []
        for tag_dir, task in self._tag_dirs(spec):
            workers = task.workers if task else []
            out.append(
                {
                    "tag": tag_dir.name,
                    "has_task": task is not None,
                    "created_at": task.created_at if task else None,
                    "workers": len(workers),
                    # Slots the operator wants running, gated ones included (the
                    # scheduler resumes those itself). Desired rather than
                    # observed state, so listing tags costs no ssh or cloud
                    # round trips.
                    "active_workers": sum(w.desired_state == "running" for w in workers),
                    "state": self.state(spec, task, queue.find(spec.name, tag_dir.name)),
                    "progress": self.progress(spec, task) if task else [],
                    "disk_bytes": _disk_bytes(tag_dir),
                    "pace": _pace(spec, tag_dir),
                    "last_active": _last_active(tag_dir),
                }
            )
        return out
