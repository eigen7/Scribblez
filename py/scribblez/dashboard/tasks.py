"""Task records for the master dashboard.

A task is one (workload, tag) pair with frozen params, worker slots and
machines, persisted as task.json in the tag's root. A tag directory without a
task.json still appears in listings, read-only.

A process holds one TaskRecord per task: load_task returns the same object
until the file changes under it, and save_task writes that object. The
dashboard reads and mutates a task from several places at once (the reconcile
pass across its blocking steps, request handlers, status polls). With a copy
each, the last save would win: an operator's pause, saved by its handler,
would be overwritten seconds later by the pass's copy, loaded as "running"
before the click, and the worker started again. With one shared object there
is nothing stale to save.
"""

import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from scribblez.dashboard.worker_stats_figures import read_stats
from scribblez.workloads import WorkloadSpec, resolve


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
    task.json has to agree with it."""

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
    # The bundle every ssh worker of this task runs, pinned when the first one
    # launches, so the fleet stays homogeneous and code edited mid-run does not
    # silently reach it. The source digest it was built from is kept so drift
    # is a local comparison (WorkerManager.bundle_drift), not a bucket read.
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
        for m in self.machines:
            if m.name == name:
                return m
        raise KeyError(f"no machine '{name}'")

    def slots_on(self, machine: str) -> list[WorkerRecord]:
        return [w for w in self.workers if w.machine == machine]


def task_path(spec: WorkloadSpec, tag: str) -> Path:
    return spec.data_dir(tag) / "task.json"


# The process's shared records (see the module docstring), by path, each with
# the file mtime it matches. A file whose mtime has moved was written by
# someone else, such as a CLI tool migrating params, and is read afresh.
_records: dict[Path, tuple[TaskRecord, int]] = {}
_records_lock = threading.Lock()


def _declared(cls, raw: dict) -> dict:
    """`raw` restricted to `cls`'s fields, so a stored field the dataclass no
    longer declares is dropped (and gone after the next save) instead of
    failing the load."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in names}


def _from_stored(cls, raw: dict):
    return cls(**_declared(cls, raw))


def _read_task(path: Path) -> TaskRecord:
    raw = _declared(TaskRecord, json.loads(path.read_text()))
    raw["workers"] = [_from_stored(WorkerRecord, w) for w in raw.get("workers", [])]
    raw["machines"] = [_from_stored(MachineRecord, m) for m in raw.get("machines", [])]
    return TaskRecord(**raw)


def load_task(spec: WorkloadSpec, tag: str) -> TaskRecord | None:
    path = task_path(spec, tag)
    with _records_lock:
        try:
            stamp = path.stat().st_mtime_ns
        except FileNotFoundError:
            _records.pop(path, None)
            return None
        held = _records.get(path)
        if held is not None and held[1] == stamp:
            return held[0]
        task = _read_task(path)
        _records[path] = (task, stamp)
        return task


def save_task(spec: WorkloadSpec, task: TaskRecord):
    """Write the record atomically, so two threads saving at once (the pass and
    a handler) never interleave in the file."""
    path = task_path(spec, task.tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".task.", suffix=".json")
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(asdict(task), indent=2) + "\n")
    with _records_lock:
        os.replace(tmp, path)
        _records[path] = (task, path.stat().st_mtime_ns)


def create_task(
    spec: WorkloadSpec, tag: str, raw_params: dict, profile: str | None = None
) -> TaskRecord:
    """Create and persist a task. Params resolve as `raw_params` over `profile`
    (the workload's default profile when None) over the workload's defaults.
    Raises params.ParamsError on bad values, AssertionError on a taken tag or
    unknown profile.

    The workload's `finalize` hook runs on the validated params: its last
    chance to resolve derived fields, since workers read task.json verbatim."""
    assert tag and all(c.isalnum() or c in "._-" for c in tag), f"invalid tag name '{tag}'"
    assert load_task(spec, tag) is None, f"tag '{tag}' already has a task"
    profile_name, validated = spec.resolve_params(profile, raw_params)
    if spec.finalize:
        validated = resolve(spec.finalize)(spec, tag, validated)
    task = TaskRecord(
        workload=spec.name,
        tag=tag,
        params=asdict(validated),
        created_at=time.time(),
        profile=profile_name,
    )
    save_task(spec, task)
    return task


def delete_tag(spec: WorkloadSpec, tag: str):
    """Delete a tag's local dir (task record, data, stats, logs). The tag's
    copy in the results bucket is deliberately left alone; purge it by hand.

    The tag must have no worker slots left, since the task record is what
    tracks their containers and machines. Callers go through
    WorkerManager.delete_task, which removes the slots first.
    """
    task = load_task(spec, tag)
    assert task is None or not task.workers, "remove the tag's workers first"
    tag_dir = spec.data_dir(tag)
    assert tag_dir.is_dir(), f"no such tag '{tag}'"
    shutil.rmtree(tag_dir)
    with _records_lock:
        _records.pop(task_path(spec, tag), None)


def progress(spec: WorkloadSpec, tag: str) -> list:
    """The workload's [label, value] progress counters for the tag."""
    if not spec.progress:
        return []
    return [list(pair) for pair in resolve(spec.progress)(spec, tag)]


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
    stamps = [r["updated_at"] for r in read_stats(tag_dir / "stats")]
    data = tag_dir / "data"
    if data.is_dir():
        for p in data.iterdir():
            stamps.append(p.stat().st_mtime)
            if p.is_dir():
                stamps += [c.stat().st_mtime for c in p.iterdir()]
    return max(stamps, default=0)


def list_tags(spec: WorkloadSpec) -> list[dict]:
    """Every tag under the workload's tags root, with listing metadata."""
    tags_root = spec.tags_root
    if not tags_root.is_dir():
        return []
    out = []
    for tag_dir in tags_root.iterdir():
        if not tag_dir.is_dir():
            continue
        task = load_task(spec, tag_dir.name)
        workers = task.workers if task else []
        out.append(
            {
                "tag": tag_dir.name,
                "has_task": task is not None,
                "created_at": task.created_at if task else None,
                "workers": len(workers),
                # Slots the operator wants running, gated ones included (the
                # scheduler resumes those itself). Desired rather than observed
                # state, so listing tags costs no ssh or cloud round trips.
                "active_workers": sum(w.desired_state == "running" for w in workers),
                "progress": progress(spec, tag_dir.name),
                "last_active": _last_active(tag_dir),
            }
        )
    return sorted(out, key=lambda r: r["tag"])
