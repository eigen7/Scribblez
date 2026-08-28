"""Task records for the master dashboard.

A task is one (workload, tag) pair with a frozen parameter set and a list of
worker slots, persisted as task.json in the tag's root dir. Tags that predate
the dashboard (no task.json) still appear in listings, read-only.
"""

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from scribblez import params as params_mod
from scribblez.dashboard.worker_stats_figures import read_stats
from scribblez.workloads import WorkloadSpec, resolve


@dataclass
class WorkerRecord:
    """One worker slot: durable identity + desired state. The backing process
    or pod's actual state is observed live by the WorkerManager, which also
    accrues the spend estimate here (observed running time x the pod's rate;
    Runpod's billing API returns nothing for CPU pods, so spend is estimated
    from our own observations)."""

    worker_id: str
    role: str  # which of the workload's roles this slot runs
    kind: str  # "local" | "cloud" | "ssh"
    desired_state: str  # "running" | "paused"
    threads: int | None = None  # local/ssh: engine thread count (None: all cores)
    host: str | None = None  # ssh: SSH destination ("user@host" or an ssh-config alias)
    # ssh: whether the slot's container is known to have been created. False
    # from add until a start confirms it -- or until a probe finds a container
    # an in-doubt start (ssh link lost mid-command) did create. While False, an
    # unreachable probe reads as "missing", so the slot stays manageable --
    # removable even when the host is bogus or offline (the host string is
    # unvalidated until first start). Defaults True so records saved before
    # this field existed (adds used to launch immediately) keep the stricter
    # unreachable handling.
    launched: bool = True
    pid: int | None = None  # local: OS pid of the backing subprocess, if spawned
    vcpus: int | None = None  # cloud CPU pod: vCPU count
    flavor: str | None = None  # cloud CPU pod: Runpod CPU flavor
    gpu_type_id: str | None = None  # cloud GPU pod: Runpod gpuTypeId
    gpu_count: int | None = None  # cloud GPU pod: number of GPUs
    pod_id: str | None = None  # cloud: the backing pod, created on first start (None until then)
    # cloud/ssh: the bundle the pod/container was created with. A pod's or
    # container's environment fixes its bundle at creation, so a slot whose id
    # no longer matches its task's is replaced rather than restarted.
    bundle_id: str | None = None
    # ssh: delivered files the container still holds. Zero from the moment the
    # container is created (it cannot hold anything yet, which is what lets one
    # that never came up be replaced), then whatever each collection finds --
    # and None whenever that is not known: before a container exists, and again
    # if a collection fails, since a stale count would go on claiming "drained"
    # while the container fills up. Durable because it decides whether
    # replacing the container is safe, and a dashboard restart must not turn
    # "holding six hours of work" into "nothing known, go ahead".
    undelivered: int | None = None
    cost_per_hr: float | None = None  # cloud: last observed rental rate
    spend: float = 0.0  # estimated dollars spent by this slot so far
    observed_at: float | None = None  # when the slot was last observed
    observed_running: bool = False  # whether it was running then


@dataclass
class TaskRecord:
    workload: str
    tag: str
    params: dict  # raw param values (validated against the workload's schema)
    created_at: float
    workers: list[WorkerRecord] = field(default_factory=list)
    # Roles the workload's scheduler has parked (role -> reason). Distinct from
    # operator pause: a gated worker keeps desired_state="running" and resumes
    # automatically when the scheduler releases the gate.
    gates: dict = field(default_factory=dict)
    # Estimated spend of worker slots that have since been removed, so the
    # task's cumulative total survives slot removal.
    retired_spend: float = 0.0
    # The bundle every bucket-delivering worker of this task runs, pinned when
    # the first one launches: an experiment's fleet stays homogeneous, and code
    # edited mid-run cannot silently become what the fleet is executing. The
    # source digest it was built from rides along so drift is a local
    # comparison (see WorkerManager.bundle_drift) rather than a bucket read.
    bundle_id: str | None = None
    bundle_source_hash: str = ""

    def worker(self, worker_id: str) -> WorkerRecord:
        for w in self.workers:
            if w.worker_id == worker_id:
                return w
        raise KeyError(f"no worker '{worker_id}'")


def task_path(spec: WorkloadSpec, tag: str) -> Path:
    return spec.data_dir(tag) / "task.json"


def load_task(spec: WorkloadSpec, tag: str) -> TaskRecord | None:
    path = task_path(spec, tag)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    raw["workers"] = [WorkerRecord(**w) for w in raw.get("workers", [])]
    return TaskRecord(**raw)


def save_task(spec: WorkloadSpec, task: TaskRecord):
    path = task_path(spec, task.tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(task), indent=2) + "\n")


def create_task(spec: WorkloadSpec, tag: str, raw_params: dict) -> TaskRecord:
    """Validate params against the workload's schema and persist a fresh task.
    Raises params.ParamsError on bad values, AssertionError on a taken tag.

    A workload's `finalize` hook runs after validation, on the typed params:
    its last chance to resolve derived fields (e.g. pin a "latest" reference to
    a concrete generation) before they are frozen, since task.json is re-read
    verbatim on every worker start with no dynamic step."""
    assert tag and all(c.isalnum() or c in "._-" for c in tag), f"invalid tag name '{tag}'"
    assert load_task(spec, tag) is None, f"tag '{tag}' already has a task"
    validated = params_mod.validate(spec.params_cls, raw_params)
    if spec.finalize:
        validated = resolve(spec.finalize)(spec, tag, validated)
    task = TaskRecord(workload=spec.name, tag=tag, params=asdict(validated), created_at=time.time())
    save_task(spec, task)
    return task


def delete_tag(spec: WorkloadSpec, tag: str):
    """Delete a tag's local dir (task record, data, stats, logs). Any cloud
    archive of the tag in the results bucket is deliberately untouched --
    purge it manually if truly done with it.

    The tag must have no worker slots left: this deletes the task record that
    tracks their pods and containers, so deleting past one would orphan the
    thing it was renting. Callers go through WorkerManager.delete_task, which
    tears the slots down first.
    """
    task = load_task(spec, tag)
    assert task is None or not task.workers, "remove the tag's workers first"
    tag_dir = spec.data_dir(tag)
    assert tag_dir.is_dir(), f"no such tag '{tag}'"
    shutil.rmtree(tag_dir)


def progress(spec: WorkloadSpec, tag: str) -> list:
    """The workload's [label, value] progress counters for the tag."""
    if not spec.progress:
        return []
    return [list(pair) for pair in resolve(spec.progress)(spec, tag)]


def _last_active(tag_dir: Path) -> float:
    """The tag's most recent real activity: the latest cycle any worker has
    published, plus the data subdirs' mtimes.

    Stats are read for their own `updated_at`, not the stats file's mtime: an
    ssh slot's records are re-copied from its container on every 5s reconcile
    pass regardless of whether the worker actually completed a cycle since
    the last one, so the file's mtime tracks the poll, not the work -- a
    stalled worker's container that is still reachable would otherwise read
    as active forever.

    The data scan goes two levels under `data/`, not one: a flat workload's
    files land directly in a `data/` subdir (e.g. `data/slogs/`), whose own
    mtime already bumps on arrival, but a generational workload nests one
    deeper (`data/generations/gen_NNNNNN/`), and a new file landing inside an
    existing generation dir only bumps *that* dir's mtime, not its parent's.

    Returns 0 for a tag nothing has happened in yet, rather than the tag
    dir's own creation time.
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
                # Slots the operator has running, a gated one included: the
                # scheduler resumes that one on its own. Read off desired
                # state rather than observed, so listing every tag stays free
                # of ssh and cloud round trips.
                "active_workers": sum(w.desired_state == "running" for w in workers),
                "progress": progress(spec, tag_dir.name),
                "last_active": _last_active(tag_dir),
            }
        )
    return sorted(out, key=lambda r: r["tag"])
