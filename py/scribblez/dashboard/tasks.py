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
    kind: str  # "local" | "cloud"
    desired_state: str  # "running" | "paused"
    threads: int | None = None  # local: engine thread count
    vcpus: int | None = None  # cloud: pod size
    flavor: str | None = None  # cloud: Runpod CPU flavor
    pod_id: str | None = None  # cloud: the backing pod
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
    Raises params.ParamsError on bad values, AssertionError on a taken tag."""
    assert tag and all(c.isalnum() or c in "._-" for c in tag), f"invalid tag name '{tag}'"
    assert load_task(spec, tag) is None, f"tag '{tag}' already has a task"
    validated = params_mod.validate(spec.params_cls, raw_params)
    task = TaskRecord(workload=spec.name, tag=tag, params=asdict(validated), created_at=time.time())
    save_task(spec, task)
    return task


def delete_tag(spec: WorkloadSpec, tag: str):
    """Delete a tag's local dir (task record, data, stats, logs). Only
    workerless tags may be deleted. Any cloud archive of the tag in the results
    bucket is deliberately untouched -- purge it manually if truly done with it.
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
    """The tag's most recent activity: worker stats records update every cycle,
    and the data subdir mtimes bump whenever files land."""
    stamps = [tag_dir.stat().st_mtime]
    stats = tag_dir / "stats"
    if stats.is_dir():
        stamps += [p.stat().st_mtime for p in stats.iterdir()]
    data = tag_dir / "data"
    if data.is_dir():
        stamps.append(data.stat().st_mtime)
        stamps += [p.stat().st_mtime for p in data.iterdir()]
    return max(stamps)


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
        out.append(
            {
                "tag": tag_dir.name,
                "has_task": task is not None,
                "created_at": task.created_at if task else None,
                "workers": len(task.workers) if task else 0,
                "progress": progress(spec, tag_dir.name),
                "last_active": _last_active(tag_dir),
            }
        )
    return sorted(out, key=lambda r: r["tag"])
