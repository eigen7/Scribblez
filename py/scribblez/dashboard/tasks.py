"""Task records for the master dashboard.

A task is one (workload, tag) pair with a frozen parameter set and a list of
worker slots, persisted as task.json in the workload's per-tag data dir. Tags
that predate the dashboard (no task.json) still appear in listings, read-only.
"""

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from scribblez import params as params_mod
from scribblez.workloads import WorkloadSpec


@dataclass
class WorkerRecord:
    """One worker slot: durable identity + desired state. The backing process
    or pod's actual state is observed live by the WorkerManager."""

    worker_id: str
    kind: str  # "local" | "cloud"
    desired_state: str  # "running" | "paused"
    threads: int | None = None  # local: engine thread count
    vcpus: int | None = None  # cloud: pod size
    flavor: str | None = None  # cloud: Runpod CPU flavor
    pod_id: str | None = None  # cloud: the backing pod


@dataclass
class TaskRecord:
    workload: str
    tag: str
    params: dict  # raw param values (validated against the workload's schema)
    created_at: float
    workers: list[WorkerRecord] = field(default_factory=list)

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
    """Delete a tag's local data dir (task record, slogs, stats, logs). Only
    workerless tags may be deleted. Any cloud archive of the tag in the results
    bucket is deliberately untouched -- purge it manually if truly done with it.
    """
    task = load_task(spec, tag)
    assert task is None or not task.workers, "remove the tag's workers first"
    tag_dir = spec.data_dir(tag)
    assert tag_dir.is_dir(), f"no such tag '{tag}'"
    shutil.rmtree(tag_dir)


def _pair_count(tag_dir: Path) -> int:
    return sum(1 for _ in (tag_dir / "slogs").glob("*.sobs"))


def _last_active(tag_dir: Path) -> float:
    stamps = [tag_dir.stat().st_mtime]
    slogs = tag_dir / "slogs"
    if slogs.is_dir():
        stamps += [p.stat().st_mtime for p in slogs.iterdir()]
    return max(stamps)


def list_tags(spec: WorkloadSpec) -> list[dict]:
    """Every tag under the workload's data root, with listing metadata."""
    if not spec.data_root.is_dir():
        return []
    out = []
    for tag_dir in spec.data_root.iterdir():
        if not tag_dir.is_dir():
            continue
        task = load_task(spec, tag_dir.name)
        out.append(
            {
                "tag": tag_dir.name,
                "has_task": task is not None,
                "created_at": task.created_at if task else None,
                "workers": len(task.workers) if task else 0,
                "pairs": _pair_count(tag_dir),
                "last_active": _last_active(tag_dir),
            }
        )
    return sorted(out, key=lambda r: r["tag"])
