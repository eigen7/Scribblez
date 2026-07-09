"""The workload-spec contract: everything a launchable kind of work declares.

A WorkloadSpec ties together a workload's name, its frozen parameter dataclass
(see scribblez/params.py -- the single source of truth from which CLI flags,
worker env vars, the dashboard's config form, and validation all derive), its
worker roles (parallel generators, a singleton trainer, ...), an optional
controller-side scheduler, and how to summarize a tag's progress. Consumers:

  - the master dashboard (task creation, worker slots, the Stats tab, progress)
  - the cloud fleet CLI (scripts/cloud_fleet.py)
  - the worker entrypoint (py/cloud/worker_entrypoint.py), which reads
    SCZ_WORKLOAD + SCZ_ROLE and dispatches to the role's runner

Heavy code -- runners, deps fetchers, schedulers -- is referenced by dotted
path ("pkg.module:attr") and imported only when it runs, so the registry stays
importable on machines without torch or a GPU (cloud CPU pods).
"""

import importlib
from dataclasses import dataclass, field
from pathlib import Path

from scribblez import params as params_mod
from scribblez.paths import TagPaths


def resolve(dotted: str):
    """Import a "pkg.module:attr" reference and return the attribute."""
    module, _, attr = dotted.partition(":")
    assert attr, f"expected 'pkg.module:attr', got '{dotted}'"
    return getattr(importlib.import_module(module), attr)


@dataclass(frozen=True)
class StatsSpec:
    """The shape of a role's per-cycle stats samples, driving the generic
    Stats tab (summary table + throughput/breakdown/timeline figures)."""

    unit: str  # what a cycle delivers: "pairs", "games", "rows"
    phases: dict[str, str]  # sample key -> display label, in stacking order


@dataclass(frozen=True)
class RoleSpec:
    """One kind of worker slot a workload's tasks can hold."""

    name: str  # "generate", "train"
    title: str  # shown in the dashboard's add-worker forms
    runner: str  # dotted path to run(ctx: WorkerContext) -> int
    deps: str = ""  # dotted path to a fetch-runtime-deps callable, or ""
    singleton: bool = False  # at most one slot per task (the trainer)
    kinds: tuple[str, ...] = ("local", "cloud")
    # Whether cloud pods for this role are rented interruptible (spot):
    # cheaper, but Runpod may stop them at any time. Only for roles that
    # tolerate preemption (the reconcile loop restarts reclaimed pods).
    interruptible: bool = False
    stats: StatsSpec | None = None


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    title: str  # human-readable, shown in the dashboard's workload picker
    params_cls: type
    roles: tuple[RoleSpec, ...]
    # Dotted path to a controller-side per-task tick
    # tick(spec, task, hooks: SchedulerHooks), run by the dashboard server's
    # reconcile loop; "" for workloads without generation lifecycle to manage.
    scheduler: str = ""
    # Dotted path to progress(spec, tag) -> list[(label, value)]: the counters
    # shown in the tag listing and the task Overview.
    progress: str = ""
    # data/ subdirectories cloud workers deliver into; cloud_sync pulls exactly
    # these bucket prefixes (plus stats/ and params/) down to the local mount.
    sync_data_dirs: tuple[str, ...] = ()

    def paths(self, tag: str, mount_root=None) -> TagPaths:
        return TagPaths(tag, self.name, *([mount_root] if mount_root else []))

    def data_dir(self, tag: str) -> Path:
        """The tag's root (task.json, logs/, stats/, data/, ...)."""
        return self.paths(tag).root

    @property
    def tags_root(self) -> Path:
        """Parent directory of every tag of this workload."""
        return TagPaths("placeholder", self.name).root.parent

    def role(self, name: str) -> RoleSpec:
        for r in self.roles:
            if r.name == name:
                return r
        raise KeyError(f"workload '{self.name}' has no role '{name}'")

    def worker_env(self, tag: str, params, role: str) -> dict[str, str]:
        """The SCZ_* environment defining this work for a worker entrypoint
        (local or cloud); worker-level knobs (sink, threads, worker id) and R2
        credentials are layered on top by the launcher."""
        self.role(role)  # validate
        return {
            "SCZ_WORKLOAD": self.name,
            "SCZ_ROLE": role,
            "SCZ_TAG": tag,
            **params_mod.to_env(params),
        }


@dataclass
class SchedulerHooks:
    """The narrow surface a scheduler tick gets from the dashboard server.

    gate(role, reason) parks every worker of `role` (distinct from operator
    pause; shown as "waiting" with the reason) or, with reason=None, releases
    it. mirror(chunk_name, dest_rel), when present, replays a local staging
    ingest in the results bucket so the bucket layout keeps mirroring the local
    corpus and the sync watcher never re-downloads an ingested chunk.
    """

    gate: object  # callable(role: str, reason: str | None)
    mirror: object = None  # callable(chunk_name: str, dest_rel: str) | None


@dataclass
class WorkerContext:
    """Everything a role runner needs, assembled by the worker entrypoint."""

    spec: WorkloadSpec
    role: RoleSpec
    tag: str
    params: object
    worker_id: str
    threads: int
    max_cycles: int  # 0 = run until stopped
    sink: object  # cloud.sinks.LocalSink | R2Sink
    provenance: dict = field(default_factory=dict)

    def tag_paths(self) -> TagPaths:
        return self.spec.paths(self.tag)
