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
from dataclasses import asdict, dataclass, field
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
    Stats tab (aggregate tiles, rate/breakdown figures, summary table)."""

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
    # Worker kinds this role's slots may run as: a "local" subprocess, a
    # "cloud" pod, an "ssh" container on an operator-owned machine.
    kinds: tuple[str, ...] = ("local", "cloud", "ssh")
    # Whether this role runs on GPU hardware. Cloud pods for a GPU role rent a
    # GPU instance (a gpuTypeId + gpu count) rather than a CPU flavor, and the
    # dashboard's add-worker form offers GPU instances instead of CPU flavors.
    gpu: bool = False
    # Whether cloud pods for this role are rented interruptible (spot):
    # cheaper, but Runpod may stop them at any time. Only for roles that
    # tolerate preemption (the reconcile loop restarts reclaimed pods).
    interruptible: bool = False
    # Dotted path to a controller-side tick for this role,
    # dispatch(spec, tag, params, slots) -- for a role whose work the
    # controller assigns rather than the worker choosing it, and whose results
    # it ingests. `slots` holds one scribblez/dashboard/slot_files.py handle
    # per running slot of the role, the only way into a worker's filesystem;
    # since a rented pod has no such handle, such a role's kinds are local and
    # ssh. "" for the self-directing roles (a generator picks its own work).
    dispatch: str = ""
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
    # Dotted path to finalize(spec, tag, params) -> params: a creation-time step
    # that resolves derived/pinned fields before the params are frozen into
    # task.json -- e.g. pinning a "latest" reference to a concrete generation so
    # it cannot drift under a later worker restart. "" leaves params as validated.
    finalize: str = ""
    # data/ subdirectories cloud workers deliver into; cloud_sync pulls exactly
    # these bucket prefixes (plus stats/ and params/) down to the local mount.
    sync_data_dirs: tuple[str, ...] = ()
    # data/ subdirectories only local and ssh workers deliver into. An ssh
    # collection takes these as well (collected_dirs below), but they never
    # exist in the bucket, so asking cloud_sync for them would be one rclone
    # per watcher cycle against a prefix nothing can ever write.
    local_data_dirs: tuple[str, ...] = ()
    # The parameters the dashboard's new-tag form shows up front, in this
    # order; the rest are folded into its collapsed "Advanced" section. The
    # ordering is the form's, not the dataclass's, which groups fields by
    # subject instead. Empty means every parameter is shown up front.
    primary_params: tuple[str, ...] = ()
    # Parameter profiles (scribblez/params.py): name -> the values it sets over
    # the dataclass defaults. The new-tag form starts from `default_profile`
    # and lets the operator switch; the CLI takes --profile. A profile may set
    # any subset of the params, and its values must validate. Empty for a
    # workload with one recipe.
    profiles: dict[str, dict] = field(default_factory=dict, hash=False)
    default_profile: str = ""

    def __post_init__(self):
        names = {f.name for f in params_mod.schema(self.params_cls)}
        unknown = [n for n in self.primary_params if n not in names]
        assert not unknown, f"workload '{self.name}': primary_params names no such param {unknown}"
        assert len(set(self.primary_params)) == len(self.primary_params), (
            f"workload '{self.name}': duplicate primary_params"
        )
        for profile, values in self.profiles.items():
            try:
                params_mod.validate(self.params_cls, values)
            except params_mod.ParamsError as e:
                reasons = "; ".join(str(a) for a in e.args)
                raise AssertionError(
                    f"workload '{self.name}': profile '{profile}': {reasons}"
                ) from None
        if self.profiles:
            assert self.default_profile in self.profiles, (
                f"workload '{self.name}': default_profile {self.default_profile!r} is not a profile"
            )
        else:
            assert not self.default_profile, (
                f"workload '{self.name}': default_profile without profiles"
            )

    def add_cli_arguments(self, parser):
        """The params' argparse flags, plus --profile when the workload has profiles."""
        params_mod.add_arguments(parser, self.params_cls, self.profiles, self.default_profile)

    def params_from_args(self, args):
        """Params from `add_cli_arguments` flags: defaults under the chosen
        profile under the flags given."""
        return params_mod.from_args(self.params_cls, args, self.profiles)

    def resolve_params(self, profile: str | None, raw: dict):
        """(profile name, params): the dataclass defaults under the named
        profile's values -- the default profile when `profile` is None, none
        when the workload has none -- under `raw`, validated. What a new tag
        freezes."""
        name = self.default_profile if profile is None else profile
        assert not name or name in self.profiles, f"workload '{self.name}': no profile {name!r}"
        return name, params_mod.validate(self.params_cls, raw, base=self.profiles.get(name, {}))

    def profile_defaults(self, profile: str) -> dict:
        """Every param's value under `profile` alone (the dataclass defaults
        where it is silent) -- what the new-tag form shows before any edit."""
        return asdict(params_mod.validate(self.params_cls, {}, base=self.profiles.get(profile, {})))

    def profile_diff(self, profile: str, params: dict) -> list[dict]:
        """How a tag's frozen `params` depart from `profile`'s defaults, as
        [{name, profile, task}] -- the provenance the task view shows."""
        defaults = self.profile_defaults(profile)
        return [
            {"name": name, "profile": value, "task": params[name]}
            for name, value in defaults.items()
            if name in params and params[name] != value
        ]

    @property
    def collected_dirs(self) -> tuple[str, ...]:
        """Every data/ subdirectory a worker delivers into -- what a collection
        from an ssh container looks through."""
        return self.sync_data_dirs + self.local_data_dirs

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
    # The slot kind reported in stats. In-process runners (CLI tools, tests)
    # are local by construction; only a launcher of remote workers overrides it.
    kind: str = "local"
    provenance: dict = field(default_factory=dict)

    def tag_paths(self) -> TagPaths:
        return self.spec.paths(self.tag)
