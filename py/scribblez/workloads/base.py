"""The workload-spec contract: everything a launchable kind of work declares.

A WorkloadSpec ties together a workload's name, its frozen parameter dataclass,
its worker roles (parallel generators, a singleton trainer, ...), an optional
controller-side scheduler, and how to summarize a tag's progress. The params
dataclass is the single source from which CLI flags, worker env vars, the
dashboard's new-tag form and validation all derive (scribblez/params.py).

Two consumers read a spec:

  - the master dashboard: task creation, worker slots, the Stats tab, progress.
  - the worker entrypoint (py/cloud/worker_entrypoint.py), which reads
    SCZ_WORKLOAD and SCZ_ROLE and calls the role's runner.

Heavy code (runners, deps fetchers, schedulers) is referenced by a dotted
"pkg.module:attr" path and imported only when it runs. That keeps the registry
importable in processes without torch or a GPU: the dashboard, and CPU-only
worker containers.
"""

import importlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cloud.runtime_abi import RUNTIME_ENGINE, RUNTIMES

from scribblez import params as params_mod
from scribblez.paths import TagPaths


def resolve(dotted: str):
    """Import a "pkg.module:attr" reference and return the attribute."""
    module, _, attr = dotted.partition(":")
    assert attr, f"expected 'pkg.module:attr', got '{dotted}'"
    return getattr(importlib.import_module(module), attr)


@dataclass(frozen=True)
class StatsSpec:
    """The shape of a role's per-cycle stats samples, from which the generic
    Stats tab builds its tiles, rate and breakdown figures, and summary table."""

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
    # Worker kinds this role's slots may run as: "local" (a subprocess on the
    # controller's machine) or "ssh" (a container on a machine reached over
    # ssh, either the operator's own or one rented for the task).
    kinds: tuple[str, ...] = ("local", "ssh")
    # Whether the role needs a GPU. Its container gets the machine's GPUs, and
    # a machine of known shape refuses the slot when none is free.
    gpu: bool = False
    # The worker image a remote slot runs on (cloud/runtime_abi.py): "engine"
    # when the binaries and the FFI cover the role, "torch" when it imports the
    # training stack. Independent of `gpu`: match eval uses a GPU on the engine
    # image.
    runtime: str = RUNTIME_ENGINE
    # Dotted path to a controller-side tick, dispatch(spec, tag, params, slots),
    # for a role whose work the controller assigns and whose results it
    # collects (match eval). `slots` holds one scribblez/dashboard/slot_files.py
    # handle per running slot, the controller's only way into a worker's
    # filesystem. "" for roles that pick their own work, like generators.
    dispatch: str = ""
    # Dotted path to a controller-side tick, ingest(spec, tag), that writes what
    # the role has delivered under the tag into dashboard.db (a trainer's
    # records: generational/train_ingest.py). It reads only the tag on the
    # controller's mount, so unlike dispatch it works for a slot of any kind.
    # "" for roles that deliver nothing the controller has to write.
    ingest: str = ""
    # Dotted path to inputs(params) -> {rel: Path}: files the role reads from
    # outside its own tag (another tag's model export, say), keyed by the
    # tag-relative name the worker looks for them under. A local worker reads
    # each source in place. For a remote slot the controller stages a copy
    # before the slot needs it: into the bucket for a bucket-delivering slot,
    # into the container otherwise. resolve_input below finds whichever copy
    # exists. "" when every input is in the bundle, the runtime deps, or the
    # tag itself.
    inputs: str = ""
    stats: StatsSpec | None = None


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    title: str  # human-readable, shown in the dashboard's workload picker
    params_cls: type
    roles: tuple[RoleSpec, ...]
    # Dotted path to a controller-side per-task tick,
    # tick(spec, task, hooks: SchedulerHooks), run by the dashboard server's
    # reconcile loop. "" for workloads with nothing to schedule.
    scheduler: str = ""
    # Dotted path to progress(spec, tag) -> list[(label, value)]: the counters
    # shown in the tag listing and the task Overview.
    progress: str = ""
    # Dotted path to finalize(spec, tag, params) -> params, run at task creation
    # before the params are frozen into task.json. It resolves fields that must
    # not drift later, such as pinning a "latest" reference to a concrete
    # generation so a worker restart cannot pick up a newer one. "" leaves the
    # params as validated.
    finalize: str = ""
    # data/ subdirectories bucket-delivering workers write into.
    # scripts/cloud_sync.py pulls exactly these bucket prefixes (plus stats/
    # and params/) down to the local mount.
    sync_data_dirs: tuple[str, ...] = ()
    # data/ subdirectories that only local and ssh workers write into. A
    # collection from an ssh container takes these too (collected_dirs), but
    # they never exist in the bucket, so they are kept out of sync_data_dirs:
    # cloud_sync would otherwise spend an rclone call per cycle on a prefix
    # nothing writes.
    local_data_dirs: tuple[str, ...] = ()
    # The parameters the dashboard's new-tag form shows up front, in this
    # order; the rest fold into its collapsed "Advanced" section. The dataclass
    # groups fields by subject, so this order is independent of it. Empty
    # shows every parameter up front.
    primary_params: tuple[str, ...] = ()
    # Parameter profiles (scribblez/params.py): profile name -> the values it
    # sets over the dataclass defaults. A profile may set any subset of the
    # params, and its values must validate. The new-tag form starts from
    # `default_profile` and lets the operator switch; the CLI takes --profile.
    # Empty for a workload with a single recipe.
    profiles: dict[str, dict] = field(default_factory=dict, hash=False)
    default_profile: str = ""

    def __post_init__(self):
        for role in self.roles:
            assert role.runtime in RUNTIMES, (
                f"workload '{self.name}': role '{role.name}' names no such runtime {role.runtime!r}"
            )
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
        """Params from `add_cli_arguments` flags. Precedence, lowest first: the
        dataclass defaults, the chosen profile, the flags given."""
        return params_mod.from_args(self.params_cls, args, self.profiles)

    def resolve_params(self, profile: str | None, raw: dict):
        """The (profile name, validated params) a new tag freezes.

        Precedence, lowest first: the dataclass defaults, the profile, `raw`.
        `profile` None means the default profile (or none, for a workload
        without profiles)."""
        name = self.default_profile if profile is None else profile
        assert not name or name in self.profiles, f"workload '{self.name}': no profile {name!r}"
        return name, params_mod.validate(self.params_cls, raw, base=self.profiles.get(name, {}))

    def profile_defaults(self, profile: str) -> dict:
        """Every param's value under `profile` alone, with dataclass defaults
        where it is silent: what the new-tag form shows before any edit."""
        return asdict(params_mod.validate(self.params_cls, {}, base=self.profiles.get(profile, {})))

    def profile_diff(self, profile: str, params: dict) -> list[dict]:
        """How a tag's frozen `params` depart from `profile`'s values, as
        [{name, profile, task}] rows for the task view's provenance table."""
        defaults = self.profile_defaults(profile)
        return [
            {"name": name, "profile": value, "task": params[name]}
            for name, value in defaults.items()
            if name in params and params[name] != value
        ]

    @property
    def collected_dirs(self) -> tuple[str, ...]:
        """Every data/ subdirectory a worker delivers into: what a collection
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
        """The SCZ_* environment that tells a worker entrypoint what to run. The
        launcher adds worker-level settings (sink, threads, worker id) and the
        bucket credentials on top."""
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

    gate(role, reason)
        Park every worker of `role`, shown as "waiting" with the reason; this is
        separate from an operator pause. reason=None releases the gate.
    mirror(chunk_name, dest_rel), optional
        Repeat a local staging ingest (a chunk moved into a generation) in the
        results bucket. The bucket keeps mirroring the local corpus, so the
        sync watcher never downloads an ingested chunk a second time.
    publish(dest_rel), optional
        Upload a complete generation to the bucket, chunks first and manifest
        last, for a trainer that runs elsewhere and reads it from there. The
        scheduler calls it once per generation and records success in the
        manifest, so a failed call is retried on the next tick.
    """

    gate: object  # callable(role: str, reason: str | None)
    mirror: object = None  # callable(chunk_name: str, dest_rel: str) | None
    publish: object = None  # callable(dest_rel: str) | None


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
    # The slot kind, reported in stats and consulted by resolve_input.
    # In-process runners (CLI tools, tests) are local; only a launcher of
    # remote workers overrides it.
    kind: str = "local"
    provenance: dict = field(default_factory=dict)
    # Root of the tag tree this worker reads and writes: the mount dir, unless
    # the launcher points it elsewhere (a bucket-delivering trainer on a
    # machine whose mount dir belongs to the controller).
    mount_root: Path | None = None

    def tag_paths(self) -> TagPaths:
        return self.spec.paths(self.tag, self.mount_root)


# How long resolve_input waits for a staged copy. The controller pushes a
# container's inputs right after creating it, so a wait this long means the
# staging failed, not that it is slow.
INPUT_WAIT_SECONDS = 600
INPUT_POLL_SECONDS = 5


def resolve_input(ctx: WorkerContext, rel: str, source: Path) -> Path:
    """Where a runner reads input `rel` (a RoleSpec.inputs key) from.

    `source` itself when it exists, as it does for a local worker sharing the
    controller's mount. Otherwise the staged copy at `rel` under the tag root:
    either pushed into the container already, or fetched through the sink by a
    bucket-delivering slot. A remote slot polls until the copy arrives; a local
    worker, for which nothing is staged, checks once and never asks the sink.
    Raises FileNotFoundError when no copy turns up."""
    if source.is_file():
        return source
    staged = ctx.tag_paths().root / rel
    missing = FileNotFoundError(f"input {rel} was neither at {source} nor staged at {staged}")
    if ctx.kind == "local":
        if staged.is_file():
            return staged
        raise missing
    deadline = time.monotonic() + INPUT_WAIT_SECONDS
    while not (staged.is_file() or ctx.sink.fetch_file(rel, staged)):
        if time.monotonic() >= deadline:
            raise missing
        time.sleep(INPUT_POLL_SECONDS)
    return staged
