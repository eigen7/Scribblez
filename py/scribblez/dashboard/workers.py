"""The WorkerManager: reconciles worker slots with real processes and
containers, and the task's machines with rented instances.

Owned by the dashboard API process. Local worker slots are backed by
subprocesses of this process running the worker entrypoint with the local
results sink; ssh slots are backed by worker-image containers on machines
reached over ssh (cloud/ssh_machine.py) -- the operator's own, or ones the
dashboard rents from a provider (cloud/providers/) and records as the task's
machines. A slot on a rented machine, and a trainer anywhere remote, delivers
through the results bucket, so while a task has any such slot, a cloud_sync
--watch subprocess streams that tag's bucket results into the local mount.

Adding a slot only records it, paused: nothing launches until the operator
starts it; the first start spawns the process / container. Renting a machine
launches its instance at once, which bills from then on.

Desired state lives in task.json (dashboard/tasks.py); actual state is observed
live -- local workers by their durable pid (worker_pid_alive reads /proc, so a
worker is observable and stoppable no matter which dashboard instance spawned
it, even across a restart), ssh slots by a docker probe over ssh, rented
machines by the provider's listing plus that probe. reconcile() drives
observed toward desired in both directions: it relaunches local workers that
should be running (e.g. after a dashboard restart), starts a stopped machine a
slot wants, stops one nothing has run on, and stops workers that are running
but should not be. It also
runs each workload's scheduler tick (generation lifecycle + fleet pacing): a
scheduler may *gate* a role -- park its workers without touching the operator's
desired state -- and reconcile stops a gated worker just as it stops a paused
one.

Cloud operations need <mount>/cloud/credentials.json; credentials load lazily
so a local-only dashboard works without them.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path

from cloud import runtime_abi
from cloud.bundles import BundleManifest, deploy_current_tree, source_hash
from cloud.credentials import CloudCredentials, CredentialsError, load_credentials
from cloud.providers.aws import AwsProvider
from cloud.providers.base import Instance, LaunchRequest, Provider, ProviderError
from cloud.r2 import bucket_path, rclone
from cloud.ssh_machine import SshMachine, SshMachineError
from cloud.ssh_transfer import pull_results, push_file, sweep_stopped
from cloud.worker_env import bundle_worker_env
from tornado.ioloop import IOLoop

from scribblez import params as params_mod
from scribblez import workloads
from scribblez.dashboard import tasks
from scribblez.dashboard.slot_files import LocalSlotFiles, SshSlotFiles
from scribblez.generational.lifecycle import MANIFEST_NAME
from scribblez.hardware import default_thread_count
from scribblez.paths import CONTROLS_REL, DEFAULT_MOUNT_ROOT, REPO_ROOT
from scribblez.workloads.base import SchedulerHooks

CLOUD_SYNC = REPO_ROOT / "py" / "scripts" / "cloud_sync.py"
SYNC_INTERVAL_SECONDS = 30


def _slot_sink(spec: workloads.WorkloadSpec, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> str:
    """Where slot `w`'s worker delivers (cloud/sinks.py's SCZ_SINK): "local"
    for a local subprocess, and for an ssh container on the operator's own
    machine whose output the reconcile pass reads back over the control link
    (cloud/ssh_transfer.py); "r2" for an ssh container on a rented
    machine (a datacenter link to the bucket, where collection over ssh would
    haul every chunk to the controller and publish it back up from a home
    uplink), and for an ssh container running a role with inputs as well as
    outputs anywhere -- a trainer, whose generations arrive and whose
    exports, checkpoint and records leave through the bucket
    (docs/plans/cloud_machines.md). Everything the controller does for a
    bucket-delivering slot -- the sync watcher, the scheduler's publish and
    mirror hooks, the controls push -- keys off this, not off the kind."""
    if w.kind == "local":
        return "local"
    if w.kind == "ssh" and not spec.role(w.role).ingest and not _rented(task, w):
        return "local"
    return "r2"


def _rented(task: tasks.TaskRecord, w: tasks.WorkerRecord) -> bool:
    return w.machine is not None and task.machine(w.machine).instance_id is not None


def _has_bucket_slots(spec: workloads.WorkloadSpec, task) -> bool:
    return any(_slot_sink(spec, task, w) == "r2" for w in task.workers)


def _bucket_trainer(spec: workloads.WorkloadSpec, task) -> bool:
    """Whether a slot whose role delivers records the controller ingests (a
    trainer) runs through the bucket -- the case that has the sync pull its
    outputs and the controls file pushed up for it."""
    return any(_slot_sink(spec, task, w) == "r2" and spec.role(w.role).ingest for w in task.workers)


# After an ssh machine fails a probe, how long it is assumed still unreachable
# before probing again -- so a powered-off machine costs one connect timeout
# per reconcile pass, not one per pass on a host that is simply off.
SSH_REPROBE_SECONDS = 30.0

# A container that dies as fast as it is started is not going to be fixed by
# starting it again: restarts back off from the pass interval up to this, and
# reset the moment one is observed running.
MAX_RESTART_BACKOFF_SECONDS = 300.0

# How long an observation of a container or machine stands in for a fresh one.
# Only the reconcile pass observes; status requests read what it left, so a
# browser polling every 3 seconds costs no ssh round trips at all.
OBSERVATION_TTL_SECONDS = 5.0

# A rented machine on which nothing has run for this long is stopped (its
# disk kept, its rate no longer charged): the "Pause all" and the finished
# run that cost money on a provider without a real suspend.
IDLE_STOP_SECONDS = 600.0

# The nice level local workers run at. A worker is the long-running background
# job on this machine; everything else that competes with it -- a bundle build
# for a fleet that is billing while it waits, the test suite, an editor's build
# -- is short and should win. At this level the scheduler gives a nice-0
# process about ten times a worker's CPU share under contention, and an idle
# machine still gives the worker all of it.
LOCAL_WORKER_NICE = 10
# Per-machine key material for rented machines (known_hosts files).
MACHINES_DIR = Path("/workspace/mount/cloud/machines")
# How long the rent form's spot rates are served from the last fetch.
SPOT_PRICES_TTL_SECONDS = 300.0
# A rented instance whose ssh does not answer is still coming up for this
# long after its launch or start before it reads as unreachable.
BOOT_GRACE_SECONDS = 300.0


# What a slot should be doing, from operator intent plus scheduler gating.
# "park" and "stop" both mean not-working; they differ in how much of the
# worker survives it, which matters where restarting is expensive.
RUN, PARK, STOP = "run", "park", "stop"


def _role_inputs(spec, role, params) -> dict[str, Path]:
    """The files a slot of `role` reads from outside its tag (RoleSpec.inputs),
    resolved against the controller's mount; empty for a role with none."""
    return workloads.resolve(role.inputs)(params) if role.inputs else {}


def _require_inputs(inputs: dict[str, Path]):
    """Every input source must exist before a slot is started for it."""
    for rel, src in inputs.items():
        if not src.is_file():
            raise SshMachineError(f"input {rel} is missing: {src} is not a readable file")


def _stage_inputs_in_container(machine, container: str, spec, tag: str, inputs: dict[str, Path]):
    """A container on the operator's own machine delivers over the control
    link, and takes its inputs the same way: pushed into it right after it
    is created, under the tag root there (the runner waits for them)."""
    root = str(spec.paths(tag).root)
    for rel, src in inputs.items():
        push_file(machine, container, remote_root=root, rel_dest=rel, src=src)


def _transfer_target(spec, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> dict:
    """Where slot `w`'s output lives, on both machines. The container runs the
    controller's own layout under the same mount root, so the two roots read
    alike -- but they are different machines' paths, and both collection paths
    (a batched pull, a sweep of a stopped container) need all four."""
    paths = spec.paths(task.tag)
    return {
        "container": _container_name(spec, task.tag, w.worker_id),
        "remote_root": str(paths.root),
        "local_root": paths.root,
        "data_dirs": [f"data/{sub}" for sub in spec.collected_dirs],
    }


def _ssh_machine(task: tasks.TaskRecord, w: tasks.WorkerRecord) -> SshMachine:
    """The link to slot `w`'s machine -- the only place one is built. A slot
    on one of the task's machines (TaskRecord.machines) takes the record's
    address and key material; a bare-host slot dials its host string."""
    if w.machine is None:
        return SshMachine(w.host)
    return _machine_link(task.machine(w.machine))


def _machine_link(m: tasks.MachineRecord) -> SshMachine:
    return SshMachine(m.host, m.identity_file, m.known_hosts_file)


def _machine_key(spec: workloads.WorkloadSpec, tag: str, name: str) -> str:
    return _key(spec, tag, f"machine:{name}")


def _next_machine_name(task: tasks.TaskRecord, provider: str) -> str:
    """`<provider>-N`, the first N no machine of the task has."""
    taken = {m.name for m in task.machines}
    n = 1
    while f"{provider}-{n}" in taken:
        n += 1
    return f"{provider}-{n}"


def _owner(spec: workloads.WorkloadSpec, tag: str, name: str) -> str:
    """The ownership tag a rented instance carries: which task's machine it
    is. One no task's machines name is an orphan."""
    return f"{spec.name}/{tag}/{name}"


def _accrue_machine(m: tasks.MachineRecord, billing: bool):
    """Advance a rented machine's spend to now, as _accrue does a slot's:
    the interval since the last observation is charged if it was billing
    then (an instance bills while pending or running, not while stopped)."""
    with _ACCRUE_LOCK:
        now = time.time()
        if m.observed_up and m.observed_at is not None:
            m.spend += (now - m.observed_at) / 3600 * (m.cost_per_hr or 0.0)
        m.observed_at = now
        m.observed_up = billing


def _rented_state(m: tasks.MachineRecord, inst: Instance | None, probe: str | None) -> str:
    """A rented machine's display state from what its provider says and,
    when the instance is running, what its ssh probe found: `launching`
    while the instance is pending or freshly running and not yet answering,
    `preparing` while its first-boot script is still pulling the images,
    `up` when it can host containers, `stopping` / `stopped` when
    suspended, `gone` once terminated -- or listed by nobody, which after a
    real listing means the same."""
    if inst is None or inst.state == "terminated":
        return "gone"
    if inst.state in ("pending", "stopping", "stopped"):
        return {"pending": "launching"}.get(inst.state, inst.state)
    if probe == "unreachable":
        since = m.launched_at or 0.0
        return "launching" if time.time() - since < BOOT_GRACE_SECONDS else "unreachable"
    return probe or "checking"


def _ssh_host(task: tasks.TaskRecord, w: tasks.WorkerRecord) -> str:
    return w.host if w.machine is None else task.machine(w.machine).host


def _forget_empty(w: tasks.WorkerRecord):
    """Give up a count of zero, keeping any other.

    Zero is the one value that authorises destroying a container, and it is
    only worth anything while it is current: a machine that has been off the
    network for hours has a worker that went on filling it up the whole time.
    A positive count is kept because it only ever refuses -- being wrong about
    it costs nothing.
    """
    if w.undelivered == 0:
        w.undelivered = None


def _holds_nothing(spec: workloads.WorkloadSpec, task: tasks.TaskRecord, w: tasks.WorkerRecord):
    """An ssh slot delivering through the bucket keeps nothing in its
    container for the controller to collect, so its count is always zero:
    what the replace rule and the Remove dialog read."""
    if w.kind == "ssh" and _slot_sink(spec, task, w) == "r2":
        w.undelivered = 0


def _note_finished(w: tasks.WorkerRecord):
    """Slot `w`'s worker exited 0: its role's terminal condition (a trainer's
    max_rows, a generator's cycle cap) is reached. Flip it to paused so
    reconcile does not restart it every backoff period forever -- a machine
    whose trainer finished would otherwise never idle -- and remember why."""
    if w.desired_state == "running":
        w.desired_state = "paused"
        w.finished = True


def _replaceable(w: tasks.WorkerRecord, task: tasks.TaskRecord) -> bool:
    """Whether slot `w`'s container may be thrown away for one on the task's
    bundle: it has to be on a different bundle, and known to be holding
    nothing. A container created but never collected from reports zero from
    the moment it is created, so one that never came up at all -- the state a
    redeploy is often trying to fix -- is replaceable rather than restarted
    forever."""
    return w.bundle_id != task.bundle_id and w.undelivered == 0


def _intent(w: tasks.WorkerRecord, task: tasks.TaskRecord) -> str:
    if w.desired_state != "running":
        return STOP
    return PARK if w.role in task.gates else RUN


def check_worker_images_current():
    """Refuse to deploy a bundle a published worker image cannot load.

    Bundles are compiled here, in the dev container, and run there, against
    the worker images' libraries -- so a dev image whose toolchain moved
    produces binaries no worker can start (August 2026: gcc-16's libstdc++,
    which crash-looped every ssh worker at import). The worker images are
    rebuilt by hand after such a change; this catches the case where they
    were not.

    Every recorded image is checked, whichever runtime this deploy's slots
    will use: a task's slots can run either, and both are rebuilt together.
    Says nothing when no push has recorded what the images provide, which is
    all that can honestly be said about them.
    """
    records = runtime_abi.read_records(DEFAULT_MOUNT_ROOT)
    if records is None:
        return
    local = runtime_abi.local_versions()
    for record in records.values():
        stale = runtime_abi.stale_libraries(record.get("versions", {}), local)
        assert not stale, (
            f"the worker image ({record.get('image')}) is older than this dev container on "
            f"{', '.join(stale)}; bundles built here will not load on it. Rebuild it from the "
            "host: ./build_and_push_worker_image.py"
        )


def _key(spec: workloads.WorkloadSpec, tag: str, worker_id: str = "") -> str:
    return f"{spec.name}/{tag}/{worker_id}"


def _next_worker_id(task: tasks.TaskRecord, prefix: str) -> str:
    """The first free "<prefix>-<n>" worker id in the task."""
    taken = {w.worker_id for w in task.workers}
    return next(f"{prefix}-{i}" for i in range(len(taken) + 1) if f"{prefix}-{i}" not in taken)


def _container_name(spec: workloads.WorkloadSpec, tag: str, worker_id: str) -> str:
    """An ssh slot's container name on its machine: qualified by workload and
    tag so one machine can serve several tasks without collisions."""
    return f"scz-{spec.name}-{tag}-{worker_id}"


# The entrypoint module name every local worker runs, matched in its /proc
# cmdline to tell a live worker from a reused pid.
_WORKER_ENTRYPOINT = "cloud.worker_entrypoint"


def _proc_env(pid: int) -> dict[bytes, bytes]:
    """The process's environment as a byte-keyed dict (empty if unreadable)."""
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    return dict(e.split(b"=", 1) for e in raw.split(b"\0") if b"=" in e)


def worker_pid_alive(pid: int | None, worker_id: str, tag: str) -> bool:
    """Whether `pid` is a live worker-entrypoint process for exactly this slot.

    Reads /proc directly rather than a subprocess handle, so liveness is
    observable no matter which dashboard instance spawned the worker -- and
    survives a dashboard restart. Confirms both the command and the identifying
    env (SCZ_WORKER_ID / SCZ_TAG) so a recycled pid (the original worker died
    and the number was reused) never reads as alive. A zombie's cmdline is
    empty, so it too reads as dead."""
    if pid is None:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    if _WORKER_ENTRYPOINT.encode() not in cmdline:
        return False
    env = _proc_env(pid)
    return env.get(b"SCZ_WORKER_ID") == worker_id.encode() and env.get(b"SCZ_TAG") == tag.encode()


def _local_state(desired: str, alive: bool, gated: bool, finished: bool = False) -> str:
    """The honest display state of a local slot from the two observed axes
    (operator intent + real liveness) plus scheduler gating. `stopping` is the
    in-flight state where a paused slot's process has not yet exited; `exited`
    is an unexpected death of a slot that should be running (reconcile respawns
    it); `finished` is the exit that reached the role's terminal condition."""
    if gated:
        return "waiting"
    if desired == "paused":
        if finished and not alive:
            return "finished"
        return "stopping" if alive else "paused"
    return "running" if alive else "exited"


def _ssh_state(desired: str, probe: str, gated: bool, finished: bool = False) -> str:
    """The honest display state of an ssh slot from its container probe
    (cloud/ssh_machine.py's probe states, plus "unknown" for a slot the
    reconcile pass has not observed yet). `unreachable` is its own display
    state rather than a guess either way: the machine may be powered off with
    the worker gone, or merely off the network with the worker still running.
    A gated slot reads `waiting` whether its container is paused (the usual
    case) or still winding down.

    A slot that should be running and has no container is `starting`, not
    `exited`: the two probe states mean different things to whoever is
    watching. `stopped` is a container that ran and died, which is the alarm
    the word carries; `missing` is one that does not exist yet, which is the
    ordinary state of a slot between the operator's Start and the moment its
    container exists -- and that is not a moment, it is however long the
    machine takes to pull an image measured in gigabytes. Reading that as
    `exited` announced a dead worker every time one was created."""
    if probe == "unknown":
        return "checking"
    if gated:
        return "waiting"
    if probe == "unreachable":
        return "unreachable"
    if desired == "paused":
        if finished and probe == "stopped":
            return "finished"
        return "stopping" if probe in ("running", "paused") else "paused"
    if probe == "missing":
        return "starting"
    if probe == "running":
        return "running"
    # Paused while it is meant to be running: a gate released between the pass
    # that parked it and the one that will resume it. Reporting that as
    # "exited" -- next to an empty exit reason, because nothing exited --
    # describes a healthy worker as a dead one.
    return "starting" if probe == "paused" else "exited"


_ACCRUE_LOCK = threading.Lock()


class WorkerManager:
    def __init__(self):
        self._local: dict[str, subprocess.Popen] = {}  # slot key -> live process
        # task key -> (sync watcher, the argv it runs): a watcher is replaced
        # when what it should pull changes.
        self._sync: dict[str, tuple[subprocess.Popen, list[str]]] = {}
        # task key -> controls.json mtime as last pushed to the bucket.
        self._controls_pushed: dict[str, int] = {}
        self._creds_cache: CloudCredentials | None = None
        self._ssh_down: dict[str, float] = {}  # host -> time of last failed probe
        # Slot key -> (probe state, when observed). Written by the reconcile
        # pass, read by everything else (see _probe_container).
        self._probes: dict[str, tuple[str, float]] = {}
        # Tasks whose recorded counts this process has already vetted; see
        # _forget_stale_counts.
        self._counted_from: set[str] = set()
        # Slot key -> why its container is not running (exit code + last log
        # line), so a crash-looping worker explains itself on the dashboard
        # instead of only in `docker logs`.
        self._exits: dict[str, str] = {}
        # Slot key -> (consecutive restarts, when the next one is allowed).
        self._restarts: dict[str, tuple[int, float]] = {}
        # Machine key -> (probe state, when observed): SshMachine.probe for
        # each of a task's machines, refreshed by the reconcile pass ahead of
        # its slots (see machine_status).
        self._machine_probes: dict[str, tuple[str, float]] = {}
        # Machine key -> its last composite state (machine_status), read by
        # the slot rules (a slot on a gone machine is removable outright).
        self._machine_states: dict[str, str] = {}
        # Machine key -> when it was first seen idle (see _reconcile_machines).
        self._idle_since: dict[str, float] = {}
        # (instances by id, when listed): the provider's view of every
        # instance it tagged ours, refreshed by the reconcile pass like the
        # container probes above.
        self._instances: tuple[dict[str, Instance], float] = ({}, 0.0)
        # Why the pass's last fleet listing failed (no credentials, a provider
        # error), shown by the burn strip in place of a listing; None when
        # the last one succeeded.
        self._fleet_error: str | None = None
        self._provider_client: Provider | None = None
        self._account: str | None = None  # the provider's account line, once asked
        # (spot rate by type, when fetched): the rent form's, refreshed lazily.
        self._spot_prices: tuple[dict[str, float], float] = ({}, 0.0)
        # Where every blocking step runs (see _offload). One thread: the point
        # is to keep the event loop free, not to do two of these at once.
        self._blocking = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scz-blocking")
        # Where a redeploy's build runs (see redeploy): off the blocking
        # thread, which it would otherwise hold for minutes.
        self._builds = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scz-build")
        # A task's first-use bundle build in flight (task key -> Future), see
        # _bundle_for_start.
        self._pending_builds: dict[str, Future] = {}
        # Per-file digests behind source_hash, so the drift check every status
        # poll makes costs a stat walk rather than 20 MB of hashing.
        self._source_digests: dict = {}

    # ---- bundle deployment -------------------------------------------------

    def deploy(self, spec, task: tasks.TaskRecord) -> str:
        """Build the controller's tree for the task's archs, push it unless
        the bucket already has it, and pin the task to the result. Returns
        the bundle id."""
        return self._pin_bundle(spec, task, self._build_bundle(self._needed_archs(spec, task)))

    async def redeploy(self, spec, task: tasks.TaskRecord) -> str:
        """The operator's Redeploy: `deploy`, with the build on its own thread.

        Building and pushing takes minutes and touches no record; run through
        `offload` it held the one blocking thread that long, and every Pause
        and Remove clicked meanwhile landed after it -- on machines that had
        gone on billing. Only the arch survey and the repin are serialized
        steps.
        """
        archs = await self.offload(self._needed_archs, spec, task)
        manifest = await IOLoop.current().run_in_executor(self._builds, self._build_bundle, archs)
        return await self.offload(self._pin_bundle, spec, task, manifest)

    def _build_bundle(self, archs: list[str]) -> BundleManifest:
        check_worker_images_current()
        creds = self._creds()
        return deploy_current_tree(creds.r2, archs, cache=self._source_digests)

    def _pin_bundle(self, spec, task: tasks.TaskRecord, manifest: BundleManifest) -> str:
        task.bundle_id = manifest.bundle_id
        task.bundle_source_hash = manifest.source_hash
        task.bundle_archs = list(manifest.archs)
        tasks.save_task(spec, task)
        return manifest.bundle_id

    def _slot_arch(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> str:
        """The CPU microarchitecture slot `w`'s machine reports -- what its
        bundle must be built for. A rented machine's is its catalog entry's; a
        registered machine or a bare host is asked once, over ssh, through the
        worker image's compiler (the container's own start-up detection), and
        the answer kept on its record."""
        holder = task.machine(w.machine) if w.machine is not None else w
        if holder.arch:
            return holder.arch
        image = self._creds().registry.image_for(spec.role(w.role).runtime)
        machine = _ssh_machine(task, w)
        machine.pull_image(image)
        holder.arch = machine.detect_arch(image)
        tasks.save_task(spec, task)
        return holder.arch

    def _needed_archs(self, spec, task: tasks.TaskRecord) -> list[str]:
        """The archs the task's bundle must cover: every ssh slot's machine's,
        plus whatever its current bundle already covers (a slot removed since
        does not un-need its arch for the containers still running it)."""
        archs = set(task.bundle_archs)
        for w in task.workers:
            if w.kind == "ssh":
                archs.add(self._slot_arch(spec, task, w))
        return sorted(archs)

    def _bundle_for_start(
        self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord, key: str
    ) -> str | None:
        """The bundle this task's remote workers run, deployed on first use --
        or None while that deployment is still building.

        Deployment is not an operator step: a task that has never launched a
        remote worker gets the controller's current tree built for its
        machines' archs and pushed, and every later worker joins that same
        bundle. Pinning is what keeps an experiment homogeneous -- editing
        code mid-run leaves the fleet on the code it started with, and moving
        it is the explicit redeploy action. A later slot whose arch the bundle
        lacks is the one exception: the same tree is built again with that
        arch added, and the task repinned (the containers on the old bundle
        run identical code; the repin lets them be replaced at their next
        restart as any redeploy does).

        The build is minutes, and slot starts run on the blocking thread; built
        there it held up every Pause and Remove clicked meanwhile (redeploy's
        lesson). So it goes to the build thread, this pass leaves the slot
        `starting` with the reason on its row, and the pass that finds the
        build done pins the task and starts the slot. A build that fails is
        the slot's exit reason, and the restart backoff paces the retry.
        """
        arch = self._slot_arch(spec, task, w)
        if task.bundle_id and arch in task.bundle_archs:
            return task.bundle_id
        task_key = f"{spec.name}/{task.tag}"
        future = self._pending_builds.get(task_key)
        if future is None:
            archs = self._needed_archs(spec, task)
            future = self._pending_builds[task_key] = self._builds.submit(self._build_bundle, archs)
        if not future.done():
            self._exits[key] = (
                f"building the worker bundle for {', '.join(self._needed_archs(spec, task))}"
            )
            return None
        del self._pending_builds[task_key]
        try:
            manifest = future.result()
        except Exception as e:
            self._exits[key] = f"bundle build failed: {e}"
            raise
        return self._pin_bundle(spec, task, manifest)

    def bundle_drift(self, task: tasks.TaskRecord) -> bool:
        """Whether the controller's tree has changed since the task pinned its
        bundle -- what the dashboard badges, and the cue to redeploy. False
        while nothing is pinned, and while an arch is unbuilt (there is no
        tree to compare until a build produces one)."""
        if not task.bundle_source_hash or not task.bundle_archs:
            return False
        current = source_hash(task.bundle_archs, self._source_digests)
        return current is not None and current != task.bundle_source_hash

    # ---- cloud plumbing --------------------------------------------------

    def _creds(self) -> CloudCredentials:
        if self._creds_cache is None:
            self._creds_cache = load_credentials()
        return self._creds_cache

    def _provider(self) -> Provider:
        if self._provider_client is None:
            creds = self._creds()
            self._provider_client = AwsProvider(creds.aws, creds.registry)
        return self._provider_client

    def _instance_index(self, observe: bool) -> dict[str, Instance]:
        """The provider's instances by id, listed at most once per
        OBSERVATION_TTL_SECONDS and only by the reconcile pass."""
        instances, at = self._instances
        if observe and time.time() - at >= OBSERVATION_TTL_SECONDS:
            self._instances = instances, _ = (self._provider().describe(), time.time())
        return instances

    def _ensure_sync(self, spec: workloads.WorkloadSpec, task: tasks.TaskRecord):
        """Keep exactly one sync watcher alive per task with bucket-delivering
        slots, pulling what those slots deliver: a watcher whose argv no
        longer matches (a trainer slot appeared) is replaced."""
        key = _key(spec, task.tag)
        has_bucket = _has_bucket_slots(spec, task)
        argv = [
            sys.executable, str(CLOUD_SYNC),
            "--workload", spec.name, "-t", task.tag,
            "--watch", "--interval", str(SYNC_INTERVAL_SECONDS),
            *(["--trainer-outputs"] if _bucket_trainer(spec, task) else []),
        ]  # fmt: skip
        entry = self._sync.get(key)
        if entry is not None and (
            not has_bucket or entry[0].poll() is not None or entry[1] != argv
        ):
            if entry[0].poll() is None:
                entry[0].terminate()
            del self._sync[key]
            entry = None
        if has_bucket and entry is None:
            log = self._log_file(spec, task.tag, "cloud_sync")
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            self._sync[key] = (proc, argv)

    def _push_controls(self, spec: workloads.WorkloadSpec, task: tasks.TaskRecord):
        """Put the operator's controls file in the bucket for a trainer that
        runs through it, whenever the file changed: the trainer reads it
        there (generational/records.py) as a local one reads the file. One
        copy per set, none per quiet pass."""
        if not _bucket_trainer(spec, task):
            return
        path = spec.paths(task.tag).controls_path
        try:
            stamp = path.stat().st_mtime_ns
        except FileNotFoundError:
            return
        key = _key(spec, task.tag)
        if self._controls_pushed.get(key) == stamp:
            return
        creds = self._creds()
        dest = bucket_path(creds.r2, spec.name, task.tag, CONTROLS_REL)
        res = rclone(creds.r2, "copyto", str(path), dest, capture=True)
        assert res.returncode == 0, f"pushing {CONTROLS_REL} failed: {res.stderr}"
        self._controls_pushed[key] = stamp

    def _stage_inputs_in_bucket(self, r2, spec, tag: str, inputs: dict[str, Path]):
        """A bucket-delivering slot's out-of-tag inputs (RoleSpec.inputs),
        put under the tag's prefix before the container exists to look for
        them; a copy already there at the same size is skipped."""
        for rel, src in inputs.items():
            dest = bucket_path(r2, spec.name, tag, *rel.split("/"))
            res = rclone(r2, "copyto", "--size-only", str(src), dest, capture=True)
            if res.returncode != 0:
                raise SshMachineError(f"staging input {rel} in the bucket failed: {res.stderr}")

    # ---- local plumbing --------------------------------------------------

    def _log_file(self, spec: workloads.WorkloadSpec, tag: str, name: str):
        log_dir = spec.paths(tag).logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        return open(log_dir / f"{name}.log", "ab")

    def _spawn_local(self, spec: workloads.WorkloadSpec, task: tasks.TaskRecord, w):
        params = params_mod.validate(spec.params_cls, task.params)
        env = os.environ | spec.worker_env(task.tag, params, w.role) | {
            "SCZ_SINK": "local",
            "SCZ_THREADS": str(w.threads),
            "SCZ_WORKER_ID": w.worker_id,
            "SCZ_WORKER_KIND": "local",
        }  # fmt: skip
        log = self._log_file(spec, task.tag, w.worker_id)
        proc = subprocess.Popen(
            [sys.executable, "-m", "cloud.worker_entrypoint"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: os.nice(LOCAL_WORKER_NICE),  # inherited by its sim threads
        )
        self._local[_key(spec, task.tag, w.worker_id)] = proc
        w.pid = proc.pid  # durable, so any instance can observe and stop this worker
        tasks.save_task(spec, task)

    def _local_alive(self, spec, task: tasks.TaskRecord, w) -> bool:
        """Whether slot `w`'s worker process is really running. Reaps our own
        exited child first (so it does not linger as a zombie), then probes by
        durable pid -- catching workers spawned by a previous or concurrent
        dashboard instance that this process holds no handle to."""
        proc = self._local.get(_key(spec, task.tag, w.worker_id))
        if proc is not None:
            proc.poll()
        return worker_pid_alive(w.pid, w.worker_id, task.tag)

    def _local_exit_code(self, spec, task: tasks.TaskRecord, w) -> int | None:
        """How slot `w`'s worker exited, when it was this process's child and
        has; None otherwise (still running, or spawned by another instance)."""
        proc = self._local.get(_key(spec, task.tag, w.worker_id))
        return None if proc is None else proc.returncode

    def _stop_local(self, spec, task: tasks.TaskRecord, w):
        """SIGTERM slot `w`'s worker by durable pid (workers flush completed
        output and exit cleanly on SIGTERM). No-op if it is not running."""
        if worker_pid_alive(w.pid, w.worker_id, task.tag):
            try:
                os.kill(w.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    # ---- ssh plumbing ----------------------------------------------------

    def _observe_container(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> str:
        """Probe slot `w`'s container over ssh and remember the answer.

        Negative caching stands: after a failed probe the host is assumed
        unreachable for SSH_REPROBE_SECONDS rather than paying a connect
        timeout every pass.

        A not-yet-launched slot is probed like any other -- an in-doubt first
        start (the ssh link dying after `docker run` was dispatched) may have
        left a live container, and a probe that finds one flips the slot to
        launched."""
        tag, host = task.tag, _ssh_host(task, w)
        down_since = self._ssh_down.get(host)
        if down_since is not None and time.time() - down_since < SSH_REPROBE_SECONDS:
            probe = "unreachable"
        else:
            probe = _ssh_machine(task, w).container_state(_container_name(spec, tag, w.worker_id))
            if probe == "unreachable":
                self._ssh_down[host] = time.time()
            else:
                self._ssh_down.pop(host, None)
        if not w.launched and probe not in ("unreachable", "missing"):
            w.launched = True  # the in-doubt start did create the container
        key = _key(spec, tag, w.worker_id)
        self._probes[key] = (probe, time.time())
        if probe == "stopped":
            self._exits[key] = _ssh_machine(task, w).container_exit(
                _container_name(spec, tag, w.worker_id)
            )
            if self._exits[key].startswith("exit 0:"):
                _note_finished(w)
        elif probe in ("running", "paused"):
            self._exits.pop(key, None)
        # A probe that finds no container clears nothing: what it is likely to
        # find is a slot whose creation failed, and that failure is the only
        # account of why it is not running.
        if probe == "running":
            self._restarts.pop(key, None)  # it came up; it is not looping
        if probe == "unreachable":
            _forget_empty(w)
        return probe

    def _probe_container(
        self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord, *, observe: bool
    ) -> str:
        """Slot `w`'s container probe state, freshly observed or remembered.

        Only the reconcile pass observes (`observe=True`); a status request
        reads what that pass left, so a browser polling every few seconds costs
        no ssh at all -- and the dashboard cannot be stalled by a machine that
        is slow to answer. A slot no pass has reached yet reads "unknown"
        rather than a guess.

        An unlaunched slot's `unreachable` maps to `missing`: with no container
        confirmed to exist, the slot must stay manageable (in particular,
        removable) even when the host is bogus or offline."""
        key = _key(spec, task.tag, w.worker_id)
        remembered, at = self._probes.get(key, ("unknown", 0.0))
        if observe and time.time() - at >= OBSERVATION_TTL_SECONDS:
            probe = self._observe_container(spec, task, w)
        else:
            probe = remembered
        if not w.launched and probe == "unreachable":
            return "missing"
        return probe

    def _restart_allowed(self, key: str) -> bool:
        """Whether slot `key` may be (re)started now. A container that keeps
        dying gets progressively longer between attempts, so a broken worker
        costs one ssh round trip every few minutes rather than one per pass --
        and its failure stays on screen instead of scrolling past."""
        _, next_at = self._restarts.get(key, (0, 0.0))
        return time.time() >= next_at

    def _note_restart(self, key: str):
        attempts = self._restarts.get(key, (0, 0.0))[0] + 1
        backoff = min(MAX_RESTART_BACKOFF_SECONDS, OBSERVATION_TTL_SECONDS * 2 ** (attempts - 1))
        self._restarts[key] = (attempts, time.time() + backoff)

    def _run_ssh_container(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord):
        """Create + start slot `w`'s container on its machine, on the task's
        bundle."""
        key = _key(spec, task.tag, w.worker_id)
        bundle_id = self._bundle_for_start(spec, task, w, key)
        if bundle_id is None:
            # Not an attempt: the slot is waiting on the build, not failing to
            # come up, so the restart backoff must not grow across the wait.
            self._restarts.pop(key, None)
            return
        w.bundle_id = bundle_id
        creds = self._creds()
        params = params_mod.validate(spec.params_cls, task.params)
        env = bundle_worker_env(
            creds, spec, task.tag, params,
            role=w.role, bundle_id=w.bundle_id, worker_id=w.worker_id,
        )  # fmt: skip
        env["SCZ_SINK"] = _slot_sink(spec, task, w)
        if w.threads:
            env["SCZ_THREADS"] = str(w.threads)
        machine = _ssh_machine(task, w)
        role = spec.role(w.role)
        inputs = _role_inputs(spec, role, params)
        try:
            # An input that is not there to stage is the slot's reason, on
            # its row and paced by the restart backoff like a machine that
            # cannot serve the role -- not an exception in the reconcile log.
            _require_inputs(inputs)
            if env["SCZ_SINK"] == "r2":
                self._stage_inputs_in_bucket(creds.r2, spec, task.tag, inputs)
            # Creating a container is the moment to take a rebuilt worker
            # image; `docker run --pull=never` below then fails fast rather
            # than pulling under the dashboard.
            image = creds.registry.image_for(role.runtime)
            machine.pull_image(image)
            name = _container_name(spec, task.tag, w.worker_id)
            machine.run_container(name, image, env, gpus=role.gpu)
            if env["SCZ_SINK"] != "r2":
                _stage_inputs_in_container(machine, name, spec, task.tag, inputs)
        except SshMachineError as e:
            # The slot will read `starting` until this succeeds, since nothing
            # of it exists to have exited. Recording why keeps that from being
            # the whole story a machine that cannot serve the role ever tells:
            # a missing NVIDIA toolkit, an image nobody logged in to pull.
            self._exits[key] = str(e)
            raise
        self._exits.pop(key, None)
        w.launched = True
        w.undelivered = 0  # a container just created is holding nothing
        tasks.save_task(spec, task)

    # ---- slot operations -------------------------------------------------

    def _check_role(
        self,
        spec,
        task: tasks.TaskRecord,
        role: str,
        kind: str,
        machine: tasks.MachineRecord | None = None,
    ):
        role_spec = spec.role(role)
        assert kind in role_spec.kinds, f"role '{role}' does not support {kind} workers"
        if machine is not None and role_spec.gpu and machine.gpu_count is not None:
            # Refused here rather than by `docker run --gpus all` on the
            # remote, after a bundle deploy: a machine of known shape says
            # what it can host. An unknown count (a manual machine) is not
            # checked, as a bare host never was.
            taking = [w.worker_id for w in task.slots_on(machine.name) if spec.role(w.role).gpu]
            assert len(taking) < machine.gpu_count, (
                f"machine '{machine.name}' has {machine.gpu_count} GPU(s), "
                f"{'already taken by ' + ', '.join(taking) if taking else 'none for'} role '{role}'"
            )
        if role_spec.singleton:
            taken = [w.worker_id for w in task.workers if w.role == role]
            assert not taken, f"role '{role}' already has a worker ({taken[0]})"
        return role_spec

    def add_local(
        self, spec, task: tasks.TaskRecord, role: str, threads: int | None
    ) -> tasks.WorkerRecord:
        self._check_role(spec, task, role, "local")
        w = tasks.WorkerRecord(
            worker_id=_next_worker_id(task, "local"),
            role=role,
            kind="local",
            desired_state="paused",
            threads=threads or default_thread_count(),
        )
        task.workers.append(w)
        tasks.save_task(spec, task)
        return w

    def add_ssh(
        self,
        spec,
        task: tasks.TaskRecord,
        role: str,
        *,
        host: str | None = None,
        machine: str | None = None,
        threads: int | None,
    ) -> tasks.WorkerRecord:
        """An ssh slot on a bare host string, or on one of the task's
        machines by name (exactly one of the two)."""
        assert (host is None) != (machine is None), "an ssh slot names a host or a machine"
        record = task.machine(machine) if machine is not None else None
        self._check_role(spec, task, role, "ssh", machine=record)
        w = tasks.WorkerRecord(
            worker_id=_next_worker_id(task, "ssh"),
            role=role,
            kind="ssh",
            desired_state="paused",
            host=host,
            machine=machine,
            launched=False,
            threads=threads,
        )
        task.workers.append(w)
        tasks.save_task(spec, task)
        self._ensure_sync(spec, task)
        return w

    # ---- machines ----------------------------------------------------------

    def add_machine(
        self,
        spec,
        task: tasks.TaskRecord,
        name: str,
        host: str,
        identity_file: str | None = None,
        gpu_count: int | None = None,
    ) -> tasks.MachineRecord:
        """Register a machine the operator owns or launched themself, for the
        task's ssh slots to run on. Prepared by hand as docs/master_dashboard.md
        says; nothing here touches it."""
        assert name and host, "a machine needs a name and a host"
        assert all(m.name != name for m in task.machines), f"machine '{name}' exists"
        m = tasks.MachineRecord(
            name=name,
            provider="manual",
            host=host,
            identity_file=identity_file or None,
            gpu_count=gpu_count,
        )
        task.machines.append(m)
        tasks.save_task(spec, task)
        return m

    def rental_offer(self) -> dict:
        """What the rent form shows: the provider, the account it rents as
        (asked once per process), and the catalog."""
        provider = self._provider()
        if self._account is None:
            self._account = provider.account()
        if time.time() - self._spot_prices[1] >= SPOT_PRICES_TTL_SECONDS:
            self._spot_prices = (provider.spot_prices(), time.time())
        return {
            "provider": provider.name,
            "account": self._account,
            "types": [asdict(t) for t in provider.catalog()],
            "spot_prices": self._spot_prices[0],
        }

    def rent_machine(
        self, spec, task: tasks.TaskRecord, name: str, type_id: str, *, spot: bool = False
    ):
        """Launch an instance of `type_id` for the task and record it as one
        of its machines, under `name` or a generated one. A refusal (a quota
        of 0, no capacity) reaches the form as the provider's sentence;
        nothing is recorded for it."""
        provider = self._provider()
        name = name or _next_machine_name(task, provider.name)
        assert all(m.name != name for m in task.machines), f"machine '{name}' exists"
        mtype = next((t for t in provider.catalog() if t.id == type_id), None)
        assert mtype is not None, f"no machine type '{type_id}'"
        try:
            inst = provider.launch(LaunchRequest(type_id, _owner(spec, task.tag, name), spot=spot))
        except ProviderError as e:
            raise AssertionError(provider.refusal(e, type_id)) from e
        # Into the listing now: the record names an instance the last listing
        # predates, which would read `gone` until the next pass relists.
        self._instances[0][inst.id] = inst
        known_hosts = MACHINES_DIR / name / "known_hosts"
        known_hosts.parent.mkdir(parents=True, exist_ok=True)
        known_hosts.write_text("")  # a relaunch is a new name, so never a stale key
        m = tasks.MachineRecord(
            name=name,
            provider=provider.name,
            host=f"{provider.ssh_user}@{inst.address or 'pending'}",
            identity_file=provider.identity_file,
            known_hosts_file=str(known_hosts),
            arch=mtype.arch,
            gpu_count=mtype.gpu_count,
            instance_id=inst.id,
            instance_type=mtype.id,
            spot=spot,
            region=getattr(provider, "region", None),
            cost_per_hr=inst.cost_per_hr if inst.cost_per_hr is not None else mtype.cost_per_hr,
            launched_at=inst.launched_at or time.time(),
        )
        _accrue_machine(m, True)
        task.machines.append(m)
        tasks.save_task(spec, task)
        return m

    def remove_machine(self, spec, task: tasks.TaskRecord, name: str):
        """Remove a machine and the slots on it -- each under the slot rule
        (not running, reachable, and the operator warned of what it holds),
        so a machine is never dropped out from under a working container --
        and terminate it if it was rented. A gone instance's slots are
        removable outright: their containers went with its disk."""
        m = task.machine(name)
        key = _machine_key(spec, task.tag, name)
        for w in task.slots_on(name):
            self.remove_worker(spec, task, w.worker_id)
        if m.instance_id is not None and self._machine_states.get(key) != "gone":
            self._terminate(m.instance_id, m.instance_type)
        for cache in (self._machine_probes, self._machine_states, self._idle_since, self._exits):
            cache.pop(key, None)
        self._restarts.pop(key, None)
        _accrue_machine(m, False)
        task.retired_spend += m.spend
        task.machines.remove(m)
        tasks.save_task(spec, task)

    def _machine_gone(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> bool:
        return (
            w.machine is not None
            and self._machine_states.get(_machine_key(spec, task.tag, w.machine)) == "gone"
        )

    def _list_fleet(self):
        """The pass's fleet step: list every instance the provider tagged
        ours, whether or not any task names one. The per-task machine step
        lists only for tasks with rented machines, so without this a
        task.json that lost its machines would leave their instances
        unlisted -- billing, and invisible. A failure (no credentials, the
        provider unreachable) is kept for the burn strip to show, and
        printed once per change rather than every pass."""
        try:
            self._instance_index(observe=True)
            error = None
        except Exception as e:  # noqa: BLE001 -- the fleet step must not stop the pass
            error = str(e)
        if error != self._fleet_error and error is not None:
            print(f"fleet listing: {error}")
        self._fleet_error = error

    def fleet(self) -> dict:
        """What the burn strip shows: every non-terminated instance tagged
        ours with its hourly rate, and the rate they add up to right now
        (those pending or running -- what _accrue_machine charges for).
        Read from the last listing, like orphans and machine_status;
        `observed_at` lets the strip flag a listing that has stopped
        refreshing."""
        instances, at = self._instances
        owned = self._owned()
        rows = [
            {
                "instance_id": inst.id,
                "type_id": inst.type_id,
                "state": inst.state,
                "owner": inst.owner,
                "tracked": inst.owner in owned,
                "spot": inst.spot,
                "cost_per_hr": self._rate(inst, owned.get(inst.owner)),
                "uptime_s": int(time.time() - inst.launched_at) if inst.launched_at else None,
            }
            for inst in instances.values()
            if inst.state != "terminated"
        ]
        return {
            "observed_at": at or None,
            "error": self._fleet_error,
            "instances": rows,
            "burn_per_hr": sum(
                r["cost_per_hr"] or 0.0 for r in rows if r["state"] in ("pending", "running")
            ),
        }

    def _rate(self, inst: Instance, record: tasks.MachineRecord | None) -> float | None:
        """An instance's hourly rate: the task's record of it (a spot
        instance's rate is known only at launch, and lives there), else the
        listing's own, else its type's catalog rate (None for a type the
        catalog no longer lists)."""
        if record is not None and record.cost_per_hr is not None:
            return record.cost_per_hr
        if inst.cost_per_hr is not None:
            return inst.cost_per_hr
        mtype = next((t for t in self._provider().catalog() if t.id == inst.type_id), None)
        return mtype.cost_per_hr if mtype is not None else None

    def _owned(self) -> dict[str, tasks.MachineRecord]:
        """Every task's machines by the ownership tag each carries."""
        return {
            _owner(spec, task.tag, m.name): m
            for spec, task in self._all_tasks()
            for m in task.machines
        }

    def orphans(self, observe: bool = False) -> list[dict]:
        """Instances the provider tagged ours that no task's machines name:
        shown with a Terminate button, never terminated on their own (a
        task.json restored from an older copy must not kill a running
        experiment)."""
        owned = self._owned()
        return [
            {
                "instance_id": inst.id,
                "type_id": inst.type_id,
                "state": inst.state,
                "owner": inst.owner,
                "uptime_s": int(time.time() - inst.launched_at) if inst.launched_at else None,
            }
            for inst in self._instance_index(observe).values()
            if inst.state != "terminated" and inst.owner not in owned
        ]

    def terminate_orphan(self, instance_id: str):
        inst = self._instance_index(False).get(instance_id)
        assert inst is not None, f"no instance {instance_id} in the last listing"
        self._terminate(instance_id, inst.type_id)
        self._instances = ({}, 0.0)  # relisted next pass

    def _terminate(self, instance_id: str, type_id: str | None):
        """Terminate through the provider, a refusal reaching the operator as
        its sentence (what happened, what to do), as a launch's does."""
        provider = self._provider()
        try:
            provider.terminate(instance_id)
        except ProviderError as e:
            raise AssertionError(provider.refusal(e, type_id or "instance")) from e
        # Out of the listing now, as the next pass will find it: until then
        # a terminated machine's instance would read as a running orphan.
        cached = self._instances[0].get(instance_id)
        if cached is not None:
            cached.state = "terminated"

    def machine_status(self, spec, task: tasks.TaskRecord, *, observe: bool = False) -> list[dict]:
        """One dict per machine: the record plus its probe state (`up`,
        `no docker`, `unreachable`; `checking` before the first pass). Like
        the slot probes, only the reconcile pass observes; a status request
        reads what it left."""
        out = []
        rented = any(m.instance_id is not None for m in task.machines)
        index = self._instance_index(observe) if rented else {}
        for m in task.machines:
            key = _machine_key(spec, task.tag, m.name)
            inst = index.get(m.instance_id) if m.instance_id is not None else None
            if (
                inst is not None
                and inst.address
                and m.host != f"{m.host.split('@')[0]}@{inst.address}"
            ):
                m.host = f"{m.host.split('@')[0]}@{inst.address}"  # it moved on a stop/start
            probe, at = self._machine_probes.get(key, (None, 0.0))
            if observe and time.time() - at >= OBSERVATION_TTL_SECONDS:
                # A probe is worth making only on a machine that can answer.
                probe = (
                    self._observe_machine(m) if inst is None or inst.state == "running" else None
                )
                self._machine_probes[key] = (probe, time.time())
            if m.instance_id is None:
                state = probe or "checking"
            else:
                state = _rented_state(m, inst, probe)
                _accrue_machine(m, inst is not None and inst.state in ("pending", "running"))
            self._machine_states[key] = state
            info = {
                "name": m.name,
                "provider": m.provider,
                "host": m.host,
                "gpu_count": m.gpu_count,
                "instance_type": m.instance_type,
                "instance_id": m.instance_id,
                "spot": m.spot,
                "cost_per_hr": m.cost_per_hr,
                "spend": m.spend,
                "state": state,
                "slots": [w.worker_id for w in task.slots_on(m.name)],
            }
            reason = self._exits.get(key)
            if reason:
                info["exit_reason"] = reason  # why the last start was refused
                _, next_at = self._restarts.get(key, (0, 0.0))
                info["retry_in_s"] = max(0, int(next_at - time.time()))
            out.append(info)
        if observe and rented:
            tasks.save_task(spec, task)
        return out

    def _reconcile_machines(self, spec, task: tasks.TaskRecord, status: list[dict]):
        """Drive each rented machine toward what its slots want: start a
        stopped instance that a slot wants running (with the growing backoff
        a refused start gets, and its reason on the machine's row), and stop
        one on which nothing has run for IDLE_STOP_SECONDS. Idle is read
        from the slots' remembered probes: a gated generator is a paused
        container and a finished trainer an exited one, so a run that ends
        stops its machine; a slot that wants running and has no container
        yet is a pending start, not idleness."""
        provider = None
        for info in status:
            m = next((x for x in task.machines if x.name == info["name"]), None)
            if m is None or m.instance_id is None:
                continue
            key = _machine_key(spec, task.tag, m.name)
            slots = task.slots_on(m.name)
            wanted = [w for w in slots if w.desired_state == "running"]
            if info["state"] == "stopped":
                self._idle_since.pop(key, None)
                if wanted and self._restart_allowed(key):
                    provider = provider or self._provider()
                    try:
                        provider.start(m.instance_id)
                    except ProviderError as e:
                        self._exits[key] = provider.refusal(e, m.instance_type or "")
                        self._note_restart(key)
                        raise
                    self._exits.pop(key, None)
                    self._restarts.pop(key, None)
                    m.launched_at = time.time()
                    self._instances = ({}, 0.0)  # relisted next pass
                    tasks.save_task(spec, task)
                continue
            if info["state"] != "up":
                self._idle_since.pop(key, None)
                continue
            busy = any(
                self._probes.get(_key(spec, task.tag, w.worker_id), ("unknown", 0.0))[0]
                in ("running", "unknown")
                for w in slots
            ) or any(w in wanted for w in slots)
            if busy:
                self._idle_since.pop(key, None)
                continue
            since = self._idle_since.setdefault(key, time.time())
            if time.time() - since >= IDLE_STOP_SECONDS:
                provider = provider or self._provider()
                provider.stop(m.instance_id)
                self._idle_since.pop(key, None)
                self._instances = ({}, 0.0)

    def _observe_machine(self, m: tasks.MachineRecord) -> str:
        """Probe a machine, under the same negative cache as its slots: a
        host that just failed is not dialed again for SSH_REPROBE_SECONDS."""
        down_since = self._ssh_down.get(m.host)
        if down_since is not None and time.time() - down_since < SSH_REPROBE_SECONDS:
            return "unreachable"
        ready = self._provider().ready_file if m.instance_id is not None else None
        state = _machine_link(m).probe(ready)
        if state == "unreachable":
            self._ssh_down[m.host] = time.time()
        else:
            self._ssh_down.pop(m.host, None)
        return state

    def set_worker_state(self, spec, task: tasks.TaskRecord, worker_id: str, run: bool):
        w = task.worker(worker_id)
        w.desired_state = "running" if run else "paused"
        if run:
            w.finished = False
        tasks.save_task(spec, task)
        start = run and w.role not in task.gates  # a gated slot starts when released
        if w.kind == "local":
            if start and not self._local_alive(spec, task, w):
                self._spawn_local(spec, task, w)
            elif not run:
                self._stop_local(spec, task, w)
        else:
            # Freshly observed: dispatching a start or stop off a remembered
            # state would send the wrong command (or, for a slot no pass has
            # reached yet, none at all). An unreachable machine gets no action
            # either way: the desired state is saved, and reconcile enforces it
            # once probes succeed.
            probe = self._probe_container(spec, task, w, observe=True)
            name = _container_name(spec, task.tag, w.worker_id)
            if start and probe == "stopped":
                _ssh_machine(task, w).start_container(name)
            elif start and probe == "missing":
                self._run_ssh_container(spec, task, w)
            elif not run and probe == "running":
                _ssh_machine(task, w).stop_container(name)

    def remove_worker(self, spec, task: tasks.TaskRecord, worker_id: str):
        """Remove a slot. Only non-running workers may be removed (pause
        first), so a removal never silently discards an in-flight cycle."""
        w = task.worker(worker_id)
        if w.kind == "local":
            assert not self._local_alive(spec, task, w), f"{worker_id} is running; pause it first"
            self._local.pop(_key(spec, task.tag, worker_id), None)
        elif w.kind == "ssh" and self._machine_gone(spec, task, w):
            pass  # its container went with the instance's disk; nothing to check or clean
        elif w.kind == "ssh":
            # Freshly observed: a removal must not act on a remembered state.
            probe = self._probe_container(spec, task, w, observe=True)
            assert probe not in ("running", "paused"), f"{worker_id} is running; pause it first"
            # Removing while unreachable would orphan a possibly-live container
            # that keeps generating into the tag with nothing tracking it.
            assert probe != "unreachable", (
                f"{_ssh_host(task, w)} is unreachable; bring it online (or clean up its "
                f"container by hand) before removing {worker_id}"
            )
            if probe == "stopped":
                _ssh_machine(task, w).remove_container(_container_name(spec, task.tag, w.worker_id))
            # A future slot may be assigned this same worker_id (a freed id is
            # the first one _next_worker_id hands out again), and a deleted
            # tag can be recreated under the same name -- both reproduce this
            # key. Without this, the new slot would start out narrating the
            # old container's exit and backoff.
            key = _key(spec, task.tag, worker_id)
            self._exits.pop(key, None)
            self._restarts.pop(key, None)
        task.workers.remove(w)
        tasks.save_task(spec, task)
        self._ensure_sync(spec, task)

    def delete_task(self, spec, tag: str):
        """Delete a tag: tear its worker slots down, then delete its local dir.

        Idle slots are removed on the operator's behalf rather than refused --
        they are how a container gets released, and the task record
        about to be deleted is the only thing tracking it. A slot that is
        still running refuses, so a fleet at work is never deleted out from
        under itself.

        Slots go one at a time, so a refusal partway through leaves the
        earlier ones removed. Hence the intent check up front: the ordinary
        refusal -- an operator who has not paused the fleet -- costs nothing,
        and what is left to discover slot by slot is the paused one whose
        backing process turns out to be alive after all.
        """
        task = tasks.load_task(spec, tag)
        if task is not None:
            running = [w.worker_id for w in task.workers if w.desired_state == "running"]
            assert not running, f"pause {', '.join(running)} first"
            for w in list(task.workers):
                self.remove_worker(spec, task, w.worker_id)
        tasks.delete_tag(spec, tag)

    # ---- observation -----------------------------------------------------

    def worker_status(self, spec, task: tasks.TaskRecord, *, observe: bool = False) -> list[dict]:
        """One dict per slot: the durable record plus observed live state.
        Every call is also a spend-accrual observation point.

        `observe` is the reconcile pass's privilege: it goes to the machines
        and the cloud API, and leaves what it learns behind -- persisted, the
        only save here. Every other caller -- every status request a browser
        makes -- reads those observations, so serving the dashboard never
        waits on ssh or the provider, and accrues in memory only: the record is the
        pass's own object, so the next pass saves what the polls accrued.
        """
        out = []
        for w in task.workers:
            gated = w.role in task.gates
            info = {
                "worker_id": w.worker_id,
                "role": w.role,
                "kind": w.kind,
                "desired_state": w.desired_state,
                "threads": w.threads,
                "host": _ssh_host(task, w) if w.kind == "ssh" else None,
                "machine": w.machine,
                "bundle_id": w.bundle_id,
                "launched": w.launched,
                # Zero and None differ to anyone about to remove the slot:
                # drained, versus nothing known about what it holds.
                "undelivered": w.undelivered,
            }
            if gated:
                info["gate_reason"] = task.gates[w.role]
            if w.kind == "local":
                alive = self._local_alive(spec, task, w)
                if not alive and self._local_exit_code(spec, task, w) == 0:
                    _note_finished(w)
                info["state"] = _local_state(w.desired_state, alive, gated, w.finished)
            else:
                _holds_nothing(spec, task, w)
                probe = self._probe_container(spec, task, w, observe=observe)
                alive = probe == "running"
                info["state"] = _ssh_state(w.desired_state, probe, gated, w.finished)
                info["ssh_probe"] = probe  # reconcile keys its enforcement off this
                info["ssh"] = f"ssh {_ssh_host(task, w)}"
                reason = self._exits.get(_key(spec, task.tag, w.worker_id))
                if reason and not alive:
                    info["exit_reason"] = reason
            # Real liveness, for reconcile's desired-vs-observed enforcement.
            info["observed_running"] = alive
            out.append(info)
        if observe and task.workers:
            tasks.save_task(spec, task)
        return out

    # ---- the scheduler's hook surface --------------------------------------

    def _scheduler_hooks(self, spec, task: tasks.TaskRecord) -> SchedulerHooks:
        def gate(role: str, reason: str | None):
            changed = (
                task.gates.pop(role, None) is not None
                if reason is None
                else task.gates.get(role) != reason
            )
            if reason is not None:
                task.gates[role] = reason
            if changed:
                tasks.save_task(spec, task)

        return SchedulerHooks(
            gate=gate,
            mirror=self._make_mirror(spec, task),
            publish=self._make_publish(spec, task),
        )

    def _make_publish(self, spec, task: tasks.TaskRecord):
        """Bucket-side copy of a completed generation (the scheduler's publish
        hook): its chunks -- the cloud-origin ones are there already after the
        mirror move and are skipped by size, the local- and ssh-origin ones
        upload -- and then its manifest, last, so a manifest in the bucket
        means the whole generation is. What a trainer running elsewhere reads
        (docs/plans/cloud_training.md); and with it the bucket holds every
        generation of a cloud-fed tag complete, not just its cloud chunks.
        None for tasks without bucket-delivering slots, as for the mirror."""
        if not _has_bucket_slots(spec, task):
            return None
        try:
            creds = self._creds()
        except (CredentialsError, FileNotFoundError):
            return None
        r2 = creds.r2
        paths = spec.paths(task.tag)

        def publish(dest_rel: str):
            gen_dir = paths.data_dir / dest_rel
            dest = bucket_path(r2, spec.name, task.tag, *dest_rel.split("/"))
            res = rclone(
                r2, "copy", "--size-only", "--exclude", MANIFEST_NAME, str(gen_dir), dest,
                capture=True,
            )  # fmt: skip
            assert res.returncode == 0, f"publishing {dest_rel} failed: {res.stderr}"
            res = rclone(
                r2, "copyto", str(gen_dir / MANIFEST_NAME), f"{dest}/{MANIFEST_NAME}", capture=True
            )
            assert res.returncode == 0, f"publishing {dest_rel}'s manifest failed: {res.stderr}"

        return publish

    def _make_mirror(self, spec, task: tasks.TaskRecord):
        """Bucket-side replay of staging ingests: when the scheduler assigns a
        chunk locally, its bucket object (if any -- the chunk may be
        local-origin) moves to the matching generation prefix, so the bucket
        keeps mirroring the local corpus and the sync watcher never
        re-downloads an ingested chunk. None for tasks without
        bucket-delivering slots."""
        if not _has_bucket_slots(spec, task):
            return None
        try:
            creds = self._creds()
        except (CredentialsError, FileNotFoundError):
            return None
        r2 = creds.r2
        staging = bucket_path(r2, spec.name, task.tag, "staging")
        staged_names: set[str] | None = None  # bucket listing, fetched on first use

        def mirror(chunk_name: str, dest_rel: str):
            nonlocal staged_names
            if staged_names is None:
                res = rclone(r2, "lsf", staging, capture=True)
                staged_names = set(res.stdout.split()) if res.returncode == 0 else set()
            if chunk_name not in staged_names:
                return  # local-origin chunk; nothing to mirror
            rclone(
                r2,
                "moveto",
                f"{staging}/{chunk_name}",
                bucket_path(r2, spec.name, task.tag, *dest_rel.split("/"), chunk_name),
                capture=True,
            )

        return mirror

    # ---- reconciliation ----------------------------------------------------

    def _all_tasks(self):
        for spec in workloads.WORKLOADS.values():
            for row in tasks.list_tags(spec):
                if not row["has_task"]:
                    continue
                task = tasks.load_task(spec, row["tag"])
                self._forget_stale_counts(spec, task)
                yield spec, task

    def _forget_stale_counts(self, spec, task: tasks.TaskRecord):
        """The first time this process sees a task, drop any recorded zero.

        The count is durable so that a restart cannot forget a container is
        holding hours of work -- but it must not let one inherit the opposite
        claim either, since whatever the workers did while nothing was
        watching is exactly what a zero from before the restart does not
        cover.
        """
        key = _key(spec, task.tag)
        if key in self._counted_from:
            return
        self._counted_from.add(key)
        for w in task.workers:
            if w.kind == "ssh":
                _forget_empty(w)
        tasks.save_task(spec, task)

    async def offload(self, fn, *args, **kwargs):
        """Run one blocking step off the event loop, one at a time.

        Everything reconcile does is blocking IO measured in seconds: ssh to a
        laptop, rclone to R2, the provider's API, a build. Run inline it froze the
        whole dashboard -- an 8-second stall at every scheduler gate flip, with
        the UI hanging on requests it could otherwise have served. The executor
        holds a single thread, so these steps stay serialized with each other
        (they mutate the same task records) while the loop stays free.
        """
        return await IOLoop.current().run_in_executor(self._blocking, partial(fn, *args, **kwargs))

    async def reconcile(self):
        """One pass over every task: run the workload's scheduler tick, then
        drive each worker's observed state toward its desired state -- respawn
        local workers that should be running but are not (dashboard restart,
        crashed process), start and stop rented machines as their slots want,
        and park or stop workers that are running but should not be (a
        scheduler gate or an operator pause). Enforcement keys off real
        liveness by durable pid or container probe, so it holds a paused
        worker down even across a dashboard restart or a second dashboard
        instance.

        It is also where the controller's half of a dispatch-driven role runs
        (RoleSpec.dispatch): assigning those slots their next piece of work and
        ingesting what they have delivered, once the pass knows which of them
        are really running -- and where a role's ingest tick (RoleSpec.ingest)
        takes a trainer's delivered records into dashboard.db.

        This pass is the only observer: it refreshes the container probes and
        the provider's instance listing everything else reads, which also
        makes it the spend-accrual heartbeat when no browser is polling.
        """
        await self.offload(self._list_fleet)
        for spec, task in self._all_tasks():
            if spec.scheduler:
                try:
                    await self.offload(self._tick_scheduler, spec, task)
                except Exception as e:  # noqa: BLE001 -- scheduling must keep ticking
                    print(f"scheduler {spec.name}/{task.tag}: {e}")
            machines = await self.offload(self.machine_status, spec, task, observe=True)
            try:
                await self.offload(self._reconcile_machines, spec, task, machines)
            except Exception as e:  # noqa: BLE001 -- one task's machines must not stop the pass
                print(f"machines {spec.name}/{task.tag}: {e}")
            down = {m["name"] for m in machines if m["state"] != "up"}
            status = await self.offload(self.worker_status, spec, task, observe=True)
            for info in status:
                # A handler runs between this pass's steps; the slot it removed
                # is not enforced.
                w = task.find(info["worker_id"])
                if w is None:
                    continue
                if w.machine is not None and w.machine in down:
                    continue  # nothing on a machine that is not up can be acted on
                # Collect before enforcing: this slot may be about to be
                # parked, and a pull needs its container running.
                if (
                    w.kind == "ssh"
                    and info["ssh_probe"] == "running"
                    and _slot_sink(spec, task, w) == "local"
                ):
                    try:
                        await self.offload(self._collect_ssh, spec, task, w)
                    except Exception as e:  # noqa: BLE001 -- one slot must not stop the pass
                        print(f"collect {spec.name}/{task.tag}/{w.worker_id}: {e}")
                # Contained per slot: one slot's failing enforcement (an ssh
                # machine vanishing mid-action, an instance launch that keeps
                # failing on an out-of-stock type) must not starve the rest
                # of the pass -- reconcile is the only enforcement some slots
                # get (e.g. stopping gated machines that are still billing).
                try:
                    await self.offload(
                        self._reconcile_worker, spec, task, w, _intent(w, task), info
                    )
                except Exception as e:  # noqa: BLE001 -- enforcement must keep ticking
                    print(f"reconcile {spec.name}/{task.tag}/{w.worker_id}: {e}")
            for role in spec.roles:
                if role.dispatch:
                    try:
                        await self.offload(self._dispatch_role, spec, task, role, status)
                    except Exception as e:  # noqa: BLE001 -- one role must not stop the pass
                        print(f"dispatch {spec.name}/{task.tag}/{role.name}: {e}")
                if role.ingest:
                    try:
                        await self.offload(workloads.resolve(role.ingest), spec, task.tag)
                    except Exception as e:  # noqa: BLE001 -- one role must not stop the pass
                        print(f"ingest {spec.name}/{task.tag}/{role.name}: {e}")
            self._ensure_sync(spec, task)
            try:
                await self.offload(self._push_controls, spec, task)
            except Exception as e:  # noqa: BLE001 -- one task must not stop the pass
                print(f"controls push {spec.name}/{task.tag}: {e}")

    def _collect_ssh(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord):
        """Read a batch of slot `w`'s finished outputs out of its container
        into the tag, and remember how much it still holds.

        The container can stop between the pass's probe and this pull: an
        operator pausing the slot stops it synchronously (set_worker_state),
        so a pull racing a "Pause all" reaches a container `docker exec` can no
        longer read. That is a benign race, not a collection failure -- a
        re-probe that finds it stopped (or gone) means the output it flushed on
        the way down waits in place for the next start (or the sweep that
        precedes a replacement) to take it. Any other failure -- a slow link
        timing out while the container is still up, an unreachable host -- is
        real, and propagates."""
        machine = _ssh_machine(task, w)
        # Unknown until this pull says otherwise. A collection that fails --
        # a link slow enough to keep hitting the transfer timeout, say, while
        # the far cheaper probe still reports the container running -- must not
        # leave the last count standing in for knowledge: it would go on
        # claiming "drained" while the container fills up.
        w.undelivered = None
        try:
            result = pull_results(machine, **_transfer_target(spec, task, w))
        except SshMachineError:
            name = _container_name(spec, task.tag, w.worker_id)
            if machine.container_state(name) not in ("stopped", "missing"):
                raise
            return
        w.undelivered = result.remaining
        tasks.save_task(spec, task)

    def _dispatch_role(self, spec, task: tasks.TaskRecord, role, status: list[dict]):
        """Run one role's controller-side tick: hand its running slots their
        next piece of work, and take in what they have delivered.

        Only running slots are offered: a paused container cannot be written
        to, and an assignment is worth making when there is something to act on
        it. Ingest is not conditional on any of that -- results already
        collected must reach the database even when every slot of the role is
        gone -- so the tick runs whether or not the list is empty.
        """
        slots = [
            self._slot_files(spec, task, w)
            for info in status
            if (w := task.find(info["worker_id"])) is not None
            and w.role == role.name
            and info["observed_running"]
        ]
        params = params_mod.validate(spec.params_cls, task.params)
        workloads.resolve(role.dispatch)(spec, task.tag, params, slots)

    def _slot_files(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord):
        """The way into slot `w`'s filesystem (dashboard/slot_files.py)."""
        paths = spec.paths(task.tag)
        assert w.kind in ("local", "ssh"), (
            f"a {w.kind} slot has no reachable filesystem; a dispatch-driven role's "
            "kinds are local and ssh"
        )
        if w.kind == "ssh":
            return SshSlotFiles(
                w.worker_id,
                _ssh_machine(task, w),
                _container_name(spec, task.tag, w.worker_id),
                str(paths.root),
            )
        return LocalSlotFiles(w.worker_id, paths.root)

    def _tick_scheduler(self, spec, task: tasks.TaskRecord):
        workloads.resolve(spec.scheduler)(spec, task, self._scheduler_hooks(spec, task))

    def _reconcile_worker(self, spec, task: tasks.TaskRecord, w, intent: str, info: dict):
        """Close one slot's desired-vs-observed gap. A rented machine that is booting
        (`starting`) is left alone -- it is already on its way up. An
        unreachable or not-yet-observed ssh machine is left alone too. A failure
        here only skips this slot's tick (the caller contains it): enforcement
        resumes on the next pass."""
        alive = info["observed_running"]
        if w.kind == "local":
            # A local worker restarts in about a second, so parking it and
            # stopping it are the same thing.
            if intent == RUN and not alive:
                self._spawn_local(spec, task, w)
            elif intent != RUN and alive:
                self._stop_local(spec, task, w)
        else:
            self._reconcile_ssh(spec, task, w, intent, info["ssh_probe"])

    def _reconcile_ssh(self, spec, task: tasks.TaskRecord, w, intent: str, probe: str):
        """Enforce one ssh slot's intent.

        A gate parks the container by pausing it rather than stopping it. A
        stop costs the worker its in-flight chunk, and the next start re-runs
        the image's bootstrap -- refetching and unpacking the bundle before the
        first game -- which on a gate that flips every minute was eating a
        third of the machine's duty cycle. A pause is instant in both
        directions and resumes the work mid-chunk.
        """
        machine = _ssh_machine(task, w)
        name = _container_name(spec, task.tag, w.worker_id)
        key = _key(spec, task.tag, w.worker_id)
        if intent == RUN:
            if probe == "paused":
                machine.unpause_container(name)  # resuming a parked worker is not a restart
            elif probe == "running":
                if _replaceable(w, task):
                    # The task has redeployed past this container and it has
                    # handed everything over: stopping it is how it gets
                    # replaced, which the next pass does. As with any stop, the
                    # cycle in flight is lost.
                    machine.stop_container(name)
            elif probe in ("missing", "stopped") and self._restart_allowed(key):
                # Anything else -- unreachable, or not observed yet -- is not
                # something this pass can act on.
                self._note_restart(key)
                self._start_or_replace(machine, name, spec, task, w, probe)
        elif intent == PARK and probe == "running":
            machine.pause_container(name)
        elif intent == STOP and probe in ("running", "paused"):
            if probe == "paused":
                machine.unpause_container(name)  # docker stop cannot signal a frozen process
            machine.stop_container(name)

    def _start_or_replace(self, machine, name, spec, task: tasks.TaskRecord, w, probe: str):
        """Bring a container that is not running back up.

        A container's bundle is fixed in the environment it was created with,
        so a slot joins a bundle the task has moved to by being replaced. That
        destroys anything the container never handed over, so it waits until a
        collection has reported the container empty -- and a container holding
        output is started instead, which is what lets the next passes drain it.

        A container that will not stay up is never collected from, so a count
        it had when it stopped staying up is the last word on it: one that was
        holding a backlog stays pinned to its bundle, restarted and visibly
        down, because recovering the slot would mean discarding an amount
        nothing can measure any more -- the operator's call from the workers
        table, where Remove says what would go. One that never held anything
        is replaceable as ever; the rule is _replaceable's, not "a collection
        said so".
        """
        if probe == "stopped" and not _replaceable(w, task):
            machine.start_container(name)
            return
        if probe == "stopped":
            # Take what it flushed on the way down before the container (and
            # its filesystem) go. If that fails, so does the replacement: the
            # slot keeps running on the old bundle, which is recoverable, where
            # throwing the output away is not.
            self._sweep_ssh(machine, spec, task, w)
            machine.remove_container(name)
        self._run_ssh_container(spec, task, w)

    def _sweep_ssh(self, machine, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord):
        """Collect from a stopped container, the last chance to do so."""
        sweep_stopped(machine, **_transfer_target(spec, task, w))

    def shutdown(self):
        """Stop owned subprocesses (workers flush completed output on SIGTERM);
        ssh containers are unaffected -- their work continues across
        dashboard restarts."""
        self._blocking.shutdown(wait=False, cancel_futures=True)
        self._builds.shutdown(wait=False, cancel_futures=True)
        for proc in [*self._local.values(), *(p for p, _ in self._sync.values())]:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
