"""The WorkerManager: reconciles worker slots with real processes and
containers, and the task's machines with rented instances.

It lives in the dashboard API process. A local slot is a subprocess of that
process running the worker entrypoint. An ssh slot is a worker-image container
on a machine reached over ssh (cloud/ssh_machine.py): the operator's own, or
one the dashboard rents from a provider (cloud/providers/) and records as a
task machine. Every ssh slot delivers into its own container, and the
reconcile pass collects from it over ssh (cloud/ssh_transfer.py); for a tag
whose data home is an ssh machine, the chunks collected here are relayed on
into it.

Adding a slot only records it, paused; its first start spawns the process or
container. Renting a machine launches its instance at once, and it bills from
then on.

Desired state lives in task.json (tasks.py). Actual state is observed live:
local workers by their durable pid, so any dashboard instance can observe and
stop them, even after a restart; ssh slots by a docker probe over ssh; rented
machines by the provider's listing plus that probe. reconcile() drives
observed state toward desired state in both directions, starting and stopping
workers and machines. It also runs each workload's scheduler tick, which may
*gate* a role: park its workers without touching the operator's desired state.

Cloud operations need <mount>/cloud/credentials.json. Credentials load lazily,
so a local-only dashboard works without them.
"""

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path

from cloud import runtime_abi
from cloud.bundles import BundleManifest, deploy_current_tree, source_hash
from cloud.credentials import CloudCredentials, CredentialsError, load_credentials
from cloud.providers.aws import CATALOG as AWS_CATALOG
from cloud.providers.aws import AwsProvider
from cloud.providers.base import Instance, LaunchRequest, Provider, ProviderError
from cloud.r2 import bucket_path, rclone
from cloud.ssh_machine import SshMachine, SshMachineError
from cloud.ssh_transfer import pull_results, push_file, relay_files, sweep_stopped
from cloud.worker_entrypoint import EXIT_INTERRUPTED
from cloud.worker_env import bundle_worker_env, r2_env
from tornado.ioloop import IOLoop

from scribblez import params as params_mod
from scribblez import workloads
from scribblez.dashboard import pool as pool_mod
from scribblez.dashboard import queue as queue_mod
from scribblez.dashboard import tasks
from scribblez.dashboard.control_store import IMPORTED_JSON, ControlStore
from scribblez.dashboard.slot_files import LocalSlotFiles, SshSlotFiles
from scribblez.generational import state_pair
from scribblez.generational.lifecycle import MANIFEST_NAME
from scribblez.generational.scheduler import (
    DATA_PLANE_HOME,
    DATA_PLANE_LEGACY,
    GENERATE_ROLE,
    TICK_FOR_TASK,
)
from scribblez.hardware import default_thread_count
from scribblez.paths import (
    CONTROLS_REL,
    DEFAULT_MOUNT_ROOT,
    SCHEDULER_STATE_REL,
    TRAINER_OUTPUT_DIRS,
)
from scribblez.workloads.base import SchedulerHooks

# After an ssh machine fails a probe, how long it is assumed still unreachable
# before probing again, so a powered-off machine costs one connect timeout per
# this interval rather than one per pass.
SSH_REPROBE_SECONDS = 30.0

# A container that dies as fast as it is started will not be fixed by starting
# it again: restarts back off from the observation TTL up to this, and reset
# once the container is observed running.
MAX_RESTART_BACKOFF_SECONDS = 300.0

# How long an observation of a container or machine stands in for a fresh one.
# Only the reconcile pass observes; status requests read what it left, so a
# browser polling every 3 seconds costs no ssh round trips at all.
OBSERVATION_TTL_SECONDS = 5.0

# A rented machine on which nothing has run for this long is stopped, keeping
# its disk, so a paused or finished run stops paying the provider's rate.
IDLE_STOP_SECONDS = 600.0

# How long a slot meant to run but not alive still keeps the tag queue off its
# machine (WorkerManager._holds_machine): long enough to cover a restart or a
# slow start, short of holding a machine for a slot that never comes back.
DEAD_SLOT_GRACE_SECONDS = 600.0

# The nice level local workers run at. A worker is the long-running background
# job on this machine; everything else that competes with it -- a bundle build
# for a fleet that is billing while it waits, the test suite, an editor's build
# -- is short and should win. At this level the scheduler gives a nice-0
# process about ten times a worker's CPU share under contention, and an idle
# machine still gives the worker all of it.
LOCAL_WORKER_NICE = 10
# Per-machine key material for rented machines (known_hosts files), under
# <workload>/<tag>/<name>: a machine's name is unique only within its task
# (_next_machine_name), so two tasks' `aws-1` must not share a file.
MACHINES_DIR = Path("/workspace/mount/cloud/machines")
# How long the rent form's spot rates are served from the last fetch.
SPOT_PRICES_TTL_SECONDS = 300.0
# A rented instance whose ssh does not answer is still coming up for this
# long after its launch or start before it reads as unreachable.
BOOT_GRACE_SECONDS = 300.0


# What a slot should be doing, from operator intent plus scheduler gating.
# PARK (a gated slot) and STOP (a paused one) both mean not working; a parked
# ssh container is frozen rather than stopped, because restarting one is
# expensive (see _reconcile_ssh).
RUN, PARK, STOP = "run", "park", "stop"


def _is_crash(code: int | None) -> bool:
    """Whether a worker's exit code is a crash, which counts toward failing
    its tag: non-zero, and not EXIT_INTERRUPTED. That one is a SIGTERM from
    outside the worker, which a worker never sends itself: a gate or pause
    from this dashboard, a docker stop, or its host stopping under a spot
    interruption. None, an exit not observed, is no crash either."""
    return code not in (None, 0, EXIT_INTERRUPTED)


def _is_ssh_crash(reason: str) -> bool:
    """Whether a stopped container's exit reason, "exit <code>: <message>", is
    a crash. A reason with no readable code (the read failed, or the container
    went between the probe and the read) counts as one: the container did
    stop, and a slot that keeps stopping for unreadable reasons must still
    fail its tag rather than restart forever."""
    head = reason.split(":", 1)[0]
    code = head.removeprefix("exit ")
    if code == head or not code.lstrip("-").isdigit():
        return True
    return _is_crash(int(code))


def _role_inputs(spec, role, params, mount_root: Path) -> dict[str, Path]:
    """The files a slot of `role` reads from outside its tag (RoleSpec.inputs),
    resolved against the controller's mount; empty for a role with none."""
    return workloads.resolve(role.inputs)(params, mount_root) if role.inputs else {}


def _require_inputs(inputs: dict[str, Path]):
    """Every input source must exist before a slot is started for it."""
    for rel, src in inputs.items():
        if not src.is_file():
            raise SshMachineError(f"input {rel} is missing: {src} is not a readable file")


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
    """The ownership tag a rented instance carries, naming the task machine it
    is. An instance whose tag no task machine matches is an orphan."""
    return f"{spec.name}/{tag}/{name}"


def _accrue_machine(m: tasks.MachineRecord, billing: bool):
    """Advance a rented machine's spend to now (MachineRecord.spend_now), and
    record whether it bills from here on (pending or running, not stopped)."""
    now = time.time()
    m.spend = m.spend_now(now)
    m.observed_at = now
    m.observed_up = billing


def _record_instance(
    record: tasks.MachineRecord, provider, inst: Instance, mtype, *, spot: bool, known_hosts: Path
):
    """Fill `record` from a rented instance just launched or adopted: its
    address and key material, type, rate and launch time, and start its spend
    accrual. `known_hosts` is emptied, since a new instance has a new host key.
    Until the instance has an address its host is a placeholder unique to it,
    so two machines still coming up never read as the same host."""
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    known_hosts.write_text("")
    record.provider = provider.name
    record.host = f"{provider.ssh_user}@{inst.address or f'pending-{inst.id}'}"
    record.identity_file = provider.identity_file
    record.known_hosts_file = str(known_hosts)
    record.arch = mtype.arch
    record.gpu_count = mtype.gpu_count
    record.instance_id = inst.id
    record.instance_type = mtype.id
    record.spot = spot
    record.region = getattr(provider, "region", None)
    record.cost_per_hr = inst.cost_per_hr if inst.cost_per_hr is not None else mtype.cost_per_hr
    record.launched_at = inst.launched_at or time.time()
    _accrue_machine(record, True)


def _moved_host(record: tasks.MachineRecord, inst: Instance | None) -> str | None:
    """`record`'s host at `inst`'s current address when it has moved (a
    stop/start, or the first address after launch); None otherwise."""
    if inst is None or not inst.address:
        return None
    host = f"{record.host.split('@')[0]}@{inst.address}"
    return host if host != record.host else None


def _rented_state(m: tasks.MachineRecord, inst: Instance | None, probe: str | None) -> str:
    """A rented machine's display state, from the provider's listing and, once
    the instance runs, its ssh probe:

      launching           pending, or running but not answering ssh yet
      preparing           its first-boot script is still pulling the images
      up                  it can host containers
      stopping, stopped   suspended
      gone                terminated, or absent from the listing
      unreachable         not answering past BOOT_GRACE_SECONDS
      checking            running, not probed yet
    """
    if inst is None or inst.state == "terminated":
        return "gone"
    if inst.state in ("pending", "stopping", "stopped"):
        return {"pending": "launching"}.get(inst.state, inst.state)
    if probe == "unreachable":
        since = m.launched_at or 0.0
        return "launching" if time.time() - since < BOOT_GRACE_SECONDS else "unreachable"
    return probe or "checking"


# The target key of every local slot (_slot_target).
LOCAL_TARGET = "localhost"


def _slot_target(kind: str, machine: tasks.MachineRecord | None, host: str | None) -> str:
    """Which physical machine a slot runs on, comparable across tasks and
    spellings: LOCAL_TARGET for a local slot, else the canonical host name."""
    if kind == "local":
        return LOCAL_TARGET
    return pool_mod.canonical_host(machine.host if machine is not None else host)


def _is_pool_machine(m: pool_mod.PoolMachine, target: str) -> bool:
    """Whether the slot target `target` (_slot_target) is pool machine `m`."""
    if m.kind == "local":
        return target == LOCAL_TARGET
    return target in pool_mod.host_names(m)


def _gpu_need(spec, task: tasks.TaskRecord, role: str) -> float | None:
    """The measured GiB a `role` slot of `task` needs (WorkloadSpec.gpu_need),
    None when its workload has no figure for it."""
    if not spec.gpu_need:
        return None
    params = params_mod.validate(spec.params_cls, task.params)
    return workloads.resolve(spec.gpu_need)(params, role)


def _forget_empty(w: tasks.WorkerRecord):
    """Downgrade an `undelivered` count of zero to unknown, keeping any other.

    Zero is the one value that authorizes destroying a container, so it is
    worth trusting only while current: a machine off the network for hours may
    have a worker that kept filling it the whole time. A positive count only
    ever refuses, so keeping a stale one costs nothing.
    """
    if w.undelivered == 0:
        w.undelivered = None


def _note_finished(w: tasks.WorkerRecord) -> bool:
    """Record that slot `w` reached its role's terminal condition: its worker
    exited 0, or the scheduler finished the role. Pausing it keeps reconcile
    from restarting it forever, which would also keep its machine from ever
    going idle. Returns whether the slot changed."""
    if w.desired_state != "running":
        return False
    w.desired_state = "paused"
    w.finished = True
    return True


def _trainer_finished(spec: workloads.WorkloadSpec, task: tasks.TaskRecord) -> bool:
    """Whether the task has a trainer slot (a role with an ingest tick) and
    every one has finished: exited at its terminal condition, e.g. max_rows."""
    trainers = [w for w in task.workers if spec.role(w.role).ingest]
    return bool(trainers) and all(w.finished for w in trainers)


def _home_trainer(task: tasks.TaskRecord, role: workloads.RoleSpec) -> bool:
    """Whether `role` is the trainer of a tag whose data plane runs beside it."""
    return task.data_plane == DATA_PLANE_HOME and bool(role.ingest)


# The data sink of a slot on its tag's data home on an ssh machine: the tag's
# named volume there, shared with the other slots on that machine and handed
# to the worker as SCZ_DATA_SINK=local.
DATA_SINK_HOME = "home"


def _trainer_slot(spec, task: tasks.TaskRecord) -> tasks.WorkerRecord | None:
    return next((w for w in task.workers if spec.role(w.role).ingest), None)


def _same_machine(a: tasks.WorkerRecord, b: tasks.WorkerRecord) -> bool:
    return (a.kind, a.machine, a.host) == (b.kind, b.machine, b.host)


def _tag_volume(spec, task: tasks.TaskRecord) -> str:
    """The named volume holding a tag's tree on its data home."""
    return _container_name(spec, task.tag, "data")


def _finish_role(task: tasks.TaskRecord, role: str) -> bool:
    """Finish every slot of `role` that wants to run and drop the role's gate
    (the scheduler's finish hook). Returns whether the task changed."""
    finished = [_note_finished(w) for w in task.workers if w.role == role]
    ungated = task.gates.pop(role, None) is not None
    return any(finished) or ungated


def _replaceable(w: tasks.WorkerRecord, task: tasks.TaskRecord) -> bool:
    """Whether slot `w`'s container may be thrown away for one on the task's
    bundle: it is on a different bundle and known to hold nothing. A new
    container counts zero from creation, so one that never came up (often what
    a redeploy is trying to fix) is replaceable rather than restarted
    forever."""
    return w.bundle_id != task.bundle_id and w.undelivered == 0


def _intent(w: tasks.WorkerRecord, task: tasks.TaskRecord) -> str:
    if w.desired_state != "running":
        return STOP
    return PARK if w.role in task.gates else RUN


def check_worker_images_current(mount_root: Path):
    """Refuse to deploy a bundle a published worker image cannot load.

    Bundles are compiled in the dev container but run against the worker
    images' libraries, so a toolchain upgrade here (a newer libstdc++, say)
    produces binaries every worker crashes on at import. The worker images are
    rebuilt by hand after such a change; this catches a forgotten rebuild.

    Every recorded image is checked, whichever runtime this deploy's slots use:
    a task's slots can run either, and both are rebuilt together. Passes when
    no image push has recorded its library versions, since then nothing is
    known.
    """
    records = runtime_abi.read_records(mount_root)
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


def _local_state(
    desired: str, alive: bool, gated: bool, finished: bool = False, failed: bool = False
) -> str:
    """A local slot's display state, from operator intent, real liveness and
    gating. `stopping` is a paused slot whose process has not exited yet;
    `exited` is an unexpected death of a slot that should be running
    (reconcile respawns it); `finished` is an exit at the role's terminal
    condition."""
    if gated:
        return "waiting"
    if desired == "paused":
        if finished and not alive:
            return "finished"
        if failed and not alive:
            return "failed"
        return "stopping" if alive else "paused"
    return "running" if alive else "exited"


def _ssh_state(
    desired: str, probe: str, gated: bool, finished: bool = False, failed: bool = False
) -> str:
    """An ssh slot's display state, from its container probe
    (cloud/ssh_machine.py's states, plus "unknown" before the reconcile pass
    has observed it), operator intent and gating.

    `unreachable` stays its own state rather than a guess: the machine may be
    off with the worker gone, or just off the network with it still running.
    A gated slot reads `waiting` whether its container is paused or still
    winding down.

    A slot that should be running shows `exited` only for a container that ran
    and died (`stopped`). No container yet (`missing`) is `starting`: the
    ordinary state between the operator's Start and the container's creation,
    which lasts as long as the machine takes to pull a multi-gigabyte image."""
    if probe == "unknown":
        return "checking"
    if gated:
        return "waiting"
    if probe == "unreachable":
        return "unreachable"
    if desired == "paused":
        if finished and probe == "stopped":
            return "finished"
        if failed and probe in ("stopped", "missing"):
            return "failed"
        return "stopping" if probe in ("running", "paused") else "paused"
    if probe == "missing":
        return "starting"
    if probe == "running":
        return "running"
    # Paused while meant to be running: a gate was released and the next pass
    # will resume it. Nothing exited, so it is not `exited`.
    return "starting" if probe == "paused" else "exited"


class WorkerManager:
    def __init__(self, mount_root: Path):
        """Everything this manager reads and writes lives under `mount_root`:
        the tag trees and the control database (control_store.py). The dashboard passes its
        --mount-root; a test or simulation passes a scratch dir."""
        self.mount_root = Path(mount_root)
        # The control records (control_store.py), written only by the thread
        # claim_writer names; every other thread reads committed copies.
        self.control = ControlStore(self.mount_root)
        self.tasks = tasks.TaskStore(self.mount_root, self.control)
        self.pool_store = pool_mod.pool_store(self.control)
        self.queue_store = queue_mod.queue_store(self.control)
        self._local: dict[str, subprocess.Popen] = {}  # slot key -> live process
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
        # Slot key -> [(when, why)] of each restart after a crash (a non-zero
        # exit), which the tag queue reads to fail a crash-looping slot.
        self._crashes: dict[str, list[tuple[float, str]]] = {}
        # Slot key -> when a slot meant to run was first seen down since it was
        # last alive (see _holds_machine).
        self._down_since: dict[str, float] = {}
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
        # Where every blocking step runs (see offload). One thread: the point is
        # to keep the event loop free, not to run these steps concurrently.
        self._blocking = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scz-blocking")
        # Where bundle builds run (see redeploy and _bundle_for_start): off the
        # blocking thread, which a build would otherwise hold for minutes.
        self._builds = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scz-build")
        # Machine keys of task rentals to stop as soon as their slots are down,
        # without the IDLE_STOP_SECONDS wait (stop_task_rentals).
        self._stop_now: set[str] = set()
        # Where generation uploads run (_make_publish), also off the blocking
        # thread: a tag moving onto a rented trainer uploads every generation
        # it has, which held every request behind it for minutes.
        self._uploads = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scz-upload")
        # (task key, dest_rel) -> its upload in flight or finished but not yet
        # collected by the scheduler's next publish call.
        self._publishing: dict[tuple[str, str], Future] = {}
        # Task key -> its first-use bundle build in flight (_bundle_for_start).
        self._pending_builds: dict[str, Future] = {}
        # Where collections run (_collect_step), one thread per slot: a trainer's
        # exports and checkpoints are tens to hundreds of megabytes, which on the
        # blocking thread would hold every other tag's pass behind them, and one
        # slot's slow link must not hold up another slot's collection.
        self._transfer_pools: dict[str, Executor] = {}
        self._new_transfer_pool = lambda: ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="scz-transfer"
        )
        # Slot key -> its collection in flight, or done and not yet recorded.
        self._collecting: dict[str, Future] = {}
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

        Building and pushing take minutes and touch no record. On the blocking
        thread they would delay every Pause and Remove clicked meanwhile, while
        the machines went on billing. Only the arch survey and the repin need
        to be serialized with other steps.
        """
        archs = await self.offload(self._needed_archs, spec, task)
        manifest = await IOLoop.current().run_in_executor(self._builds, self._build_bundle, archs)
        return await self.offload(self._pin_bundle, spec, task, manifest)

    def _build_bundle(self, archs: list[str]) -> BundleManifest:
        check_worker_images_current(self.mount_root)
        creds = self._creds()
        return deploy_current_tree(creds.r2, archs, cache=self._source_digests)

    def _pin_bundle(self, spec, task: tasks.TaskRecord, manifest: BundleManifest) -> str:
        task.bundle_id = manifest.bundle_id
        task.bundle_source_hash = manifest.source_hash
        task.bundle_archs = list(manifest.archs)
        self.tasks.save(spec, task)
        return manifest.bundle_id

    def _slot_arch(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> str:
        """The CPU microarchitecture slot `w`'s bundle must be built for. A
        rented machine's comes from the catalog. A registered machine or bare
        host is asked once over ssh, through the worker image's own start-up
        detection, and the answer is kept on its record."""
        holder = self._machine_record(task, w.machine) if w.machine is not None else w
        if holder.arch:
            return holder.arch
        image = self._creds().registry.image_for(spec.role(w.role).runtime)
        machine = self._ssh_machine(task, w)
        machine.pull_image(image)
        holder.arch = machine.detect_arch(image)
        self.tasks.save(spec, task)
        if w.machine is not None and task.find_machine(w.machine) is None:
            self.pool_store.save(self.pool_store.load())  # a leased pool machine's record
        return holder.arch

    def _needed_archs(self, spec, task: tasks.TaskRecord) -> list[str]:
        """The archs the task's bundle must cover: every ssh slot's, plus those
        its current bundle already covers, since containers of a removed slot's
        arch may still be running it."""
        archs = set(task.bundle_archs)
        for w in task.workers:
            if w.kind == "ssh":
                archs.add(self._slot_arch(spec, task, w))
        return sorted(archs)

    def _bundle_for_start(
        self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord, key: str
    ) -> str | None:
        """The bundle this task's remote workers run, deployed on first use, or
        None while that deployment is still building.

        Deployment is not an operator step: the first remote start builds the
        controller's current tree for the task's archs, pushes it, and pins
        it, and every later worker joins that bundle. Pinning keeps an
        experiment homogeneous; moving the fleet to new code is the explicit
        redeploy. The one exception is a later slot whose arch the bundle
        lacks: the tree is rebuilt with that arch added and the task repinned.
        Containers on the old bundle then get replaced as after any redeploy.

        A build takes minutes, and slot starts run on the blocking thread, so
        the build goes to the build thread instead. Meanwhile the slot shows
        `starting` with the reason on its row, and the pass that finds the
        build done pins the task and starts the slot. A failed build becomes
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
        bundle: the dashboard's cue to redeploy. False while nothing is pinned
        or the tree's hash cannot be computed yet (an arch not built here)."""
        if not task.bundle_source_hash or not task.bundle_archs:
            return False
        current = source_hash(task.bundle_archs, self._source_digests)
        return current is not None and current != task.bundle_source_hash

    # ---- cloud plumbing --------------------------------------------------

    def _bucket_env(self) -> dict[str, str]:
        """Bucket credentials for a local data home's trainer, which restores
        the window a remote home uploaded; none when the controller has no
        credentials file."""
        try:
            return r2_env(self._creds())
        except (CredentialsError, FileNotFoundError):
            return {}

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

    def _push_controls(self, spec, task: tasks.TaskRecord, status: list[dict]):
        """Copy the operator's controls file into each running ssh trainer's
        container when it has changed since that container last got it; the
        trainer reads it from its own tree each generation
        (generational/records.py). A local trainer reads the controller's."""
        path = self.tasks.paths(spec, task.tag).controls_path
        try:
            stamp = path.stat().st_mtime_ns
        except FileNotFoundError:
            return
        for info in status:
            w = task.find(info["worker_id"])
            if w is None or w.kind != "ssh" or not spec.role(w.role).ingest:
                continue
            key = _key(spec, task.tag, w.worker_id)
            if info.get("ssh_probe") != "running" or self._controls_pushed.get(key) == stamp:
                continue
            push_file(
                self._ssh_machine(task, w),
                _container_name(spec, task.tag, w.worker_id),
                remote_root=str(self.tasks.paths(spec, task.tag).root),
                rel_dest=CONTROLS_REL,
                src=path,
            )
            self._controls_pushed[key] = stamp

    def _log_file(self, spec: workloads.WorkloadSpec, tag: str, name: str):
        log_dir = self.tasks.paths(spec, tag).logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        return open(log_dir / f"{name}.log", "ab")

    def _spawn_local(self, spec: workloads.WorkloadSpec, task: tasks.TaskRecord, w):
        params = params_mod.validate(spec.params_cls, task.params)
        env = os.environ | spec.worker_env(task.tag, params, w.role) | {
            "SCZ_SINK": "local",
            "SCZ_MOUNT_ROOT": str(self.mount_root),
            "SCZ_THREADS": str(w.threads),
            "SCZ_WORKER_ID": w.worker_id,
            "SCZ_WORKER_KIND": "local",
        }  # fmt: skip
        if _home_trainer(task, spec.role(w.role)):
            env |= {"SCZ_DATA_PLANE": DATA_PLANE_HOME, **self._bucket_env()}
        log = self._log_file(spec, task.tag, w.worker_id)
        proc = subprocess.Popen(
            [sys.executable, "-m", "cloud.worker_entrypoint"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: os.nice(LOCAL_WORKER_NICE),  # inherited by its sim threads
            # Its own session: a Ctrl-C in the dashboard's terminal must not
            # interrupt it mid-step. The dashboard's shutdown SIGTERMs it, which
            # it answers by flushing what it has and exiting.
            start_new_session=True,
        )
        self._local[_key(spec, task.tag, w.worker_id)] = proc
        w.pid = proc.pid  # durable, so any instance can observe and stop this worker
        self.tasks.save(spec, task)

    def _local_alive(self, spec, task: tasks.TaskRecord, w) -> bool:
        """Whether slot `w`'s worker process is really running, by its durable
        pid, which also covers workers another dashboard instance spawned."""
        return worker_pid_alive(w.pid, w.worker_id, task.tag)

    def _local_exit_code(self, spec, task: tasks.TaskRecord, w) -> int | None:
        """Slot `w`'s worker's exit code, or None if it is still running or was
        not this process's child."""
        proc = self._local.get(_key(spec, task.tag, w.worker_id))
        return None if proc is None else proc.returncode

    def _stop_local(self, spec, task: tasks.TaskRecord, w):
        """SIGTERM slot `w`'s worker, which flushes completed output and exits.
        No-op if it is not running."""
        if worker_pid_alive(w.pid, w.worker_id, task.tag):
            try:
                os.kill(w.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    # ---- ssh plumbing ----------------------------------------------------

    def _observe_container(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> str:
        """Probe slot `w`'s container over ssh (or skip, for a host that failed
        within SSH_REPROBE_SECONDS) and remember the answer, with the exit
        reason of a stopped container.

        An unlaunched slot is probed too: an in-doubt first start (ssh lost
        after `docker run` was sent) may have left a live container, and
        finding one marks the slot launched."""
        tag, host = task.tag, self._ssh_host(task, w)
        down_since = self._ssh_down.get(host)
        if down_since is not None and time.time() - down_since < SSH_REPROBE_SECONDS:
            probe = "unreachable"
        else:
            probe = self._ssh_machine(task, w).container_state(
                _container_name(spec, tag, w.worker_id)
            )
            if probe == "unreachable":
                self._ssh_down[host] = time.time()
            else:
                self._ssh_down.pop(host, None)
        if not w.launched and probe not in ("unreachable", "missing"):
            w.launched = True  # the in-doubt start did create the container
        key = _key(spec, tag, w.worker_id)
        self._probes[key] = (probe, time.time())
        if probe == "stopped":
            self._exits[key] = self._ssh_machine(task, w).container_exit(
                _container_name(spec, tag, w.worker_id)
            )
            if self._exits[key].startswith("exit 0:"):
                _note_finished(w)
        elif probe in ("running", "paused"):
            self._exits.pop(key, None)
        # Finding no container clears nothing: that is usually a slot whose
        # creation failed, and the failure is the only account of why.
        if probe == "running":
            self._restarts.pop(key, None)  # it came up; it is not looping
        if probe == "unreachable":
            _forget_empty(w)
        return probe

    def _refresh_probe(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> str:
        """_probe_container, observed afresh unless the last observation is
        younger than OBSERVATION_TTL_SECONDS. For the writer only."""
        _, at = self._probes.get(_key(spec, task.tag, w.worker_id), ("unknown", 0.0))
        if time.time() - at >= OBSERVATION_TTL_SECONDS:
            self._observe_container(spec, task, w)
        return self._probe_container(spec, task, w)

    def _probe_container(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> str:
        """Slot `w`'s container probe state, as the pass last observed it
        (_observe_slots). Status requests read it too, so browser polling costs
        no ssh and cannot be stalled by a slow machine. A slot no pass has
        reached yet reads "unknown".

        An unlaunched slot's `unreachable` reads as `missing`: with no
        container known to exist, the slot must stay removable even when its
        host is bogus or offline."""
        probe, _ = self._probes.get(_key(spec, task.tag, w.worker_id), ("unknown", 0.0))
        if not w.launched and probe == "unreachable":
            return "missing"
        return probe

    def _note_crash(self, key: str, why: str):
        """Record that slot `key` is being restarted after exiting non-zero."""
        self._crashes.setdefault(key, []).append((time.time(), why))

    def _forget_slot(self, key: str):
        """Drop everything remembered about removed slot `key`. A later slot
        reuses the key (_next_worker_id hands out the freed id again, a requeued
        tag gets its layout's ids back, a deleted tag can be recreated), and
        must not inherit the old worker's process handle, its container's last
        probe and exit reason, its backoff, crashes or downtime: a stale
        "stopped" probe read as the new slot's own became a phantom crash and
        a start of a container that does not exist."""
        for memory in (
            self._local, self._exits, self._restarts, self._probes, self._crashes,
            self._down_since,
        ):  # fmt: skip
            memory.pop(key, None)
        self._collecting.pop(key, None)
        pool = self._transfer_pools.pop(key, None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def forget_crashes(self, spec, tag: str, worker_id: str):
        self._crashes.pop(_key(spec, tag, worker_id), None)

    def recent_crashes(self, spec, tag: str, worker_id: str, window: float) -> list[str]:
        """Why slot `worker_id` crashed, for each crash in the last `window`
        seconds (see _note_crash)."""
        cutoff = time.time() - window
        crashes = self._crashes.get(_key(spec, tag, worker_id), [])
        return [why for when, why in crashes if when >= cutoff]

    def _restart_allowed(self, key: str) -> bool:
        """Whether slot or machine `key` may be (re)started now. Attempts back
        off, so a broken worker costs one ssh round trip every few minutes
        rather than one per pass, and its failure stays on screen."""
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
            # Waiting on the build is not a failed attempt, so the restart
            # backoff must not grow across the wait.
            self._restarts.pop(key, None)
            return
        w.bundle_id = bundle_id
        creds = self._creds()
        params = params_mod.validate(spec.params_cls, task.params)
        env = bundle_worker_env(
            creds, spec, task.tag, params,
            role=w.role, bundle_id=w.bundle_id, worker_id=w.worker_id,
        )  # fmt: skip
        env["SCZ_SINK"] = self._slot_records_sink(spec, task, w)
        data_sink = self._slot_data_sink(spec, task, w)
        data_env = "local" if data_sink == DATA_SINK_HOME else data_sink
        if data_env != env["SCZ_SINK"]:
            # Only when it differs: a bundle predating SCZ_DATA_SINK refuses
            # to start on an SCZ_* variable it does not know.
            env["SCZ_DATA_SINK"] = data_env
        if _home_trainer(task, spec.role(w.role)):
            env["SCZ_DATA_PLANE"] = DATA_PLANE_HOME
            # Its generations reach the bucket only from the home itself.
            env["SCZ_HOME_UPLOADS"] = "1"
        # A new trainer container starts from the controller's copy of the
        # state (state_pair's cursor rule decides against what it may hold).
        seed = (
            spec.role(w.role).ingest
            and self.tasks.paths(spec, task.tag).rolling_checkpoint.exists()
        )
        if seed:
            env["SCZ_STATE_SEED"] = "1"
        self._controls_pushed.pop(_key(spec, task.tag, w.worker_id), None)
        # The whole tag tree, where the worker's default mount root puts it,
        # so every rename stays within one filesystem.
        volume = (
            (_tag_volume(spec, task), str(spec.paths(task.tag, DEFAULT_MOUNT_ROOT).root))
            if data_sink == DATA_SINK_HOME
            else None
        )
        if w.threads:
            env["SCZ_THREADS"] = str(w.threads)
        machine = self._ssh_machine(task, w)
        role = spec.role(w.role)
        inputs = _role_inputs(spec, role, params, self.mount_root)
        try:
            # A missing input becomes the slot's exit reason, shown on its row
            # and paced by the restart backoff.
            _require_inputs(inputs)
            # Pull here so a new container picks up a rebuilt worker image;
            # run_container then never pulls on its own.
            image = creds.registry.image_for(role.runtime)
            machine.pull_image(image)
            name = _container_name(spec, task.tag, w.worker_id)
            if volume is not None:
                machine.create_volume(volume[0])
            machine.run_container(name, image, env, gpus=role.gpu, volume=volume)
            self._stage_inputs_in_container(machine, name, spec, task.tag, inputs)
            if seed:
                self._seed_state(machine, name, spec, task)
        except SshMachineError as e:
            # The slot reads `starting` until this succeeds. Recording why shows
            # the operator what the machine lacks (an NVIDIA toolkit, a
            # registry login) instead of an endless `starting`.
            self._exits[key] = str(e)
            raise
        self._exits.pop(key, None)
        w.launched = True
        w.undelivered = 0  # a container just created is holding nothing
        self.tasks.save(spec, task)

    # ---- slot operations -------------------------------------------------

    def _check_role(
        self,
        spec,
        task: tasks.TaskRecord,
        role: str,
        kind: str,
        machine: tasks.MachineRecord | None = None,
        host: str | None = None,
        check_gpu: bool = True,
    ):
        role_spec = spec.role(role)
        assert kind in role_spec.kinds, f"role '{role}' does not support {kind} workers"
        if machine is not None and role_spec.gpu and machine.gpu_count == 0:
            # Refused here rather than by `docker run --gpus all` on the
            # remote, after a bundle deploy. Only a machine known to have no
            # GPU is refused: every GPU container runs under `--gpus all`, so
            # GPU slots share a machine's GPUs the way local workers share the
            # controller's (a move-set-eval generator and trainer on one L4),
            # and a count of them would pin nothing. An unknown count (a
            # manual machine) is not checked, as a bare host never was.
            raise AssertionError(f"machine '{machine.name}' has no GPU for role '{role}'")
        if role_spec.singleton:
            taken = [w.worker_id for w in task.workers if w.role == role]
            assert not taken, f"role '{role}' already has a worker ({taken[0]})"
        if task.data_plane == DATA_PLANE_HOME and not role_spec.dispatch:
            self._check_data_home_order(spec, task, role_spec)
        if role_spec.gpu and check_gpu:
            refusal = self._gpu_fit_refusal(spec, task, role, kind, machine, host)
            assert refusal is None, refusal
        return role_spec

    def _check_data_home_order(self, spec, task: tasks.TaskRecord, role: workloads.RoleSpec):
        """A data-home tag's trainer decides where its other data-plane slots
        deliver, and a running slot cannot move (its sinks and mount are fixed
        when its worker starts). So a generator needs a trainer, and a trainer
        joins only while the others are stopped; _rehome then moves them."""
        if role.ingest:
            moving = [
                w.worker_id
                for w in task.workers
                if not spec.role(w.role).dispatch
                and (w.desired_state == "running" or self._seen_alive(spec, task, w))
            ]
            assert not moving, (
                "pause this tag's other slots before adding its trainer: where it runs "
                f"decides where they deliver ({', '.join(moving)} still running)"
            )
        else:
            assert _trainer_slot(spec, task) is not None, (
                "add this tag's trainer first: its machine is where the generators deliver"
            )

    def _rehome(self, spec, task: tasks.TaskRecord, joined: tasks.WorkerRecord):
        """After data-home trainer `joined` is added to a tag that already has
        other slots: recreate their stopped ssh containers at their next start,
        with the sinks and mount the new home gives them, and remove the tag's
        volume wherever no slot now works in it. What a moved generator had
        staged in the old home's volume goes with it; generations come back
        from the bucket, and the checkpoint from the controller (_seed_state)."""
        if not _home_trainer(task, spec.role(joined.role)):
            return
        for w in task.workers:
            if w is not joined and w.kind == "ssh" and not self._machine_gone(spec, task, w):
                self._discard_container(spec, task, w)
        homes = {
            (w.machine, w.host) for w in task.workers if self._on_remote_data_home(spec, task, w)
        }
        for w in task.workers:
            if w.kind == "ssh" and (w.machine, w.host) not in homes:
                if not self._machine_gone(spec, task, w):
                    self._ssh_machine(task, w).remove_volume(_tag_volume(spec, task))
                homes.add((w.machine, w.host))  # once per machine

    def _gpu_fit_refusal(self, spec, task, role: str, kind: str, machine, host) -> str | None:
        """Why a new `role` slot would not fit the target machine's GPU memory
        alongside the GPU slots already there, or None: the task's own, and
        other tags' while they hold the machine (_holds_machine), so a paused
        or long-dead slot elsewhere does not refuse it. The same measured
        needs as placement (WorkloadSpec.gpu_need); checked only when every need
        and the machine's memory are known, so a hand placement is refused
        only on evidence."""
        capacity = self._gpu_capacity(kind, machine, host)
        if capacity is None:
            return None
        target = _slot_target(kind, machine, host)
        others = [(s, t) for s, t in self.all_tasks() if (s.name, t.tag) != (spec.name, task.tag)]
        needs = [_gpu_need(spec, task, role)]
        for s, t in [(spec, task), *others]:
            for w in t.workers:
                if (
                    not s.role(w.role).gpu
                    or _slot_target(w.kind, None, self._slot_host(t, w)) != target
                ):
                    continue
                if t is task or self._holds_machine(s, t, w):
                    needs.append(_gpu_need(s, t, w.role))
        if any(n is None for n in needs):
            return None
        if sum(needs) > capacity:
            return (
                f"{role} would need {sum(needs):.1f} GiB of GPU memory on this machine "
                f"with the GPU slots already there, and it has {capacity:.1f}"
            )
        return None

    def _gpu_capacity(self, kind: str, machine, host) -> float | None:
        """GiB of GPU memory slots may use on the target: a pool machine's
        memory per GPU, or a rented type's catalog figure;
        None when unknown (a bare host or registered machine not in the pool)."""
        target = _slot_target(kind, machine, host)
        for m in self.pool_store.load().machines:
            if _is_pool_machine(m, target):
                return m.gpu_capacity_gb
        if machine is not None and machine.instance_type:
            mtype = next((t for t in AWS_CATALOG if t.id == machine.instance_type), None)
            return mtype.gpu_memory_gb if mtype is not None else None
        return None

    def add_local(
        self,
        spec,
        task: tasks.TaskRecord,
        role: str,
        threads: int | None,
        *,
        check_gpu: bool = True,
    ) -> tasks.WorkerRecord:
        """A local slot. `check_gpu` False skips the GPU-fit check, for the tag
        queue, whose placement has already made it (with any override)."""
        self._check_role(spec, task, role, "local", check_gpu=check_gpu)
        w = tasks.WorkerRecord(
            worker_id=_next_worker_id(task, "local"),
            role=role,
            kind="local",
            desired_state="paused",
            threads=threads or default_thread_count(),
        )
        task.workers.append(w)
        self._rehome(spec, task, w)
        self.tasks.save(spec, task)
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
        check_gpu: bool = True,
    ) -> tasks.WorkerRecord:
        """An ssh slot on a bare host string, or on one of the task's
        machines by name (exactly one of the two). `check_gpu` as for
        add_local."""
        assert (host is None) != (machine is None), "an ssh slot names a host or a machine"
        record = self._machine_record(task, machine) if machine is not None else None
        self._check_role(spec, task, role, "ssh", machine=record, host=host, check_gpu=check_gpu)
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
        self._rehome(spec, task, w)
        self.tasks.save(spec, task)
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
        """Register a machine the operator prepared by hand (see
        docs/master_dashboard.md) for the task's ssh slots. Nothing here
        contacts it."""
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
        self.tasks.save(spec, task)
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
        """Launch an instance of `type_id` and record it as a task machine
        named `name` (or a generated name). A provider refusal (zero quota, no
        capacity) reaches the form as the provider's explanation, and nothing
        is recorded."""
        provider = self._provider()
        name = name or _next_machine_name(task, provider.name)
        assert all(m.name != name for m in task.machines), f"machine '{name}' exists"
        mtype = next((t for t in provider.catalog() if t.id == type_id), None)
        assert mtype is not None, f"no machine type '{type_id}'"
        try:
            inst = provider.launch(LaunchRequest(type_id, _owner(spec, task.tag, name), spot=spot))
        except ProviderError as e:
            raise AssertionError(provider.refusal(e, type_id)) from e
        # Add it to the cached listing now; otherwise the machine reads `gone`
        # until the next pass relists.
        self._instances[0][inst.id] = inst
        m = tasks.MachineRecord(name=name, provider=provider.name, host="")
        known_hosts = MACHINES_DIR / spec.name / task.tag / name / "known_hosts"
        _record_instance(m, provider, inst, mtype, spot=spot, known_hosts=known_hosts)
        task.machines.append(m)
        self.tasks.save(spec, task)
        return m

    def remove_machine(self, spec, task: tasks.TaskRecord, name: str):
        """Remove a machine and its slots, terminating it if rented. Each slot
        goes through remove_worker's checks, so a machine is never dropped from
        under a working container; the slots of a gone instance are removed
        outright, since their containers went with its disk."""
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
        self.tasks.save(spec, task)

    def _machine_gone(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> bool:
        return (
            w.machine is not None
            and self._machine_states.get(_machine_key(spec, task.tag, w.machine)) == "gone"
        )

    def _list_fleet(self):
        """The pass's fleet step: list every instance the provider tagged
        ours, whether or not a task names it. The per-task step lists only
        for tasks with rented machines, so without this an instance whose
        task.json lost its record would bill invisibly. A failure is kept for
        the burn strip and printed only when it changes."""
        try:
            self._instance_index(observe=True)
            error = None
        except Exception as e:  # noqa: BLE001 -- the fleet step must not stop the pass
            error = str(e)
        if error != self._fleet_error and error is not None:
            print(f"fleet listing: {error}")
        self._fleet_error = error

    def fleet(self) -> dict:
        """The burn strip: every live instance tagged ours with its hourly rate,
        and the total rate of those billing now (pending or running). Read
        from the last listing; `observed_at` lets the strip flag a listing
        that has stopped refreshing."""
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
        """An instance's hourly rate: its task record's (the only place a spot
        rate, known at launch, is kept), else the listing's, else its type's
        catalog rate, else None."""
        if record is not None and record.cost_per_hr is not None:
            return record.cost_per_hr
        if inst.cost_per_hr is not None:
            return inst.cost_per_hr
        mtype = next((t for t in self._provider().catalog() if t.id == inst.type_id), None)
        return mtype.cost_per_hr if mtype is not None else None

    def _owned(self) -> dict[str, tasks.MachineRecord]:
        """Every task's machines and every pool rental, by the ownership tag
        each carries."""
        owned = {
            _owner(spec, task.tag, m.name): m
            for spec, task in self.all_tasks()
            for m in task.machines
        }
        for m in self.pool_store.load().machines:
            if m.capacity is not None:
                owned[pool_mod.owner_tag(m.name)] = m.machine
        return owned

    def orphans(self, observe: bool = False) -> list[dict]:
        """Instances tagged ours that no task machine names. They are shown with
        a Terminate button but never terminated automatically: a task.json
        restored from an older copy must not kill a running experiment."""
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

    def task_rentals(self) -> list[tuple]:
        """(spec, task, machine) for every task-owned rented machine whose
        instance bills or may: not yet known stopped."""
        states = {i.id: i.state for i in self._instance_index(False).values()}
        return [
            (spec, task, m)
            for spec, task in self.all_tasks()
            for m in task.machines
            if m.instance_id is not None and states.get(m.instance_id) in ("pending", "running")
        ]

    def stop_task_rentals(self) -> list[str]:
        """Pause every slot on a task-owned rented machine and stop the machine
        as soon as its slots are down (Stop all cloud spending). Stopped keeps
        its disk, and the task's data on it, for a later Start. Returns
        "<workload>/<tag>/<machine>" of each."""
        out = []
        for spec, task, m in self.task_rentals():
            for w in task.slots_on(m.name):
                w.desired_state = "paused"
            self.tasks.save(spec, task)
            self._stop_now.add(_machine_key(spec, task.tag, m.name))
            out.append(f"{spec.name}/{task.tag}/{m.name}")
        return out

    def terminate_orphan(self, instance_id: str):
        inst = self._instance_index(False).get(instance_id)
        assert inst is not None, f"no instance {instance_id} in the last listing"
        self._terminate(instance_id, inst.type_id)
        self._instances = ({}, 0.0)  # relisted next pass

    def _terminate(self, instance_id: str, type_id: str | None):
        """Terminate through the provider; a refusal reaches the operator as
        the provider's explanation, as for a launch."""
        provider = self._provider()
        try:
            provider.terminate(instance_id)
        except ProviderError as e:
            raise AssertionError(provider.refusal(e, type_id or "instance")) from e
        # Mark it terminated in the cached listing now; until the next pass it
        # would otherwise read as a running orphan.
        cached = self._instances[0].get(instance_id)
        if cached is not None:
            cached.state = "terminated"

    def machine_status(self, spec, task: tasks.TaskRecord, *, observe: bool = False) -> list[dict]:
        """One dict per machine: the record plus its display state. A registered
        machine's state is its ssh probe (`up`, `preparing`, `no docker`,
        `unreachable`; `checking` before the first pass); a rented one's is
        _rented_state. Only the reconcile pass observes, and records what it
        learns: a moved host, the spend accrued, the state the slots read. A
        status request changes nothing; it shows spend advanced to now."""
        out = []
        leased = self._leased_records(task)
        rented = any(m.instance_id is not None for m in [*task.machines, *leased])
        index = self._instance_index(observe) if rented else {}
        for m in [*task.machines, *leased]:
            pooled = any(m is x for x in leased)
            key = _machine_key(spec, task.tag, m.name)
            inst = index.get(m.instance_id) if m.instance_id is not None else None
            moved = _moved_host(m, inst) if observe else None
            if moved is not None:
                m.host = moved
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
                if observe and not pooled:  # a pool rental accrues in the pool's own step
                    _accrue_machine(m, inst is not None and inst.state in ("pending", "running"))
            if observe:
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
                "spend": m.spend_now(time.time()),
                "state": state,
                "slots": [w.worker_id for w in task.slots_on(m.name)],
                # A pool machine this task leases: the pool, not the task,
                # owns it, so the task view offers no Remove.
                "pool": pooled,
            }
            reason = self._exits.get(key)
            if reason:
                info["exit_reason"] = reason  # why the last start was refused
                _, next_at = self._restarts.get(key, (0, 0.0))
                info["retry_in_s"] = max(0, int(next_at - time.time()))
            out.append(info)
        if observe and rented:
            self.tasks.save(spec, task)
        return out

    def _reconcile_machines(self, spec, task: tasks.TaskRecord, status: list[dict]):
        """Drive each rented machine toward what its slots want: start a
        stopped instance a slot wants running (a refused start backs off, with
        its reason on the machine's row), and stop one on which nothing has
        run for IDLE_STOP_SECONDS.

        Idleness is read from the slots' remembered probes, so a machine whose
        slots are all operator-paused or finished (exited containers) stops. A
        slot that wants to run counts as busy even with no container yet, and
        so does a gated one: a gate is expected to lift. A role that is done
        is finished by its scheduler (SchedulerHooks.finish) instead."""
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
                    self.tasks.save(spec, task)
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
            if key in self._stop_now or time.time() - since >= IDLE_STOP_SECONDS:
                provider = provider or self._provider()
                provider.stop(m.instance_id)
                self._idle_since.pop(key, None)
                self._stop_now.discard(key)
                self._instances = ({}, 0.0)

    def _observe_machine(self, m: tasks.MachineRecord) -> str:
        """Probe a machine, skipping a host that failed within
        SSH_REPROBE_SECONDS, as for its slots."""
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
            w.failed = None
            self._crashes.pop(_key(spec, task.tag, worker_id), None)
        self.tasks.save(spec, task)
        start = run and w.role not in task.gates  # a gated slot starts when released
        if w.kind == "local":
            if start and not self._local_alive(spec, task, w):
                self._spawn_local(spec, task, w)
            elif not run:
                self._stop_local(spec, task, w)
        else:
            # Observe afresh: a remembered state could send the wrong command,
            # or none for a slot no pass has reached. An unreachable machine
            # gets no command; reconcile enforces the saved desired state once
            # it answers.
            probe = self._refresh_probe(spec, task, w)
            name = _container_name(spec, task.tag, w.worker_id)
            key = _key(spec, task.tag, worker_id)
            # Each command makes the probe just taken stale: the next pass
            # must observe afresh, or it would repeat the command (a second
            # `docker run` fails on the name now in use).
            if start and probe == "stopped":
                self._expire_probe(key)
                self._ssh_machine(task, w).start_container(name)
            elif start and probe == "missing":
                self._expire_probe(key)
                self._run_ssh_container(spec, task, w)
            elif not run and probe == "running":
                self._expire_probe(key)
                self._ssh_machine(task, w).stop_container(name)

    def set_data_plane(self, spec, task: tasks.TaskRecord, data_plane: str):
        """Move the tag's generation data plane (TaskRecord.data_plane). Only
        while every slot is stopped, so the old and new schedulers never run
        side by side, and only for a workload whose scheduler has a data home."""
        assert spec.scheduler == TICK_FOR_TASK, f"workload '{spec.name}' has no data home"
        assert data_plane in (DATA_PLANE_LEGACY, DATA_PLANE_HOME), f"no data plane '{data_plane}'"
        running = [w.worker_id for w in task.workers if w.desired_state == "running"]
        assert not running, f"pause every slot first ({', '.join(running)} still set to run)"
        alive = [w.worker_id for w in task.workers if self._seen_alive(spec, task, w)]
        assert not alive, f"wait for every slot to stop ({', '.join(alive)} still alive)"
        assert data_plane == DATA_PLANE_HOME or self._remote_data_home(spec, task) is None, (
            "a data home on an ssh machine cannot go back to legacy: the bucket holds "
            "generations it numbered, which the controller's scheduler would reuse"
        )
        for w in task.workers:
            if w.kind == "ssh" and not self._machine_gone(spec, task, w):
                self._discard_container(spec, task, w)
        if any(w.kind == "ssh" for w in task.workers):
            task.bundle_id = None  # built afresh at the next start, with data home support
        task.data_plane = data_plane
        task.gates.pop(GENERATE_ROLE, None)  # the new scheduler decides afresh
        self.tasks.save(spec, task)

    def _discard_container(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord):
        """Remove stopped ssh slot `w`'s container, collecting what it holds
        first, so its next start creates one with the sinks and mount it now
        gets."""
        probe = self._refresh_probe(spec, task, w)
        assert probe in ("stopped", "missing"), f"{w.worker_id} is {probe}"
        if probe == "stopped":
            machine = self._ssh_machine(task, w)
            self._sweep_ssh(machine, spec, task, w)
            machine.remove_container(_container_name(spec, task.tag, w.worker_id))
            # The cached probe still says "stopped": a Start acting on it
            # within OBSERVATION_TTL_SECONDS would `docker start` a container
            # that is gone.
            self._expire_probe(_key(spec, task.tag, w.worker_id))
        w.launched = False

    def remove_worker(self, spec, task: tasks.TaskRecord, worker_id: str):
        """Remove a slot. Its worker must not be running, so a removal never
        silently discards an in-flight cycle."""
        w = task.worker(worker_id)
        if w.kind == "local":
            assert not self._local_alive(spec, task, w), f"{worker_id} is running; pause it first"
        elif w.kind == "ssh" and self._machine_gone(spec, task, w):
            pass  # its container went with the instance's disk; nothing to check or clean
        elif w.kind == "ssh":
            # Observe afresh: a removal must not act on a remembered state.
            probe = self._refresh_probe(spec, task, w)
            assert probe not in ("running", "paused"), f"{worker_id} is running; pause it first"
            # Removing while unreachable could orphan a live container that
            # keeps generating into the tag with nothing tracking it.
            assert probe != "unreachable", (
                f"{self._ssh_host(task, w)} is unreachable; bring it online (or clean up its "
                f"container by hand) before removing {worker_id}"
            )
            if probe == "stopped":
                machine = self._ssh_machine(task, w)
                if spec.role(w.role).ingest:
                    # A trainer's last flush holds its final state pair, which
                    # no collection reached; it is small, unlike a generator's
                    # backlog (the Remove dialog warns about that).
                    self._sweep_ssh(machine, spec, task, w)
                machine.remove_container(_container_name(spec, task.tag, w.worker_id))
        if task.data_plane == DATA_PLANE_HOME and w.kind == "ssh":
            self._release_tag_volume(spec, task, w)
        self._forget_slot(_key(spec, task.tag, worker_id))
        task.workers.remove(w)
        self.tasks.save(spec, task)

    def _release_tag_volume(self, spec, task: tasks.TaskRecord, leaving: tasks.WorkerRecord):
        """Remove a data-home tag's volume from the machine ssh slot `leaving`
        is about to leave, when no other slot of the tag stays there to mount
        it. Keyed on the machine rather than on the trainer, which may already
        be gone; removing a volume that was never made is harmless."""
        if self._machine_gone(spec, task, leaving):
            return  # the volume went with the instance's disk
        if not any(_same_machine(w, leaving) for w in task.workers if w is not leaving):
            self._ssh_machine(task, leaving).remove_volume(_tag_volume(spec, task))

    def delete_task(self, spec, tag: str):
        """Delete a tag: remove its worker slots, then its local dir.

        Idle slots are removed on the operator's behalf, since removal is what
        releases a container and the task record about to go is the only
        thing tracking it. A running slot refuses, so a working fleet is never
        deleted from under itself.

        Slots are removed one at a time, so a refusal partway leaves the
        earlier ones gone. The desired-state check up front catches the usual
        case (the fleet was not paused) before anything is removed; only a
        paused slot whose process is still alive is discovered midway.
        """
        task = self.tasks.load(spec, tag)
        if task is not None:
            running = [w.worker_id for w in task.workers if w.desired_state == "running"]
            assert not running, f"pause {', '.join(running)} first"
            for w in list(task.workers):
                self.remove_worker(spec, task, w.worker_id)
        self.tasks.delete(spec, tag)

    # ---- the machine pool (dashboard/pool.py) -----------------------------

    def add_pool_machine(
        self,
        name: str,
        host: str | None = None,
        *,
        identity_file: str | None = None,
        aliases: list[str] | None = None,
        generator_threads: int | None = None,
    ) -> pool_mod.PoolMachine:
        """Add a machine to the pool: this one (no `host`) or a registered ssh
        machine, whose hardware is probed now so eligibility never has to
        guess. A registered machine must already be prepared as for any ssh
        slot (docs/master_dashboard.md)."""
        pool = self.pool_store.load()
        assert name, "a pool machine needs a name"
        assert pool.find(name) is None, f"pool machine '{name}' exists"
        if host is None:
            assert all(m.kind != "local" for m in pool.machines), "this machine is already pooled"
            m = pool_mod.PoolMachine(name=name, kind="local", hardware=pool_mod.local_hardware())
        else:
            record = tasks.MachineRecord(
                name=name, provider="manual", host=host, identity_file=identity_file or None
            )
            hardware = pool_mod.parse_hardware(_machine_link(record).hardware_report())
            record.gpu_count = hardware.gpu_count
            m = pool_mod.PoolMachine(name=name, kind="ssh", machine=record, hardware=hardware)
        m.aliases = list(aliases or [])
        m.generator_threads = generator_threads
        pool.machines.append(m)
        self.pool_store.save(pool)
        return m

    def edit_pool_machine(self, name: str, **changes):
        """Change a pool machine's operator-set fields (aliases, generator
        threads)."""
        editable = {"aliases", "generator_threads"}
        assert set(changes) <= editable, f"not editable: {sorted(set(changes) - editable)}"
        pool = self.pool_store.load()
        m = pool.machine(name)
        for key, value in changes.items():
            setattr(m, key, value)
        self.pool_store.save(pool)

    def reprobe_pool_machine(self, name: str):
        """Re-read a pool machine's hardware (after a GPU swap, say)."""
        pool = self.pool_store.load()
        m = pool.machine(name)
        if m.machine is None:
            m.hardware = pool_mod.local_hardware()
        else:
            m.hardware = pool_mod.parse_hardware(_machine_link(m.machine).hardware_report())
            m.machine.gpu_count = m.hardware.gpu_count
        self.pool_store.save(pool)

    def remove_pool_machine(self, name: str):
        """Take a machine out of the pool, terminating it if the pool rented
        it. Refused while a tag leases it or any slot names it: those slots
        would lose their machine, and every lookup of it (the pool page, the
        reconcile pass) would fail."""
        pool = self.pool_store.load()
        m = pool.machine(name)
        assert m.lease is None, f"{name} is leased by {m.lease.workload}/{m.lease.tag}"
        naming = [
            f"{spec.name}/{task.tag}/{w.worker_id}"
            for spec, task in self.all_tasks()
            for w in task.workers
            if w.machine == name and task.find_machine(name) is None
        ]
        assert not naming, f"slots still name {name}: {', '.join(naming)}; remove them first"
        if m.capacity is not None and m.machine.instance_id is not None:
            self._terminate(m.machine.instance_id, m.machine.instance_type)
            _accrue_machine(m.machine, False)
        pool.machines.remove(m)
        self.pool_store.save(pool)

    def add_capacity(self, name: str, instance_type: str, *, spot: bool, cap: int):
        """Let the pool rent up to `cap` instances of `instance_type`
        (pool.Capacity). The name also prefixes its machines' names and owner
        tags, so it is restricted to letters, digits and underscores."""
        assert name and all(c.isalnum() or c == "_" for c in name), (
            "a capacity name uses only letters, digits and underscores"
        )
        assert cap >= 1, "the cap is at least 1"
        assert any(t.id == instance_type for t in self._provider().catalog()), (
            f"no machine type '{instance_type}'"
        )
        pool = self.pool_store.load()
        assert all(c.name != name for c in pool.capacity), f"capacity '{name}' exists"
        assert pool.find(name) is None, f"'{name}' names a pool machine"
        pool.capacity.append(pool_mod.Capacity(name, instance_type, spot, cap))
        self.pool_store.save(pool)

    def set_capacity_cap(self, name: str, cap: int):
        """Change a capacity entry's cap. Lowering it rents no more; machines
        already rented finish their tags and are terminated once idle."""
        assert cap >= 0, "the cap is at least 0"
        pool = self.pool_store.load()
        entry = next((c for c in pool.capacity if c.name == name), None)
        assert entry is not None, f"no capacity '{name}'"
        entry.cap = cap
        self.pool_store.save(pool)

    def remove_capacity(self, name: str):
        """Stop renting under a capacity entry. Its rented machines stay until
        their tags finish and they idle out."""
        pool = self.pool_store.load()
        pool.capacity = [c for c in pool.capacity if c.name != name]
        self.pool_store.save(pool)

    def lease_spend(self, task: tasks.TaskRecord) -> float:
        """What the task's current leases of pool rentals have cost so far."""
        return sum(
            pool_mod.lease_spend(m)
            for m in self.pool_store.load().machines
            if m.lease is not None and (m.lease.workload, m.lease.tag) == (task.workload, task.tag)
        )

    def pool_status(self) -> list[dict]:
        """One dict per pool machine: its record, and `occupants`, the slots
        outside its lease that make it busy (see occupants). Reads only
        remembered observations, like every status request."""
        tasks_now = list(self.all_tasks())
        out = []
        for m in self.pool_store.load().machines:
            occupants = self.occupants(m, tasks_now)
            info = asdict(m)
            info["occupants"] = occupants
            info["state"] = "leased" if m.lease else "busy" if occupants else "free"
            out.append(info)
        return out

    def occupants(self, m: pool_mod.PoolMachine, tasks_now) -> list[str]:
        """The slots on pool machine `m`, outside the tag leasing it, that make
        it busy: `workload/tag/worker_id` of each _holds_machine counts.
        Matching an ssh slot to `m` goes by canonical host name, since tags
        spell one machine several ways."""
        out = []
        for spec, task in tasks_now:
            if m.lease and m.lease.held_by(spec.name, task.tag):
                continue
            for w in task.workers:
                if not _is_pool_machine(m, _slot_target(w.kind, None, self._slot_host(task, w))):
                    continue
                if self._holds_machine(spec, task, w):
                    out.append(f"{spec.name}/{task.tag}/{w.worker_id}")
        return out

    def _holds_machine(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> bool:
        """Whether slot `w` keeps the queue off its machine: it is alive, or
        gated (its gate will lift), or meant to run and down for less than
        DEAD_SLOT_GRACE_SECONDS. The grace covers a slot between restarts or
        just started. Past it, a slot meant to run that never comes up (an
        `exited` one) is dead weight, and holding a machine for it would idle
        the machine indefinitely. Paused and finished slots hold nothing."""
        if self._seen_alive(spec, task, w):
            return True
        if w.desired_state != "running":
            return False
        if w.role in task.gates:
            return True
        since = self._down_since.get(_key(spec, task.tag, w.worker_id))
        return since is None or time.time() - since < DEAD_SLOT_GRACE_SECONDS

    def _note_down(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord):
        """Start or clear slot `w`'s down clock, which _holds_machine's grace
        runs on: it starts when a slot meant to run is first seen down, and a
        gated slot's is left as it was."""
        key = _key(spec, task.tag, w.worker_id)
        if self._seen_alive(spec, task, w) or w.desired_state != "running":
            self._down_since.pop(key, None)
        elif w.role not in task.gates:
            self._down_since.setdefault(key, time.time())

    def _seen_alive(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> bool:
        """Whether slot `w`'s worker is alive, from what is already known: its
        pid for a local slot, its last probe for an ssh one."""
        if w.kind == "local":
            return worker_pid_alive(w.pid, w.worker_id, task.tag)
        probe, _ = self._probes.get(_key(spec, task.tag, w.worker_id), ("unknown", 0.0))
        return probe in ("running", "paused")

    # ---- observation -----------------------------------------------------

    def worker_status(self, spec, task: tasks.TaskRecord, *, observe: bool = False) -> list[dict]:
        """One dict per slot: the durable record plus observed live state.

        Only the reconcile pass passes `observe`: it probes the machines and
        records what it learns (_observe_slots). Every other caller, such as a
        browser's status poll, reads those observations and changes nothing,
        so serving the dashboard never waits on ssh.
        """
        seen = self._observe_slots(spec, task) if observe else {}
        out = []
        for w in task.workers:
            # Shown only on a slot meant to run: pausing the trainer gates the
            # generators too, and a slot the operator paused must read paused,
            # not waiting to resume.
            gated = w.role in task.gates and w.desired_state == "running"
            info = {
                "worker_id": w.worker_id,
                "role": w.role,
                "kind": w.kind,
                "desired_state": w.desired_state,
                "threads": w.threads,
                "host": self._ssh_host(task, w) if w.kind == "ssh" else None,
                "machine": w.machine,
                "bundle_id": w.bundle_id,
                "launched": w.launched,
                # Zero means drained, None means unknown; the Remove dialog
                # tells them apart.
                "undelivered": w.undelivered,
                "failed": w.failed,
            }
            if gated:
                info["gate_reason"] = task.gates[w.role]
            if w.kind == "local":
                alive = seen[w.worker_id] if observe else self._local_alive(spec, task, w)
                info["state"] = _local_state(
                    w.desired_state, alive, gated, w.finished, w.failed is not None
                )
            else:
                probe = self._probe_container(spec, task, w)
                alive = probe == "running"
                info["state"] = _ssh_state(
                    w.desired_state, probe, gated, w.finished, w.failed is not None
                )
                info["ssh_probe"] = probe  # reconcile keys its enforcement off this
                info["ssh"] = f"ssh {self._ssh_host(task, w)}"
                reason = self._slot_reason(spec, task, w)
                if reason and not alive:
                    info["exit_reason"] = reason
            if w.failed and not alive:
                info["exit_reason"] = w.failed  # why the tag queue gave up on it
            # Real liveness, for reconcile's desired-vs-observed enforcement.
            info["observed_running"] = alive
            out.append(info)
        return out

    def _observe_slots(self, spec, task: tasks.TaskRecord) -> dict[str, bool]:
        """The pass's look at every slot, and what it records: each container's
        fresh probe (at most once per OBSERVATION_TTL_SECONDS), that a local worker which
        exited 0 finished, and since when a slot meant to run has been down
        (_holds_machine). Returns each local slot's liveness as observed here,
        for the pass to act on: a worker seen alive and then found gone
        without its exit read would be respawned though it finished. The
        local slots go last, after the seconds of ssh the probes take."""
        for w in task.workers:
            if w.kind == "ssh":
                self._refresh_probe(spec, task, w)
        seen = {}
        for w in task.workers:
            if w.kind == "local":
                proc = self._local.get(_key(spec, task.tag, w.worker_id))
                if proc is not None:
                    proc.poll()  # reap our own exited child, so it is not seen alive
                seen[w.worker_id] = self._local_alive(spec, task, w)
                if not seen[w.worker_id] and self._local_exit_code(spec, task, w) == 0:
                    _note_finished(w)
        for w in task.workers:
            self._note_down(spec, task, w)
        if task.workers:
            self.tasks.save(spec, task)
        return seen

    def _slot_reason(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> str | None:
        """Why ssh slot `w` is not running. While its machine is not up the
        reconcile pass leaves the slot alone, so the slot's own last reason (a
        bundle build long finished, say) goes stale; the machine's state, and
        why its last start was refused, is the answer then."""
        mkey = _machine_key(spec, task.tag, w.machine) if w.machine else None
        mstate = self._machine_states.get(mkey) if mkey else None
        if mstate is not None and mstate != "up":
            why = self._exits.get(mkey)
            return f"waiting for machine {w.machine} ({mstate}{': ' + why if why else ''})"
        return self._exits.get(_key(spec, task.tag, w.worker_id))

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
                self.tasks.save(spec, task)

        def finish(role: str):
            if _finish_role(task, role):
                self.tasks.save(spec, task)

        return SchedulerHooks(
            paths=self.tasks.paths(spec, task.tag),
            gate=gate,
            finish=finish,
            publish=self._make_publish(spec, task),
            role_running=lambda role: any(
                self._seen_alive(spec, task, w) for w in task.workers if w.role == role
            ),
        )

    def _make_publish(self, spec, task: tasks.TaskRecord):
        """The scheduler's publish hook: copy a completed generation to the
        bucket, where an ssh trainer on the legacy data plane reads it
        (docs/plans/cloud_training.md). None for a task without one, and for a
        data home, whose trainer reads its own tree.

        The copy runs on the upload thread, so the hook only starts it and
        reports whether it is done (SchedulerHooks.publish); a failed copy
        raises on the call that collects it, and the next call starts it
        again. Chunks already uploaded are skipped by size. The manifest goes
        last, so a manifest in the bucket means the whole generation is."""
        if task.data_plane == DATA_PLANE_HOME or not self._has_bucket_data(spec, task):
            return None
        try:
            creds = self._creds()
        except (CredentialsError, FileNotFoundError):
            return None
        r2 = creds.r2
        paths = self.tasks.paths(spec, task.tag)

        def upload(dest_rel: str):
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

        def publish(dest_rel: str) -> bool:
            key = (_key(spec, task.tag), dest_rel)
            future = self._publishing.get(key)
            if future is None:
                self._publishing[key] = self._uploads.submit(upload, dest_rel)
                return False
            if not future.done():
                return False
            del self._publishing[key]
            future.result()  # a failure raises; the next call uploads again
            return True

        return publish

    # ---- reconciliation ----------------------------------------------------

    def all_tasks(self):
        """(spec, task) for every task; changes nothing."""
        return self.tasks.load_all()

    def _forget_stale_counts(self, spec, task: tasks.TaskRecord):
        """The first time this process sees a task, forget its slots' recorded
        zero `undelivered` counts. The count is durable so a restart cannot
        forget a container holds hours of work, but a zero from before the
        restart says nothing about what workers produced while no dashboard
        was watching.
        """
        key = _key(spec, task.tag)
        if key in self._counted_from:
            return
        self._counted_from.add(key)
        for w in task.workers:
            if w.kind == "ssh":
                _forget_empty(w)
        self.tasks.save(spec, task)

    def claim_writer(self):
        """Make the blocking thread the only one that may write the stores:
        every change is then a command or a pass step run through offload,
        and a status read, served on the event loop, reads committed copies
        (control_store.py). The dashboard claims it at startup; a test that
        calls the manager from one thread need not."""
        self.control.writer.claim(self._blocking.submit(threading.current_thread).result())

    def import_json_stores(self) -> list[str]:
        """Move the control state kept in JSON files before the control store
        existed into it, once (docs/plans/dashboard_state_model.md §10):
        pool.json, queue.json, and every task.json's slots, machines, gates
        and spend, in one transaction. The two store files are then renamed
        aside (*.pre-control-db), kept for reference; task.json files lose
        their control fields on their next save. What it moved, one line
        each; empty once done."""
        if self.control.meta(IMPORTED_JSON) is not None:
            return []
        moved = []
        with self.control.transaction():
            for name, store, decode in (
                ("pool.json", self.pool_store, pool_mod.decode),
                ("queue.json", self.queue_store, queue_mod.decode),
            ):
                path = self.mount_root / name
                if path.is_file():
                    store.save(decode(json.loads(path.read_text())))
                    moved.append(name)
            moved += self.tasks.import_json()
            self.control.set_meta(IMPORTED_JSON, str(time.time()))
        for name in ("pool.json", "queue.json"):
            path = self.mount_root / name
            if path.is_file():
                path.rename(path.with_suffix(".pre-control-db.json"))
        return moved

    def export_json_stores(self) -> list[str]:
        """Undo import_json_stores, to run code from before the control store:
        write every task's whole record to its task.json, and the pool and
        queue to pool.json and queue.json, then clear the store's records and
        its import mark, so that a later start imports the JSON files afresh.
        For a stopped dashboard only (py/scripts/export_control_db.py). What
        it wrote, one line each."""
        written = []
        for spec, task in list(self.all_tasks()):
            self.tasks.export_json(spec, task)
            written.append(f"{spec.name}/{task.tag}")
        for name, store in (("pool.json", self.pool_store), ("queue.json", self.queue_store)):
            (self.mount_root / name).write_text(json.dumps(asdict(store.load()), indent=2) + "\n")
            written.append(name)
        self.control.clear()
        return written

    async def offload(self, fn, *args, **kwargs):
        """Run one blocking step off the event loop, one at a time.

        Everything reconcile does is blocking IO measured in seconds: ssh,
        rclone, the provider's API. Inline, it would freeze every request the
        dashboard serves (a scheduler gate flip alone can take 8 seconds). The
        executor has a single thread, so the steps stay serialized with each
        other, as they mutate the same task records.
        """
        return await IOLoop.current().run_in_executor(self._blocking, partial(fn, *args, **kwargs))

    async def reconcile(self):
        """One pass over every task. For each: run the workload's scheduler
        tick; start and stop rented machines as their slots want; collect from
        ssh slots; drive each worker toward its intent, respawning, parking or
        stopping it; run dispatch ticks (RoleSpec.dispatch) and ingest ticks
        (RoleSpec.ingest); keep the bucket sync and controls push current.

        Enforcement keys off real liveness (durable pid or container probe),
        so it holds a paused worker down even across a dashboard restart.

        This pass is the only observer: it refreshes the container probes and
        the instance listing that everything else reads, and it alone
        accrues rented machines' spend.
        """
        await self.offload(self._list_fleet)
        # Loaded on the writer, for its live records (control_store.py).
        for spec, task in await self.offload(lambda: list(self.all_tasks())):
            await self.offload(self._forget_stale_counts, spec, task)
            if spec.scheduler:
                try:
                    await self.offload(self._tick_scheduler, spec, task)
                except Exception as e:  # noqa: BLE001 -- scheduling must keep ticking
                    print(f"scheduler {spec.name}/{task.tag}: {e}")
            # An unobservable fleet (a failed provider listing, say) must not stop
            # enforcement for this tag's slots or any later tag's; its machines,
            # and every slot on one, are left alone this pass.
            try:
                machines = await self.offload(self.machine_status, spec, task, observe=True)
            except Exception as e:  # noqa: BLE001 -- see above
                print(f"machines {spec.name}/{task.tag}: {e}")
                machines = None
            if machines is not None:
                try:
                    await self.offload(self._reconcile_machines, spec, task, machines)
                except Exception as e:  # noqa: BLE001 -- one task's machines must not stop the pass
                    print(f"machines {spec.name}/{task.tag}: {e}")
            down = {m["name"] for m in machines or () if m["state"] != "up"}
            status = await self.offload(self.worker_status, spec, task, observe=True)
            for info in status:
                # A handler may have removed the slot between this pass's steps.
                w = task.find(info["worker_id"])
                if w is None:
                    continue
                if w.machine is not None and (machines is None or w.machine in down):
                    continue  # nothing on a machine not known to be up can be acted on
                # Collect before enforcing: this slot may be about to be
                # parked, and a pull needs its container running.
                if (
                    w.kind == "ssh"
                    and info["ssh_probe"] == "running"
                    and self._remote_data_home(spec, task) is w
                ):
                    try:
                        await self.offload(self._pull_scheduler_state, spec, task, w)
                    except Exception as e:  # noqa: BLE001 -- the gate then reads a stale record
                        print(f"scheduler state {spec.name}/{task.tag}/{w.worker_id}: {e}")
                if w.kind == "ssh" and info["ssh_probe"] == "running":
                    try:
                        await self.offload(self._collect_step, spec, task, w)
                    except Exception as e:  # noqa: BLE001 -- one slot must not stop the pass
                        print(f"collect {spec.name}/{task.tag}/{w.worker_id}: {e}")
                # Contained per slot: one slot's failure (a machine vanishing
                # mid-action, say) must not starve the rest of the pass, which
                # is the only enforcement some slots get.
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
                        await self.offload(
                            workloads.resolve(role.ingest), spec, self.tasks.paths(spec, task.tag)
                        )
                    except Exception as e:  # noqa: BLE001 -- one role must not stop the pass
                        print(f"ingest {spec.name}/{task.tag}/{role.name}: {e}")
            try:
                await self.offload(self._push_controls, spec, task, status)
            except Exception as e:  # noqa: BLE001 -- one task must not stop the pass
                print(f"controls push {spec.name}/{task.tag}: {e}")

    def _collect_step(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord):
        """The pass's collection for slot `w` (blocking thread): record the
        last pull if it has finished, and start the next on the slot's transfer
        thread unless one is still running or the slot is about to be parked
        or replaced, which _reconcile_ssh does only once no pull is in flight
        (a stop needs no wait: a pull it cuts short is the benign stop race
        _pull_ssh describes). The bytes move off this thread; the slot's record
        changes only here."""
        key = _key(spec, task.tag, w.worker_id)
        running = self._collecting.get(key)
        if running is not None and not running.done():
            return
        if running is not None:
            self._record_pull(spec, task, w, self._collecting.pop(key))
        if _intent(w, task) == PARK or _replaceable(w, task):
            return
        pool = self._transfer_pools.setdefault(key, self._new_transfer_pool())
        future = pool.submit(self._transfer_ssh, spec, task, w)
        if future.done():  # a synchronous pool (the simulation's)
            self._record_pull(spec, task, w, future)
        else:
            self._collecting[key] = future

    def _pulling(self, key: str) -> bool:
        """Whether slot `key` has a collection in flight."""
        running = self._collecting.get(key)
        return running is not None and not running.done()

    def _record_pull(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord, done: Future):
        """Record a finished pull's count of what the container still holds.
        A failed pull, or one the container stopped under, leaves the count
        unknown: a failing collection (a link too slow for the transfer timeout
        while probes still pass) must not leave an old zero claiming "drained"
        while the container fills up."""
        try:
            result = done.result()
        except Exception as e:  # noqa: BLE001 -- the next pass pulls again
            print(f"collect {spec.name}/{task.tag}/{w.worker_id}: {e}")
            result = None
        w.undelivered = None if result is None else result.remaining
        self.tasks.save(spec, task)

    def _transfer_ssh(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord):
        """One collection for slot `w`, on its transfer thread: the pull, then,
        when `w` is the trainer of a remote data home, the relay of chunks
        staged here into it (_relay_staging). Returns the pull's result."""
        result = self._pull_ssh(spec, task, w)
        if result is not None and self._remote_data_home(spec, task) is w:
            self._relay_staging(self._ssh_machine(task, w), spec, task, w)
        return result

    def _pull_ssh(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord):
        """Pull a batch of slot `w`'s finished output from its container into
        the tag, installing any state pairs it brought; None when the container
        stopped under it. Touches only files, so it runs off the blocking
        thread.

        A pull can race an operator's pause, which stops the container
        between the pass's probe and the pull. That is benign: if a re-probe
        finds the container stopped or gone, what it flushed on the way down
        waits for the next start, or for the sweep before a replacement. Any
        other failure propagates."""
        machine = self._ssh_machine(task, w)
        try:
            result = pull_results(machine, **self._transfer_target(spec, task, w))
        except SshMachineError:
            name = _container_name(spec, task.tag, w.worker_id)
            if machine.container_state(name) not in ("stopped", "missing"):
                raise
            return None
        self._install_pulled_pairs(spec, task, result.pulled)
        return result

    def _relay_staging(self, machine, spec, task: tasks.TaskRecord, home: tasks.WorkerRecord):
        """Push a batch of the chunks staged here into the staging of the tag's
        data home on an ssh machine, whose scheduler assigns them: what the
        tag's generators elsewhere delivered, a local slot's directly and an
        ssh slot's through collection. It runs after each pull from the home's
        trainer (_transfer_ssh), so only while that container runs, and the
        chunks wait here while it does not. A failed relay keeps them here for the next one; it
        must not fail the pull before it, whose output is already in place."""
        paths = self.tasks.paths(spec, task.tag)
        try:
            relay_files(
                machine,
                _container_name(spec, task.tag, home.worker_id),
                remote_root=str(paths.root),
                local_root=paths.root,
                rel=str(paths.staging_dir.relative_to(paths.root)),
            )
        except SshMachineError as e:
            print(f"relay {spec.name}/{task.tag}/{home.worker_id}: {e}")

    def _install_pulled_pairs(self, spec, task: tasks.TaskRecord, pulled: list[str]):
        """Install the state pairs a pull brought in, oldest first, each under
        the cursor rule; a pair that loses (or arrived torn) is discarded."""
        paths = self.tasks.paths(spec, task.tag)
        pairs = sorted(
            {"/".join(n.split("/")[:2]) for n in pulled if n.startswith(f"{state_pair.STATE_DIR}/")}
        )
        for rel in pairs:
            state_pair.install(paths.root / rel, paths)

    def _pull_scheduler_state(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord):
        """Copy a remote data home's scheduler record (its gate and heartbeat)
        into the tag here, in a small read of its own each pass, so a slow
        bulk collection never makes the heartbeat look dead."""
        paths = self.tasks.paths(spec, task.tag)
        record = self._ssh_machine(task, w).read_from_container(
            _container_name(spec, task.tag, w.worker_id),
            [
                "sh",
                "-c",
                f"cat {shlex.quote(str(paths.root / SCHEDULER_STATE_REL))} 2>/dev/null || true",
            ],
        )
        if record.strip():
            tmp = paths.root / f".{SCHEDULER_STATE_REL}.tmp"
            tmp.write_bytes(record)
            os.replace(tmp, paths.root / SCHEDULER_STATE_REL)

    def _dispatch_role(self, spec, task: tasks.TaskRecord, role, status: list[dict]):
        """Run one role's dispatch tick: hand its running slots their next piece
        of work, and take in what they have delivered. Once the tick reports
        nothing outstanding and the task's trainer has finished, finish the
        role, so an idle worker does not hold its rented machine up forever.

        Only running slots are offered, since a paused container cannot be
        written to. The tick runs even with no slots, because results already
        collected must still reach the database.
        """
        slots = [
            self._slot_files(spec, task, w)
            for info in status
            if (w := task.find(info["worker_id"])) is not None
            and w.role == role.name
            and info["observed_running"]
        ]
        params = params_mod.validate(spec.params_cls, task.params)
        outstanding = workloads.resolve(role.dispatch)(
            spec, self.tasks.paths(spec, task.tag), params, slots
        )
        if not outstanding and _trainer_finished(spec, task) and _finish_role(task, role.name):
            self.tasks.save(spec, task)

    def _slot_files(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord):
        """The way into slot `w`'s filesystem (dashboard/slot_files.py)."""
        paths = self.tasks.paths(spec, task.tag)
        assert w.kind in ("local", "ssh"), (
            f"a {w.kind} slot has no reachable filesystem; a dispatch-driven role's "
            "kinds are local and ssh"
        )
        if w.kind == "ssh":
            return SshSlotFiles(
                w.worker_id,
                self._ssh_machine(task, w),
                _container_name(spec, task.tag, w.worker_id),
                str(paths.root),
            )
        return LocalSlotFiles(w.worker_id, paths.root)

    def _tick_scheduler(self, spec, task: tasks.TaskRecord):
        workloads.resolve(spec.scheduler)(spec, task, self._scheduler_hooks(spec, task))

    def _reconcile_worker(self, spec, task: tasks.TaskRecord, w, intent: str, info: dict):
        """Close one slot's desired-vs-observed gap. A failure skips only this
        slot; enforcement resumes on the next pass."""
        alive = info["observed_running"]
        if w.kind == "local":
            # A local worker restarts in about a second, so parking it and
            # stopping it are the same thing.
            if intent == RUN and not alive:
                code = self._local_exit_code(spec, task, w)
                if _is_crash(code):
                    self._note_crash(_key(spec, task.tag, w.worker_id), f"exit {code}")
                self._spawn_local(spec, task, w)
            elif intent != RUN and alive:
                self._stop_local(spec, task, w)
        else:
            self._reconcile_ssh(spec, task, w, intent, info["ssh_probe"])

    def _reconcile_ssh(self, spec, task: tasks.TaskRecord, w, intent: str, probe: str):
        """Enforce one ssh slot's intent. An unreachable or unobserved container
        gets no command.

        A gate parks the container by pausing it rather than stopping it. A
        stop loses the in-flight chunk, and the next start re-runs the image's
        bootstrap (fetching and unpacking the bundle), which under a gate that
        flips every minute costs about a third of the machine's time. A pause
        is instant both ways and resumes mid-chunk.
        """
        machine = self._ssh_machine(task, w)
        name = _container_name(spec, task.tag, w.worker_id)
        key = _key(spec, task.tag, w.worker_id)
        if intent == RUN:
            if probe == "paused":
                self._expire_probe(key)
                machine.unpause_container(name)  # resuming a parked worker is not a restart
            elif probe == "running":
                if _replaceable(w, task) and not self._pulling(key):
                    # The task has redeployed past this container and it holds
                    # nothing: stop it, and the next pass replaces it. As with
                    # any stop, the cycle in flight is lost.
                    self._expire_probe(key)
                    machine.stop_container(name)
            elif probe in ("missing", "stopped") and self._restart_allowed(key):
                # Unreachable or unobserved: nothing this pass can act on.
                self._note_restart(key)
                why = self._exits.get(key, "")
                if probe == "stopped" and _is_ssh_crash(why):
                    self._note_crash(key, why)
                self._expire_probe(key)
                self._start_or_replace(machine, name, spec, task, w, probe)
        elif intent == PARK and probe == "running" and not self._pulling(key):
            # A pause would freeze a pull in flight until it times out, so the
            # park waits for it; _collect_step starts no further pull.
            self._expire_probe(key)
            machine.pause_container(name)
        elif intent == STOP and probe in ("running", "paused"):
            self._expire_probe(key)
            if probe == "paused":
                machine.unpause_container(name)  # docker stop cannot signal a frozen process
            machine.stop_container(name)

    def _expire_probe(self, key: str):
        """Make the next pass observe slot `key`'s container afresh, before a
        command changes its state. The pass cadence matches the probe TTL, so
        without this the next pass could read the probe from before the
        command and repeat it: a second pause fails with "already paused"."""
        probe, _ = self._probes.get(key, ("unknown", 0.0))
        self._probes[key] = (probe, 0.0)

    def _start_or_replace(self, machine, name, spec, task: tasks.TaskRecord, w, probe: str):
        """Bring a container that is not running back up, replacing it when
        _replaceable allows.

        A container's bundle is fixed at creation, so a slot moves to its
        task's new bundle only by being replaced. Replacing destroys whatever
        the container has not handed over, so a container holding output is
        restarted instead, which lets later passes drain it.

        A container that will not stay up is never collected from, so its last
        count stands. Unless that count is zero, the slot stays on its old
        bundle, restarting and visibly down: recovering it would discard an
        amount nothing can measure any more, which is the operator's call
        (Remove says what would go).
        """
        if probe == "stopped" and not _replaceable(w, task):
            machine.start_container(name)
            return
        if probe == "stopped":
            # Collect what it flushed on the way down before the container
            # goes. If that fails, so does the replacement: staying on the old
            # bundle is recoverable, losing the output is not.
            self._sweep_ssh(machine, spec, task, w)
            machine.remove_container(name)
        self._run_ssh_container(spec, task, w)

    def _sweep_ssh(self, machine, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord):
        """Collect from a stopped container, the last chance to do so: what it
        flushed on the way down, a trainer's final state pair included, which
        is installed under the cursor rule."""
        pulled = sweep_stopped(machine, **self._transfer_target(spec, task, w))
        self._install_pulled_pairs(spec, task, pulled)

    def shutdown(self):
        """SIGTERM this process's local workers (they flush and exit) and sync
        watchers. ssh containers keep running across a dashboard restart."""
        self._blocking.shutdown(wait=False, cancel_futures=True)
        self._builds.shutdown(wait=False, cancel_futures=True)
        self._uploads.shutdown(wait=False, cancel_futures=True)
        for pool in self._transfer_pools.values():
            pool.shutdown(wait=False, cancel_futures=True)
        for proc in self._local.values():
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)

    # ---- a slot's machine and delivery (read through the pool store) -----

    def _slot_records_sink(
        self, spec: workloads.WorkloadSpec, task: tasks.TaskRecord, w: tasks.WorkerRecord
    ) -> str:
        """Where slot `w`'s worker sends its records (SCZ_SINK, cloud/sinks.py):
        stats, params, a trainer's records, exports and state pairs. Always
        "local": a local subprocess writes the tag tree here, and an ssh
        container writes its own, which the reconcile pass collects over ssh
        (cloud/ssh_transfer.py; _transfer_target says what is taken). The
        controller is the hub every record reaches it through."""
        return "local"

    def _slot_data_sink(
        self, spec: workloads.WorkloadSpec, task: tasks.TaskRecord, w: tasks.WorkerRecord
    ) -> str:
        """Where slot `w`'s worker delivers into and reads from the tag's data/
        store (SCZ_DATA_SINK, cloud/sinks.py). DATA_SINK_HOME on the machine of
        a tag's data home on an ssh machine. Otherwise "local": a local slot
        delivers into the tag tree here, and an ssh slot into its container,
        which the reconcile pass collects (and, for a data home elsewhere,
        relays on: _relay_staging). The one exception is an ssh trainer on the
        legacy data plane, which reads the generations the controller
        publishes to the bucket."""
        if self._on_remote_data_home(spec, task, w):
            return DATA_SINK_HOME
        if w.kind == "ssh" and spec.role(w.role).ingest:
            return "r2"
        return "local"

    def _remote_data_home(self, spec, task: tasks.TaskRecord) -> tasks.WorkerRecord | None:
        """The trainer slot of a tag whose data plane runs beside it on an ssh
        machine; None for any other tag."""
        if task.data_plane != DATA_PLANE_HOME:
            return None
        trainer = _trainer_slot(spec, task)
        return trainer if trainer is not None and trainer.kind == "ssh" else None

    def _on_remote_data_home(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> bool:
        """Whether slot `w` works in its tag's data home on an ssh machine."""
        home = self._remote_data_home(spec, task)
        return home is not None and not spec.role(w.role).dispatch and _same_machine(w, home)

    def _has_bucket_data(self, spec: workloads.WorkloadSpec, task) -> bool:
        """Whether any slot reads the tag's data/ store from the bucket, which
        the scheduler's publish hook keeps in step with the local one."""
        return any(self._slot_data_sink(spec, task, w) == "r2" for w in task.workers)

    def _machine_record(self, task: tasks.TaskRecord, name: str) -> tasks.MachineRecord:
        """The machine `name` a slot of `task` runs on: one of the task's own, or
        a pool machine the task leases (dashboard/pool.py). Pool machines are
        resolved here rather than copied into the task, so the pool stays their
        one owner."""
        own = task.find_machine(name)
        if own is not None:
            return own
        record = pool_mod.leased_record(self.pool_store.load(), task.workload, task.tag, name)
        if record is None:
            raise KeyError(f"no machine '{name}'")
        return record

    def _leased_records(self, task: tasks.TaskRecord) -> list[tasks.MachineRecord]:
        """The records of the ssh pool machines `task` leases."""
        return [
            m.machine
            for m in self.pool_store.load().machines
            if m.machine is not None
            and m.lease is not None
            and m.lease.held_by(task.workload, task.tag)
        ]

    def _ssh_machine(self, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> SshMachine:
        """The ssh link to slot `w`'s machine: its machine record's address and
        key material, or its bare host string."""
        if w.machine is None:
            return SshMachine(w.host)
        return _machine_link(self._machine_record(task, w.machine))

    def _ssh_host(self, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> str:
        return w.host if w.machine is None else self._machine_record(task, w.machine).host

    def _slot_host(self, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> str | None:
        return self._ssh_host(task, w) if w.kind == "ssh" else None

    def _transfer_target(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord) -> dict:
        """Where slot `w`'s output lives on both machines, as the collection calls
        (a batched pull, a sweep of a stopped container) take it. The two roots
        read alike, since the container uses the controller's layout, but they are
        paths on different machines."""
        paths = self.tasks.paths(spec, task.tag)
        target = {
            "container": _container_name(spec, task.tag, w.worker_id),
            "remote_root": str(paths.root),
            "local_root": paths.root,
            # A data home's staging is its own scheduler's to take.
            "data_dirs": []
            if self._slot_data_sink(spec, task, w) == DATA_SINK_HOME
            else [f"data/{sub}" for sub in spec.collected_dirs],
        }
        if spec.role(w.role).ingest:
            target["data_dirs"] += list(TRAINER_OUTPUT_DIRS)
            target["pair_dirs"] = {state_pair.STATE_DIR: state_pair.CURSOR_NAME}
        return target

    def _seed_state(self, machine, container: str, spec, task: tasks.TaskRecord):
        """Push the controller's checkpoint and cursor into a freshly created
        trainer container as a state pair (state_pair.SEED_DIR), the cursor
        last as its commit marker. Its trainer installs it under the cursor
        rule before resuming."""
        paths = self.tasks.paths(spec, task.tag)
        root = str(paths.root)
        for src, name in (
            (paths.rolling_checkpoint, state_pair.MODEL_NAME),
            (paths.train_state_path, state_pair.CURSOR_NAME),
        ):
            push_file(
                machine,
                container,
                remote_root=root,
                rel_dest=f"{state_pair.SEED_DIR}/{name}",
                src=src,
            )

    def _stage_inputs_in_container(
        self, machine, container: str, spec, tag: str, inputs: dict[str, Path]
    ):
        """Push a role's inputs into a freshly created container on the operator's
        own machine, under the tag root there. Its runner waits for them."""
        root = str(self.tasks.paths(spec, tag).root)
        for rel, src in inputs.items():
            push_file(machine, container, remote_root=root, rel_dest=rel, src=src)
