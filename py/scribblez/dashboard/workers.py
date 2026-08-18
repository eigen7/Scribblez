"""The WorkerManager: reconciles worker slots with real processes, pods, and
ssh machines.

Owned by the dashboard API process. Local worker slots are backed by
subprocesses of this process running the worker entrypoint with the local
results sink; cloud slots are backed by Runpod pods (one pod per slot),
created through the same pod spec as the fleet CLI; ssh slots are backed by
worker-image containers on operator-owned machines (cloud/ssh_machine.py),
booting the same image + bundle flow as a pod. Cloud and ssh slots both
deliver through the results bucket, so while a task has any, a cloud_sync
--watch subprocess streams that tag's bucket results into the local mount.

Adding a slot only records it, paused: nothing launches until the operator
starts it. For local and ssh slots the first start spawns the process /
container; for cloud slots it creates the backing pod itself (a Runpod pod
boots on creation), so an added-but-never-started slot costs nothing.

Desired state lives in task.json (dashboard/tasks.py); actual state is observed
live -- local workers by their durable pid (worker_pid_alive reads /proc, so a
worker is observable and stoppable no matter which dashboard instance spawned
it, even across a restart), cloud slots by pod runtime, ssh slots by a docker
probe over ssh. reconcile() drives
observed toward desired in both directions: it relaunches local workers that
should be running (e.g. after a dashboard restart), restarts interruptible pods
Runpod reclaimed, and stops workers that are running but should not be. It also
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
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from cloud.bundles import deploy_current_tree, source_hash
from cloud.credentials import CloudCredentials, CredentialsError, load_credentials
from cloud.r2 import bucket_path, rclone
from cloud.runpod_api import RunpodClient
from cloud.ssh_machine import SshMachine
from cloud.ssh_transfer import pull_results
from scripts.cloud_fleet import (
    CpuResources,
    GpuResources,
    bundle_worker_env,
    new_pod_name,
    pod_create_spec,
)
from tornado.ioloop import IOLoop

from scribblez import params as params_mod
from scribblez import workloads
from scribblez.dashboard import tasks
from scribblez.hardware import default_thread_count
from scribblez.paths import REPO_ROOT
from scribblez.workloads.base import SchedulerHooks

CLOUD_SYNC = REPO_ROOT / "py" / "scripts" / "cloud_sync.py"
SYNC_INTERVAL_SECONDS = 30

# Worker kinds that deliver through the results bucket, and so need the
# per-task sync watcher and the scheduler's bucket mirror. An ssh worker does
# not: it delivers into its own container and the reconcile pass reads its
# output back over the control link (cloud/ssh_transfer.py).
BUCKET_KINDS = ("cloud",)

# After an ssh machine fails a probe, how long it is assumed still unreachable
# before probing again -- so a powered-off machine costs one connect timeout
# per reconcile pass, not one per pass on a host that is simply off.
SSH_REPROBE_SECONDS = 30.0

# A container that dies as fast as it is started is not going to be fixed by
# starting it again: restarts back off from the pass interval up to this, and
# reset the moment one is observed running.
MAX_RESTART_BACKOFF_SECONDS = 300.0

# How long an observation of a container or pod stands in for a fresh one.
# Only the reconcile pass observes; status requests read what it left, so a
# browser polling every 3 seconds costs no ssh round trips at all.
OBSERVATION_TTL_SECONDS = 5.0


# What a slot should be doing, from operator intent plus scheduler gating.
# "park" and "stop" both mean not-working; they differ in how much of the
# worker survives it, which matters where restarting is expensive.
RUN, PARK, STOP = "run", "park", "stop"


def _intent(w: tasks.WorkerRecord, task: tasks.TaskRecord) -> str:
    if w.desired_state != "running":
        return STOP
    return PARK if w.role in task.gates else RUN


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


def _local_state(desired: str, alive: bool, gated: bool) -> str:
    """The honest display state of a local slot from the two observed axes
    (operator intent + real liveness) plus scheduler gating. `stopping` is the
    in-flight state where a paused slot's process has not yet exited; `exited`
    is an unexpected death of a slot that should be running (reconcile respawns
    it)."""
    if gated:
        return "waiting"
    if desired == "paused":
        return "stopping" if alive else "paused"
    return "running" if alive else "exited"


def _ssh_state(desired: str, probe: str, gated: bool) -> str:
    """The honest display state of an ssh slot from its container probe
    (cloud/ssh_machine.py's probe states, plus "unknown" for a slot the
    reconcile pass has not observed yet). `unreachable` is its own display
    state rather than a guess either way: the machine may be powered off with
    the worker gone, or merely off the network with the worker still running.
    A gated slot reads `waiting` whether its container is paused (the usual
    case) or still winding down."""
    if probe == "unknown":
        return "checking"
    if gated:
        return "waiting"
    if probe == "unreachable":
        return "unreachable"
    if desired == "paused":
        return "stopping" if probe in ("running", "paused") else "paused"
    return "running" if probe == "running" else "exited"


def _cloud_state(desired: str, alive: bool, gated: bool, desired_status: str | None) -> str:
    """The honest display state of a cloud slot. `alive` is real pod liveness
    (a running runtime), independent of the pod's own desiredStatus. `stopping`
    / `starting` are the in-flight states where intent and reality disagree;
    `interrupted` is a pod Runpod reclaimed out from under a should-run slot.
    `desired_status` is None for a slot whose pod has not been created yet
    (pods are created on first start): such a slot is `paused` until started,
    then `starting` while reconcile creates its pod."""
    if gated:
        return "waiting"
    if desired == "paused":
        return "stopping" if alive else "paused"
    if alive:
        return "running"
    return "starting" if desired_status is None or desired_status == "RUNNING" else "interrupted"


def _resource_record_fields(resources: CpuResources | GpuResources) -> dict:
    """The WorkerRecord cloud-resource fields for a pod's hardware selection."""
    if isinstance(resources, GpuResources):
        return {"gpu_type_id": resources.gpu_type_id, "gpu_count": resources.gpu_count}
    return {"vcpus": resources.vcpus, "flavor": resources.flavor}


def _worker_resources(w: tasks.WorkerRecord) -> CpuResources | GpuResources:
    """The inverse of _resource_record_fields: the slot's recorded hardware
    selection, for creating its pod at start time."""
    if w.gpu_type_id:
        return GpuResources(gpu_type_id=w.gpu_type_id, gpu_count=w.gpu_count)
    return CpuResources(vcpus=w.vcpus, flavor=w.flavor)


def _accrue(w: tasks.WorkerRecord, running: bool, cost_per_hr: float | None):
    """Advance a slot's spend estimate to now: if it was running at its last
    observation, the elapsed interval is charged at the last observed rate.
    Every observation point (status polls, the reconcile tick, state changes)
    calls this, so the estimate only drifts across dashboard-server downtime.
    """
    now = time.time()
    if w.observed_running and w.observed_at is not None:
        w.spend += (now - w.observed_at) / 3600 * (w.cost_per_hr or 0.0)
    w.observed_at = now
    w.observed_running = running
    if cost_per_hr is not None:
        w.cost_per_hr = cost_per_hr


class WorkerManager:
    def __init__(self):
        self._local: dict[str, subprocess.Popen] = {}  # slot key -> live process
        self._sync: dict[str, subprocess.Popen] = {}  # task key -> sync watcher
        self._creds: CloudCredentials | None = None
        self._client: RunpodClient | None = None
        self._ssh_down: dict[str, float] = {}  # host -> time of last failed probe
        # Slot key -> (probe state, when observed). Written by the reconcile
        # pass, read by everything else (see _probe_container).
        self._probes: dict[str, tuple[str, float]] = {}
        # Slot key -> why its container is not running (exit code + last log
        # line), so a crash-looping worker explains itself on the dashboard
        # instead of only in `docker logs`.
        self._exits: dict[str, str] = {}
        # Slot key -> (consecutive restarts, when the next one is allowed).
        self._restarts: dict[str, tuple[int, float]] = {}
        # (pods by id, when listed), the cloud counterpart of _probes.
        self._pods: tuple[dict, float] = ({}, 0.0)
        # Where every blocking step runs (see _offload). One thread: the point
        # is to keep the event loop free, not to do two of these at once.
        self._blocking = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scz-blocking")
        # Per-file digests behind source_hash, so the drift check every status
        # poll makes costs a stat walk rather than 20 MB of hashing.
        self._source_digests: dict = {}

    # ---- bundle deployment -------------------------------------------------

    def deploy(self, spec, task: tasks.TaskRecord) -> str:
        """Build the controller's tree, push it unless the bucket already has
        it, and pin the task to the result. Returns the bundle id."""
        creds, _ = self._cloud()
        manifest = deploy_current_tree(creds.r2, cache=self._source_digests)
        task.bundle_id = manifest.bundle_id
        task.bundle_source_hash = manifest.source_hash
        tasks.save_task(spec, task)
        return manifest.bundle_id

    def task_bundle_id(self, spec, task: tasks.TaskRecord) -> str:
        """The bundle this task's remote workers run, deployed on first use.

        Deployment is not an operator step: a task that has never launched a
        bucket-delivering worker gets the controller's current tree built and
        pushed here, and every later worker joins that same bundle. Pinning is
        what keeps an experiment homogeneous -- editing code mid-run leaves the
        fleet on the code it started with, and moving it is the explicit
        redeploy action.
        """
        return task.bundle_id or self.deploy(spec, task)

    def bundle_drift(self, task: tasks.TaskRecord) -> bool:
        """Whether the controller's tree has changed since the task pinned its
        bundle -- what the dashboard badges, and the cue to redeploy. False
        while nothing is pinned, and while an arch is unbuilt (there is no
        tree to compare until a build produces one)."""
        if not task.bundle_source_hash:
            return False
        current = source_hash(self._source_digests)
        return current is not None and current != task.bundle_source_hash

    # ---- cloud plumbing --------------------------------------------------

    def _cloud(self) -> tuple[CloudCredentials, RunpodClient]:
        if self._client is None:
            self._creds = load_credentials()
            self._client = RunpodClient(self._creds.runpod.api_key)
        return self._creds, self._client

    def _create_pod(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord):
        """Create slot `w`'s backing pod (first start), on the latest bundle,
        under the slot's recorded name and hardware. The pod boots and runs on
        creation."""
        creds, client = self._cloud()
        params = params_mod.validate(spec.params_cls, task.params)
        w.bundle_id = self.task_bundle_id(spec, task)
        body = pod_create_spec(
            creds, spec, task.tag, params,
            name=w.worker_id, role=w.role,
            bundle_id=w.bundle_id,
            resources=_worker_resources(w),
        )  # fmt: skip
        w.pod_id = client.create_pod(body)["id"]
        tasks.save_task(spec, task)

    def _ensure_sync(self, spec: workloads.WorkloadSpec, task: tasks.TaskRecord):
        """Keep exactly one sync watcher alive per task with bucket-delivering
        slots."""
        key = _key(spec, task.tag)
        has_bucket = any(w.kind in BUCKET_KINDS for w in task.workers)
        proc = self._sync.get(key)
        if has_bucket and (proc is None or proc.poll() is not None):
            log = self._log_file(spec, task.tag, "cloud_sync")
            self._sync[key] = subprocess.Popen(
                [
                    sys.executable, str(CLOUD_SYNC),
                    "--workload", spec.name, "-t", task.tag,
                    "--watch", "--interval", str(SYNC_INTERVAL_SECONDS),
                ],
                stdout=log, stderr=subprocess.STDOUT,
            )  # fmt: skip
        elif not has_bucket and proc is not None:
            proc.terminate()
            del self._sync[key]

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

    def _stop_local(self, spec, task: tasks.TaskRecord, w):
        """SIGTERM slot `w`'s worker by durable pid (workers flush completed
        output and exit cleanly on SIGTERM). No-op if it is not running."""
        if worker_pid_alive(w.pid, w.worker_id, task.tag):
            try:
                os.kill(w.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    # ---- ssh plumbing ----------------------------------------------------

    def _observe_container(self, spec, tag: str, w: tasks.WorkerRecord) -> str:
        """Probe slot `w`'s container over ssh and remember the answer.

        Negative caching stands: after a failed probe the host is assumed
        unreachable for SSH_REPROBE_SECONDS rather than paying a connect
        timeout every pass.

        A not-yet-launched slot is probed like any other -- an in-doubt first
        start (the ssh link dying after `docker run` was dispatched) may have
        left a live container, and a probe that finds one flips the slot to
        launched."""
        down_since = self._ssh_down.get(w.host)
        if down_since is not None and time.time() - down_since < SSH_REPROBE_SECONDS:
            probe = "unreachable"
        else:
            probe = SshMachine(w.host).container_state(_container_name(spec, tag, w.worker_id))
            if probe == "unreachable":
                self._ssh_down[w.host] = time.time()
            else:
                self._ssh_down.pop(w.host, None)
        if not w.launched and probe not in ("unreachable", "missing"):
            w.launched = True  # the in-doubt start did create the container
        key = _key(spec, tag, w.worker_id)
        self._probes[key] = (probe, time.time())
        if probe == "stopped":
            self._exits[key] = SshMachine(w.host).container_exit(
                _container_name(spec, tag, w.worker_id)
            )
        else:
            self._exits.pop(key, None)
        if probe == "running":
            self._restarts.pop(key, None)  # it came up; it is not looping
        return probe

    def _probe_container(self, spec, tag: str, w: tasks.WorkerRecord, *, observe: bool) -> str:
        """Slot `w`'s container probe state, freshly observed or remembered.

        Only the reconcile pass observes (`observe=True`); a status request
        reads what that pass left, so a browser polling every few seconds costs
        no ssh at all -- and the dashboard cannot be stalled by a machine that
        is slow to answer. A slot no pass has reached yet reads "unknown"
        rather than a guess.

        An unlaunched slot's `unreachable` maps to `missing`: with no container
        confirmed to exist, the slot must stay manageable (in particular,
        removable) even when the host is bogus or offline."""
        key = _key(spec, tag, w.worker_id)
        remembered, at = self._probes.get(key, ("unknown", 0.0))
        if observe and time.time() - at >= OBSERVATION_TTL_SECONDS:
            probe = self._observe_container(spec, tag, w)
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
        bundle (matching what a fresh pod would run)."""
        creds, _ = self._cloud()
        params = params_mod.validate(spec.params_cls, task.params)
        w.bundle_id = self.task_bundle_id(spec, task)
        env = bundle_worker_env(
            creds, spec, task.tag, params,
            role=w.role, bundle_id=w.bundle_id, worker_id=w.worker_id, kind="ssh",
        )  # fmt: skip
        env["SCZ_SINK"] = "local"  # collected over ssh, not uploaded (ssh_transfer.py)
        if w.threads:
            env["SCZ_THREADS"] = str(w.threads)
        SshMachine(w.host).run_container(
            _container_name(spec, task.tag, w.worker_id), creds.registry.worker_image, env
        )
        w.launched = True
        tasks.save_task(spec, task)

    # ---- slot operations -------------------------------------------------

    def _check_role(self, spec, task: tasks.TaskRecord, role: str, kind: str, gpu: bool = False):
        role_spec = spec.role(role)
        assert kind in role_spec.kinds, f"role '{role}' does not support {kind} workers"
        if kind == "cloud":
            want = "GPU" if role_spec.gpu else "CPU"
            assert gpu == role_spec.gpu, f"role '{role}' requires {want} instances"
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
        self, spec, task: tasks.TaskRecord, role: str, host: str, threads: int | None
    ) -> tasks.WorkerRecord:
        role_spec = self._check_role(spec, task, role, "ssh")
        assert not role_spec.gpu, f"role '{role}' needs GPU hardware; the worker image is CPU-only"
        w = tasks.WorkerRecord(
            worker_id=_next_worker_id(task, "ssh"),
            role=role,
            kind="ssh",
            desired_state="paused",
            host=host,
            launched=False,
            threads=threads,
        )
        task.workers.append(w)
        tasks.save_task(spec, task)
        self._ensure_sync(spec, task)
        return w

    def add_cloud(
        self,
        spec,
        task: tasks.TaskRecord,
        role: str,
        count: int,
        resources: CpuResources | GpuResources,
    ) -> list[tasks.WorkerRecord]:
        gpu = isinstance(resources, GpuResources)
        role_spec = self._check_role(spec, task, role, "cloud", gpu=gpu)
        assert not (role_spec.singleton and count > 1), f"role '{role}' allows one worker"
        added = []
        for _ in range(count):
            w = tasks.WorkerRecord(
                worker_id=new_pod_name(task.tag),
                role=role,
                kind="cloud",
                desired_state="paused",
                **_resource_record_fields(resources),
            )
            task.workers.append(w)
            added.append(w)
        tasks.save_task(spec, task)
        self._ensure_sync(spec, task)
        return added

    def set_worker_state(self, spec, task: tasks.TaskRecord, worker_id: str, run: bool):
        w = task.worker(worker_id)
        w.desired_state = "running" if run else "paused"
        _accrue(w, run, None)
        tasks.save_task(spec, task)
        start = run and w.role not in task.gates  # a gated slot starts when released
        if w.kind == "local":
            if start and not self._local_alive(spec, task, w):
                self._spawn_local(spec, task, w)
            elif not run:
                self._stop_local(spec, task, w)
        elif w.kind == "ssh":
            # An unreachable machine gets no action either way: the desired
            # state is saved, and reconcile enforces it once probes succeed.
            probe = self._probe_container(spec, task.tag, w)
            name = _container_name(spec, task.tag, w.worker_id)
            if start and probe == "stopped":
                SshMachine(w.host).start_container(name)
            elif start and probe == "missing":
                self._run_ssh_container(spec, task, w)
            elif not run and probe == "running":
                SshMachine(w.host).stop_container(name)
        else:
            if start and w.pod_id is None:
                self._create_pod(spec, task, w)
            elif start:
                _, client = self._cloud()
                client.start_pod(w.pod_id)
            elif not run and w.pod_id is not None:
                _, client = self._cloud()
                client.stop_pod(w.pod_id)

    def remove_worker(self, spec, task: tasks.TaskRecord, worker_id: str):
        """Remove a slot. Only non-running workers may be removed (pause
        first), so a removal never silently discards an in-flight cycle."""
        w = task.worker(worker_id)
        if w.kind == "local":
            assert not self._local_alive(spec, task, w), f"{worker_id} is running; pause it first"
            self._local.pop(_key(spec, task.tag, worker_id), None)
        elif w.kind == "ssh":
            # Freshly observed: a removal must not act on a remembered state.
            probe = self._probe_container(spec, task.tag, w, observe=True)
            assert probe not in ("running", "paused"), f"{worker_id} is running; pause it first"
            # Removing while unreachable would orphan a possibly-live container
            # that keeps generating into the tag with nothing tracking it.
            assert probe != "unreachable", (
                f"{w.host} is unreachable; bring it online (or clean up its container "
                f"by hand) before removing {worker_id}"
            )
            if probe == "stopped":
                SshMachine(w.host).remove_container(_container_name(spec, task.tag, w.worker_id))
        elif w.pod_id is not None:  # a never-started cloud slot has no pod to delete
            _, client = self._cloud()
            pod = next((p for p in client.list_pods() if p["id"] == w.pod_id), None)
            assert pod is None or pod.get("runtime") is None, (
                f"{worker_id} is running; pause it first"
            )
            if pod is not None:
                client.delete_pod(w.pod_id)
        _accrue(w, False, None)
        task.retired_spend += w.spend
        task.workers.remove(w)
        tasks.save_task(spec, task)
        self._ensure_sync(spec, task)

    # ---- observation -----------------------------------------------------

    def _pod_index(self, observe: bool) -> dict:
        """Runpod's pods by id, listed at most once per OBSERVATION_TTL_SECONDS
        and only by the reconcile pass -- the cloud counterpart of the container
        probe cache."""
        pods, at = self._pods
        if observe and time.time() - at >= OBSERVATION_TTL_SECONDS:
            _, client = self._cloud()
            self._pods = pods, _ = ({p["id"]: p for p in client.list_pods()}, time.time())
        return pods

    def worker_status(self, spec, task: tasks.TaskRecord, *, observe: bool = False) -> list[dict]:
        """One dict per slot: the durable record plus observed live state.
        Every call is also a spend-accrual observation point (persisted).

        `observe` is the reconcile pass's privilege: it goes to the machines
        and the cloud API, and leaves what it learns behind. Every other caller
        -- every status request a browser makes -- reads those observations, so
        serving the dashboard never waits on ssh or Runpod.
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
                "vcpus": w.vcpus,
                "flavor": w.flavor,
                "gpu_type_id": w.gpu_type_id,
                "gpu_count": w.gpu_count,
                "pod_id": w.pod_id,
                "host": w.host,
                "bundle_id": w.bundle_id,
            }
            if gated:
                info["gate_reason"] = task.gates[w.role]
            if w.kind == "local":
                alive = self._local_alive(spec, task, w)
                info["state"] = _local_state(w.desired_state, alive, gated)
                _accrue(w, alive, None)
            elif w.kind == "ssh":
                probe = self._probe_container(spec, task.tag, w, observe=observe)
                alive = probe == "running"
                info["state"] = _ssh_state(w.desired_state, probe, gated)
                info["ssh_probe"] = probe  # reconcile keys its enforcement off this
                info["ssh"] = f"ssh {w.host}"
                reason = self._exits.get(_key(spec, task.tag, w.worker_id))
                if reason and not alive:
                    info["exit_reason"] = reason
                _accrue(w, alive, None)
            elif w.pod_id is None:  # not yet started, so no pod to observe
                alive = False
                info["state"] = _cloud_state(w.desired_state, False, gated, None)
                _accrue(w, False, None)
            else:
                pod = self._pod_index(observe).get(w.pod_id)
                if pod is None:
                    # Nothing listed for this pod: terminated if we just
                    # looked, merely unobserved if we are reading a listing
                    # that has not covered it yet.
                    alive = False if observe else w.observed_running
                    info["state"] = (
                        "terminated" if observe else _cloud_state(
                            w.desired_state, alive, gated, None
                        )
                    )  # fmt: skip
                    _accrue(w, alive, None)
                else:
                    alive = pod.get("runtime") is not None  # real liveness, not desiredStatus
                    info["state"] = _cloud_state(
                        w.desired_state, alive, gated, pod.get("desiredStatus", "")
                    )
                    info["cost_per_hr"] = pod.get("costPerHr")
                    info["public_ip"] = pod.get("publicIp")
                    info["ssh"] = f"ssh {w.pod_id}@ssh.runpod.io"
                    _accrue(w, alive, pod.get("costPerHr"))
            # Real liveness, for reconcile's desired-vs-observed enforcement.
            info["observed_running"] = alive
            info["spend"] = w.spend
            out.append(info)
        if task.workers:
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

        return SchedulerHooks(gate=gate, mirror=self._make_mirror(spec, task))

    def _make_mirror(self, spec, task: tasks.TaskRecord):
        """Bucket-side replay of staging ingests: when the scheduler assigns a
        chunk locally, its bucket object (if any -- the chunk may be
        local-origin) moves to the matching generation prefix, so the bucket
        keeps mirroring the local corpus and the sync watcher never
        re-downloads an ingested chunk. None for tasks without
        bucket-delivering slots."""
        if not any(w.kind in BUCKET_KINDS for w in task.workers):
            return None
        try:
            creds, _ = self._cloud()
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
                yield spec, tasks.load_task(spec, row["tag"])

    async def offload(self, fn, *args, **kwargs):
        """Run one blocking step off the event loop, one at a time.

        Everything reconcile does is blocking IO measured in seconds: ssh to a
        laptop, rclone to R2, the Runpod API, a build. Run inline it froze the
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
        crashed process), restart interruptible pods Runpod reclaimed, and park
        or stop workers that are running but should not be (a scheduler gate or
        an operator pause). Enforcement keys off real liveness by durable
        pid/pod runtime, so it holds a paused worker down even across a
        dashboard restart or a second dashboard instance.

        This pass is the only observer: it refreshes the container probes and
        pod listing everything else reads, which also makes it the spend-accrual
        heartbeat when no browser is polling.
        """
        for spec, task in self._all_tasks():
            if spec.scheduler:
                try:
                    await self.offload(self._tick_scheduler, spec, task)
                except Exception as e:  # noqa: BLE001 -- scheduling must keep ticking
                    print(f"scheduler {spec.name}/{task.tag}: {e}")
            if not task.workers:
                continue
            status = await self.offload(self.worker_status, spec, task, observe=True)
            for w, info in zip(task.workers, status, strict=True):
                # Collect before enforcing: this slot may be about to be
                # parked, and a pull needs its container running.
                if w.kind == "ssh" and info["ssh_probe"] == "running":
                    try:
                        await self.offload(self._collect_ssh, spec, task, w)
                    except Exception as e:  # noqa: BLE001 -- one slot must not stop the pass
                        print(f"collect {spec.name}/{task.tag}/{w.worker_id}: {e}")
                # Contained per slot: one slot's failing enforcement (an ssh
                # machine vanishing mid-action, a pod creation that keeps
                # failing on an out-of-stock flavor) must not starve the rest
                # of the pass -- reconcile is the only enforcement some slots
                # get (e.g. stopping gated pods that are still billing).
                try:
                    await self.offload(
                        self._reconcile_worker, spec, task, w, _intent(w, task), info
                    )
                except Exception as e:  # noqa: BLE001 -- enforcement must keep ticking
                    print(f"reconcile {spec.name}/{task.tag}/{w.worker_id}: {e}")
            self._ensure_sync(spec, task)

    def _collect_ssh(self, spec, task: tasks.TaskRecord, w: tasks.WorkerRecord):
        """Read slot `w`'s finished outputs out of its container into the tag."""
        paths = spec.paths(task.tag)
        pull_results(
            SshMachine(w.host), _container_name(spec, task.tag, w.worker_id),
            remote_root=str(paths.root), local_root=paths.root,
            data_dirs=[f"data/{sub}" for sub in spec.sync_data_dirs],
        )  # fmt: skip

    def _tick_scheduler(self, spec, task: tasks.TaskRecord):
        workloads.resolve(spec.scheduler)(spec, task, self._scheduler_hooks(spec, task))

    def _reconcile_worker(self, spec, task: tasks.TaskRecord, w, intent: str, info: dict):
        """Close one slot's desired-vs-observed gap. A cloud pod that is booting
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
        elif w.kind == "ssh":
            self._reconcile_ssh(spec, task, w, intent, info["ssh_probe"])
        else:
            # Parking a pod has to stop it: an idle pod bills like a busy one.
            if intent == RUN and w.pod_id is None:
                # A should-run slot with no pod: its creation failed on start,
                # or a gate released it before its first start. Create it now.
                self._create_pod(spec, task, w)
            elif intent == RUN and info["state"] == "interrupted":
                _, client = self._cloud()
                client.start_pod(w.pod_id)
            elif intent != RUN and alive:
                _, client = self._cloud()
                client.stop_pod(w.pod_id)

    def _reconcile_ssh(self, spec, task: tasks.TaskRecord, w, intent: str, probe: str):
        """Enforce one ssh slot's intent.

        A gate parks the container by pausing it rather than stopping it. A
        stop costs the worker its in-flight chunk, and the next start re-runs
        the image's bootstrap -- refetching and unpacking the bundle before the
        first game -- which on a gate that flips every minute was eating a
        third of the machine's duty cycle. A pause is instant in both
        directions and resumes the work mid-chunk.
        """
        machine, name = SshMachine(w.host), _container_name(spec, task.tag, w.worker_id)
        key = _key(spec, task.tag, w.worker_id)
        if intent == RUN:
            if probe == "paused":
                machine.unpause_container(name)  # resuming a parked worker is not a restart
                return
            if probe not in ("missing", "stopped"):
                return  # running, or nothing this pass can act on
            if not self._restart_allowed(key):
                return
            self._note_restart(key)
            if probe == "missing":
                self._run_ssh_container(spec, task, w)
            elif probe == "stopped" and w.bundle_id != task.bundle_id:
                # Redeployed since this container was created. Its bundle is
                # fixed in the environment it was created with, so the slot
                # joins the task's new bundle by being replaced, not restarted.
                machine.remove_container(name)
                self._run_ssh_container(spec, task, w)
            elif probe == "stopped":
                machine.start_container(name)
        elif intent == PARK and probe == "running":
            machine.pause_container(name)
        elif intent == STOP and probe in ("running", "paused"):
            if probe == "paused":
                machine.unpause_container(name)  # docker stop cannot signal a frozen process
            machine.stop_container(name)

    def shutdown(self):
        """Stop owned subprocesses (workers flush completed output on SIGTERM);
        pods and ssh containers are unaffected -- their work continues across
        dashboard restarts."""
        self._blocking.shutdown(wait=False, cancel_futures=True)
        for proc in [*self._local.values(), *self._sync.values()]:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
