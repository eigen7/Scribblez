"""A fake world for simulating the dashboard's control plane
(docs/plans/dashboard_state_model.md §7).

The real WorkerManager, TagQueue and PoolRentals run against it unchanged;
everything they touch outside the process is faked here: a clock, local worker
processes (Popen, /proc liveness, signals), ssh machines and their containers,
the cloud provider's instances, the results bucket and the cloud_sync watchers
that pull from it, and the thread pools, which run each job at once so a seed
replays exactly.

The workload is a stand-in with position_eval's shape: a trainer and a
generator placed by a layout, a row budget as the end condition, and a
scheduler that gates the generator while it runs ahead and finishes it once
the trainer is done. Its workers do nothing but advance counters, each step
the world takes, in the files a real trainer writes: train_state.json (the
cursor placement's state_home reads) in the tag dir for a local trainer, in
the bucket for a bucket-delivering one, from where a watcher pulls it.

What it checks is orderings and restarts at step granularity; the executors
are synchronous, so thread interleavings are out of its reach.
"""

import dataclasses
import json
import signal
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

from cloud.bundles import BundleManifest
from cloud.credentials import RegistryConfig
from cloud.providers.base import Instance, MachineType, ProviderError
from cloud.ssh_machine import SshMachineError
from cloud.worker_entrypoint import EXIT_INTERRUPTED
from scribblez import params as params_mod
from scribblez.workloads.base import RoleSpec, SlotPlan, WorkloadSpec

WORKLOAD = "sim"
ROWS_PER_STEP = 100  # a live trainer's progress per world step
ROWS_PER_GENERATION = 200  # what one generator step feeds
AHEAD_LIMIT = 3  # generations a generator may run ahead before its gate closes
STEP_SECONDS = 5.0
TRAIN_GPU_GB = 8.0


# ---- the workload ----------------------------------------------------------


@dataclasses.dataclass
class SimParams:
    max_rows: int = params_mod.param(1000, "rows the trainer trains, then finishes", end=True)


def layout(params, vcpus: int, generator_threads: int | None) -> list[SlotPlan]:
    return [
        SlotPlan("train", None, TRAIN_GPU_GB),
        SlotPlan("generate", generator_threads or vcpus, 0.0),
    ]


def gpu_need(params, role: str) -> float:
    return TRAIN_GPU_GB if role == "train" else 0.0


def read_rows(tag_root: Path) -> int:
    try:
        return json.loads((tag_root / "train_state.json").read_text())["rows_trained"]
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def read_generations(tag_root: Path) -> int:
    try:
        return int((tag_root / "data" / "generations").read_text())
    except (FileNotFoundError, ValueError):
        return 0


def scheduler(spec, task, hooks):
    """Gate the generator while it is AHEAD_LIMIT generations past what the
    trainer has consumed; finish it once the trainer reached max_rows."""
    params = params_mod.validate(spec.params_cls, task.params)
    rows = read_rows(hooks.paths.root)
    if params_mod.reached(rows, params.max_rows):
        hooks.finish("generate")
        return
    ahead = read_generations(hooks.paths.root) - rows // ROWS_PER_GENERATION
    hooks.gate("generate", "ahead of trainer" if ahead >= AHEAD_LIMIT else None)


def progress(spec, paths, params) -> list[tuple[str, object]]:
    return [("rows", read_rows(paths.root))]


def ingest(spec, paths):
    """The trainer's controller-side tick: nothing to ingest here, but its
    presence is what makes the trainer a trainer (bucket delivery on ssh)."""


def never_run(ctx) -> int:
    raise AssertionError("simulated workers never run for real")


SPEC = WorkloadSpec(
    name=WORKLOAD,
    title="Simulation",
    params_cls=SimParams,
    roles=(
        RoleSpec(
            "train", "Trainer", "sim_world:never_run", singleton=True, gpu=True,
            ingest="sim_world:ingest",
        ),
        RoleSpec("generate", "Generator", "sim_world:never_run"),
    ),
    scheduler="sim_world:scheduler",
    progress="sim_world:progress",
    layout="sim_world:layout",
    gpu_need="sim_world:gpu_need",
)  # fmt: skip


# ---- clock and executors ---------------------------------------------------


class Clock:
    def __init__(self, start: float = 1_800_000_000.0):
        self.now = start

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float):
        self.now += seconds

    def module(self):
        """A stand-in for the `time` module, for the modules under test."""
        return SimpleNamespace(time=self.time, monotonic=self.monotonic, sleep=self.sleep)


class SyncExecutor:
    """A ThreadPoolExecutor that runs each job at once, on the caller's thread,
    so a run is a pure function of its seed."""

    def submit(self, fn, *args, **kwargs) -> Future:
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001 -- delivered through the future
            future.set_exception(e)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


class WriterThread:
    """The manager's blocking executor as one real thread that runs each job
    to completion before submit returns. The dashboard's writer rule then
    holds as it does live (shared_json): the pass and every command run on
    this thread, and the sim's own reads, on the main thread, see committed
    copies. A run stays a pure function of its seed."""

    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim-writer")

    def submit(self, fn, *args, **kwargs) -> Future:
        future = self._pool.submit(fn, *args, **kwargs)
        wait([future])
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)


# ---- the world -------------------------------------------------------------


@dataclasses.dataclass
class Worker:
    """What a simulated worker is doing wherever it runs: its role, tag, and
    where it delivers ("local": the tag dir; "r2": the bucket)."""

    role: str
    tag: str
    sink: str
    max_rows: int


class FakeProc:
    """A local worker process (subprocess.Popen's surface the manager uses)."""

    def __init__(self, world, pid: int, env: dict, worker: Worker | None, watcher=None):
        self.world = world
        self.pid = pid
        self.env = env
        self.worker = worker
        self.watcher = watcher  # (tag, trainer_outputs) for a cloud_sync watcher
        self.returncode = None
        self._exit_pending: int | None = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def send_signal(self, sig):
        if self.returncode is None:
            self._exit_pending = EXIT_INTERRUPTED

    def exit(self, code: int):
        self.returncode = code
        self.world.exited.append((self.env.get("SCZ_TAG"), self.env.get("SCZ_WORKER_ID"), code))

    @property
    def alive(self) -> bool:
        return self.returncode is None


@dataclasses.dataclass
class Container:
    state: str  # running | paused | stopped
    worker: Worker
    env: dict
    exit_reason: str = ""


class Host:
    """An ssh-reachable machine: a registered one or a rented instance."""

    def __init__(self, name: str, hardware: str, instance_id: str | None = None):
        self.name = name
        self.hardware = hardware
        self.instance_id = instance_id
        self.containers: dict[str, Container] = {}


class FakeProvider:
    """The cloud provider: instances that boot, stop when interrupted, and a
    listing that can fail."""

    name = "aws"
    ssh_user = "ubuntu"
    identity_file = "/k/sim.pem"
    ready_file = "/var/lib/scribblez/ready"
    region = "sim-1"

    def __init__(self, world):
        self.world = world
        self.instances: dict[str, Instance] = {}
        self.listing_fails = False

    def account(self):
        return "sim account"

    def catalog(self):
        return [MachineType("g6.2xlarge", 8, 1, "L4", "znver3", 1.0, gpu_memory_gb=22.5)]

    def prepare(self):
        pass

    def spot_prices(self):
        return {"g6.2xlarge": 0.4}

    def launch(self, request):
        n = len(self.instances) + 1
        inst = Instance(
            id=f"i-{n}", state="pending", type_id=request.type_id, owner=request.owner,
            address=None, launched_at=self.world.clock.now, spot=request.spot,
            cost_per_hr=0.4 if request.spot else None,
        )  # fmt: skip
        self.instances[inst.id] = inst
        return inst

    def describe(self):
        if self.listing_fails:
            raise ProviderError("the listing failed (simulated)")
        return {i.id: dataclasses.replace(i) for i in self.instances.values()}

    def start(self, instance_id):
        self.instances[instance_id].state = "pending"

    def stop(self, instance_id):
        self.instances[instance_id].state = "stopping"

    def terminate(self, instance_id):
        self.instances[instance_id].state = "terminated"

    def refusal(self, error, type_id):
        return f"refused {type_id}: {error}"


class World:
    """Everything outside the dashboard process, advanced one step at a time."""

    def __init__(self, mount_root: Path):
        self.mount_root = mount_root
        self.clock = Clock()
        self.procs: dict[int, FakeProc] = {}
        self._next_pid = 1000
        self.hosts: dict[str, Host] = {}
        self.provider = FakeProvider(self)
        self.bucket: dict[str, str] = {}
        self.exited: list[tuple] = []  # (tag, worker id, code) of every process exit
        self.crashed: set[tuple[str, str]] = set()  # (tag, worker id) the world crashed

    # -- processes

    def subprocess_module(self):
        """A stand-in for the `subprocess` module, for workers.py: Popen starts
        a simulated worker or watcher, and run (the drain's one-off
        cloud_sync) pulls once."""
        return SimpleNamespace(
            Popen=self.popen,
            run=self.run,
            STDOUT=subprocess.STDOUT,
            PIPE=subprocess.PIPE,
            CalledProcessError=subprocess.CalledProcessError,
        )

    def run(self, argv, check=False, **kwargs):
        if any("cloud_sync" in str(a) for a in argv):
            tag = argv[argv.index("-t") + 1]
            self._sync_down(tag, "--trainer-outputs" in argv)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def popen(self, argv, env=None, **kwargs) -> FakeProc:
        env = env or {}
        pid = self._next_pid
        self._next_pid += 1
        if any("cloud_sync" in str(a) for a in argv):
            tag = argv[argv.index("-t") + 1]
            proc = FakeProc(self, pid, env, None, watcher=(tag, "--trainer-outputs" in argv))
        else:
            proc = FakeProc(self, pid, env, _worker_from_env(env, "local"))
        self.procs[pid] = proc
        return proc

    def pid_alive(self, pid, worker_id: str, tag: str) -> bool:
        proc = self.procs.get(pid)
        return (
            proc is not None
            and proc.alive
            and proc.env.get("SCZ_WORKER_ID") == worker_id
            and proc.env.get("SCZ_TAG") == tag
        )

    def kill(self, pid: int, sig):
        proc = self.procs.get(pid)
        if proc is None or not proc.alive:
            raise ProcessLookupError(pid)
        proc.send_signal(sig)

    def crash_local(self, tag: str, worker_id: str) -> bool:
        for proc in self.procs.values():
            if (
                proc.alive
                and proc.env.get("SCZ_TAG") == tag
                and proc.env.get("SCZ_WORKER_ID") == worker_id
            ):
                self.crashed.add((tag, worker_id))
                proc.exit(1)
                return True
        return False

    # -- ssh machines

    def add_host(self, name: str, hardware: str = "8\n23034\n") -> Host:
        self.hosts[name] = host = Host(name, hardware)
        return host

    def host(self, spelling: str) -> Host | None:
        """The host an ssh destination names, if it is up."""
        name = spelling.split("@", 1)[-1].lower()
        host = self.hosts.get(name)
        if host is None:
            return None
        if host.instance_id is not None:
            inst = self.provider.instances.get(host.instance_id)
            if inst is None or inst.state != "running":
                return None
        return host

    def crash_container(self, host: Host, name: str) -> bool:
        c = host.containers.get(name)
        if c is None or c.state != "running":
            return False
        c.state, c.exit_reason = "stopped", "exit 1: simulated crash"
        self.crashed.add((c.env.get("SCZ_TAG"), c.env.get("SCZ_WORKER_ID")))
        return True

    # -- the passage of time

    def step(self):
        self.clock.sleep(STEP_SECONDS)
        for proc in list(self.procs.values()):
            if not proc.alive:
                continue
            if proc._exit_pending is not None:
                proc.exit(proc._exit_pending)
            elif proc.watcher is not None:
                self._sync_down(*proc.watcher)
            elif proc.worker is not None:
                code = self._work(proc.worker)
                if code is not None:
                    proc.exit(code)
        for inst in self.provider.instances.values():
            self._advance_instance(inst)
        for host in self.hosts.values():
            if self.host(host.name) is None:
                continue
            for c in host.containers.values():
                if c.state == "running":
                    code = self._work(c.worker)
                    if code is not None:
                        c.state, c.exit_reason = "stopped", f"exit {code}: done"

    def _advance_instance(self, inst: Instance):
        if inst.state == "pending":
            inst.state = "running"
            inst.address = f"10.0.0.{inst.id.split('-')[1]}"
            host = self.hosts.get(inst.address)
            if host is None:
                host = self.add_host(inst.address)
                host.instance_id = inst.id
        elif inst.state == "stopping":
            inst.state = "stopped"
            host = self.hosts.get(inst.address or "")
            for c in host.containers.values() if host else []:
                if c.state in ("running", "paused"):
                    c.state, c.exit_reason = "stopped", f"exit {EXIT_INTERRUPTED}: host stopped"
        if inst.state in ("stopped", "terminated") and inst.address:
            if inst.state == "terminated":
                self.hosts.pop(inst.address, None)
                inst.address = None

    def interrupt_spot(self, instance_id: str) -> bool:
        inst = self.provider.instances.get(instance_id)
        if inst is None or inst.state != "running" or not inst.spot:
            return False
        inst.state = "stopping"
        return True

    def vanish(self, instance_id: str) -> bool:
        inst = self.provider.instances.get(instance_id)
        if inst is None or inst.state == "terminated":
            return False
        inst.state = "terminated"
        self._advance_instance(inst)
        return True

    # -- what workers do

    def _tag_root(self, tag: str) -> Path:
        return self.mount_root / "tags" / WORKLOAD / tag

    def _read(self, w: Worker, rel: str) -> str | None:
        if w.sink == "r2":
            return self.bucket.get(f"{w.tag}/{rel}")
        path = self._tag_root(w.tag) / rel
        return path.read_text() if path.is_file() else None

    def _write(self, w: Worker, rel: str, text: str):
        if w.sink == "r2":
            self.bucket[f"{w.tag}/{rel}"] = text
            return
        path = self._tag_root(w.tag) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def _work(self, w: Worker) -> int | None:
        """One step of a live worker; its exit code once it is done."""
        if w.role == "generate":
            gens = int(self._read(w, "data/generations") or 0)
            self._write(w, "data/generations", str(gens + 1))
            return None
        state = json.loads(self._read(w, "train_state.json") or '{"rows_trained": 0}')
        if params_mod.reached(state["rows_trained"], w.max_rows):
            return 0
        state["rows_trained"] += ROWS_PER_STEP
        self._write(w, "train_state.json", json.dumps(state))
        return None

    def _sync_down(self, tag: str, trainer_outputs: bool):
        """A cloud_sync watcher's pass: pull what bucket slots delivered, the
        trainer's cursor included (as the real one does, whatever is local)."""
        rels = ["data/generations"] + (["train_state.json"] if trainer_outputs else [])
        for rel in rels:
            text = self.bucket.get(f"{tag}/{rel}")
            if text is not None:
                path = self._tag_root(tag) / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)

    def billing(self) -> list[Instance]:
        return [i for i in self.provider.instances.values() if i.state in ("pending", "running")]


def _worker_from_env(env: dict, default_sink: str) -> Worker | None:
    if env.get("SCZ_WORKLOAD") != WORKLOAD:
        return None
    return Worker(
        role=env["SCZ_ROLE"],
        tag=env["SCZ_TAG"],
        sink=env.get("SCZ_SINK", default_sink),
        max_rows=int(env.get("SCZ_MAX_ROWS", "0") or 0),
    )


class FakeSshMachine:
    """cloud.ssh_machine.SshMachine's surface, over the world's hosts."""

    world: World  # set by install()

    def __init__(self, host, identity_file=None, known_hosts_file=None):
        self.spelling = host

    def _host(self) -> Host:
        host = self.world.host(self.spelling)
        if host is None:
            raise SshMachineError(f"{self.spelling} is unreachable")
        return host

    def probe(self, ready_file=None) -> str:
        return "up" if self.world.host(self.spelling) is not None else "unreachable"

    def hardware_report(self) -> str:
        return self._host().hardware

    def pull_image(self, image):
        self._host()

    def detect_arch(self, image) -> str:
        return "znver3"

    def container_state(self, name: str) -> str:
        host = self.world.host(self.spelling)
        if host is None:
            return "unreachable"
        c = host.containers.get(name)
        return "missing" if c is None else c.state

    def container_exit(self, name: str) -> str:
        c = self._host().containers.get(name)
        return c.exit_reason if c is not None else ""

    def container_logs(self, name: str) -> str:
        return ""

    def run_container(self, name, image, env, gpus=False):
        host = self._host()
        worker = _worker_from_env(env, "r2")
        host.containers[name] = Container("running", worker, dict(env))

    def start_container(self, name: str):
        c = self._host().containers[name]
        c.state, c.exit_reason = "running", ""

    def stop_container(self, name: str):
        c = self._host().containers.get(name)
        if c is not None and c.state in ("running", "paused"):
            c.state, c.exit_reason = "stopped", f"exit {EXIT_INTERRUPTED}: stopped"

    def pause_container(self, name: str):
        c = self._host().containers[name]
        if c.state == "running":
            c.state = "paused"

    def unpause_container(self, name: str):
        c = self._host().containers[name]
        if c.state == "paused":
            c.state = "running"

    def remove_container(self, name: str):
        self._host().containers.pop(name, None)


def pull_results(machine, container, **target):
    return SimpleNamespace(remaining=0, pulled=[])


def sweep_stopped(machine, container, **target):
    return []


def push_file(machine, container, **kw):
    pass


def fake_rclone(r2, *args, capture=False, input_text=None):
    return SimpleNamespace(returncode=0, stdout="", stderr="")


SIM_CREDS = SimpleNamespace(
    registry=RegistryConfig(worker_image="repo/worker"),
    r2=SimpleNamespace(account_id="a", access_key_id="k", secret_access_key="s", bucket="sim"),
    aws=None,
)


def install(monkeypatch, world: World):
    """Point the dashboard's modules at the world, for one test."""
    from scribblez import workloads
    from scribblez.dashboard import pool as pool_mod
    from scribblez.dashboard import pool_rentals as rentals_mod
    from scribblez.dashboard import tag_queue as tq_mod
    from scribblez.dashboard import tasks as tasks_mod
    from scribblez.dashboard import workers as workers_mod
    from scribblez.dashboard.pool import Hardware

    monkeypatch.setitem(workloads.WORKLOADS, WORKLOAD, SPEC)
    fake_time = world.clock.module()
    for mod in (workers_mod, tq_mod, rentals_mod, tasks_mod):
        monkeypatch.setattr(mod, "time", fake_time)
    monkeypatch.setattr(workers_mod, "subprocess", world.subprocess_module())
    monkeypatch.setattr(workers_mod, "worker_pid_alive", world.pid_alive)
    monkeypatch.setattr(tq_mod, "worker_pid_alive", world.pid_alive)
    monkeypatch.setattr(workers_mod.os, "kill", world.kill)
    FakeSshMachine.world = world
    monkeypatch.setattr(workers_mod, "SshMachine", FakeSshMachine)
    for mod, name, fn in (
        (workers_mod, "pull_results", pull_results),
        (workers_mod, "sweep_stopped", sweep_stopped),
        (tq_mod, "sweep_stopped", sweep_stopped),
        (workers_mod, "push_file", push_file),
        (workers_mod, "rclone", fake_rclone),
    ):
        monkeypatch.setattr(mod, name, fn)
    monkeypatch.setattr(pool_mod, "canonical_host", lambda h: h.split("@", 1)[-1].lower())
    monkeypatch.setattr(pool_mod, "local_hardware", lambda: Hardware(28, 1, 16.0))
    monkeypatch.setattr(workers_mod, "check_worker_images_current", lambda mount_root: None)


def wire_manager(monkeypatch, manager, world: World):
    """Per-manager fakes: its provider, credentials, bundle builds and pools."""
    monkeypatch.setattr(manager, "_provider", lambda: world.provider)
    monkeypatch.setattr(manager, "_creds", lambda: SIM_CREDS)
    monkeypatch.setattr(
        manager,
        "_build_bundle",
        lambda archs: BundleManifest(
            bundle_id="b-sim", git_sha="0", git_dirty=False, archs=list(archs), source_hash="sim"
        ),
    )
    manager._blocking = WriterThread()
    for name in ("_builds", "_uploads"):
        setattr(manager, name, SyncExecutor())
