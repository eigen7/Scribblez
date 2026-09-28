"""The machine pool (docs/plans/tag_queue.md §2): the machines the tag queue
may place tags on, owned by the pool rather than by any task.

A task-owned machine (tasks.MachineRecord in a task.json) lives and dies with
its task. A pool machine outlives every tag that uses it: a tag holds it under
a `Lease`, and the lease, not the machine, moves from tag to tag. So nothing
about the machine (its address, key material, arch, rented instance) has to be
copied between task records or kept in agreement with them. A leased
machine's slots name it like any machine, and the dashboard resolves the name
through the pool (`leased_record`).

An entry is `localhost` (this machine, running local slots) or a registered
ssh machine. Each records the facts placement checks: vCPUs, GPU count, memory
per GPU, and a GPU reserve for memory no slot accounts for (the dashboard's
inference and the test suite on localhost).

pool.json lives under the mount root beside the workload tag trees, held as
one shared object per process (shared_json.py).
"""

import subprocess
from dataclasses import dataclass, field, fields

from cloud.ssh_machine import HARDWARE_COMMAND

from scribblez.dashboard.shared_json import SharedJson
from scribblez.dashboard.tasks import MachineRecord
from scribblez.paths import DEFAULT_MOUNT_ROOT

POOL_PATH = DEFAULT_MOUNT_ROOT / "pool.json"
LOCALHOST = "localhost"
KINDS = ("local", "ssh")


@dataclass
class Lease:
    """A tag's hold on a pool machine. `phase` follows placement and release
    (docs/plans/tag_queue.md §4-5): reserved, starting, running, releasing,
    and held (a failed tag's machine kept for investigation)."""

    workload: str
    tag: str
    phase: str
    since: float
    reason: str = ""  # why a failed or held lease is where it is


@dataclass
class Hardware:
    """What placement checks a machine against. `gpu_memory_gb` is per GPU,
    in GiB as nvidia-smi reports it (the smallest, if the GPUs differ); None
    before a probe has answered."""

    vcpus: int | None = None
    gpu_count: int | None = None
    gpu_memory_gb: float | None = None


@dataclass
class PoolMachine:
    """One machine the pool owns.

    `machine` is the ssh machine's record (address, key material, arch),
    the same type a task-owned machine has, so everything that reaches a
    machine over ssh takes either; None for localhost. `aliases` are other
    host strings slots may use for it (see canonical_host)."""

    name: str
    kind: str  # "local" | "ssh"
    machine: MachineRecord | None = None
    aliases: list[str] = field(default_factory=list)
    hardware: Hardware = field(default_factory=Hardware)
    gpu_reserve_gb: float = 0.0
    # The generator threads a layout gives this machine, when not its own
    # vCPU arithmetic (docs/plans/tag_queue.md §3).
    generator_threads: int | None = None
    lease: Lease | None = None


@dataclass
class Pool:
    machines: list[PoolMachine] = field(default_factory=list)

    def find(self, name: str) -> PoolMachine | None:
        return next((m for m in self.machines if m.name == name), None)

    def machine(self, name: str) -> PoolMachine:
        m = self.find(name)
        if m is None:
            raise KeyError(f"no pool machine '{name}'")
        return m


def parse_hardware(report: str) -> Hardware:
    """Hardware from HARDWARE_COMMAND's output (cloud/ssh_machine.py)."""
    lines = [line.strip() for line in report.splitlines() if line.strip()]
    gpus = [float(line) / 1024 for line in lines[1:]]
    return Hardware(
        vcpus=int(lines[0]),
        gpu_count=len(gpus),
        gpu_memory_gb=min(gpus) if gpus else 0.0,
    )


def local_hardware() -> Hardware:
    """This machine's hardware, by the same command a registered one runs."""
    res = subprocess.run(HARDWARE_COMMAND, capture_output=True, text=True, check=True)
    return parse_hardware(res.stdout)


_canonical: dict[str, str] = {}


def canonical_host(host: str) -> str:
    """The hostname ssh would connect to for `host`, which may be an ssh-config
    alias and may carry a `user@`. Slots spell one machine several ways (today's
    tags have both `asus-laptop` and `dshin@asus-laptop`); this is what makes
    them compare equal. `ssh -G` only evaluates the config, never connects."""
    bare = host.split("@", 1)[-1]
    if bare not in _canonical:
        res = subprocess.run(["ssh", "-G", bare], capture_output=True, text=True)
        names = [
            ln.split(None, 1)[1] for ln in res.stdout.splitlines() if ln.startswith("hostname ")
        ]
        _canonical[bare] = names[0].lower() if res.returncode == 0 and names else bare.lower()
    return _canonical[bare]


def host_names(m: PoolMachine) -> set[str]:
    """Every canonical host name slots may use for ssh pool machine `m`."""
    assert m.machine is not None, f"{m.name} is not an ssh machine"
    return {canonical_host(h) for h in [m.machine.host, *m.aliases]}


def leased_record(pool: Pool, workload: str, tag: str, name: str) -> MachineRecord | None:
    """The machine record of pool machine `name`, if (workload, tag) holds its
    lease: how a leased machine's name resolves for that task's slots."""
    m = pool.find(name)
    if m is None or m.machine is None or m.lease is None:
        return None
    if (m.lease.workload, m.lease.tag) != (workload, tag):
        return None
    return m.machine


def _from_stored(cls, raw: dict):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in known})


def _decode(raw: dict) -> Pool:
    machines = []
    for m in raw.get("machines", []):
        m = dict(m)
        m["machine"] = _from_stored(MachineRecord, m["machine"]) if m.get("machine") else None
        m["hardware"] = _from_stored(Hardware, m.get("hardware") or {})
        m["lease"] = _from_stored(Lease, m["lease"]) if m.get("lease") else None
        machines.append(_from_stored(PoolMachine, m))
    return Pool(machines=machines)


_store = SharedJson(lambda: POOL_PATH, _decode, Pool)


def load_pool() -> Pool:
    """The process's shared Pool; an empty one before pool.json exists."""
    return _store.load()


def save_pool(pool: Pool):
    """Write `pool` atomically and keep it as the shared object."""
    _store.save(pool)
