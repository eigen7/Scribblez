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
ssh machine. Each records the facts placement checks: vCPUs, GPU count, and
memory per GPU.

pool.json lives under the mount root beside the workload tag trees, held by
the dashboard's pool store (pool_store, shared_json.py).
"""

import subprocess
from dataclasses import dataclass, field, fields
from pathlib import Path

from cloud.ssh_machine import HARDWARE_COMMAND

from scribblez.dashboard.shared_json import SharedJson, Writer
from scribblez.dashboard.tasks import MachineRecord

LOCALHOST = "localhost"
POOL_OWNER_PREFIX = "pool/"
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
    # A rented machine's spend when the lease began: what the lease costs the
    # tag is the machine's spend since.
    spend_start: float = 0.0
    # Put the tag back at the head of the queue once released (Requeue). Its
    # own field, so no reason text (a drain error, say) can erase it.
    requeue: bool = False
    # The queue entry's eligibility, kept so a requeued tag returns with it.
    machines: list[str] = field(default_factory=list)
    memory_override_gb: float | None = None

    def held_by(self, workload: str, tag: str) -> bool:
        return (self.workload, self.tag) == (workload, tag)


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
    # The generator threads a layout gives this machine, when not its own
    # vCPU arithmetic (docs/plans/tag_queue.md §3).
    generator_threads: int | None = None
    lease: Lease | None = None
    # The Capacity entry a machine the pool rented belongs to; None for a
    # machine the operator added. A rented one is terminated once idle.
    capacity: str | None = None
    # A rental being given up (Stop all cloud spending): the queue places
    # nothing more on it, and it is terminated as soon as no lease holds it.
    retiring: bool = False

    @property
    def gpu_capacity_gb(self) -> float:
        """GPU memory slots may use: the memory per GPU."""
        return self.hardware.gpu_memory_gb or 0.0


@dataclass
class Capacity:
    """Machines the pool may rent (docs/plans/tag_queue.md §2, §4): up to
    `cap` instances of catalog type `instance_type` at a time, rented only for
    a queued tag that no free owned machine can take. The operator sets the
    cap to what the account's quota allows."""

    name: str
    instance_type: str
    spot: bool = False
    cap: int = 1


@dataclass
class Pool:
    machines: list[PoolMachine] = field(default_factory=list)
    capacity: list[Capacity] = field(default_factory=list)

    def find(self, name: str) -> PoolMachine | None:
        return next((m for m in self.machines if m.name == name), None)

    def machine(self, name: str) -> PoolMachine:
        m = self.find(name)
        if m is None:
            raise KeyError(f"no pool machine '{name}'")
        return m


def parse_hardware(report: str) -> Hardware:
    """Hardware from HARDWARE_COMMAND's output (cloud/ssh_machine.py). Only
    numeric lines after the first are GPUs: nvidia-smi prints its failures
    ("couldn't communicate with the NVIDIA driver") on stdout, and a machine in
    that state has no usable GPU."""
    lines = [line.strip() for line in report.splitlines() if line.strip()]
    gpus = [float(line) / 1024 for line in lines[1:] if _is_number(line)]
    return Hardware(
        vcpus=int(lines[0]),
        gpu_count=len(gpus),
        gpu_memory_gb=min(gpus) if gpus else 0.0,
    )


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


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


def owner_tag(name: str) -> str:
    """The ownership tag of the instance behind rented pool machine `name`:
    how the provider's listing names it, and how an instance a crash left
    unrecorded is found again. Task machines' tags have three parts
    (workload/tag/machine); a pool machine's has this one prefix."""
    return f"{POOL_OWNER_PREFIX}{name}"


def lease_spend(m: PoolMachine) -> float:
    """What pool machine `m`'s current lease has cost its tag: a rented
    machine's spend since the lease began; 0 for an owned machine."""
    if m.lease is None or m.machine is None or m.machine.instance_id is None:
        return 0.0
    return max(0.0, m.machine.spend - m.lease.spend_start)


def leased_record(pool: Pool, workload: str, tag: str, name: str) -> MachineRecord | None:
    """The machine record of pool machine `name`, if (workload, tag) holds its
    lease: how a leased machine's name resolves for that task's slots."""
    m = pool.find(name)
    if m is None or m.machine is None or m.lease is None or not m.lease.held_by(workload, tag):
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
    capacity = [_from_stored(Capacity, c) for c in raw.get("capacity", [])]
    return Pool(machines=machines, capacity=capacity)


def pool_store(mount_root: Path, writer: Writer | None = None) -> SharedJson:
    """The store of pool.json under `mount_root`: load() gives the Pool (an
    empty one before the file exists), live on the writer thread and the last
    committed copy elsewhere (shared_json); save(pool) writes it atomically."""
    return SharedJson(Path(mount_root) / "pool.json", _decode, Pool, writer or Writer())
