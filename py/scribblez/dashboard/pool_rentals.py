"""Machines the pool rents (docs/plans/tag_queue.md §2, §4-5): renting one for
a queued tag, and driving each rented instance's lifecycle.

A task-owned rented machine's lifecycle runs per task (WorkerManager's
machine steps). A pool-rented one belongs to no task, so it runs here, once
per pass, whatever its lease:

  leased, reserved/running/releasing   started if stopped (a spot
                                       interruption), since its tag's slots or
                                       its drain need it up
  leased, held                         stopped: a failed tag's machine is kept
                                       for investigation, paying only for its
                                       disk, until a queued tag needs it
  unleased                             terminated after IDLE_TERMINATE_SECONDS;
                                       the pool rents again on demand

Spend accrues on the machine's record; a lease's share goes to its tag
(pool.lease_spend).

Renting is journaled: the pool machine and its lease are written before the
provider is asked, and the instance carries the machine's owner tag
(pool.owner_tag). A crash after the launch but before its id is recorded
leaves an instance the next pass adopts by that tag, rather than a second
rental.
"""

import time

from cloud.providers.base import LaunchRequest, ProviderError

from scribblez.dashboard import pool as pool_mod
from scribblez.dashboard import tasks
from scribblez.dashboard.pool import Capacity, Hardware, Lease, Pool, PoolMachine
from scribblez.dashboard.workers import (
    IDLE_STOP_SECONDS,
    MACHINES_DIR,
    WorkerManager,
    _accrue_machine,
)

# An unleased rented machine is terminated after this long, like a task's
# rented machine is stopped: nothing is queued for it, and a stopped one would
# keep paying for its disk.
IDLE_TERMINATE_SECONDS = IDLE_STOP_SECONDS
BILLING = ("pending", "running")


class PoolRentals:
    def __init__(self, manager: WorkerManager):
        self._m = manager
        # Rented pool machine name -> when it was first seen unleased.
        self._idle_since: dict[str, float] = {}

    # ---- the catalog's view of a capacity entry ------------------------------

    def _mtype(self, cap: Capacity):
        mtype = next((t for t in self._m._provider().catalog() if t.id == cap.instance_type), None)
        assert mtype is not None, f"no machine type '{cap.instance_type}'"
        return mtype

    def prospect(self, cap: Capacity) -> PoolMachine:
        """The machine `cap` would rent, as eligibility sees it: the catalog's
        hardware and arch, no instance yet."""
        mtype = self._mtype(cap)
        record = tasks.MachineRecord(
            name=cap.name, provider="aws", host="", arch=mtype.arch, gpu_count=mtype.gpu_count
        )
        return PoolMachine(
            name=cap.name,
            kind="ssh",
            machine=record,
            hardware=Hardware(mtype.vcpus, mtype.gpu_count, mtype.gpu_memory_gb),
            capacity=cap.name,
        )

    def in_use(self, cap: Capacity, pool: Pool) -> int:
        """How many instances count against `cap`: the pool's machines rented
        under it, plus any live instance tagged as one of its machines that
        the pool has no record of (a crash between launch and record)."""
        recorded = {m.name for m in pool.machines if m.capacity == cap.name}
        prefix = pool_mod.owner_tag(f"{cap.name}-")
        stray = {
            inst.owner
            for inst in self._m._instance_index(False).values()
            if inst.state != "terminated"
            and inst.owner
            and inst.owner.startswith(prefix)
            and inst.owner.removeprefix(pool_mod.POOL_OWNER_PREFIX) not in recorded
        }
        return len(recorded) + len(stray)

    # ---- renting -------------------------------------------------------------

    def rent(self, cap: Capacity, pool: Pool, workload: str, tag: str) -> PoolMachine:
        """Rent a machine under `cap` for (workload, tag): record it with a
        reserved lease, then launch. A refused launch removes the record and
        raises with the provider's explanation."""
        m = self.prospect(cap)
        m.name = _next_name(pool, cap.name)
        m.machine.name = m.name
        m.lease = Lease(workload, tag, "reserved", time.time())
        pool.machines.append(m)
        pool_mod.save_pool(pool)
        try:
            self._launch(m, cap)
        except ProviderError as e:
            pool.machines.remove(m)
            pool_mod.save_pool(pool)
            raise AssertionError(self._m._provider().refusal(e, cap.instance_type)) from e
        pool_mod.save_pool(pool)
        return m

    def _launch(self, m: PoolMachine, cap: Capacity):
        provider = self._m._provider()
        inst = provider.launch(
            LaunchRequest(cap.instance_type, pool_mod.owner_tag(m.name), spot=cap.spot)
        )
        self._m._instances[0][inst.id] = inst  # listed now, not gone until the next relist
        self._adopt(m, inst, cap.spot)

    def _adopt(self, m: PoolMachine, inst, spot: bool):
        """Make `inst` the instance behind rented pool machine `m`."""
        provider = self._m._provider()
        known_hosts = MACHINES_DIR / "pool" / m.name / "known_hosts"
        known_hosts.parent.mkdir(parents=True, exist_ok=True)
        known_hosts.write_text("")  # a new instance: never a stale key
        mtype = next(t for t in provider.catalog() if t.id == inst.type_id)
        record = m.machine
        record.provider = provider.name
        record.host = f"{provider.ssh_user}@{inst.address or 'pending'}"
        record.identity_file = provider.identity_file
        record.known_hosts_file = str(known_hosts)
        record.instance_id = inst.id
        record.instance_type = inst.type_id
        record.spot = spot
        record.region = getattr(provider, "region", None)
        record.cost_per_hr = inst.cost_per_hr if inst.cost_per_hr is not None else mtype.cost_per_hr
        record.launched_at = inst.launched_at or time.time()
        _accrue_machine(record, True)

    # ---- the pass ------------------------------------------------------------

    def reconcile(self, pool: Pool):
        """Drive every rented pool machine toward what its lease wants (see
        the module docstring). Contained per machine."""
        rented = [m for m in pool.machines if m.capacity is not None]
        if not rented:
            return
        index = self._m._instance_index(True)
        for m in list(rented):
            try:
                self._reconcile_one(m, pool, index)
            except Exception as e:  # noqa: BLE001 -- retried next pass
                print(f"pool rental {m.name}: {e}")
        pool_mod.save_pool(pool)

    def _reconcile_one(self, m: PoolMachine, pool: Pool, index: dict):
        record = m.machine
        if record.instance_id is None:
            self._recover_launch(m, pool, index)
            return
        inst = index.get(record.instance_id)
        if inst is None or inst.state == "terminated":
            if m.lease is None:
                pool.machines.remove(m)  # terminated and unleased: nothing left to track
            else:
                m.lease.reason = "its instance is gone"
            return
        if inst.address and record.host != f"{record.host.split('@')[0]}@{inst.address}":
            record.host = f"{record.host.split('@')[0]}@{inst.address}"  # moved on a stop/start
        _accrue_machine(record, inst.state in BILLING)
        provider = self._m._provider()
        if m.lease is None:
            since = self._idle_since.setdefault(m.name, time.time())
            if time.time() - since >= IDLE_TERMINATE_SECONDS:
                provider.terminate(record.instance_id)
                _accrue_machine(record, False)
                pool.machines.remove(m)
                self._idle_since.pop(m.name, None)
            return
        self._idle_since.pop(m.name, None)
        if m.lease.phase == "held":
            if inst.state == "running":
                provider.stop(record.instance_id)
        elif inst.state == "stopped" and self._m._restart_allowed(f"pool:{m.name}"):
            self._m._note_restart(f"pool:{m.name}")
            provider.start(record.instance_id)
            record.launched_at = time.time()

    def _recover_launch(self, m: PoolMachine, pool: Pool, index: dict):
        """A machine recorded without an instance: a launch a crash cut short,
        or one that never ran. Adopt the instance tagged as it, else launch."""
        tagged = next(
            (
                i
                for i in index.values()
                if i.owner == pool_mod.owner_tag(m.name) and i.state != "terminated"
            ),
            None,
        )
        cap = next((c for c in pool.capacity if c.name == m.capacity), None)
        if tagged is not None:
            self._adopt(m, tagged, tagged.spot)
        elif cap is not None:
            self._launch(m, cap)
        else:
            pool.machines.remove(m)  # its capacity entry is gone; nothing to launch under


def _next_name(pool: Pool, base: str) -> str:
    """`<capacity name>-N`, the first N no pool machine has."""
    taken = {m.name for m in pool.machines}
    n = 1
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"
