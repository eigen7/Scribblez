"""Which queued tag goes on which pool machine (docs/plans/tag_queue.md §3-4):
eligibility of one (tag, machine) pair, and the queue-order matching of tags
to free machines. Pure functions over records, so the rules can be tested
without a dashboard.
"""

from collections.abc import Callable

from scribblez import params as params_mod
from scribblez.dashboard import tasks
from scribblez.dashboard.pool import PoolMachine
from scribblez.dashboard.queue import BUNDLE_READY, QueueEntry
from scribblez.generational import lifecycle
from scribblez.paths import TagPaths
from scribblez.workloads import WorkloadSpec, resolve
from scribblez.workloads.base import SlotPlan

# Where a tag's training state lives (state_home).
HOME_LOCAL, HOME_BUCKET = "local", "bucket"


def plan_for(spec: WorkloadSpec, params, m: PoolMachine) -> list[SlotPlan]:
    """The slots the workload's layout asks for on machine `m`."""
    return resolve(spec.layout)(params, m.hardware.vcpus, m.generator_threads)


def gpu_total(spec: WorkloadSpec, plan: list[SlotPlan], entry: QueueEntry) -> float | None:
    """The GPU memory the planned slots need together, since GPU slots on one
    machine share its GPUs: the entry's override if set, else the sum of the
    measured needs, None if any GPU role has no figure."""
    if entry.memory_override_gb is not None:
        return entry.memory_override_gb
    needs = [p.gpu_gb for p in plan if spec.role(p.role).gpu]
    return None if any(n is None for n in needs) else sum(needs)


def state_home(paths: TagPaths, task: tasks.TaskRecord) -> str | None:
    """Where the tag's training state (checkpoint, cursor, generations) lives,
    once it has any: HOME_BUCKET when its trainer delivered through the
    results bucket, where a trainer on any machine resumes it (and localhost
    holds the copy the sync pulls back); else HOME_LOCAL, only this machine's
    tag dir. None before any row is trained, when a tag may start anywhere.
    A tag with a data home follows the same rule: on an ssh machine its
    trainer uploads its checkpoint and its generations to the bucket
    (generational/data_home.py), where a fresh home restores them."""
    if not lifecycle.read_train_state(paths).get("rows_trained"):
        return None
    return HOME_BUCKET if task.trainer_sink == "r2" else HOME_LOCAL


def refusal(
    spec: WorkloadSpec,
    params,
    home: str | None,
    entry: QueueEntry,
    m: PoolMachine,
    *,
    need_bundle: bool = True,
) -> str | None:
    """Why pool machine `m` cannot take the queued tag, or None if it can.
    Checks the entry's machine list, where the tag's training state is (`home`,
    state_home: a trainer on an ssh machine cannot resume state that is only
    on localhost; it would start over, and its checkpoints would then replace
    the local ones), the kinds each planned role allows, and GPU memory; with
    `need_bundle`, also that an ssh machine has the tag's pinned bundle to
    run. Whether `m` is free is the caller's concern."""
    if entry.machines and m.name not in entry.machines:
        return "not among the machines this tag may use"
    if home == HOME_LOCAL and m.kind == "ssh":
        return "its training state is only on localhost, so its trainer must stay there"
    plan = plan_for(spec, params, m)
    for p in plan:
        role = spec.role(p.role)
        if m.kind not in role.kinds:
            return f"role {p.role} cannot run on a {m.kind} machine"
        if role.gpu and not m.hardware.gpu_count:
            return f"role {p.role} needs a GPU"
    need = gpu_total(spec, plan, entry)
    if need is None:
        return "no GPU memory figure for this configuration (set an override)"
    have = m.gpu_capacity_gb
    if need > have:
        return f"needs {need:.1f} GiB of GPU memory, has {have:.1f}"
    if need_bundle and m.kind == "ssh" and entry.bundle != BUNDLE_READY:
        return f"its bundle is {entry.bundle}"
    return None


def has_end_condition(spec: WorkloadSpec, params) -> bool:
    """Whether the tag finishes on its own: every end parameter it declares
    is bounded (docs/plans/tag_queue.md §1)."""
    ends = [f.name for f in params_mod.schema(spec.params_cls) if f.end]
    return bool(ends) and not any(params_mod.unbounded(getattr(params, n)) for n in ends)


def match(
    entries: list[QueueEntry],
    machines: list[PoolMachine],
    eligible: Callable[[QueueEntry, PoolMachine], bool],
) -> dict[tuple[str, str], str]:
    """Queued tags matched to free machines: entry key -> machine name.

    Entries are taken in queue order, each by an augmenting path: an earlier
    entry may move to another machine it is eligible for when that lets a
    later one start, but never loses its placement. So no free machine idles
    while an entry eligible for it waits behind a greedy choice. Machines are
    tried in the given order, which is the tie-break."""
    holder: dict[str, int] = {}  # machine name -> index of the entry holding it

    def assign(i: int, seen: set[str]) -> bool:
        for m in machines:
            if m.name in seen or not eligible(entries[i], m):
                continue
            seen.add(m.name)
            j = holder.get(m.name)
            if j is None or assign(j, seen):
                holder[m.name] = i
                return True
        return False

    for i in range(len(entries)):
        assign(i, set())
    return {entries[i].key: name for name, i in holder.items()}
