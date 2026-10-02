"""The tag queue's records (docs/plans/tag_queue.md §4): tags waiting for a
pool machine, in the order they are to be placed.

The queue is one record of the control store (queue_store, control_store.py).
It is global across workloads; each entry's eligibility decides which pool
machines may take it.
"""

from dataclasses import dataclass, field, fields

from scribblez.dashboard.control_store import ControlStore, SharedRecord

# The states of a queued tag's bundle (QueueEntry.bundle). A tag that may land
# on an ssh machine has its bundle built and pinned when it is enqueued, so
# its remote slots run the code the operator enqueued rather than whatever the
# tree holds hours later.
BUNDLE_NONE = "none"  # only localhost is eligible: local slots run the checkout
BUNDLE_BUILDING = "building"
BUNDLE_READY = "ready"
# A failed build's state is this prefix and why: "failed: <why>".
BUNDLE_FAILED_PREFIX = "failed: "


@dataclass
class QueueEntry:
    """One queued tag.

    `machines` narrows eligibility to the named pool machines; empty means
    any. `memory_override_gb` stands in for the summed GPU need of the tag's
    layout, for a configuration with no measured figure. `bundle` is one of
    the BUNDLE_* states, or "failed: <why>"."""

    workload: str
    tag: str
    enqueued_at: float
    machines: list[str] = field(default_factory=list)
    memory_override_gb: float | None = None
    bundle: str = BUNDLE_NONE

    @property
    def key(self) -> tuple[str, str]:
        return (self.workload, self.tag)


@dataclass
class Queue:
    entries: list[QueueEntry] = field(default_factory=list)

    def find(self, workload: str, tag: str) -> QueueEntry | None:
        return next((e for e in self.entries if e.key == (workload, tag)), None)

    def entry(self, workload: str, tag: str) -> QueueEntry:
        e = self.find(workload, tag)
        if e is None:
            raise KeyError(f"{workload}/{tag} is not queued")
        return e


def decode(raw: dict) -> Queue:
    known = {f.name for f in fields(QueueEntry)}
    return Queue(
        entries=[QueueEntry(**{k: v for k, v in e.items() if k in known}) for e in raw["entries"]]
    )


def queue_store(control: ControlStore) -> SharedRecord:
    """The queue's record: load() gives the Queue (an empty one before the
    first save), live on the writer thread and the last committed copy
    elsewhere (control_store.py); save(queue) writes it."""
    return SharedRecord(control, "queue", "", decode, Queue)
