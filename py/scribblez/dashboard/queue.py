"""The tag queue's records (docs/plans/tag_queue.md §4): tags waiting for a
pool machine, in the order they are to be placed.

queue.json lives under the mount root beside pool.json, held by the
dashboard's queue store (queue_store, shared_json.py). The queue is global across workloads;
each entry's eligibility decides which pool machines may take it.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path

from scribblez.dashboard.shared_json import SharedJson, Writer

# The states of a queued tag's bundle (QueueEntry.bundle). A tag that may land
# on an ssh machine has its bundle built and pinned when it is enqueued, so
# its remote slots run the code the operator enqueued rather than whatever the
# tree holds hours later.
BUNDLE_NONE = "none"  # only localhost is eligible: local slots run the checkout
BUNDLE_BUILDING = "building"
BUNDLE_READY = "ready"


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


def _decode(raw: dict) -> Queue:
    known = {f.name for f in fields(QueueEntry)}
    return Queue(
        entries=[QueueEntry(**{k: v for k, v in e.items() if k in known}) for e in raw["entries"]]
    )


def queue_store(mount_root: Path, writer: Writer | None = None) -> SharedJson:
    """The store of queue.json under `mount_root`: load() gives the Queue (an
    empty one before the file exists), live on the writer thread and the last
    committed copy elsewhere (shared_json); save(queue) writes it."""
    return SharedJson(Path(mount_root) / "queue.json", _decode, Queue, writer or Writer())
