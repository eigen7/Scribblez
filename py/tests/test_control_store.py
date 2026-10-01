"""The control store (control_store.py): one thread writes and holds the live
objects; every other thread reads the last committed copy, and cannot write;
a transaction commits several records together."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from scribblez.dashboard import pool as pool_mod
from scribblez.dashboard import queue as queue_mod
from scribblez.dashboard import tasks
from scribblez.dashboard.control_store import ControlStore, Writer
from scribblez.dashboard.pool import PoolMachine
from scribblez.dashboard.queue import QueueEntry
from scribblez.workloads.position_eval import SPEC


@pytest.fixture
def writer(tmp_path):
    """A control store whose writer is a thread of its own, and a way to run
    a job on that thread."""
    thread = ThreadPoolExecutor(max_workers=1)
    w = Writer()
    w.claim(thread.submit(threading.current_thread).result())
    yield ControlStore(tmp_path, w), lambda fn, *a: thread.submit(fn, *a).result()
    thread.shutdown()


def test_a_save_off_the_writer_fails(writer):
    control, on_writer = writer
    store = pool_mod.pool_store(control)
    pool = on_writer(store.load)
    with pytest.raises(AssertionError, match="off the writer thread"):
        store.save(pool)
    on_writer(store.save, pool)  # the writer may


def test_a_reader_sees_what_was_committed_and_nothing_else(writer):
    control, on_writer = writer
    store = pool_mod.pool_store(control)
    live = on_writer(store.load)
    live.machines.append(PoolMachine(name="box", kind="ssh"))
    assert store.load().machines == []  # a step in progress is not visible
    on_writer(store.save, live)
    read = store.load()
    assert [m.name for m in read.machines] == ["box"]
    assert read is not live  # the reader's copy is not the writer's record
    live.machines.clear()
    assert [m.name for m in store.load().machines] == ["box"]


def test_the_writer_cannot_save_a_readers_copy(writer):
    """The mistake the rule invites: a handler loads on the event loop and
    hands the record to a command. Saving it would put a stale copy back."""
    control, on_writer = writer
    store = pool_mod.pool_store(control)
    on_writer(store.save, on_writer(store.load))
    copy = store.load()
    with pytest.raises(AssertionError, match="reader's copy"):
        on_writer(store.save, copy)


def test_a_transaction_commits_its_records_together(writer):
    """A reader sees none of a transition until all of it is committed: a tag
    never reads as both queued and placed."""
    control, on_writer = writer
    pool, queue = pool_mod.pool_store(control), queue_mod.queue_store(control)

    def place() -> tuple:
        with control.transaction():
            p = pool.load()
            p.machines.append(PoolMachine(name="box", kind="ssh"))
            pool.save(p)
            with control.transaction():  # nested: commits with the outer one
                q = queue.load()
                q.entries.append(QueueEntry("position_eval", "t", 0.0))
                queue.save(q)
            return _read_elsewhere(pool, queue)

    assert on_writer(place) == ([], [])  # mid-transaction, from another thread
    assert _read(pool, queue) == (["box"], ["t"])


def test_a_transaction_that_raises_still_commits_what_it_wrote(writer):
    """The live objects keep what the step changed, so the rows must too."""
    control, on_writer = writer
    store = pool_mod.pool_store(control)

    def fail_midway():
        with control.transaction():
            p = store.load()
            p.machines.append(PoolMachine(name="box", kind="ssh"))
            store.save(p)
            raise RuntimeError("the provider refused")

    with pytest.raises(RuntimeError):
        on_writer(fail_midway)
    assert [m.name for m in store.load().machines] == ["box"]


def test_a_commit_during_a_read_is_seen_by_the_next_read(writer, tmp_path, monkeypatch):
    """A reader caches its copy under the version from before its read, so a
    commit landing mid-read is picked up next time, not hidden until some
    unrelated commit."""
    control, on_writer = writer
    store = tasks.TaskStore(tmp_path, control)
    on_writer(store.create, SPEC, "t", {})
    entry = store._entry(SPEC, "t")
    read = entry._read

    def read_then_commit(stamp):
        copy = read(stamp)
        # The operator's command commits meanwhile. It writes the row alone:
        # a save would wait on the entry lock this read holds, which a live
        # reader never keeps while waiting on the writer.
        stored = {**json.loads(control.get("task", f"{SPEC.name}/t")), "retired_spend": 42.0}
        on_writer(control.put, "task", f"{SPEC.name}/t", json.dumps(stored))
        return copy

    monkeypatch.setattr(entry, "_read", read_then_commit)
    assert store.load(SPEC, "t").retired_spend == 0.0  # read before the commit
    monkeypatch.setattr(entry, "_read", read)
    assert store.load(SPEC, "t").retired_spend == 42.0


def test_a_task_is_deleted_only_on_the_writer(writer, tmp_path):
    control, on_writer = writer
    store = tasks.TaskStore(tmp_path, control)
    on_writer(store.create, SPEC, "t", {})
    assert store.load(SPEC, "t") is not None
    with pytest.raises(AssertionError, match="off the writer thread"):
        store.delete(SPEC, "t")
    on_writer(store.delete, SPEC, "t")
    assert store.load(SPEC, "t") is None


def test_a_late_save_of_a_deleted_tag_does_not_bring_it_back(writer, tmp_path):
    """Seen live: a Delete ran between two steps of a reconcile pass, and the
    pass's next save of the record it had loaded re-created the tag. A new
    tag of the same name is a new record, and saves as usual."""
    control, on_writer = writer
    store = tasks.TaskStore(tmp_path, control)
    on_writer(store.create, SPEC, "t", {})
    held = on_writer(store.load, SPEC, "t")
    on_writer(store.delete, SPEC, "t")
    on_writer(store.save, SPEC, held)
    assert on_writer(store.load, SPEC, "t") is None
    assert not store.task_path(SPEC, "t").exists()
    on_writer(store.create, SPEC, "t", {})
    fresh = on_writer(store.load, SPEC, "t")
    fresh.retired_spend = 1.0
    on_writer(store.save, SPEC, fresh)
    assert store.load(SPEC, "t").retired_spend == 1.0


def test_unclaimed_every_thread_reads_and_writes_the_live_object(tmp_path):
    """A single-threaded test or tool never claims a writer."""
    store = pool_mod.pool_store(ControlStore(tmp_path))
    pool = store.load()
    store.save(pool)
    assert store.load() is pool


def _read(pool, queue) -> tuple[list[str], list[str]]:
    return [m.name for m in pool.load().machines], [e.tag for e in queue.load().entries]


def _read_elsewhere(pool, queue) -> tuple[list[str], list[str]]:
    with ThreadPoolExecutor(max_workers=1) as reader:
        return reader.submit(_read, pool, queue).result()
