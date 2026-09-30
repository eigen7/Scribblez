"""The stores' writer rule (shared_json): one thread writes and holds the live
objects; every other thread reads the last committed copy, and cannot write."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from scribblez.dashboard import pool as pool_mod
from scribblez.dashboard import tasks
from scribblez.dashboard.pool import PoolMachine
from scribblez.dashboard.shared_json import Writer
from scribblez.workloads.position_eval import SPEC


@pytest.fixture
def writer():
    """A writer thread claimed by a Writer, and a way to run a job on it."""
    thread = ThreadPoolExecutor(max_workers=1)
    w = Writer()
    w.claim(thread.submit(threading.current_thread).result())
    yield w, lambda fn, *a: thread.submit(fn, *a).result()
    thread.shutdown()


def test_a_save_off_the_writer_fails(tmp_path, writer):
    w, on_writer = writer
    store = pool_mod.pool_store(tmp_path, w)
    pool = on_writer(store.load)
    with pytest.raises(AssertionError, match="off the writer thread"):
        store.save(pool)
    on_writer(store.save, pool)  # the writer may


def test_a_reader_sees_what_was_committed_and_nothing_else(tmp_path, writer):
    w, on_writer = writer
    store = pool_mod.pool_store(tmp_path, w)
    live = on_writer(store.load)
    live.machines.append(PoolMachine(name="box", kind="ssh"))
    assert store.load().machines == []  # a step in progress is not visible
    on_writer(store.save, live)
    read = store.load()
    assert [m.name for m in read.machines] == ["box"]
    assert read is not live  # the reader's copy is not the writer's record
    live.machines.clear()
    assert [m.name for m in store.load().machines] == ["box"]


def test_the_writer_cannot_save_a_readers_copy(tmp_path, writer):
    """The mistake the rule invites: a handler loads on the event loop and
    hands the record to a command. Saving it would put a stale copy back."""
    w, on_writer = writer
    store = pool_mod.pool_store(tmp_path, w)
    on_writer(store.save, on_writer(store.load))
    copy = store.load()
    with pytest.raises(AssertionError, match="reader's copy"):
        on_writer(store.save, copy)


def test_a_task_is_deleted_only_on_the_writer(tmp_path, writer):
    w, on_writer = writer
    store = tasks.TaskStore(tmp_path, w)
    on_writer(store.create, SPEC, "t", {})
    assert store.load(SPEC, "t") is not None
    with pytest.raises(AssertionError, match="off the writer thread"):
        store.delete(SPEC, "t")
    on_writer(store.delete, SPEC, "t")
    assert store.load(SPEC, "t") is None


def test_unclaimed_every_thread_reads_and_writes_the_live_object(tmp_path):
    """A single-threaded test or tool never claims a writer."""
    store = pool_mod.pool_store(tmp_path)
    pool = store.load()
    store.save(pool)
    assert store.load() is pool
