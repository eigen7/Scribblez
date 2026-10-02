"""The task-record store (dashboard/tasks.py): one object per task in the
store, its frozen fields in task.json and its control state in the control
store, re-read only when task.json changes under it."""

import json
import os

import pytest
from scribblez import workloads
from scribblez.dashboard import queue as queue_mod
from scribblez.dashboard import tasks
from scribblez.dashboard.control_store import ControlStore
from scribblez.dashboard.queue import QueueEntry
from scribblez.generational import lifecycle


@pytest.fixture
def spec():
    return workloads.get("kill_test")


@pytest.fixture
def store(tmp_path) -> tasks.TaskStore:
    return tasks.TaskStore(tmp_path)


def _save(store, spec, **fields) -> tasks.TaskRecord:
    task = tasks.TaskRecord(workload=spec.name, tag="t", params={}, created_at=0.0, **fields)
    store.save(spec, task)
    return task


def test_load_returns_the_saved_object_itself(store, spec):
    """Every reader of a task gets the same record, so there is no stale copy
    for a later save to write back over a newer one."""
    task = _save(store, spec)
    assert store.load(spec, "t") is task
    assert store.load(spec, "t") is store.load(spec, "t")


def test_a_file_written_by_someone_else_is_read_afresh(store, spec):
    """A CLI tool editing task.json while the dashboard runs (a params
    migration) is picked up on the next load rather than overwritten, and the
    control state stays as it was."""
    task = _save(store, spec, retired_spend=1.0)
    path = store.task_path(spec, "t")
    raw = json.loads(path.read_text())
    raw["params"] = {"migrated": True}
    path.write_text(json.dumps(raw))
    stamp = path.stat().st_mtime_ns + 1_000_000
    os.utime(path, ns=(stamp, stamp))
    fresh = store.load(spec, "t")
    assert fresh is not task
    assert fresh.params == {"migrated": True} and fresh.retired_spend == 1.0
    assert store.load(spec, "t") is fresh


def test_task_json_holds_only_the_frozen_fields(store, spec):
    """Tools outside the dashboard read task.json for params; the control
    state is the control store's, and survives a restart there."""
    task = _save(store, spec)
    task.retired_spend = 3.5
    task.gates = {"generate": "ahead"}
    store.save(spec, task)
    tag_dir = store.task_path(spec, "t").parent
    assert [p.name for p in tag_dir.iterdir()] == ["task.json"]  # no temporary left
    assert set(json.loads(store.task_path(spec, "t").read_text())) == set(tasks.FROZEN_FIELDS)
    fresh = tasks.TaskStore(store.mount_root).load(spec, "t")  # a restarted dashboard
    assert fresh.retired_spend == 3.5 and fresh.gates == {"generate": "ahead"}


def test_a_deleted_tag_is_forgotten(store, spec):
    task = _save(store, spec, retired_spend=2.0)
    store.delete(spec, "t")
    assert store.load(spec, "t") is None
    again = _save(store, spec)
    assert again is not task
    assert tasks.TaskStore(store.mount_root).load(spec, "t").retired_spend == 0.0


def test_machines_round_trip(store, spec):
    task = _save(store, spec)
    task.machines.append(
        tasks.MachineRecord(name="m1", provider="manual", host="ubuntu@1.2.3.4", gpu_count=1)
    )
    store.save(spec, task)
    loaded = tasks.TaskStore(store.mount_root).load(spec, "t")  # a fresh read
    assert loaded.machine("m1").host == "ubuntu@1.2.3.4"
    assert loaded.machine("m1").gpu_count == 1
    assert loaded.slots_on("m1") == []


def test_fields_a_record_no_longer_declares_are_dropped_on_read(tmp_path, spec):
    """A record written before a field was removed still loads: the stale keys
    are dropped, and the next save writes it without them."""
    control = ControlStore(tmp_path)
    store = tasks.TaskStore(tmp_path, control)
    _save(store, spec)
    stored = {"workers": [{"worker_id": "w0", "role": "generate", "kind": "local",
                           "desired_state": "paused", "pod_id": "abc"}], "gone": True}  # fmt: skip
    control.put("task", f"{spec.name}/t", json.dumps(stored))
    loaded = tasks.TaskStore(tmp_path, control).load(spec, "t")
    assert loaded.workers[0].worker_id == "w0"
    tasks.TaskStore(tmp_path, control).save(spec, loaded)
    raw = json.loads(control.get("task", f"{spec.name}/t"))
    assert "gone" not in raw and "pod_id" not in raw["workers"][0]


def test_a_task_json_from_before_the_control_store_is_imported_once(store, spec):
    """The migration adopts task.json's control state, leaving the file as it
    is until the row is committed; the next save drops those fields from it.
    A row already there is never overwritten by a file."""
    path = store.task_path(spec, "t")
    path.parent.mkdir(parents=True)
    old = {"workload": spec.name, "tag": "t", "params": {}, "created_at": 0.0,
           "retired_spend": 4.0, "gates": {"generate": "ahead"}}  # fmt: skip
    path.write_text(json.dumps(old))
    assert store.import_json() == [f"{spec.name}/t"]
    assert json.loads(path.read_text())["retired_spend"] == 4.0  # untouched
    task = store.load(spec, "t")
    assert task.retired_spend == 4.0 and task.gates == {"generate": "ahead"}
    store.save(spec, task)
    assert "retired_spend" not in json.loads(path.read_text())
    path.write_text(json.dumps({**old, "retired_spend": 9.0}))  # a stale copy restored
    assert store.import_json() == []
    assert tasks.TaskStore(store.mount_root).load(spec, "t").retired_spend == 4.0


def test_a_tag_deleted_while_it_is_read_reads_as_gone(store, spec):
    """A status read lists the tag dirs, then loads each; a Delete landing in
    between leaves a task.json that was there at the stat and is gone at the
    read. The read sees no tag, rather than failing the whole page."""
    _save(store, spec)
    entry = store._entry(spec, "t")
    stamp = tasks._mtime(entry.path)
    entry.path.unlink()
    assert entry._read(stamp) is None


def _slot(worker_id: str, desired_state: str, finished: bool = False) -> tasks.WorkerRecord:
    return tasks.WorkerRecord(worker_id, "generate", "local", desired_state, finished=finished)


def _failed_slot() -> tasks.WorkerRecord:
    w = _slot("b", "paused")
    w.failed = "crashed 3 times"
    return w


@pytest.mark.parametrize(
    ("slots", "expected"),
    [
        ([], tasks.IDLE),
        ([_slot("a", "paused")], tasks.PAUSED),
        ([_slot("a", "paused"), _slot("b", "running")], tasks.RUNNING),
        ([_slot("a", "paused", finished=True), _slot("b", "paused")], tasks.PAUSED),
        ([_slot("a", "paused", finished=True)], tasks.COMPLETE),
    ],
)
def test_state_of_a_workload_without_an_end_condition(store, spec, slots, expected):
    """kill_test has no WorkloadSpec.complete: it is complete while every slot
    is finished."""
    assert store.state(spec, _save(store, spec, workers=slots), None) == expected


def _entry(bundle: str = queue_mod.BUNDLE_NONE) -> QueueEntry:
    return QueueEntry("kill_test", "t", 0.0, bundle=bundle)


@pytest.mark.parametrize(
    ("fields", "entry", "expected"),
    [
        ({}, _entry(), tasks.QUEUED),
        ({"workers": [_slot("a", "running")]}, _entry(), tasks.RUNNING),
        ({}, _entry(queue_mod.BUNDLE_FAILED_PREFIX + "no credentials"), tasks.FAILED),
        ({"failure": "gen-0: exited 3 times"}, None, tasks.FAILED),
        ({"workers": [_slot("a", "running"), _failed_slot()]}, None, tasks.FAILED),
        ({"workers": [_slot("a", "paused", finished=True)], "failure": "x"}, None, tasks.COMPLETE),
    ],
)
def test_queued_and_failed_states(store, spec, fields, entry, expected):
    """Complete outranks failed, and failed outranks running and queued: a
    failure waits on the operator whatever else the tag is doing."""
    assert store.state(spec, _save(store, spec, **fields), entry) == expected


def test_state_is_complete_from_the_data_once_the_slots_are_gone(store):
    """The tag queue removes a completed tag's slots; its end condition still
    shows it complete."""
    spec = workloads.get("position_eval")
    task = tasks.TaskRecord(spec.name, "t", {"max_rows": 1000}, 0.0)
    store.save(spec, task)
    assert store.state(spec, task, None) == tasks.IDLE
    lifecycle.write_train_state(store.paths(spec, "t"), {"rows_trained": 1000})
    assert store.state(spec, task, None) == tasks.COMPLETE


def test_disk_bytes_counts_a_hard_linked_file_once(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a").write_bytes(b"x" * 100_000)
    os.link(tmp_path / "a", tmp_path / "sub" / "a_link")
    one = os.lstat(tmp_path / "a").st_blocks * 512
    assert tasks._disk_bytes(tmp_path) == one
