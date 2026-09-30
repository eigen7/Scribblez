"""Moving the control state from the JSON files into the control store, and
back (WorkerManager.import_json_stores and export_json_stores)."""

import json

from scribblez.dashboard import tasks
from scribblez.dashboard.workers import WorkerManager
from scribblez.workloads.position_eval import SPEC

_WORKER = {"worker_id": "local-0", "role": "generate", "kind": "local", "desired_state": "paused"}


def _json_mount(root):
    """A mount as the dashboard left it before the control store: a pool with
    one leased machine, a queue of one, and a tag whose task.json holds its
    slots, gates and spend."""
    lease = {"workload": SPEC.name, "tag": "a", "phase": "running", "since": 1.0}
    pool = {"machines": [{"name": "localhost", "kind": "local", "lease": lease}]}
    (root / "pool.json").write_text(json.dumps(pool))
    queue = {"entries": [{"workload": SPEC.name, "tag": "b", "enqueued_at": 2.0}]}
    (root / "queue.json").write_text(json.dumps(queue))
    for tag, extra in (("a", {"workers": [_WORKER], "gates": {"generate": "ahead"}}), ("b", {})):
        path = SPEC.paths(tag, root).root / "task.json"
        path.parent.mkdir(parents=True)
        frozen = {"workload": SPEC.name, "tag": tag, "params": {}, "created_at": 0.0}
        path.write_text(json.dumps({**frozen, "retired_spend": 1.5, **extra}))


def _state(manager: WorkerManager) -> dict:
    pool = manager.pool_store.load()
    return {
        "leases": {m.name: m.lease.tag for m in pool.machines if m.lease},
        "queued": [e.tag for e in manager.queue_store.load().entries],
        "tasks": {
            t.tag: ([w.worker_id for w in t.workers], t.gates, t.retired_spend)
            for _, t in manager.all_tasks()
        },
    }


_EXPECTED = {
    "leases": {"localhost": "a"},
    "queued": ["b"],
    "tasks": {"a": (["local-0"], {"generate": "ahead"}, 1.5), "b": ([], {}, 1.5)},
}


def test_the_json_stores_are_imported_once(tmp_path):
    _json_mount(tmp_path)
    manager = WorkerManager(tmp_path)
    assert manager.import_json_stores() == [
        "pool.json", "queue.json", f"{SPEC.name}/a", f"{SPEC.name}/b",
    ]  # fmt: skip
    assert _state(manager) == _EXPECTED
    assert not (tmp_path / "pool.json").exists()
    assert (tmp_path / "pool.pre-control-db.json").exists()
    restarted = WorkerManager(tmp_path)
    assert restarted.import_json_stores() == []
    assert _state(restarted) == _EXPECTED


def test_an_export_hands_the_state_back_to_the_json_files(tmp_path):
    """The rollback: the files hold everything again, as code from before the
    control store reads them, and a later start imports them afresh."""
    _json_mount(tmp_path)
    manager = WorkerManager(tmp_path)
    manager.import_json_stores()
    manager.export_json_stores()
    stored = json.loads(tasks.TaskStore(tmp_path).task_path(SPEC, "a").read_text())
    assert [w["worker_id"] for w in stored["workers"]] == ["local-0"]
    assert json.loads((tmp_path / "queue.json").read_text())["entries"][0]["tag"] == "b"
    assert manager.control.get("pool", "") is None  # the store holds nothing now
    again = WorkerManager(tmp_path)
    assert again.import_json_stores()  # imported afresh
    assert _state(again) == _EXPECTED
