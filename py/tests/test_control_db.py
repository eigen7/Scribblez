"""The control database in shadow mode (dashboard/control_db.py): the schema's
constraints, each row of the migration's decision table
(docs/plans/dashboard_state_model.md §10), the projection rules (§1), and the
shadow driver that compares them with the JSON stores."""

import sqlite3

import pytest
from scribblez.dashboard import control_db, tasks
from scribblez.dashboard import pool as pool_mod
from scribblez.dashboard.pool import Hardware, Lease, PoolMachine
from scribblez.dashboard.queue import QueueEntry
from scribblez.dashboard.workers import WorkerManager
from scribblez.workloads.position_eval import SPEC


@pytest.fixture
def manager(tmp_path, monkeypatch) -> WorkerManager:
    """A manager rooted at tmp_path with localhost and one ssh machine, gpu-box,
    in the pool."""
    monkeypatch.setattr(pool_mod, "local_hardware", lambda: Hardware(28, 1, 16.0))
    monkeypatch.setattr(pool_mod, "canonical_host", lambda h: h.split("@", 1)[-1].lower())
    manager = WorkerManager(tmp_path)
    manager.add_pool_machine("localhost")
    pool = manager.pool_store.load()
    record = tasks.MachineRecord(name="gpu-box", provider="manual", host="me@gpu-box")
    pool.machines.append(PoolMachine(name="gpu-box", kind="ssh", machine=record))
    manager.pool_store.save(pool)
    return manager


def _task(manager, tag: str, *slots: tasks.WorkerRecord) -> tasks.TaskRecord:
    task = tasks.TaskRecord(workload="position_eval", tag=tag, params={}, created_at=0.0)
    task.workers = list(slots)
    manager.tasks.save(SPEC, task)
    return task


def _slot(worker_id: str, desired="running", kind="local", **kw) -> tasks.WorkerRecord:
    return tasks.WorkerRecord(
        worker_id=worker_id, role="generate", kind=kind, desired_state=desired, **kw
    )


def _lease(manager, tag: str, phase: str, machine="localhost"):
    pool = manager.pool_store.load()
    pool.machine(machine).lease = Lease("position_eval", tag, phase, 0.0)
    manager.pool_store.save(pool)


def _enqueue(manager, *tags: str):
    queue = manager.queue_store.load()
    queue.entries += [QueueEntry("position_eval", t, 0.0) for t in tags]
    manager.queue_store.save(queue)


def _run(manager, tmp_path):
    conn = control_db.connect(tmp_path / "control.db")
    findings = control_db.import_stores(conn, manager)
    projected = {tag: state for (_, tag), state in control_db.project(conn).items()}
    return conn, findings, projected


def test_each_store_combination_projects_the_state_the_stores_imply(manager, tmp_path):
    """The decision table's clean rows: waiting, placed, releasing, failed, run
    by hand, idle. The projection agrees with the stores' own reading."""
    _task(manager, "waiting")
    _task(manager, "placed", _slot("local-0"))
    _task(manager, "releasing", _slot("ssh-0", kind="ssh", machine="gpu-box"))
    _task(manager, "idle")
    _enqueue(manager, "waiting")
    _lease(manager, "placed", "running")
    _lease(manager, "releasing", "releasing", machine="gpu-box")
    conn, findings, projected = _run(manager, tmp_path)
    assert findings == []
    assert projected == {
        "waiting": "queued", "placed": "running", "releasing": "releasing", "idle": "idle",
    }  # fmt: skip
    legacy = {tag: state for (_, tag), state in control_db.legacy_states(manager).items()}
    assert legacy == projected
    assert conn.execute("SELECT queue_pos FROM tag WHERE name = 'waiting'").fetchone() == (1,)
    assert conn.execute("SELECT kind, phase FROM assignment WHERE tag = 'placed'").fetchall() == [
        ("queue", "running")
    ]


def test_a_held_lease_is_a_failed_tag(manager, tmp_path):
    _task(manager, "a", _slot("local-0", desired="paused"))
    _lease(manager, "a", "held")
    conn, findings, projected = _run(manager, tmp_path)
    assert projected == {"a": "failed"} and findings == []
    assert conn.execute("SELECT result FROM tag").fetchone() == ("failed",)


def test_a_hand_run_tag_holds_its_machine_only_while_a_slot_wants_to_run(manager, tmp_path):
    """Manual either way; a manual assignment only for a slot meant to run, as
    under today's busy rule, so paused leftovers never claim a machine."""
    _task(manager, "hand", _slot("local-0"))
    _task(manager, "leftover", _slot("local-0", desired="paused"))
    conn, findings, projected = _run(manager, tmp_path)
    assert projected == {"hand": "manual", "leftover": "manual"} and findings == []
    assert conn.execute("SELECT tag, machine, kind FROM assignment").fetchall() == [
        ("hand", "localhost", "manual")
    ]


def test_a_queued_tag_with_slots_is_refused_for_the_operator(manager, tmp_path):
    """The tune-wd0.01 case: queued, and running by hand at the same time."""
    _task(manager, "a", _slot("local-0"))
    _enqueue(manager, "a")
    _, findings, projected = _run(manager, tmp_path)
    assert [f.kind for f in findings] == ["queued with slots"]
    assert projected == {"a": "queued"}


def test_a_queue_entry_beside_a_lease_is_stale(manager, tmp_path):
    _task(manager, "a", _slot("local-0"))
    _enqueue(manager, "a")
    _lease(manager, "a", "running")
    conn, findings, projected = _run(manager, tmp_path)
    assert [f.kind for f in findings] == ["stale queue entry"]
    assert projected == {"a": "running"}
    assert conn.execute("SELECT queue_pos FROM tag").fetchone() == (None,)


def test_a_manual_slot_on_a_queue_leased_machine_is_a_constraint_finding(manager, tmp_path):
    """A tag run by hand on the machine another tag holds from the queue: the
    trigger refuses the manual assignment, the finding says so, and the rest
    of the import stands."""
    _task(manager, "placed", _slot("local-0"))
    _task(manager, "hand", _slot("local-0"))
    _lease(manager, "placed", "running")
    conn, findings, projected = _run(manager, tmp_path)
    assert [(f.tag, f.kind) for f in findings] == [("hand", "constraint")]
    assert "shares its machine" in findings[0].detail
    assert projected == {"placed": "running", "hand": "manual"}
    assert conn.execute("SELECT COUNT(*) FROM slot").fetchone() == (2,)


def test_slots_resolve_to_pool_task_and_bare_host_machines(manager, tmp_path):
    """A slot's machine: the pool machine its host matches (by any alias), its
    task machine otherwise, or its bare host string."""
    task = _task(
        manager,
        "hand",
        _slot("ssh-0", kind="ssh", host="someone@GPU-BOX"),
        _slot("ssh-1", kind="ssh", machine="m1"),
        _slot("ssh-2", kind="ssh", host="elsewhere"),
    )
    task.machines = [tasks.MachineRecord(name="m1", provider="aws", host="ubuntu@1.2.3.4")]
    manager.tasks.save(SPEC, task)
    conn, findings, _ = _run(manager, tmp_path)
    assert findings == []
    rows = dict(conn.execute("SELECT worker_id, machine FROM slot").fetchall())
    assert rows == {
        "ssh-0": "gpu-box",
        "ssh-1": "task:position_eval/hand/m1",
        "ssh-2": "host:elsewhere",
    }
    origins = dict(conn.execute("SELECT name, origin FROM machine").fetchall())
    assert origins["task:position_eval/hand/m1"] == "task"
    assert origins["host:elsewhere"] == "host"


def test_a_slot_naming_a_pool_machine_its_tag_does_not_lease_is_a_finding(manager, tmp_path):
    _task(manager, "a", _slot("ssh-0", kind="ssh", machine="gpu-box"))
    _, findings, _ = _run(manager, tmp_path)
    assert [f.kind for f in findings] == ["slot machine"]


def test_a_queue_entry_without_a_task_is_a_finding(manager, tmp_path):
    _enqueue(manager, "ghost")
    _, findings, projected = _run(manager, tmp_path)
    assert [(f.tag, f.kind) for f in findings] == [("ghost", "no task")]
    assert projected == {}


def test_the_schema_refuses_a_second_queue_assignment_on_a_machine(tmp_path):
    conn = control_db.connect(tmp_path / "control.db")
    conn.execute("INSERT INTO machine VALUES ('m', 'ssh', 'pool', NULL, NULL, 0)")
    for tag in ("a", "b"):
        conn.execute("INSERT INTO tag VALUES ('w', ?, 'queued', NULL, NULL, NULL, NULL)", (tag,))
    conn.execute("INSERT INTO assignment VALUES ('w', 'a', 'm', 'queue', 'running', '', 0)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO assignment VALUES ('w', 'b', 'm', 'queue', 'running', '', 0)")
    # Manual assignments share a machine with each other, never with a queue one.
    conn.execute("INSERT INTO machine VALUES ('n', 'ssh', 'pool', NULL, NULL, 0)")
    conn.execute("INSERT INTO assignment VALUES ('w', 'a', 'n', 'manual', 'running', '', 0)")
    conn.execute("INSERT INTO assignment VALUES ('w', 'b', 'n', 'manual', 'running', '', 0)")
    conn.execute("INSERT INTO tag VALUES ('w', 'c', 'manual', NULL, NULL, NULL, NULL)")
    with pytest.raises(sqlite3.IntegrityError, match="shares its machine"):
        conn.execute("INSERT INTO assignment VALUES ('w', 'c', 'm', 'manual', 'running', '', 0)")


def test_the_shadow_rebuilds_each_pass_and_reports_a_finding_once(manager, capsys):
    _task(manager, "a", _slot("local-0"))
    _enqueue(manager, "a")
    shadow = control_db.ShadowControl(manager)
    assert [f.kind for f in shadow.sync()] == ["queued with slots"]
    assert shadow.sync() == shadow.findings  # a rebuild, not an accumulation
    assert capsys.readouterr().out.count("queued with slots") == 1  # reported once
    assert (manager.mount_root / "control.db").is_file()
    status = shadow.status()
    assert status["tags"] == [{"workload": "position_eval", "tag": "a", "state": "queued"}]
    assert status["findings"][0]["kind"] == "queued with slots"


def test_the_import_only_reads_the_tag_records(manager, tmp_path):
    """The dry run reads the live mount: loading tasks there must never write
    one back (all_tasks saves each on first sight; load_all does not)."""
    _task(manager, "a", _slot("ssh-0", kind="ssh", host="elsewhere"))
    path = manager.tasks.task_path(SPEC, "a")
    before = path.stat().st_mtime_ns
    fresh = WorkerManager(manager.mount_root)
    _run(fresh, tmp_path)
    assert path.stat().st_mtime_ns == before


def test_a_disagreement_between_the_projection_and_the_stores_is_a_finding(
    manager, monkeypatch, capsys
):
    """The point of shadow mode: where the database's projection and the
    stores' own reading of a tag differ, the pass says so."""
    _task(manager, "a")
    monkeypatch.setattr(control_db, "legacy_states", lambda m: {("position_eval", "a"): "running"})
    findings = control_db.ShadowControl(manager).sync()
    assert [(f.tag, f.kind) for f in findings] == [("a", "disagreement")]
    assert "projected idle, stores say running" in findings[0].detail
    assert "disagreement" in capsys.readouterr().out


def test_a_tags_home_follows_its_training_state(manager, tmp_path):
    """None before any training; local or bucket after, by where its trainer
    delivered (placement.state_home)."""
    _task(manager, "fresh")
    for tag, sink in (("local", "local"), ("bucket", "r2")):
        task = _task(manager, tag)
        task.trainer_sink = sink
        manager.tasks.save(SPEC, task)
        paths = manager.tasks.paths(SPEC, tag)
        paths.train_state_path.write_text('{"rows_trained": 256, "generation_index": 0}')
    conn, _, _ = _run(manager, tmp_path)
    homes = dict(conn.execute("SELECT name, home FROM tag").fetchall())
    assert homes == {"fresh": None, "local": "local", "bucket": "bucket"}


def test_a_finished_tag_projects_as_done_unless_it_still_holds_a_machine(tmp_path):
    """The result decides once no queue assignment says otherwise."""
    conn = control_db.connect(tmp_path / "control.db")
    conn.execute("INSERT INTO machine VALUES ('m', 'ssh', 'pool', NULL, NULL, 0)")
    for tag in ("finished", "releasing"):
        conn.execute("INSERT INTO tag VALUES ('w', ?, 'queued', NULL, 'done', NULL, NULL)", (tag,))
    conn.execute(
        "INSERT INTO assignment VALUES ('w', 'releasing', 'm', 'queue', 'releasing', '', 0)"
    )
    assert control_db.project(conn) == {("w", "finished"): "done", ("w", "releasing"): "releasing"}
