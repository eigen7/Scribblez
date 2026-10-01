"""The control database's normalized tables (docs/plans/dashboard_state_model.md
§1, §10), in shadow mode.

The plan keeps the dashboard's control state as normalized records with
constraints: tags, machines, assignments (a tag on a machine, queue or
manual), slots and operations. The state the operator sees for a tag is a
projection of those records, not a stored field.

In shadow mode the control store's records (control_store.py: the pool, the
queue, each task's control state) stay authoritative. Each reconcile pass
imports them into these tables (a full rebuild, in one transaction), projects
every tag's state, and compares it with the state the records imply. What it
cannot import cleanly, or where the two disagree, becomes a finding. So the
schema, its constraints and the decision table (§10) meet the live records
before the constraints are enforced at commit.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from scribblez.dashboard import placement
from scribblez.dashboard import pool as pool_mod
from scribblez.dashboard.workers import LOCAL_TARGET, WorkerManager

SCHEMA = """
CREATE TABLE IF NOT EXISTS tag (
    workload TEXT NOT NULL,
    name TEXT NOT NULL,
    -- What the operator wants: 'idle', 'queued' (placed or waiting), 'manual'.
    desire TEXT NOT NULL CHECK (desire IN ('idle', 'queued', 'manual')),
    queue_pos INTEGER,  -- 1-based, only while queued and unplaced
    result TEXT CHECK (result IN ('done', 'failed')),
    result_reason TEXT,
    home TEXT CHECK (home IN ('local', 'bucket')),  -- NULL before any training
    PRIMARY KEY (workload, name),
    CHECK (queue_pos IS NULL OR desire = 'queued')
);
CREATE TABLE IF NOT EXISTS machine (
    name TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('local', 'ssh')),
    -- 'pool': a pool machine; 'task': a task-owned machine (PR 5 moves these
    -- into the pool); 'host': a bare ssh host string no machine record names.
    origin TEXT NOT NULL CHECK (origin IN ('pool', 'task', 'host')),
    capacity TEXT,  -- the capacity entry a pool rental belongs to
    instance_id TEXT,
    retiring INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS assignment (
    workload TEXT NOT NULL,
    tag TEXT NOT NULL,
    machine TEXT NOT NULL REFERENCES machine (name),
    kind TEXT NOT NULL CHECK (kind IN ('queue', 'manual')),
    phase TEXT NOT NULL CHECK (phase IN ('reserved', 'moving', 'running', 'releasing', 'held')),
    reason TEXT NOT NULL DEFAULT '',
    requeue INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (workload, tag, machine),
    FOREIGN KEY (workload, tag) REFERENCES tag (workload, name)
);
-- A queue assignment is exclusive: one per machine and one per tag.
CREATE UNIQUE INDEX IF NOT EXISTS one_queue_per_machine ON assignment (machine)
    WHERE kind = 'queue';
CREATE UNIQUE INDEX IF NOT EXISTS one_queue_per_tag ON assignment (workload, tag)
    WHERE kind = 'queue';
-- ... and never shares a machine with a manual one.
CREATE TRIGGER IF NOT EXISTS queue_excludes_manual BEFORE INSERT ON assignment
WHEN EXISTS (
    SELECT 1 FROM assignment a
    WHERE a.machine = NEW.machine AND a.kind != NEW.kind
      AND (a.kind = 'queue' OR NEW.kind = 'queue')
)
BEGIN
    SELECT RAISE(ABORT, 'a queue assignment shares its machine with a manual one');
END;
CREATE TABLE IF NOT EXISTS slot (
    workload TEXT NOT NULL,
    tag TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    role TEXT NOT NULL,
    machine TEXT NOT NULL REFERENCES machine (name),
    wanted TEXT NOT NULL CHECK (wanted IN ('run', 'park', 'stop')),
    outcome TEXT CHECK (outcome IN ('finished', 'crashed', 'failed', 'lost')),
    outcome_source TEXT,
    PRIMARY KEY (workload, tag, worker_id),
    FOREIGN KEY (workload, tag) REFERENCES tag (workload, name)
);
-- The outbox of actions on the world (§3). Empty until PR 4 writes to it.
CREATE TABLE IF NOT EXISTS operation (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    target TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'done', 'failed')),
    created_at REAL NOT NULL,
    finished_at REAL,
    result TEXT
);
"""

# The assignment phases that project to "running" (§1).
_RUNNING_PHASES = ("reserved", "moving", "running")


@dataclass(frozen=True)
class Finding:
    """Something the shadow import could not reconcile: `kind` names the rule
    (a row of §10's decision table, a constraint), `detail` says what."""

    workload: str
    tag: str
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.workload}/{self.tag}: {self.kind}: {self.detail}"


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def import_stores(conn: sqlite3.Connection, manager: WorkerManager) -> list[Finding]:
    """Rebuild the normalized tables from the manager's stores, in one
    transaction. Returns what could not be imported cleanly, per the
    decision table in docs/plans/dashboard_state_model.md §10."""
    findings: list[Finding] = []
    pool = manager.pool_store.load()
    queue = manager.queue_store.load()
    tasks_now = list(manager.tasks.load_all())  # read only: never all_tasks' first-sight save
    with conn:
        for table in ("slot", "assignment", "machine", "tag"):
            conn.execute(f"DELETE FROM {table}")
        for m in pool.machines:
            _insert_machine(conn, m.name, m.kind, "pool", m)
        leases = {(m.lease.workload, m.lease.tag): m for m in pool.machines if m.lease}
        waiting = {(e.workload, e.tag): i for i, e in enumerate(queue.entries, 1)}
        known = {(spec.name, task.tag) for spec, task in tasks_now}
        for workload, tag in sorted(set(waiting) | set(leases)):
            if (workload, tag) not in known:
                findings.append(Finding(workload, tag, "no task", "queued or leased, no task.json"))
        # Every tag and queue assignment first: a lease is the stores' own
        # record of who holds a machine, so it must win over a manual claim.
        for spec, task in tasks_now:
            findings += _import_tag(conn, manager, spec, task, waiting, leases)
        for spec, task in tasks_now:
            findings += _import_slots(conn, pool, spec, task, leases)
    return findings


def _import_tag(conn, manager, spec, task, waiting, leases) -> list[Finding]:
    """One row of the decision table: the tag and its queue assignment."""
    findings = []
    key = (spec.name, task.tag)
    lease_machine = leases.get(key)
    queued_at = waiting.get(key)
    if queued_at is not None and lease_machine is not None:
        # The repair _drop_stale_entries already makes: the lease wins.
        findings.append(Finding(*key, "stale queue entry", "queued while it holds a lease"))
        queued_at = None
    if queued_at is not None and task.workers:
        # The tune-wd0.01 case: refused at the switch, for the operator to settle.
        findings.append(
            Finding(*key, "queued with slots", "dequeue it, or remove its slots, before PR 3")
        )
    if lease_machine is not None or queued_at is not None:
        desire = "queued"
    elif task.workers:
        desire = "manual"
    else:
        desire = "idle"
    failed = lease_machine is not None and lease_machine.lease.phase == "held"
    home = placement.state_home(manager.tasks.paths(spec, task.tag), task)
    conn.execute(
        "INSERT INTO tag VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            *key,
            desire,
            queued_at if lease_machine is None else None,
            "failed" if failed else None,
            lease_machine.lease.reason if failed else None,
            {placement.HOME_LOCAL: "local"}.get(home),
        ),
    )
    if lease_machine is not None:
        lease = lease_machine.lease
        conn.execute(
            "INSERT INTO assignment VALUES (?, ?, ?, 'queue', ?, ?, ?)",
            (*key, lease_machine.name, _phase(lease.phase), lease.reason, int(lease.requeue)),
        )
    return findings


def _import_slots(conn, pool, spec, task, leases) -> list[Finding]:
    """The tag's slots, and for a tag without a queue lease a manual
    assignment on each machine where one of its slots wants to run (running
    or gated). Paused and finished slots hold nothing, as under today's busy
    rule, so a hand-run tag the operator paused does not claim its machine."""
    findings = []
    key = (spec.name, task.tag)
    lease_machine = leases.get(key)
    manual_machines = set()
    for w in task.workers:
        machine, problem = _slot_machine(conn, pool, task, w)
        if problem:
            findings.append(Finding(*key, "slot machine", f"{w.worker_id}: {problem}"))
        wanted, outcome, source = _slot_state(task, w)
        holds = wanted != "stop"
        if lease_machine is None and holds and machine not in manual_machines:
            error = _insert_or_refuse(
                conn,
                "INSERT INTO assignment VALUES (?, ?, ?, 'manual', 'running', '', 0)",
                (*key, machine),
            )
            if error:
                findings.append(Finding(*key, "constraint", f"manual on {machine}: {error}"))
            else:
                manual_machines.add(machine)
        conn.execute(
            "INSERT INTO slot VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (*key, w.worker_id, w.role, machine, wanted, outcome, source),
        )
    return findings


def _insert_or_refuse(conn, sql: str, row: tuple) -> str | None:
    """Run one insert under a savepoint: a constraint refusal rolls back just
    this row and is returned, and the rest of the import stands."""
    conn.execute("SAVEPOINT row")
    try:
        conn.execute(sql, row)
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK TO row")
        return str(e)
    finally:
        conn.execute("RELEASE row")
    return None


def _phase(lease_phase: str) -> str:
    """A lease phase as an assignment phase ("starting" is running's start)."""
    return "running" if lease_phase == "starting" else lease_phase


def _slot_state(task, w) -> tuple[str, str | None, str | None]:
    """(wanted, outcome, outcome source) from today's slot fields."""
    wanted = "stop" if w.desired_state != "running" else "park" if w.role in task.gates else "run"
    if w.failed:
        return wanted, "failed", "queue"
    if w.finished:
        return wanted, "finished", "exited"
    return wanted, None, None


def _slot_machine(conn, pool, task, w) -> tuple[str, str | None]:
    """The machine row slot `w` runs on, inserted if it is not yet known (a
    pool machine where one matches, else its task machine or bare host), and
    what is wrong with it, if anything."""
    if w.kind == "local":
        local = next((m for m in pool.machines if m.kind == "local"), None)
        if local is not None:
            return local.name, None
        return _ensure_machine(conn, LOCAL_TARGET, "local", "host"), None
    record = task.find_machine(w.machine) if w.machine else None
    if w.machine and record is None:
        m = pool.find(w.machine)
        if m is not None and m.lease is not None and m.lease.held_by(task.workload, task.tag):
            return m.name, None
        # A slot naming a pool machine its tag no longer leases (or that is
        # gone): kept, on a placeholder row, so the finding is the only effect.
        name = _ensure_machine(conn, f"host:{w.machine}", "ssh", "host")
        return name, f"names pool machine {w.machine}, which its tag does not lease"
    target = pool_mod.canonical_host(record.host if record else w.host)
    pooled = next(
        (m for m in pool.machines if m.kind == "ssh" and target in pool_mod.host_names(m)), None
    )
    if pooled is not None:
        return pooled.name, None
    if record is not None:
        name = f"task:{task.workload}/{task.tag}/{record.name}"
        return _ensure_machine(conn, name, "ssh", "task", record), None
    return _ensure_machine(conn, f"host:{target}", "ssh", "host"), None


def _ensure_machine(conn, name: str, kind: str, origin: str, record=None) -> str:
    conn.execute(
        "INSERT OR IGNORE INTO machine VALUES (?, ?, ?, NULL, ?, 0)",
        (name, kind, origin, record.instance_id if record is not None else None),
    )
    return name


def _insert_machine(conn, name: str, kind: str, origin: str, m) -> None:
    record = m.machine
    conn.execute(
        "INSERT INTO machine VALUES (?, ?, ?, ?, ?, ?)",
        (
            name,
            kind,
            origin,
            m.capacity,
            record.instance_id if record is not None else None,
            int(m.retiring),
        ),
    )


def project(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    """Every tag's state as the operator sees it (§1's projection rules)."""
    out = {}
    for workload, name, desire, result in conn.execute(
        "SELECT workload, name, desire, result FROM tag"
    ):
        row = conn.execute(
            "SELECT phase FROM assignment WHERE workload = ? AND tag = ? AND kind = 'queue'",
            (workload, name),
        ).fetchone()
        phase = row[0] if row else None
        if result == "failed" or phase == "held":
            state = "failed"
        elif phase == "releasing":
            state = "releasing"
        elif phase in _RUNNING_PHASES:
            state = "running"
        elif result == "done":
            state = "done"
        else:
            state = desire
        out[(workload, name)] = state
    return out


def legacy_states(manager: WorkerManager) -> dict[tuple[str, str], str]:
    """Every tag's state as today's JSON stores imply it: what the projection
    must agree with while the stores stay authoritative."""
    pool = manager.pool_store.load()
    queued = {(e.workload, e.tag) for e in manager.queue_store.load().entries}
    leases = {(m.lease.workload, m.lease.tag): m.lease for m in pool.machines if m.lease}
    out = {}
    for spec, task in manager.tasks.load_all():
        key = (spec.name, task.tag)
        lease = leases.get(key)
        if lease is not None:
            out[key] = {"held": "failed", "releasing": "releasing"}.get(lease.phase, "running")
        elif key in queued:
            out[key] = "queued"
        elif task.workers:
            out[key] = "manual"
        else:
            out[key] = "idle"
    return out


class ShadowControl:
    """The shadow-mode driver: once per reconcile pass, rebuild the control
    database from the JSON stores, compare its projection with the stores'
    own reading, and log findings when they change (so a standing one is
    reported once, not every pass)."""

    def __init__(self, manager: WorkerManager):
        self._m = manager
        self.path = manager.mount_root / "control.db"
        self.findings: list[Finding] = []
        self.projected: dict[tuple[str, str], str] = {}

    def sync(self) -> list[Finding]:
        conn = connect(self.path)
        try:
            findings = import_stores(conn, self._m)
            projected = project(conn)
        finally:
            conn.close()
        legacy = legacy_states(self._m)
        for key in sorted(set(projected) | set(legacy)):
            if projected.get(key) != legacy.get(key):
                findings.append(
                    Finding(
                        *key,
                        "disagreement",
                        f"projected {projected.get(key)}, stores say {legacy.get(key)}",
                    )
                )
        for f in findings:
            if f not in self.findings:
                print(f"control db (shadow): {f}")
        self.findings = findings
        self.projected = projected
        return findings

    def status(self) -> dict:
        """The last sync's projection and findings (GET /api/control/shadow)."""
        return {
            "tags": [
                {"workload": w, "tag": t, "state": state}
                for (w, t), state in sorted(self.projected.items())
            ],
            "findings": [vars(f) for f in self.findings],
        }
