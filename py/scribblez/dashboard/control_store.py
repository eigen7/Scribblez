"""The dashboard's control records, in one SQLite database written by one
thread (docs/plans/dashboard_state_model.md §2, §9).

The records are the pool, the tag queue, and each tag's control state: its
slots, machines, gates and spend. A tag's frozen params stay in its task.json,
which tools outside the dashboard read (tasks.py). Each record is one row of
<mount>/control.db, its dataclass as JSON.

**One writer.** The worker manager's blocking thread runs the reconcile pass
and every command a request handler submits. It holds one live object per
record, and a save writes that object's row. A save commits at once, unless it
is inside `transaction()`: a transition that changes several records (placing
a tag writes the pool, the queue and the task) commits them together, so a
crash never leaves half of one.

**Readers.** Every other thread (a status request on the event loop, say)
reads the last committed copy: the row as committed, decoded apart from the
live object, over a connection of its own that sees no uncommitted write. So
a read never sees a transition half done, and nothing a read does can reach
the writer's records. A save off the writer thread fails.
"""

import json
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS record (
    kind TEXT NOT NULL,  -- 'pool', 'queue' or 'task'
    key TEXT NOT NULL,  -- '' for the pool and the queue; 'workload/tag' for a task
    body TEXT NOT NULL,  -- the record's dataclass as JSON
    PRIMARY KEY (kind, key)
);
CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

# The meta row marking that the JSON files' control state was imported
# (WorkerManager.import_json_stores): the store, not the files, is the truth.
IMPORTED_JSON = "imported_json"


class Writer:
    """The one thread allowed to write the records. Until `claim` names one,
    any thread writes and every thread reads the live objects, as a
    single-threaded test or tool expects."""

    def __init__(self):
        self.thread: threading.Thread | None = None

    def claim(self, thread: threading.Thread):
        self.thread = thread

    def here(self) -> bool:
        """Whether the calling thread may write (and so reads live objects)."""
        return self.thread is None or threading.current_thread() is self.thread

    def check(self, what: object):
        assert self.here(), (
            f"{what} written on {threading.current_thread().name}, off the writer thread "
            f"{self.thread.name}: submit the change as a command (WorkerManager.offload)"
        )


class ControlStore:
    """The control database under one mount root: record rows, and the
    transactions that write them."""

    def __init__(self, mount_root: Path, writer: Writer | None = None):
        self.path = Path(mount_root) / "control.db"
        self.writer = writer or Writer()
        self._conn: sqlite3.Connection | None = None  # the writer's, opened on first use
        self._readers = threading.local()
        self._depth = 0  # nested transaction() blocks open on the writer
        # Commits so far, which a reader's cached copy is checked against.
        self.version = 0

    def get(self, kind: str, key: str) -> str | None:
        """The row's body, as the writer last wrote it on the writer thread
        and as last committed elsewhere; None when there is no row."""
        conn = self._write_conn() if self.writer.here() else self._read_conn()
        row = conn.execute(
            "SELECT body FROM record WHERE kind = ? AND key = ?", (kind, key)
        ).fetchone()
        return row[0] if row else None

    def put(self, kind: str, key: str, body: str):
        self.writer.check(f"{kind} record {key!r}")
        self._write_conn().execute(
            "INSERT OR REPLACE INTO record (kind, key, body) VALUES (?, ?, ?)", (kind, key, body)
        )
        self._committed()

    def delete(self, kind: str, key: str):
        self.writer.check(f"{kind} record {key!r}")
        self._write_conn().execute("DELETE FROM record WHERE kind = ? AND key = ?", (kind, key))
        self._committed()

    def meta(self, name: str) -> str | None:
        """A meta row's value, read as get() reads a record."""
        conn = self._write_conn() if self.writer.here() else self._read_conn()
        row = conn.execute("SELECT value FROM meta WHERE name = ?", (name,)).fetchone()
        return row[0] if row else None

    def set_meta(self, name: str, value: str):
        self.writer.check(f"meta {name!r}")
        self._write_conn().execute(
            "INSERT OR REPLACE INTO meta (name, value) VALUES (?, ?)", (name, value)
        )
        self._committed()

    def clear(self):
        """Delete every record and the import mark, handing the control state
        back to the JSON files (WorkerManager.export_json_stores)."""
        self.writer.check("the control store")
        with self.transaction():
            self._write_conn().execute("DELETE FROM record")
            self._write_conn().execute("DELETE FROM meta WHERE name = ?", (IMPORTED_JSON,))

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit every write inside the block together, at its end. Blocks
        nest; the outermost one commits. A block that raises still commits
        what it wrote, as the live objects keep what it changed: the
        transaction makes a transition atomic against a crash, not undoable."""
        self.writer.check("a transaction")
        conn = self._write_conn()
        if self._depth == 0:
            conn.execute("BEGIN IMMEDIATE")
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            if self._depth == 0:
                conn.execute("COMMIT")
                self.version += 1

    def _committed(self):
        """After a write: outside a transaction, that write is committed."""
        if self._depth == 0:
            self.version += 1

    def _write_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            # Opened on whichever thread first writes, then used by the writer.
            self._conn = _connect(self.path, check_same_thread=False)
        return self._conn

    def _read_conn(self) -> sqlite3.Connection:
        conn = getattr(self._readers, "conn", None)
        if conn is None:
            conn = self._readers.conn = _connect(self.path)
        return conn


def _connect(path: Path, check_same_thread: bool = True) -> sqlite3.Connection:
    """A connection in autocommit mode (a statement outside BEGIN commits by
    itself), over a write-ahead log, so readers never wait on the writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=check_same_thread)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


class SharedRecord:
    """One record of the control store, as a dataclass instance: `decode(raw)`
    builds one from its JSON, `empty()` one for a record with no row yet. The
    writer gets the same live object on every load; every other thread gets
    the last committed copy, shared by every reader, who must not modify it."""

    def __init__(
        self, control: ControlStore, kind: str, key: str, decode: Callable[[dict], object], empty
    ):
        self._control = control
        self._kind, self._key = kind, key
        self._decode = decode
        self._empty = empty
        self._held: object | None = None  # the writer's live object, once loaded
        self._copy: tuple[object, int] | None = None  # (readers' copy, the version it reflects)
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if not self._control.writer.here():
                return self._committed_copy()
            if self._held is None:
                self._held = self._read()
            return self._held

    def save(self, obj):
        assert self._copy is None or obj is not self._copy[0], (
            f"{self._kind} record {self._key!r}: saving a reader's copy; "
            "load the record on the writer thread"
        )
        self._control.put(self._kind, self._key, json.dumps(asdict(obj)))
        with self._lock:
            self._held = obj

    def forget(self):
        """Delete the record's row, and both copies."""
        self._control.delete(self._kind, self._key)
        with self._lock:
            self._held = self._copy = None

    def _committed_copy(self):
        version = self._control.version
        if self._copy is None or self._copy[1] != version:
            self._copy = (self._read(), version)
        return self._copy[0]

    def _read(self):
        body = self._control.get(self._kind, self._key)
        return self._decode(json.loads(body)) if body is not None else self._empty()
