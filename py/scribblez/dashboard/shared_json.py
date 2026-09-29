"""One shared object per dashboard JSON file, written by one thread.

The dashboard's control state (task records, the pool, the tag queue) has one
writer: the worker manager's blocking thread, which runs the reconcile pass
and every command a request handler submits (docs/plans/dashboard_state_model.md
§2). That thread holds one live object per file, rereads the file only when it
changes under it (a CLI tool, another process), and saves the object
atomically.

Every other thread (a status request on the event loop, say) reads the last
committed copy instead: the file as last saved, decoded apart from the live
object. So a read never sees a step half done, and nothing a read does can
reach the writer's records. A save off the writer thread fails.

tasks.py keeps task records this way; the pool and the tag queue use this
store directly.
"""

import json
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path


class Writer:
    """The one thread allowed to write the stores that share this object.
    Until `claim` names one, any thread writes and every thread reads the
    live objects, as a single-threaded test or tool expects."""

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


class SharedJson:
    """The shared object for the file at `path`, a dataclass instance.
    `decode(raw)` builds one from the file's JSON, `empty()` one for a file
    that does not exist yet."""

    def __init__(self, path: Path, decode: Callable[[dict], object], empty, writer: Writer):
        self.path = path
        self._decode = decode
        self._empty = empty
        self._writer = writer
        self._held: tuple[object, int] | None = None  # (live object, mtime; 0 = no file)
        # The readers' copy: (object, mtime, the save it reflects).
        self._committed: tuple[object, int, int] | None = None
        self._saves = 0
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            stamp = self._stamp()
            if not self._writer.here():
                return self._committed_copy(stamp)
            if self._held is None or self._held[1] != stamp:
                self._held = (self._read(stamp), stamp)
            return self._held[0]

    def save(self, obj):
        self._writer.check(self.path)
        assert self._committed is None or obj is not self._committed[0], (
            f"{self.path}: saving a reader's copy; load the record on the writer thread"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.stem}.", suffix=".json"
        )
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(asdict(obj), indent=2) + "\n")
        with self._lock:
            os.replace(tmp, self.path)
            self._held = (obj, self._stamp())
            self._saves += 1

    def forget(self):
        """Drop both copies, after the file was deleted."""
        self._writer.check(self.path)
        with self._lock:
            self._held = self._committed = None
            self._saves += 1

    def _committed_copy(self, stamp: int):
        """The file as last committed, decoded once per save (or per outside
        change) and shared by every reader, who must not modify it."""
        c = self._committed
        if c is None or c[1] != stamp or c[2] != self._saves:
            c = self._committed = (self._read(stamp), stamp, self._saves)
        return c[0]

    def _stamp(self) -> int:
        try:
            return self.path.stat().st_mtime_ns
        except FileNotFoundError:
            return 0

    def _read(self, stamp: int):
        return self._decode(json.loads(self.path.read_text())) if stamp else self._empty()
