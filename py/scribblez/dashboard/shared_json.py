"""One process-wide object per dashboard JSON file.

The dashboard mutates its records from several places at once: the reconcile
pass across its blocking steps, request handlers, status polls. With a copy
each, the last save would win and undo another's edit. So a process holds one
object per file, rereads it only when the file changes under it (a CLI tool,
another process), and saves that object atomically. tasks.py keeps task
records this way; the pool and the tag queue use this store.
"""

import json
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path


class SharedJson:
    """The shared object for the file at `path()`, a dataclass instance.
    `decode(raw)` builds one from the file's JSON, `empty()` one for a file
    that does not exist yet. `path` is called on every access, so a test can
    redirect the module constant it reads."""

    def __init__(self, path: Callable[[], Path], decode: Callable[[dict], object], empty):
        self._path = path
        self._decode = decode
        self._empty = empty
        self._held: dict[Path, tuple[object, int]] = {}  # path -> (object, mtime; 0 = no file)
        self._lock = threading.Lock()

    def load(self):
        path = self._path()
        with self._lock:
            try:
                stamp = path.stat().st_mtime_ns
            except FileNotFoundError:
                stamp = 0
            held = self._held.get(path)
            if held is None or held[1] != stamp:
                obj = self._decode(json.loads(path.read_text())) if stamp else self._empty()
                self._held[path] = held = (obj, stamp)
            return held[0]

    def save(self, obj):
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(asdict(obj), indent=2) + "\n")
        with self._lock:
            os.replace(tmp, path)
            self._held[path] = (obj, path.stat().st_mtime_ns)

    def forget(self):
        """Drop the held objects, as a fresh process would start (tests)."""
        with self._lock:
            self._held.clear()
