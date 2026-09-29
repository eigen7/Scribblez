"""One shared object per dashboard JSON file.

The dashboard mutates its records from several places at once: the reconcile
pass across its blocking steps, request handlers, status polls. With a copy
each, the last save would win and undo another's edit. So the dashboard holds
one store per file, which rereads it only when the file changes under it (a
CLI tool, another process), and saves the held object atomically. tasks.py
keeps task records this way; the pool and the tag queue use this store.
"""

import json
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path


class SharedJson:
    """The shared object for the file at `path`, a dataclass instance.
    `decode(raw)` builds one from the file's JSON, `empty()` one for a file
    that does not exist yet."""

    def __init__(self, path: Path, decode: Callable[[dict], object], empty):
        self.path = path
        self._decode = decode
        self._empty = empty
        self._held: tuple[object, int] | None = None  # (object, mtime; 0 = no file)
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            try:
                stamp = self.path.stat().st_mtime_ns
            except FileNotFoundError:
                stamp = 0
            if self._held is None or self._held[1] != stamp:
                obj = self._decode(json.loads(self.path.read_text())) if stamp else self._empty()
                self._held = (obj, stamp)
            return self._held[0]

    def save(self, obj):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.stem}.", suffix=".json"
        )
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(asdict(obj), indent=2) + "\n")
        with self._lock:
            os.replace(tmp, self.path)
            self._held = (obj, self.path.stat().st_mtime_ns)
