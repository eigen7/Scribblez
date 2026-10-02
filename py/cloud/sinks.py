"""The results sink: where a worker's outputs go, independent of how they are made.

A role runner writes files in a private work dir and hands them to its sink,
a LocalSink over the tag tree on this machine's mount dir: data files are
renamed into it, records written in place. On an ssh slot that tree is the
container's own, which the dashboard collects (cloud/ssh_transfer.py).

Its calls take paths relative to the tag root (or, where the argument is
`data_rel`, to its data/ tree):

    deliver, push_file      send a data file / a file addressed from the root
    push_json, read_json    write / read back a small record (stats,
                            provenance, trainer records); read_json is how a
                            restarted worker recovers its counters
    count_data_files        count a data directory's files by suffix (a
                            generator's progress toward a store size)
    fetch_data_dir          whether a directory is present
    fetch_data_files        nothing to fetch: the directory is the store
    fetch_file              whether a file is present
    list_dirs               the subdirectories of a root-relative directory
    remove_tree             delete a root-relative directory and its files
    deliver_output          settle an artifact a trainer wrote in place
    remove_output(s)        delete artifacts
"""

import json
import os
import shutil
from pathlib import Path


class LocalSink:
    kind = "local"

    def __init__(self, tag_root: Path):
        self._root = tag_root

    def push_json(self, rel_path: str, obj: dict):
        """Written atomically, since the dashboard's ingest may read it at
        any moment."""
        path = self._root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2) + "\n")
        os.replace(tmp, path)

    def read_json(self, rel_path: str) -> dict | None:
        try:
            return json.loads((self._root / rel_path).read_text())
        except FileNotFoundError:
            return None

    def deliver(self, src: Path, data_rel: str) -> int:
        """Move `src` to <tag>/data/<data_rel> by atomic rename. Returns the
        bytes uploaded, always 0 here."""
        dest = self._root / "data" / data_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dest)
        return 0

    def push_file(self, src: Path, rel_path: str) -> int:
        """Move `src` to <tag>/<rel_path>. This may be a copy rather than a
        rename when `src` is on another filesystem; that is safe because the
        record naming the file is pushed only after this returns. Returns 0,
        as deliver does."""
        dest = self._root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dest)
        return 0

    def fetch_data_dir(self, data_rel: str, dest: Path) -> bool:
        """Whether <tag>/data/<data_rel>, which is `dest`, exists."""
        assert dest == self._root / "data" / data_rel, (dest, data_rel)
        return dest.is_dir()

    def fetch_file(self, rel_path: str, dest: Path) -> bool:
        """Whether <tag>/<rel_path>, which is `dest`, exists."""
        assert dest == self._root / rel_path, (dest, rel_path)
        return dest.is_file()

    def fetch_data_files(self, data_rel: str, dest: Path):
        """Nothing to pull: `dest` is <tag>/data/<data_rel> itself."""
        assert dest == self._root / "data" / data_rel, (dest, data_rel)

    def count_data_files(self, data_rel: str, suffix: str) -> int:
        """Files under <tag>/data/<data_rel> ending in `suffix`."""
        d = self._root / "data" / data_rel
        return sum(1 for _ in d.glob(f"*{suffix}")) if d.is_dir() else 0

    def remove_output(self, rel_path: str):
        """Delete <tag>/<rel_path>; absent is success."""
        (self._root / rel_path).unlink(missing_ok=True)

    def list_dirs(self, rel_path: str) -> list[str]:
        """Names of the directories in <tag>/<rel_path>, sorted."""
        d = self._root / rel_path
        return sorted(c.name for c in d.iterdir() if c.is_dir()) if d.is_dir() else []

    def remove_tree(self, rel_path: str):
        """Delete <tag>/<rel_path> and everything in it; absent is success."""
        shutil.rmtree(self._root / rel_path, ignore_errors=True)

    def remove_outputs(self, rel_paths: list[str]):
        for rel in rel_paths:
            self.remove_output(rel)

    def deliver_output(self, src: Path, rel_path: str, *, keep: bool = False):
        """An output written in place under the tag root is already
        delivered. `src` may instead be a snapshot beside it (a trainer
        delivering off its training thread links one, so the file it keeps
        rewriting is not the one in flight); the snapshot is dropped unless
        `keep`."""
        dest = self._root / rel_path
        assert dest.is_file(), (src, rel_path)
        if src != dest and not keep:
            src.unlink()


def make_sink(spec, tag: str, mount_root: Path) -> LocalSink:
    """The sink over the tag's tree under `mount_root`, the mount dir by
    default."""
    return LocalSink(spec.paths(tag, mount_root).root)
