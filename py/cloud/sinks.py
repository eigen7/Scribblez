"""Results sinks: where a worker's outputs go, independent of how they are made.

A role runner writes files in a private work dir and hands them to its sinks
(chosen by SCZ_DATA_SINK and SCZ_SINK; see make_sinks):

    LocalSink   the tag tree on this machine's mount dir is the destination:
                data files are renamed into it, records written in place
    R2Sink      files are uploaded to the bucket under <workload>/<tag>/ and
                the local copies deleted; the machine's disk is scratch

Both sinks offer the same calls, with paths relative to the tag root (or,
where the argument is `data_rel`, to its data/ tree):

    deliver, push_file      send a data file / a file addressed from the root
    push_json, read_json    write / read back a small record (stats,
                            provenance, trainer records); read_json is how a
                            restarted worker recovers its counters
    count_data_files        count a data directory's files by suffix (a
                            generator's progress toward a store size)
    fetch_data_dir          bring a published directory (manifest last)
    fetch_data_files        bring a directory of independently delivered files
    fetch_file              bring one file
    list_dirs               the subdirectories of a root-relative directory
    remove_tree             delete a root-relative directory and its files
    deliver_output          send back an artifact a trainer wrote in place
    remove_output(s)        delete artifacts wherever the sink keeps them

The fetch and deliver_output calls let a trainer run wherever its sink points.
Under LocalSink they are no-ops, since the files are already where the
trainer reads and writes them.

A worker holds two sinks, which may be of different kinds:

    data sink       the tag's data/ store: what generators deliver and
                    trainers read (deliver, count_data_files, fetch_data_dir,
                    fetch_data_files, and removing store files)
    records sink    everything else under the tag root: stats and params
                    records, trainer records and outputs, the checkpoint and
                    cursor, controls, staged inputs

In the bucket, the tag prefix flattens the tag root and its data/ tree
(data/slogs and stats sit side by side), so a root-relative path's key drops
the leading `data/`.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

from cloud.credentials import R2Credentials
from cloud.r2 import bucket_path, rclone

# The object a published generation writes last
# (scribblez/generational/lifecycle.py). Duplicated rather than imported to
# keep the sinks independent of the training package.
MANIFEST_NAME = "manifest.json"


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


class R2Sink:
    kind = "ssh"  # only ssh slots deliver through the bucket

    def __init__(self, r2: R2Credentials, workload: str, tag: str, root: Path | None = None):
        self._r2 = r2
        self._prefix = (workload, tag)
        # The tag root on this machine, where a worker keeps local copies of
        # some outputs (a trainer's checkpoint, its exports until pruned).
        # remove_output deletes those too. None if it keeps none.
        self._root = root

    def _path(self, *parts: str) -> str:
        return bucket_path(self._r2, *self._prefix, *parts)

    @staticmethod
    def _key(rel_path: str) -> str:
        """The bucket key, under the tag prefix, of a root-relative path."""
        return rel_path.removeprefix("data/")

    def push_json(self, rel_path: str, obj: dict):
        res = rclone(
            self._r2,
            "rcat",
            self._path(*rel_path.split("/")),
            capture=True,
            input_text=json.dumps(obj, indent=2) + "\n",
        )
        assert res.returncode == 0, f"upload of {rel_path} failed: {res.stderr}"

    def read_json(self, rel_path: str) -> dict | None:
        """The record at `rel_path`, or None if there is none (a first run).
        A failed read also returns None, which costs a restarted worker only
        its counter history."""
        res = rclone(self._r2, "cat", self._path(*rel_path.split("/")), capture=True)
        if res.returncode != 0:
            return None
        try:
            return json.loads(res.stdout)
        except json.JSONDecodeError:
            return None

    def deliver(self, src: Path, data_rel: str) -> int:
        """Upload `src` to <workload>/<tag>/<data_rel> and delete the local
        copy. Returns the bytes uploaded."""
        nbytes = src.stat().st_size
        res = rclone(self._r2, "copyto", str(src), self._path(*data_rel.split("/")), capture=True)
        assert res.returncode == 0, f"upload of {src.name} failed: {res.stderr}"
        src.unlink()
        return nbytes

    # The tag prefix flattens data/ (see the module docstring), so a
    # root-addressed file uploads exactly as a data file does.
    push_file = deliver

    def fetch_data_dir(self, data_rel: str, dest: Path) -> bool:
        """Pull <workload>/<tag>/<data_rel> into `dest`, or return False if it
        is not fully published yet. A generation is published with its
        manifest last, so the manifest's presence means complete. Its objects
        never change, so files `dest` already holds are skipped by size."""
        prefix = self._path(*data_rel.split("/"))
        if not rclone(self._r2, "lsf", f"{prefix}/{MANIFEST_NAME}", capture=True).stdout.strip():
            return False
        res = rclone(self._r2, "copy", "--size-only", prefix, str(dest), capture=True)
        assert res.returncode == 0, f"pull of {data_rel} failed: {res.stderr}"
        return True

    def fetch_data_files(self, data_rel: str, dest: Path):
        """Pull every object under <workload>/<tag>/<data_rel> into `dest`,
        skipping files it already holds at the same size. For a directory of
        independently delivered files (a pair store), which has no manifest;
        each object is whole once it exists."""
        res = rclone(self._r2, "copy", "--size-only", self._path(*data_rel.split("/")),
                     str(dest), capture=True)  # fmt: skip
        assert res.returncode == 0, f"pull of {data_rel} failed: {res.stderr}"

    def count_data_files(self, data_rel: str, suffix: str) -> int:
        """Objects under <workload>/<tag>/<data_rel> ending in `suffix`."""
        res = rclone(self._r2, "lsf", self._path(*data_rel.split("/")), capture=True)
        assert res.returncode == 0, f"listing {data_rel} failed: {res.stderr}"
        return sum(1 for name in res.stdout.split() if name.endswith(suffix))

    def remove_output(self, rel_path: str):
        """Delete `rel_path` from the bucket and any local copy; absent is
        success."""
        res = rclone(self._r2, "deletefile", self._path(self._key(rel_path)), capture=True)
        assert res.returncode == 0 or "not found" in res.stderr.lower(), (
            f"delete of {rel_path} failed: {res.stderr}"
        )
        self._unlink_local(rel_path)

    def remove_outputs(self, rel_paths: list[str]):
        """remove_output over many paths in one rclone run, rather than a
        round trip per object."""
        if not rel_paths:
            return
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("".join(self._key(rel) + "\n" for rel in rel_paths))
        try:
            res = rclone(self._r2, "delete", self._path(), "--files-from", f.name, capture=True)
            assert res.returncode == 0, f"delete of {len(rel_paths)} outputs failed: {res.stderr}"
        finally:
            os.unlink(f.name)
        for rel in rel_paths:
            self._unlink_local(rel)

    def list_dirs(self, rel_path: str) -> list[str]:
        """Names of the directories under <workload>/<tag>/<rel_path>, sorted."""
        res = rclone(self._r2, "lsf", "--dirs-only", self._path(*rel_path.split("/")), capture=True)
        assert res.returncode == 0, f"listing {rel_path} failed: {res.stderr}"
        return sorted(name.rstrip("/") for name in res.stdout.split())

    def remove_tree(self, rel_path: str):
        """Delete everything under <workload>/<tag>/<rel_path>; absent is
        success."""
        res = rclone(self._r2, "purge", self._path(*rel_path.split("/")), capture=True)
        assert res.returncode == 0 or "not found" in res.stderr.lower(), (
            f"delete of {rel_path} failed: {res.stderr}"
        )

    def _unlink_local(self, rel_path: str):
        if self._root is not None:
            (self._root / rel_path).unlink(missing_ok=True)

    def fetch_file(self, rel_path: str, dest: Path) -> bool:
        """Pull <workload>/<tag>/<rel_path> to `dest` if the bucket has it."""
        src = self._path(*rel_path.split("/"))
        if not rclone(self._r2, "lsf", src, capture=True).stdout.strip():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        res = rclone(self._r2, "copyto", src, str(dest), capture=True)
        assert res.returncode == 0, f"pull of {rel_path} failed: {res.stderr}"
        return True

    def deliver_output(self, src: Path, rel_path: str, *, keep: bool = False):
        """Upload `src` to <workload>/<tag>/<rel_path>, deleting the local copy
        unless `keep` (e.g. the checkpoint and cursor a resume needs)."""
        res = rclone(self._r2, "copyto", str(src), self._path(*rel_path.split("/")), capture=True)
        assert res.returncode == 0, f"upload of {rel_path} failed: {res.stderr}"
        if not keep:
            src.unlink()


def r2_from_env() -> R2Credentials:
    return R2Credentials(
        account_id=os.environ["R2_ACCOUNT_ID"],
        access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        bucket=os.environ["R2_BUCKET"],
    )


def _make_sink(kind: str, spec, tag: str, mount_root: Path):
    root = spec.paths(tag, mount_root).root
    if kind == "local":
        return LocalSink(root)
    return R2Sink(r2_from_env(), spec.name, tag, root)


def make_sinks(spec, tag: str, mount_root: Path) -> tuple:
    """(data sink, records sink). SCZ_SINK selects the records sink, "r2" (the
    default) or "local"; SCZ_DATA_SINK selects the data sink and defaults to
    the records sink's kind. The tag root is under `mount_root`, the mount dir
    by default."""
    records_kind = os.environ.get("SCZ_SINK", "r2")
    data_kind = os.environ.get("SCZ_DATA_SINK", records_kind)
    return (
        _make_sink(data_kind, spec, tag, mount_root),
        _make_sink(records_kind, spec, tag, mount_root),
    )
