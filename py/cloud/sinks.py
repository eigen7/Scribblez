"""Results sinks: where a worker's outputs go, decoupled from how they're made.

A role runner produces files in a private work dir and hands them to the sink:

    LocalSink   the mount dir IS the destination -- data files are renamed into
                the tag's data/ tree, records written as plain files
    R2Sink      each file is uploaded to the results bucket under
                <workload>/<tag>/ and the local copy deleted (the bucket is the
                destination; the machine's disk is scratch)

Both expose the same calls: `deliver(src, data_rel)` for data files (relative
to the tag's data/ tree, mirrored as the bucket prefix), `push_file(src, rel)`
for a file addressed from the tag root (a trainer's prediction arrays),
`push_json(rel, obj)` for small records (stats, provenance manifests, the
trainer's records; relative to the tag root / bucket prefix), and
`read_json(rel)` to read one back -- how a restarted worker recovers the
counters it published before, and how a trainer reads its controls.

A trainer also consumes and produces whole artifacts at the tag root -- the
generations or pair store it trains over, its exports, its rolling
checkpoint and cursor -- and those take a few more calls, which is what lets
one trainer run wherever its sink points: `fetch_data_dir(data_rel, dest)`
(a published directory, whole, manifest last), `fetch_data_files(data_rel,
dest)` (a directory of independently delivered files, whatever is there)
and `fetch_file(rel, dest)` bring an artifact to where the trainer reads it
(nothing to do under the local sink, whose mount dir is where it already
is); `deliver_output(src, rel, keep)` sends one the trainer wrote back
(again nothing to do locally); and `remove_output(rel)` / `remove_outputs`
delete ones it is done with (an export pruned, a corpus retired) wherever
the sink keeps them -- the bucket's copy and this machine's alike. Under
the R2 sink the fetches are pulls from the tag prefix, the delivery an
upload -- unlinked afterwards unless kept, since the machine's disk is
scratch and the bucket is where outputs live.

Paths are tag-relative on both sinks; a generator's `dest_dir` and a
trainer's `data_rel` name a data/ subdirectory, and `count_data_files`
reads how many files of a suffix it holds, which is how a worker that cannot
see the store on disk (a generator uploading to the bucket) reads the target
the store is grown to.

The bucket prefix flattens the tag root and its data/ tree (data/slogs and
stats sit side by side), so a root-relative path's key drops the `data/`.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

from cloud.credentials import R2Credentials
from cloud.r2 import bucket_path, rclone

# A published generation's last object (scribblez/generational/lifecycle.py);
# named here rather than imported so the sinks stay free of the training
# package.
MANIFEST_NAME = "manifest.json"


class LocalSink:
    kind = "local"

    def __init__(self, tag_root: Path):
        self._root = tag_root

    def push_json(self, rel_path: str, obj: dict):
        """Write the record atomically: a reader on this machine (the
        dashboard's ingest tick) must never see a half-written one."""
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
        """Move `src` to <tag>/data/<data_rel> (atomic rename). Returns 0: no
        bytes travel a network, so upload accounting stays zero."""
        dest = self._root / "data" / data_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dest)
        return 0

    def push_file(self, src: Path, rel_path: str) -> int:
        """Move `src` to <tag>/<rel_path>. A rename when `src` is on the
        mount's filesystem, a copy otherwise (a temp file elsewhere): the
        record that names the file is pushed after this returns, so nothing
        reads it before it is whole either way. Returns 0 as deliver does."""
        dest = self._root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dest)
        return 0

    def fetch_data_dir(self, data_rel: str, dest: Path) -> bool:
        """Whether <tag>/data/<data_rel> exists -- it is `dest` itself."""
        assert dest == self._root / "data" / data_rel, (dest, data_rel)
        return dest.is_dir()

    def fetch_file(self, rel_path: str, dest: Path) -> bool:
        """Whether <tag>/<rel_path> exists -- it is `dest` itself."""
        assert dest == self._root / rel_path, (dest, rel_path)
        return dest.is_file()

    def fetch_data_files(self, data_rel: str, dest: Path):
        """<tag>/data/<data_rel>'s files are `dest`'s own: nothing to pull."""
        assert dest == self._root / "data" / data_rel, (dest, data_rel)

    def count_data_files(self, data_rel: str, suffix: str) -> int:
        """Files under <tag>/data/<data_rel> ending in `suffix`."""
        d = self._root / "data" / data_rel
        return sum(1 for _ in d.glob(f"*{suffix}")) if d.is_dir() else 0

    def remove_output(self, rel_path: str):
        """Delete <tag>/<rel_path>; absent is success."""
        (self._root / rel_path).unlink(missing_ok=True)

    def remove_outputs(self, rel_paths: list[str]):
        for rel in rel_paths:
            self.remove_output(rel)

    def deliver_output(self, src: Path, rel_path: str, *, keep: bool = False):
        """An output written at its place under the tag root is already
        delivered. `src` may instead be a snapshot of it taken beside it (a
        trainer delivering off its training thread links one so the file it
        keeps rewriting is not the one in flight); with nothing to send, the
        snapshot is just dropped unless `keep`."""
        dest = self._root / rel_path
        assert dest.is_file(), (src, rel_path)
        if src != dest and not keep:
            src.unlink()


class R2Sink:
    kind = "ssh"  # the bucket delivers for ssh slots only

    def __init__(self, r2: R2Credentials, workload: str, tag: str, root: Path | None = None):
        self._r2 = r2
        self._prefix = (workload, tag)
        # The tag root on this machine, for the outputs a worker keeps a copy
        # of beside the bucket's (a trainer's checkpoint, its exports until
        # they are pruned); None for a worker that keeps none.
        self._root = root

    def _path(self, *parts: str) -> str:
        return bucket_path(self._r2, *self._prefix, *parts)

    @staticmethod
    def _key(rel_path: str) -> str:
        """A tag-root-relative path's key under the tag prefix, which
        flattens data/ (deliver lands data/slogs/x at slogs/x)."""
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
        """The record previously published at `rel_path`, or None if the
        bucket has none (a first run) -- also None if the read itself fails,
        which costs a restarted worker its counter history and nothing more."""
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

    # The bucket prefix flattens the tag root and its data/ tree (stats/ and
    # staging/ sit side by side), so a root-addressed file uploads exactly as
    # a data file does.
    push_file = deliver

    def fetch_data_dir(self, data_rel: str, dest: Path) -> bool:
        """Pull <workload>/<tag>/<data_rel> into `dest`, whole, if the bucket
        has it: a generation is published chunks-first and manifest-last, so
        the manifest's presence is the test, and its objects never change,
        so what `dest` already holds is skipped by size. False when the
        bucket has no such directory yet."""
        prefix = self._path(*data_rel.split("/"))
        if not rclone(self._r2, "lsf", f"{prefix}/{MANIFEST_NAME}", capture=True).stdout.strip():
            return False
        res = rclone(self._r2, "copy", "--size-only", prefix, str(dest), capture=True)
        assert res.returncode == 0, f"pull of {data_rel} failed: {res.stderr}"
        return True

    def fetch_data_files(self, data_rel: str, dest: Path):
        """Pull every object under <workload>/<tag>/data/<data_rel> into
        `dest`, skipping what it already holds at the same size. For a
        directory of independently delivered files (a pair store), where
        fetch_data_dir's manifest test does not apply: each file is whole on
        arrival, and a pair is complete when both its members are."""
        res = rclone(self._r2, "copy", "--size-only", self._path(*data_rel.split("/")),
                     str(dest), capture=True)  # fmt: skip
        assert res.returncode == 0, f"pull of {data_rel} failed: {res.stderr}"

    def count_data_files(self, data_rel: str, suffix: str) -> int:
        """Objects under <workload>/<tag>/<data_rel> ending in `suffix`, by
        one listing."""
        res = rclone(self._r2, "lsf", self._path(*data_rel.split("/")), capture=True)
        assert res.returncode == 0, f"listing {data_rel} failed: {res.stderr}"
        return sum(1 for name in res.stdout.split() if name.endswith(suffix))

    def remove_output(self, rel_path: str):
        """Delete <workload>/<tag>'s object for `rel_path` from the bucket,
        and this machine's copy if it has one; absent is success."""
        res = rclone(self._r2, "deletefile", self._path(self._key(rel_path)), capture=True)
        assert res.returncode == 0 or "not found" in res.stderr.lower(), (
            f"delete of {rel_path} failed: {res.stderr}"
        )
        self._unlink_local(rel_path)

    def remove_outputs(self, rel_paths: list[str]):
        """As remove_output over many, in one rclone run: a corpus retired
        file by file would spend a round trip per object."""
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
        """Upload `src` to <workload>/<tag>/<rel_path>; the local copy goes
        unless `keep` (the checkpoint a resume needs, the cursor)."""
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


def make_sink(spec, tag: str, mount_root=None):
    """The sink selected by SCZ_SINK ("r2", the default, or "local"); a local
    sink's root is the tag's under `mount_root` (the mount dir by default)."""
    if os.environ.get("SCZ_SINK", "r2") == "local":
        return LocalSink(spec.paths(tag, mount_root).root)
    return R2Sink(r2_from_env(), spec.name, tag, spec.paths(tag, mount_root).root)
