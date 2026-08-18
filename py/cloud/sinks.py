"""Results sinks: where a worker's outputs go, decoupled from how they're made.

A role runner produces files in a private work dir and hands them to the sink:

    LocalSink   the mount dir IS the destination -- data files are renamed into
                the tag's data/ tree, records written as plain files
    R2Sink      each file is uploaded to the results bucket under
                <workload>/<tag>/ and the local copy deleted (the bucket is the
                destination; the pod disk is scratch)

Both expose the same three calls: `deliver(src, data_rel)` for data files
(relative to the tag's data/ tree, mirrored as the bucket prefix),
`push_json(rel, obj)` for small records (stats, provenance manifests, relative
to the tag root / bucket prefix), and `read_json(rel)` to read one back -- how
a restarted worker recovers the counters it published before.
"""

import json
import os
from pathlib import Path

from cloud.credentials import R2Credentials
from cloud.r2 import bucket_path, rclone


class LocalSink:
    kind = "local"

    def __init__(self, tag_root: Path):
        self._root = tag_root

    def push_json(self, rel_path: str, obj: dict):
        path = self._root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2) + "\n")

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


class R2Sink:
    kind = "cloud"

    def __init__(self, r2: R2Credentials, workload: str, tag: str):
        self._r2 = r2
        self._prefix = (workload, tag)

    def _path(self, *parts: str) -> str:
        return bucket_path(self._r2, *self._prefix, *parts)

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


def r2_from_env() -> R2Credentials:
    return R2Credentials(
        account_id=os.environ["R2_ACCOUNT_ID"],
        access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        bucket=os.environ["R2_BUCKET"],
    )


def make_sink(spec, tag: str):
    """The sink selected by SCZ_SINK ("r2", the default, or "local")."""
    if os.environ.get("SCZ_SINK", "r2") == "local":
        return LocalSink(spec.data_dir(tag))
    return R2Sink(r2_from_env(), spec.name, tag)
