"""One worker slot's filesystem, as the controller reaches it.

Dispatch-driven roles (RoleSpec.dispatch) do not pick their own work: the
controller puts each assignment where the worker will find it (for match eval,
the ONNX export of the generation to play), lists the directory to learn
whether the work is still in flight, and removes what the exchange is finished
with. That place is the tag's own tree for a local slot and a container on
another machine for an ssh slot, so both expose the same three calls and the
dispatch code (scribblez/match_eval/dispatch.py) never branches on kind.

Paths are relative to the tag root. A container runs the controller's own
layout under the same mount root, so a relative path names the same thing on
either machine.

An ssh slot here is on a registered machine: a container on a rented machine
delivers through the bucket, which dispatch never reads, so a dispatch-driven
role is refused there when its slot is added (WorkerManager._check_role).
"""

import os
from pathlib import Path

from cloud.ssh_transfer import list_dir, push_file, remove_file


class LocalSlotFiles:
    """A local slot's files: the tag tree on this machine.

    Inputs are symlinked rather than copied: the file is already here, and a
    per-assignment copy of a model would be tens of megabytes of duplicate
    bytes.
    """

    def __init__(self, worker_id: str, tag_root: Path):
        self.worker_id = worker_id
        self._root = tag_root

    def list(self, rel: str) -> list[str]:
        try:
            return sorted(p.name for p in (self._root / rel).iterdir())
        except FileNotFoundError:
            return []

    def put(self, src: Path, rel: str):
        """Point the slot at `src` under `rel`. The link is renamed into place,
        so a worker polling the directory never sees it half made."""
        dest = self._root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.part")
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(src)
        os.replace(tmp, dest)

    def remove(self, rel: str):
        (self._root / rel).unlink(missing_ok=True)


class SshSlotFiles:
    """An ssh slot's files: its container, reached over the ssh control link
    (cloud/ssh_transfer.py)."""

    def __init__(self, worker_id: str, machine, container: str, remote_root: str):
        self.worker_id = worker_id
        self._machine = machine
        self._container = container
        self._root = remote_root

    def list(self, rel: str) -> list[str]:
        return sorted(list_dir(self._machine, self._container, remote_root=self._root, rel=rel))

    def put(self, src: Path, rel: str):
        push_file(self._machine, self._container, remote_root=self._root, rel_dest=rel, src=src)

    def remove(self, rel: str):
        remove_file(self._machine, self._container, remote_root=self._root, rel=rel)
