"""One worker slot's filesystem, as the controller reaches it.

Some roles are not self-directing: the controller decides what a slot should
work on next and puts the input where the worker will find it (match eval: the
ONNX export of the generation to play), reads the directory back to learn
whether that work is still in flight, and removes what the exchange is
finished with. Where "there" is depends on the slot kind -- the tag's own data
tree for a local worker, a container on another machine for an ssh one -- so
both are presented through the three calls a dispatch needs
(scribblez/match_eval/dispatch.py), which therefore never branches on kind.

Paths are relative to the tag root, the one layout both sides share: the
container runs the controller's own tree under the same mount root, so a
relative path names the same thing on either machine.

A cloud pod has no such surface -- nothing can reach into a rented pod's
filesystem, only the bucket it uploads to -- so a dispatch-driven role is
local or ssh, and its RoleSpec says so.
"""

import os
from pathlib import Path

from cloud.ssh_transfer import list_dir, push_file, remove_file


class LocalSlotFiles:
    """A local slot's world: the tag tree this process is already writing.

    Inputs are symlinked rather than copied -- the file the worker is being
    pointed at is right there, and a per-assignment copy of a model would be
    tens of megabytes of the same bytes.
    """

    def __init__(self, worker_id: str, tag_root: Path):
        self.worker_id = worker_id
        self._root = tag_root

    def list(self, rel: str) -> list[str]:
        try:
            return sorted(p.name for p in (self._root / rel).iterdir())
        except FileNotFoundError:
            return []  # nothing has been put there yet

    def put(self, src: Path, rel: str):
        """Point the slot at `src` under `rel`. The link is created under a
        temporary name and renamed into place, so a worker polling the
        directory never sees a link before it has a target."""
        dest = self._root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.part")
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(src)
        os.replace(tmp, dest)

    def remove(self, rel: str):
        (self._root / rel).unlink(missing_ok=True)


class SshSlotFiles:
    """An ssh slot's world: its container, over the control link (which is how
    its results come back too, cloud/ssh_transfer.py)."""

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
