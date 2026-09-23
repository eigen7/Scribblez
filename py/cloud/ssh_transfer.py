"""Moving files between the controller and a worker's container over the ssh
link the controller already uses to manage it.

A worker with the local sink (SCZ_SINK=local) delivers into its own
container, and the controller collects: each pass streams a `docker exec tar`
of finished output back over the open ssh connection. This keeps the network
out of the worker's cycle (a bucket round trip per cycle would dwarf work that
takes seconds) and needs no bucket or extra credential. The same link runs the
other way for roles whose work the controller assigns: push_file drops a file
where the worker polls for it (match eval: the ONNX of the generation to
play), and list_dir reads back what is there.

Only the controller initiates. The dev container runs no sshd, and a worker
that pushed would need a route, a stable address and a key for the
controller. This also degrades well: while the controller is down, the worker
keeps generating into its own filesystem and the next pass collects the
backlog.

A pull takes a bounded batch (BATCH files), not the whole backlog. If a pull's
cost grew with the backlog, one slow pull could exceed its timeout, skip the
deletes that follow extraction, and leave a larger backlog for the next pull,
which then also times out; the backlog would grow without bound. With a fixed
batch, every pull costs the same and drains at a steady rate.

Delivered data is moved: deleted from the container once it is on disk here.
The worker's stats and params records are copied instead, because the worker
reads its counters back from them on restart. The delete runs only after
extraction succeeds, so a transfer that dies mid-stream loses nothing. A file
pulled twice (the delete failed, or the container restarted first) is
deduplicated by the scheduler's ingest ledger.
"""

import shlex
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import IO

# Records the worker rewrites and reads back: copied, never removed.
RECORD_DIRS = ("stats", "params")

# Delivered files moved per pull. Larger drains faster but holds the pass's
# single blocking thread longer. Measured on this fleet: 32 chunks is ~20 MB
# raw, ~0.4 s to compress and ~1.4 s to move at 7.5 MB/s, about a third of a
# pass, draining ~29 net files per pass against one worker's ~2.5.
BATCH = 32

# The worker machine's CPU is busy playing games and is scarcer than the link.
# On real chunks, level 1 takes 0.2 s against the default's 0.7 s, for only
# 13% more bytes: a net win at this bandwidth.
COMPRESSION = "gzip -1"

# Time limit on the in-container tar, enforced inside the container so an
# overrunning transfer dies with its ssh client. Otherwise an abandoned tar
# keeps running, and one more accumulates each pass.
COLLECT_TIMEOUT_SECONDS = 60

# Where a pulled file is written before it is moved into place. It is under
# the tag root, so the move is a same-filesystem rename and a file appears at
# its destination whole or not at all.
INCOMING_DIR = ".incoming"

# Prefix of the spool file a sweep streams through, so the next sweep can
# recognize and remove one left by a process that died mid-copy.
SPOOL_PREFIX = "sweep-"


@dataclass(frozen=True)
class PullResult:
    pulled: list[str]  # paths relative to the tag root, as extracted
    remaining: int | None  # delivered files still waiting; None if unknown


def list_command(root: str, data_dirs: list[str], batch: int) -> list[str]:
    """The in-container command listing the next `batch` delivered files, then
    a "TOTAL <n>" line counting everything waiting.

    Names begin with their chunk's write timestamp, so sorting drains in
    production order. The command trims the list in the container so only the
    batch's names cross the wire, however large the backlog."""
    dirs = " ".join(shlex.quote(d) for d in data_dirs)
    return [
        "sh",
        "-c",
        f"cd {shlex.quote(root)} 2>/dev/null || exit 0\n"
        f'for d in {dirs}; do [ -d "$d" ] && ls -1 "$d" | sed "s|^|$d/|"; done | sort'
        f" | awk -v n={batch} 'NR<=n {{ print }} END {{ print \"TOTAL\", NR+0 }}'",
    ]


def parse_listing(output: bytes) -> tuple[list[str], int | None]:
    """The batch of names and the total waiting, from list_command's output.

    Empty output means the tag root does not exist yet: a true zero. Output
    without the TOTAL line gives a total of None, never 0, because the total
    decides whether a container may be destroyed."""
    if not output.strip():
        return [], 0
    lines = output.decode(errors="replace").split()
    if len(lines) < 2 or lines[-2] != "TOTAL":
        return [], None
    return lines[:-2], int(lines[-1])


def collect_command(root: str, names: list[str], seconds: int) -> list[str]:
    """The in-container command streaming `names` plus the record directories
    as a gzipped tar. Emits nothing when there is nothing to send, since tar
    refuses to create an empty archive."""
    quoted = " ".join(shlex.quote(name) for name in names)
    records = " ".join(RECORD_DIRS)
    return [
        "sh",
        "-c",
        f"cd {shlex.quote(root)} 2>/dev/null || exit 0\n"
        f"set -- {quoted}\n"
        f'for d in {records}; do [ -d "$d" ] && set -- "$@" "$d"; done\n'
        '[ "$#" -eq 0 ] && exit 0\n'
        f"exec timeout {seconds} tar -c --use-compress-program='{COMPRESSION}' -f - \"$@\"",
    ]


def write_command(root: str, rel_dest: str, size: int) -> list[str]:
    """The in-container command reading `size` bytes off stdin into `rel_dest`.

    The bytes land in a dotted temporary file beside the destination and are
    renamed into place, so the worker, which polls for these files, never
    opens a half-written one. A push that dies mid-stream leaves only the
    temporary, which the next push overwrites and listings skip.

    The size is checked before the rename because `cat` exits 0 on any EOF,
    including a dropped link's. A truncated model under its final name would
    wedge the slot: it reads as a match in flight, so nothing replaces it, and
    the worker crashes on it at every restart."""
    dest = f"{root}/{rel_dest}"
    parent, _, name = dest.rpartition("/")
    tmp = f"{parent}/.{name}.part"
    return [
        "sh",
        "-c",
        f"set -e\n"
        f"mkdir -p {shlex.quote(parent)}\n"
        f"cat > {shlex.quote(tmp)}\n"
        f"n=$(( $(wc -c < {shlex.quote(tmp)}) ))\n"
        f'[ "$n" -eq {size} ] || {{ echo "{name}: got $n of {size} bytes" >&2; exit 1; }}\n'
        f"mv {shlex.quote(tmp)} {shlex.quote(dest)}",
    ]


def list_dir_command(root: str, rel: str) -> list[str]:
    """The in-container command listing `rel`. A directory that does not exist
    yet lists as empty: nothing has been pushed there."""
    return ["sh", "-c", f"ls -1 {shlex.quote(f'{root}/{rel}')} 2>/dev/null || true"]


def push_file(machine, container: str, *, remote_root: str, rel_dest: str, src: Path):
    """Send `src` to `rel_dest` under the container's tag root. Raises if it
    did not arrive whole."""
    command = write_command(remote_root, rel_dest, src.stat().st_size)
    machine.write_to_container(container, command, src)


def list_dir(machine, container: str, *, remote_root: str, rel: str) -> list[str]:
    """The names under `rel` in the container's tag root."""
    output = machine.read_from_container(container, list_dir_command(remote_root, rel))
    return output.decode(errors="replace").split()


def remove_file(machine, container: str, *, remote_root: str, rel: str):
    """Delete `rel` under the container's tag root; absent is success."""
    machine.exec_in_container(container, ["rm", "-f", f"{remote_root}/{rel}"])


def _extract(
    archive: bytes | IO[bytes], root: Path, mode: str = "r:gz", prefix: str = ""
) -> list[str]:
    """Unpack `archive` (bytes, or a file object) under `root`, each file
    atomically, and return the paths written relative to `root`. `prefix` is
    prepended to every member name."""
    if isinstance(archive, bytes):
        if not archive:
            return []
        archive = BytesIO(archive)
    names = []
    with tarfile.open(fileobj=archive, mode=mode) as tar:
        # Iterate rather than call getmembers(): a stream ("r|", as
        # sweep_stopped uses) allows one forward pass, and building the member
        # list would consume the data before extraction.
        for member in tar:
            if not member.isfile():
                continue
            name = f"{prefix}{member.name}"
            staged = root / INCOMING_DIR / name
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(tar.extractfile(member).read())
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(dest)
            names.append(name)
    return names


def sweep_stopped(
    machine, container: str, *, remote_root: str, local_root: Path, data_dirs: list[str]
) -> list[str]:
    """Take everything a stopped container still holds, before it is destroyed.

    A graceful stop gives the worker a minute after SIGTERM to flush finished
    output, and that flush lands after the last collection, which needs a
    running container. `docker cp` can read a stopped container, so this is the
    last chance to save that output.

    A stopped container cannot run a listing, so each data directory is taken
    whole rather than in batches. That is affordable only because sweeps run
    on containers recorded as empty (by a collection, or because they never
    ran long enough to produce anything), so a sweep moves just the final
    flush, a few megabytes. On a container with a real backlog it would move
    gigabytes in one call. The copy is spooled through a file rather than held
    in memory all the same, and it is uncompressed; copy_from_container
    explains why.
    """
    names = []
    incoming = local_root / INCOMING_DIR
    incoming.mkdir(parents=True, exist_ok=True)
    for stale in incoming.glob(f"{SPOOL_PREFIX}*"):
        stale.unlink()  # a spool from a sweep whose process died mid-copy
    for data_dir in data_dirs:
        archive = incoming / f"{SPOOL_PREFIX}{Path(data_dir).name}.tar"
        try:
            path = f"{remote_root}/{data_dir}"
            if not machine.copy_from_container(container, path, archive):
                continue
            with open(archive, "rb") as stream:
                # docker cp names members relative to the copied directory's
                # parent; "r|" streams without seeking.
                names += _extract(stream, local_root, mode="r|", prefix=f"{Path(data_dir).parent}/")
        finally:
            archive.unlink(missing_ok=True)
    return names


def pull_results(
    machine,
    container: str,
    *,
    remote_root: str,
    local_root: Path,
    data_dirs: list[str],
    batch: int = BATCH,
) -> PullResult:
    """Collect up to `batch` of one ssh worker's finished outputs, plus its
    records, into the tag's local tree.

    `remote_root` is the tag root inside the container; `local_root` is the
    controller's. In production they are the same string, since both sides
    use the same mount layout, but they name paths on different machines.
    """
    listing = machine.read_from_container(container, list_command(remote_root, data_dirs, batch))
    taking, waiting = parse_listing(listing)

    archive = machine.read_from_container(
        container, collect_command(remote_root, taking, COLLECT_TIMEOUT_SECONDS)
    )
    names = _extract(archive, local_root)
    delivered = [name for name in names if name in taking]
    if delivered:
        paths = [f"{remote_root}/{name}" for name in delivered]
        machine.exec_in_container(container, ["rm", "-f", *paths])
    remaining = None if waiting is None else waiting - len(delivered)
    return PullResult(pulled=names, remaining=remaining)
