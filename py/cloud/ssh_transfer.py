"""Reading a worker's results out of its container, over the control ssh link.

An ssh worker runs on a machine the operator owns, one the controller already
reaches over ssh to manage its container. Its outputs used to travel the same
way a cloud pod's do -- uploaded to the results bucket and pulled back down --
which put two R2 round trips on every cycle of a job whose actual work took
1.5 seconds, and made a home network's hiccups the throughput ceiling.

So the worker delivers locally, into its own container (SCZ_SINK=local), and
the controller collects from there: a `docker exec tar` per pass streams
finished output back over the ssh connection that is already open. The
worker's cycle no longer contains a network at all.

Direction matters. The controller reaches the machine, never the reverse: the
dev container runs no sshd, and a worker that had to push would need a route
back to it, an address that survives DHCP, and a key -- for no gain, since the
bytes are the same either way. It also fails better: a controller that is down
or restarting leaves the worker generating into its own filesystem, and the
next pass collects the backlog.

A pull takes a bounded batch, not everything waiting. Taking everything made a
pull's cost grow with the backlog it existed to drain, which is a loop that
only diverges: once one pull outran its timeout, none finished, the deletes
that follow a successful extraction never ran, and the backlog grew without
bound (10,746 chunks and 6.3 GB, observed on a laptop generating a chunk every
two seconds). Bounded batches make a pull's cost fixed and the drain rate a
floor -- BATCH files per pass against however fast one worker produces them.

Delivered data is moved (deleted from the container once safely on disk here);
the workers' own stats and params records are copied, because the worker reads
its counters back from them on restart. Deleting is a separate step after the
extraction succeeds, so a transfer that dies mid-stream loses nothing. A file
that is pulled twice (the delete failed, or the container restarted before it
ran) is deduplicated by the scheduler's ingest ledger, exactly as a re-synced
bucket chunk was.
"""

import shlex
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import IO

# Records the worker keeps writing to and reading back: copied, never removed.
RECORD_DIRS = ("stats", "params")

# Delivered files moved per pull. The bound being on files rather than on the
# backlog is the point; the size trades drain rate against how long one pull
# holds the pass's single blocking thread. Measured on this fleet: 32 chunks is
# ~20 MB raw, ~0.4 s to compress and ~1.4 s to move over a link clocked at
# 7.5 MB/s -- around a third of a pass, and ~29 net files drained per pass
# against one worker's ~2.5.
BATCH = 32

# Compression level for the stream. The worker machine is busy playing games,
# and its CPU is scarcer than the link: on real chunks, level 1 costs 0.2 s and
# the default 0.7 s, for 13% fewer bytes -- a net loss at this bandwidth.
COMPRESSION = "gzip -1"

# What the in-container tar is allowed, enforced inside the container so a
# transfer that overruns dies with its ssh client instead of outliving it. An
# abandoned tar keeps reading the backlog it was asked for, and one per pass
# compounds: six were found running at once on the jammed machine.
COLLECT_TIMEOUT_SECONDS = 60

# Where a pulled file waits while the rest of the archive extracts. Under the
# tag root, so moving it to its final place is a rename on one filesystem --
# a chunk appears in staging whole or not at all.
INCOMING_DIR = ".incoming"

# Names the file a sweep streams through, so one left behind by a process that
# died mid-copy is recognisable -- and removable -- by the next sweep.
SPOOL_PREFIX = "sweep-"


@dataclass(frozen=True)
class PullResult:
    """What one pull moved, and what it left for the next one."""

    pulled: list[str]  # paths relative to the tag root, as extracted
    remaining: int | None  # delivered files still there; None if that is unclear


def list_command(root: str, data_dirs: list[str], batch: int) -> list[str]:
    """The in-container command listing the next `batch` delivered files,
    oldest name first, then a "TOTAL <n>" line counting everything waiting.

    Names begin with the timestamp their chunk was written at, so sorting them
    drains in production order. Only the batch crosses the wire: a backlog of
    ten thousand would otherwise ship a third of a megabyte of filenames every
    pass to choose sixteen of them."""
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

    No output at all means the tag root does not exist there yet -- a real
    zero. Output without the sentinel is not understood, and says so: the
    total ends up deciding whether a container may be destroyed, so a count
    nobody can vouch for must not read as "empty"."""
    if not output.strip():
        return [], 0
    lines = output.decode(errors="replace").split()
    if len(lines) < 2 or lines[-2] != "TOTAL":
        return [], None
    return lines[:-2], int(lines[-1])


def collect_command(root: str, names: list[str], seconds: int) -> list[str]:
    """The in-container command streaming `names` plus the record directories
    as a tar archive. Emits nothing when there is nothing to send rather than
    asking tar to build an empty archive, which it refuses to do."""
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


def _extract(
    archive: bytes | IO[bytes], root: Path, mode: str = "r:gz", prefix: str = ""
) -> list[str]:
    """Unpack `archive` -- bytes, or a file object to read as a stream -- under
    `root`, atomically per file, returning the paths written relative to
    `root`. `prefix` is prepended to each member name, for an archive that does
    not already name things the way this tree does."""
    if isinstance(archive, bytes):
        if not archive:
            return []
        archive = BytesIO(archive)
    names = []
    with tarfile.open(fileobj=archive, mode=mode) as tar:
        # Iterated rather than read through getmembers(), which a stream ("r|",
        # what sweep_stopped uses) cannot serve: it allows one forward pass, so
        # building the member list would consume the data before extraction.
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

    A container is stopped gracefully -- SIGTERM, then a minute in which the
    worker flushes the output it has finished -- and that flush lands after the
    last collection could run, since collecting needs a running container. Left
    there it would go into the bin with the container. `docker cp` reads a
    stopped one, so this is the last look before a replacement.

    Takes the directory whole rather than in batches, because a stopped
    container cannot be asked what is in it. That is affordable only because a
    sweep runs solely on a container recorded as holding nothing -- a
    collection said so, or it never came up long enough to hold anything -- so
    what it finds is one worker's dying gasp: were something ever to replace a
    container holding a backlog, this would be gigabytes in one call. It
    streams through a file rather than memory even so -- the size is a remote
    machine's business, not something to hold in the dashboard.
    Uncompressed, unlike a collection: what a sweep moves is a couple of
    megabytes, and the shell pipeline that would compress it costs the ability
    to tell a failed copy from an empty one (see copy_from_container).
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
                # parent, and "r|" reads without holding the archive.
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
    """Collect up to `batch` of one ssh worker's finished outputs into the
    tag's local tree, plus its records.

    `remote_root` is the tag's root inside the container and `local_root` is
    the controller's own. In production they are the same string -- the worker
    runs the same layout under the same mount root -- but they are different
    machines' paths, and only one of them can be opened here.
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
