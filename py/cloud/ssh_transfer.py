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

# Records the worker keeps writing to and reading back: copied, never removed.
RECORD_DIRS = ("stats", "params")

# Delivered files moved per pull. Sized so one pull is seconds of transfer
# rather than minutes -- a chunk is well under a megabyte, and the pass that
# calls this has other slots to serve.
BATCH = 16

# What the in-container tar is allowed, enforced inside the container so a
# transfer that overruns dies with its ssh client instead of outliving it. An
# abandoned tar keeps reading the backlog it was asked for, and one per pass
# compounds: six were found running at once on the jammed machine.
COLLECT_TIMEOUT_SECONDS = 60

# Where a pulled file waits while the rest of the archive extracts. Under the
# tag root, so moving it to its final place is a rename on one filesystem --
# a chunk appears in staging whole or not at all.
INCOMING_DIR = ".incoming"


@dataclass(frozen=True)
class PullResult:
    """What one pull moved, and what it left for the next one."""

    pulled: list[str]  # paths relative to the tag root, as extracted
    remaining: int  # delivered files still in the container


def list_command(root: str, data_dirs: list[str]) -> list[str]:
    """The in-container command listing delivered files, oldest name first.

    Names begin with the timestamp their chunk was written at, so sorting them
    drains in production order."""
    dirs = " ".join(shlex.quote(d) for d in data_dirs)
    return [
        "sh",
        "-c",
        f"cd {shlex.quote(root)} 2>/dev/null || exit 0\n"
        f'for d in {dirs}; do [ -d "$d" ] && ls -1 "$d" | sed "s|^|$d/|"; done | sort',
    ]


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
        f'exec timeout {seconds} tar -czf - "$@"',
    ]


def _extract(archive: bytes, root: Path) -> list[str]:
    """Unpack `archive` under `root`, atomically per file. Returns the archive
    member names, which are paths relative to `root`."""
    if not archive:
        return []
    names = []
    with tarfile.open(fileobj=BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            staged = root / INCOMING_DIR / member.name
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(tar.extractfile(member).read())
            dest = root / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(dest)
            names.append(member.name)
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
    listing = machine.read_from_container(container, list_command(remote_root, data_dirs))
    waiting = listing.decode(errors="replace").split()
    taking = waiting[:batch]

    archive = machine.read_from_container(
        container, collect_command(remote_root, taking, COLLECT_TIMEOUT_SECONDS)
    )
    names = _extract(archive, local_root)
    delivered = [name for name in names if name in set(taking)]
    if delivered:
        paths = [f"{remote_root}/{name}" for name in delivered]
        machine.exec_in_container(container, ["rm", "-f", *paths])
    return PullResult(pulled=names, remaining=len(waiting) - len(delivered))
