"""Reading a worker's results out of its container, over the control ssh link.

An ssh worker runs on a machine the operator owns, one the controller already
reaches over ssh to manage its container. Its outputs used to travel the same
way a cloud pod's do -- uploaded to the results bucket and pulled back down --
which put two R2 round trips on every cycle of a job whose actual work took
1.5 seconds, and made a home network's hiccups the throughput ceiling.

So the worker delivers locally, into its own container (SCZ_SINK=local), and
the controller collects from there: one `docker exec tar` per pass streams
whatever is finished back over the ssh connection that is already open. The
worker's cycle no longer contains a network at all.

Direction matters. The controller reaches the machine, never the reverse: the
dev container runs no sshd, and a worker that had to push would need a route
back to it, an address that survives DHCP, and a key -- for no gain, since the
bytes are the same either way. It also fails better: a controller that is down
or restarting leaves the worker generating into its own filesystem, and the
next pass collects the backlog.

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
from io import BytesIO
from pathlib import Path

# Records the worker keeps writing to and reading back: copied, never removed.
RECORD_DIRS = ("stats", "params")

# Where a pulled file waits while the rest of the archive extracts. Under the
# tag root, so moving it to its final place is a rename on one filesystem --
# a chunk appears in staging whole or not at all.
INCOMING_DIR = ".incoming"


def collect_command(root: str, subdirs: list[str]) -> list[str]:
    """The in-container command streaming `subdirs` of `root` as a tar archive.

    Skips whatever does not exist yet (a fresh worker has produced nothing) and
    emits nothing at all rather than asking tar to build an empty archive,
    which it refuses to do."""
    quoted = " ".join(shlex.quote(d) for d in subdirs)
    return [
        "sh",
        "-c",
        f"cd {shlex.quote(root)} 2>/dev/null || exit 0\n"
        f'set -- $(for d in {quoted}; do [ -d "$d" ] && printf "%s " "$d"; done)\n'
        '[ "$#" -eq 0 ] && exit 0\n'
        'exec tar -czf - "$@"',
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
    machine, container: str, *, remote_root: str, local_root: Path, data_dirs: list[str]
) -> list[str]:
    """Collect one ssh worker's finished outputs into the tag's local tree,
    returning the paths pulled (relative to both roots).

    `remote_root` is the tag's root inside the container and `local_root` is
    the controller's own. In production they are the same string -- the worker
    runs the same layout under the same mount root -- but they are different
    machines' paths, and only one of them can be opened here.
    """
    archive = machine.read_from_container(
        container, collect_command(remote_root, list(data_dirs) + list(RECORD_DIRS))
    )
    names = _extract(archive, local_root)
    delivered = [n for n in names if n.startswith("data/")]
    if delivered:
        paths = [f"{remote_root}/{name}" for name in delivered]
        machine.exec_in_container(container, ["rm", "-f", *paths])
    return names
