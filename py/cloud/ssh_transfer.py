"""Moving files between the controller and a worker's container over the ssh
link the controller already uses to manage it.

A worker with the local sink (SCZ_SINK=local) delivers into its own
container, and the controller collects: each pass streams a `docker exec tar`
of finished output back over the open ssh connection. This keeps the network
out of the worker's cycle (a bucket round trip per cycle would dwarf work that
takes seconds) and needs no bucket or extra credential. The same link runs the
other way for roles whose work the controller assigns: push_file drops a file
where the worker polls for it (match eval: the ONNX of the generation to
play), and list_dir reads back what is there. relay_files pushes a batch of
delivered files on into another container (chunks collected here for a data
home on an ssh machine).

Only the controller initiates. The dev container runs no sshd, and a worker
that pushed would need a route, a stable address and a key for the
controller. This also degrades well: while the controller is down, the worker
keeps generating into its own filesystem and the next pass collects the
backlog.

A pull takes a bounded batch (at most BATCH entries and BATCH_BYTES, but
always at least one entry), not the whole backlog. If a pull's cost grew with
the backlog, one slow pull could exceed its timeout, skip the deletes that
follow extraction, and leave a larger backlog for the next pull, which then
also times out; the backlog would grow without bound. With a bounded batch,
every pull costs about the same and drains at a steady rate. Its time limit
scales with the bytes it carries (MIN_RATE), so a lone 117 MB checkpoint gets
the time it needs.

Delivered data is moved: deleted from the container once it is on disk here.
That covers the flat data directories (chunks, pairs, a trainer's exports and
records) and the *pair directories*: a subdirectory, such as a trainer's
state/gen_NNNNNN pair, taken whole once its marker file (the cursor, written
last) exists, so a pair is never pulled half-written. The worker's stats and
params records, and root record files such as a data home's
scheduler_state.json, are copied instead, because the worker reads or
rewrites them in place. The delete runs only after extraction succeeds, so a
transfer that dies mid-stream loses nothing. A file pulled twice (the delete
failed, or the container restarted first) is deduplicated by whatever consumes
it: the scheduler's ingest ledger for chunks, the cursor rule for state pairs.
"""

import shlex
import shutil
import tarfile
import uuid
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

# Bytes moved per pull, past the first entry: a trainer's 40 MB export and
# 117 MB checkpoint each go in a pull of their own rather than with a batch
# of chunks behind them.
BATCH_BYTES = 64 * 1024 * 1024

# The slowest link a pull is given time for, in bytes per second. A pull's
# time limit is its bytes at this rate, and never less than
# COLLECT_TIMEOUT_SECONDS.
MIN_RATE = 1024 * 1024

# The worker machine's CPU is busy playing games and is scarcer than the link.
# On real chunks, level 1 takes 0.2 s against the default's 0.7 s, for only
# 13% more bytes: a net win at this bandwidth.
COMPRESSION = "gzip -1"

# Time limit on the in-container tar, enforced inside the container so an
# overrunning transfer dies with its ssh client. Otherwise an abandoned tar
# keeps running, and one more accumulates each pass.
COLLECT_TIMEOUT_SECONDS = 60

# How long the ssh read waits beyond the in-container tar's own limit, so the
# tar always times out first and dies with its client.
READ_MARGIN_SECONDS = 30

# Where a pulled file is written before it is moved into place. It is under
# the tag root, so the move is a same-filesystem rename and a file appears at
# its destination whole or not at all.
INCOMING_DIR = ".incoming"

# Prefix of the spool file a sweep streams through, so the next sweep can
# recognize and remove one left by a process that died mid-copy.
SPOOL_PREFIX = "sweep-"

# Prefix of the directory one extraction stages its files in (_extract).
EXTRACT_PREFIX = "x-"

# The archive a relay builds before streaming it, overwritten by the next
# (a tag has one relay at a time: its data home's).
RELAY_SPOOL = "relay.tar.gz"


@dataclass(frozen=True)
class PullResult:
    pulled: list[str]  # paths relative to the tag root, as extracted
    remaining: int | None  # delivered files still waiting; None if unknown


def list_command(
    root: str,
    data_dirs: list[str],
    batch: int,
    pair_dirs: dict[str, str] | None = None,
    batch_bytes: int = BATCH_BYTES,
) -> list[str]:
    """The in-container command listing the next batch of delivered entries,
    then "TOTAL <n>" counting every entry waiting and "BYTES <b>" summing the
    batch's sizes.

    An entry is a file in a data directory, or a subdirectory of a pair
    directory whose marker file exists (`pair_dirs`: directory -> marker).
    Entries are sorted by path, so chunks (named by write timestamp) drain in
    production order, and a trainer's exports (models/) come before the
    records (records/) that announce them. The batch is a prefix of that
    order: at most `batch` entries, and past the first, at most
    `batch_bytes`. The command trims the list in the container, so only the
    batch's names cross the wire however large the backlog. Dotted names and
    `.tmp` names are in-progress writes (a trainer's exports and records are
    written beside their final name, with a `.tmp` suffix) and are skipped."""
    dirs = " ".join(shlex.quote(d) for d in data_dirs)
    pairs = "".join(
        f'[ -d {shlex.quote(d)} ] && for p in {shlex.quote(d)}/*/; do p="${{p%/}}"; '
        f'[ -f "$p/{marker}" ] && echo "$(du -sb "$p" | cut -f1) $p"; done\n'
        for d, marker in (pair_dirs or {}).items()
    )
    take = (
        f"!stop && taken < {batch} && (taken == 0 || sum + $1 <= {batch_bytes})"
        " { print $2; taken++; sum += $1; next } { stop = 1 }"
        ' END { print "TOTAL", NR+0; print "BYTES", sum+0 }'
    )
    return [
        "sh",
        "-c",
        f"cd {shlex.quote(root)} 2>/dev/null || exit 0\n"
        "{\n"
        f'for d in {dirs}; do [ -d "$d" ] && '
        "find \"$d\" -mindepth 1 -maxdepth 1 -type f ! -name '.*' ! -name '*.tmp' "
        "-printf '%s %p\\n'; done\n"
        f"{pairs}"
        f"}} | sort -k2 | awk '{take}'",
    ]


@dataclass(frozen=True)
class Listing:
    names: list[str]  # the batch, in order
    total: int | None  # every entry waiting; None if the listing was cut off
    nbytes: int  # the batch's size


def parse_listing(output: bytes) -> Listing:
    """The batch, the total waiting and the batch's bytes, from list_command's
    output.

    Empty output means the tag root does not exist yet: a true zero. Output
    without the trailing lines gives a total of None, never 0, because the
    total decides whether a container may be destroyed."""
    if not output.strip():
        return Listing([], 0, 0)
    lines = output.decode(errors="replace").split()
    if len(lines) < 4 or lines[-4] != "TOTAL" or lines[-2] != "BYTES":
        return Listing([], None, 0)
    return Listing(lines[:-4], int(lines[-3]), int(lines[-1]))


def collect_command(
    root: str, names: list[str], seconds: int, record_files: tuple[str, ...] = ()
) -> list[str]:
    """The in-container command streaming `names` plus the record directories
    and `record_files` as a gzipped tar. Emits nothing when there is nothing
    to send, since tar refuses to create an empty archive. A file the worker
    removes between the listing and the tar (an older state pair it pruned)
    is skipped rather than failing the pull; what it was part of arrives
    torn, and its consumer discards it."""
    quoted = " ".join(shlex.quote(name) for name in names)
    records = " ".join([*RECORD_DIRS, *(shlex.quote(f) for f in record_files)])
    return [
        "sh",
        "-c",
        f"cd {shlex.quote(root)} 2>/dev/null || exit 0\n"
        f"set -- {quoted}\n"
        f'for d in {records}; do [ -e "$d" ] && set -- "$@" "$d"; done\n'
        '[ "$#" -eq 0 ] && exit 0\n'
        f"exec timeout {seconds} tar -c --ignore-failed-read"
        f" --use-compress-program='{COMPRESSION}' -f - \"$@\"",
    ]


def transfer_seconds(nbytes: int) -> int:
    """The time limit for a pull carrying `nbytes` (MIN_RATE)."""
    return max(COLLECT_TIMEOUT_SECONDS, nbytes // MIN_RATE)


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


def _batch(files: list[Path], batch: int, batch_bytes: int) -> list[Path]:
    """The prefix of `files` a pull of the same bounds would take: at most
    `batch` files and, past the first, at most `batch_bytes`."""
    taken, total = [], 0
    for f in files:
        size = f.stat().st_size
        if len(taken) == batch or (taken and total + size > batch_bytes):
            break
        taken.append(f)
        total += size
    return taken


def unpack_command(root: str, dest_rel: str, count: int) -> list[str]:
    """The in-container command unpacking a gzipped tar of `count` files off
    stdin into `dest_rel`.

    The files unpack into a work directory beside the destination, and are
    renamed in only once all `count` are there, so a stream cut short leaves
    nothing in the destination, and the worker reading it (a data home's
    scheduler, which takes every file in staging as whole) never sees a
    partial file. The work directory is named per push and removed on any
    failure."""
    dest = f"{root}/{dest_rel}"
    work = f"{root}/data/work/relay-{uuid.uuid4().hex[:12]}"
    return [
        "sh",
        "-c",
        f"w={shlex.quote(work)}; d={shlex.quote(dest)}\n"
        "trap 'rm -rf \"$w\"' EXIT\n"
        'mkdir -p "$w" "$d"\n'
        'tar -xz -C "$w" -f - || exit 1\n'
        'n=$(ls -A "$w" | wc -l)\n'
        f'[ "$n" -eq {count} ] || {{ echo "got $n of {count} files" >&2; exit 1; }}\n'
        'mv "$w"/* "$d"/',
    ]


def relay_files(
    machine,
    container: str,
    *,
    remote_root: str,
    local_root: Path,
    rel: str,
    batch: int = BATCH,
    batch_bytes: int = BATCH_BYTES,
) -> list[str]:
    """Push a batch of the files in `rel` under the controller's tag root into
    the same directory of the container's, then delete them here. Returns the
    names moved.

    The batch is the oldest files by name, bounded as a pull is. Dotted and
    `.tmp` names are writes still in progress and stay. The local copies go
    only once the container has every file of the batch in place, so a push
    that fails is pushed again whole on the next call; a file that arrives
    twice (the deletes here failed) is deduplicated by its consumer, as a
    pulled one is."""
    src = local_root / rel
    if not src.is_dir():
        return []
    ready = sorted(
        f
        for f in src.iterdir()
        if f.is_file() and not f.name.startswith(".") and not f.name.endswith(".tmp")
    )
    files = _batch(ready, batch, batch_bytes)
    if not files:
        return []
    incoming = local_root / INCOMING_DIR
    incoming.mkdir(parents=True, exist_ok=True)
    archive = incoming / RELAY_SPOOL
    try:
        with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
            for f in files:
                tar.add(f, arcname=f.name)
        machine.write_to_container(container, unpack_command(remote_root, rel, len(files)), archive)
    finally:
        archive.unlink(missing_ok=True)
    for f in files:
        f.unlink()
    return [f.name for f in files]


def _extract(
    archive: bytes | IO[bytes], root: Path, mode: str = "r:gz", prefix: str = ""
) -> list[str]:
    """Unpack `archive` (bytes, or a file object) under `root`, each file
    atomically, and return the paths written relative to `root`. `prefix` is
    prepended to every member name.

    Each call stages under a directory of its own: the slots of a data home
    share one volume, so their concurrent pulls carry the same record files
    (stats/), and a shared staging path would let one pull rename away the
    file another is about to."""
    if isinstance(archive, bytes):
        if not archive:
            return []
        archive = BytesIO(archive)
    names = []
    stage = root / INCOMING_DIR / f"{EXTRACT_PREFIX}{uuid.uuid4().hex[:12]}"
    try:
        with tarfile.open(fileobj=archive, mode=mode) as tar:
            # Iterate rather than call getmembers(): a stream ("r|", as
            # sweep_stopped uses) allows one forward pass, and building the
            # member list would consume the data before extraction.
            for member in tar:
                if not member.isfile():
                    continue
                name = f"{prefix}{member.name}"
                staged = stage / name
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_bytes(tar.extractfile(member).read())
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                staged.replace(dest)
                names.append(name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return names


def sweep_stopped(
    machine,
    container: str,
    *,
    remote_root: str,
    local_root: Path,
    data_dirs: list[str],
    pair_dirs: dict[str, str] | None = None,
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
    explains why. Pair directories are taken whole too, a pair still being
    written included; its consumer discards a pair without its marker.
    """
    names = []
    incoming = local_root / INCOMING_DIR
    incoming.mkdir(parents=True, exist_ok=True)
    for stale in incoming.glob(f"{SPOOL_PREFIX}*"):
        stale.unlink()  # a spool from a sweep whose process died mid-copy
    for data_dir in [*data_dirs, *(pair_dirs or {})]:
        archive = incoming / f"{SPOOL_PREFIX}{Path(data_dir).name}.tar"
        try:
            path = f"{remote_root}/{data_dir}"
            if not machine.copy_from_container(container, path, archive):
                continue
            with open(archive, "rb") as stream:
                # docker cp names members relative to the copied directory's
                # parent, which for a top-level directory is the root itself;
                # "r|" streams without seeking.
                parent = Path(data_dir).parent
                prefix = "" if parent == Path(".") else f"{parent}/"
                names += _extract(stream, local_root, mode="r|", prefix=prefix)
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
    pair_dirs: dict[str, str] | None = None,
    record_files: tuple[str, ...] = (),
    batch: int = BATCH,
    batch_bytes: int = BATCH_BYTES,
) -> PullResult:
    """Collect a batch of one ssh worker's finished outputs, plus its records,
    into the tag's local tree (see the module docstring for what is moved and
    what copied).

    `remote_root` is the tag root inside the container; `local_root` is the
    controller's. In production they are the same string, since both sides
    use the same mount layout, but they name paths on different machines.
    """
    listing = parse_listing(
        machine.read_from_container(
            container, list_command(remote_root, data_dirs, batch, pair_dirs, batch_bytes)
        )
    )
    seconds = transfer_seconds(listing.nbytes)
    archive = machine.read_from_container(
        container,
        collect_command(remote_root, listing.names, seconds, record_files),
        timeout=seconds + READ_MARGIN_SECONDS,
    )
    names = _extract(archive, local_root)
    delivered = [e for e in listing.names if any(n == e or n.startswith(e + "/") for n in names)]
    if delivered:
        paths = [f"{remote_root}/{entry}" for entry in delivered]
        machine.exec_in_container(container, ["rm", "-rf", *paths])
    remaining = None if listing.total is None else listing.total - len(delivered)
    return PullResult(pulled=names, remaining=remaining)
