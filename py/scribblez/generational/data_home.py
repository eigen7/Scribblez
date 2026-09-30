"""A tag's data home: the generation data plane, run beside its trainer.

For a tag whose data plane is "home" (TaskRecord.data_plane), the machine its
trainer runs on holds the whole data plane. Generators on that machine deliver
into its staging dir by rename; generators elsewhere deliver to bucket staging.
A thread of the trainer (DataHome) does the rest, every POLL_SECONDS:

  - It ingests bucket staging. Each chunk is downloaded beside staging, renamed
    in, and then deleted from the bucket, so the bucket never holds an ingested
    chunk and none is downloaded twice. If the bucket delete fails, the next
    pass downloads the chunk again, and the scheduler's ingest ledger drops the
    copy once the chunk is assigned.
  - It runs the generation scheduler (scheduler.tick) over the local tree.
  - It publishes the scheduler's state (its gate on the generate role and a
    heartbeat) through the trainer's records sink as scheduler_state.json. The
    controller parks and releases generators from that record instead of
    ticking the scheduler itself (scheduler.tick_for_task).

So to the trainer every chunk arrives the way a colocated generator's does,
and its generation reads are the local sink's no-ops. A failed bucket call is
retried on the next pass. Any other failure stops the thread and is re-raised
by `check`, which the training loop calls: the runner fails rather than leave
the generators parked behind a heartbeat that has gone quiet.
"""

import fcntl
import os
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from cloud.r2 import bucket_path, rclone
from cloud.sinks import r2_from_env

from scribblez.paths import SCHEDULER_STATE_REL, TagPaths

from . import scheduler

POLL_SECONDS = 5

# Where a chunk lands while it downloads, under data/work, so the rename into
# staging stays on one filesystem.
INGRESS_WORK_DIR = "ingress"

LOCK_NAME = "scheduler.lock"


def start_for(ctx, paths: TagPaths, params) -> "DataHome | None":
    """Start the data home beside trainer `ctx` when its tag's data plane is
    home (ctx.data_plane); None otherwise. It ingests from the bucket when the
    worker has bucket credentials."""
    if ctx.data_plane != scheduler.DATA_PLANE_HOME:
        return None
    cfg = scheduler.SchedulerConfig(
        games_per_generation=params.games_per_generation, open_ahead=params.open_ahead
    )
    r2 = r2_from_env() if "R2_BUCKET" in os.environ else None
    home = DataHome(paths, cfg, ctx.records_sink, r2)
    home.start()
    return home


@contextmanager
def _tree_lock(paths: TagPaths):
    """Exclusive use of the tag's generation structure: the scheduler has one
    writer, and this holds it to one even if a second tick ever appears."""
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.data_dir / LOCK_NAME, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


class DataHome:
    """The data plane's thread beside a trainer (see the module docstring).
    `r2` is the bucket to ingest remote generators' chunks from, or None when
    no credentials are available, in which case only colocated generators can
    feed the tag."""

    def __init__(
        self,
        paths: TagPaths,
        cfg: scheduler.SchedulerConfig,
        records_sink,
        r2=None,
        chunk_games: scheduler.ChunkGamesFn = scheduler._header_games,
    ):
        self._paths = paths
        self._cfg = cfg
        self._records = records_sink
        self._r2 = r2
        self._chunk_games = chunk_games
        self._gate: str | None = None
        self._error: Exception | None = None
        # The part of workloads.SchedulerHooks a tick uses. Nothing is mirrored
        # or published: the bucket holds no copy of this tree.
        self._hooks = SimpleNamespace(paths=paths, gate=self._set_gate, mirror=None, publish=None)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        shutil.rmtree(self._ingress_dir, ignore_errors=True)  # partial downloads
        self._thread.start()

    def check(self):
        """Raise the thread's failure, if it has stopped on one."""
        if self._error is not None:
            raise self._error

    def step(self):
        """One pass: ingest, schedule, publish the state."""
        with _tree_lock(self._paths):
            if self._r2 is not None:
                self._ingest()
            scheduler.tick(self._paths, self._cfg, self._hooks, self._chunk_games)
        self._publish_state()

    def _run(self):
        while True:
            try:
                self.step()
            except Exception as e:  # noqa: BLE001 -- re-raised by check(), not lost
                self._error = e
                return
            time.sleep(POLL_SECONDS)

    def _set_gate(self, role: str, reason: str | None):
        assert role == scheduler.GENERATE_ROLE, role
        self._gate = reason

    @property
    def _ingress_dir(self) -> Path:
        return self._paths.data_dir / "work" / INGRESS_WORK_DIR

    def _staging_prefix(self) -> str:
        return bucket_path(self._r2, self._paths.task, self._paths.tag, "staging")

    def _ingest(self):
        prefix = self._staging_prefix()
        listing = rclone(self._r2, "lsf", "--files-only", prefix, capture=True)
        if listing.returncode != 0:
            print(f"data home: listing bucket staging failed: {listing.stderr.strip()}")
            return
        self._ingress_dir.mkdir(parents=True, exist_ok=True)
        self._paths.staging_dir.mkdir(parents=True, exist_ok=True)
        for name in sorted(n for n in listing.stdout.split() if n.endswith(".slog")):
            if not self._ingest_one(prefix, name):
                return  # the bucket is failing; the next pass retries

    def _ingest_one(self, prefix: str, name: str) -> bool:
        tmp = self._ingress_dir / name
        got = rclone(self._r2, "copyto", f"{prefix}/{name}", str(tmp), capture=True)
        if got.returncode != 0:
            print(f"data home: downloading {name} failed: {got.stderr.strip()}")
            return False
        os.replace(tmp, self._paths.staging_dir / name)
        gone = rclone(self._r2, "deletefile", f"{prefix}/{name}", capture=True)
        if gone.returncode != 0:
            print(f"data home: deleting {name} from the bucket failed: {gone.stderr.strip()}")
            return False
        return True

    def _publish_state(self):
        record = {"gate": self._gate, "heartbeat": time.time()}
        try:
            self._records.push_json(SCHEDULER_STATE_REL, record)
        except AssertionError as e:  # a bucket sink's failed upload; retried next pass
            print(f"data home: publishing the scheduler state failed: {e}")
