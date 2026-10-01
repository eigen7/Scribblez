"""A tag's data home: the generation data plane, run beside its trainer.

For a tag whose data plane is "home" (TaskRecord.data_plane), the machine its
trainer runs on holds the whole data plane. Generators on that machine deliver
into its staging dir by rename. Generators elsewhere deliver where they run;
the controller collects their chunks and, for a home on an ssh machine, pushes
them into its staging (WorkerManager._relay_staging). So every chunk arrives
the way a colocated generator's does. A thread of the trainer (DataHome) does
the rest, every POLL_SECONDS:

  - It runs the generation scheduler (scheduler.tick) over the local tree.
  - It publishes the scheduler's state (its gate on the generate role and a
    heartbeat) through the trainer's records sink as scheduler_state.json. The
    controller parks and releases generators from that record instead of
    ticking the scheduler itself (scheduler.tick_for_task).

A data home on an ssh machine can vanish with its disk (a spot loss, a
released machine). So it also keeps the bucket able to resume it, on a thread
of its own, so a slow or stuck upload never holds up the scheduler or its
heartbeat:

  - Each complete generation is uploaded, chunks first and manifest last, so a
    manifest in the bucket means the whole generation is there. It is then
    marked published, and the trainer evicts no unpublished generation.
  - Generations older than the window behind the trainer's cursor are deleted
    from the bucket.
  - At start, before the thread runs, `restore` pulls the uploaded generations
    from the window onward that the machine lacks. A fresh home thereby holds
    the highest generation uploaded, and numbers new ones after it rather than
    reusing an index. The checkpoint and cursor come back through the records
    sink as before (position_eval.trainer.restore_from_sink).

The trainer's generation reads are the local sink's no-ops. A failed bucket
call is retried on the next pass. Any other failure stops the thread and is re-raised
by `check`, which the training loop calls: the runner fails rather than leave
the generators parked behind a heartbeat that has gone quiet.
"""

import fcntl
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from cloud.r2 import bucket_path, rclone
from cloud.sinks import r2_from_env

from scribblez.paths import SCHEDULER_STATE_REL, TagPaths
from scribblez.workloads.base import SchedulerHooks

from . import lifecycle, scheduler

POLL_SECONDS = 5

LOCK_NAME = "scheduler.lock"


def start_for(ctx, paths: TagPaths, params) -> "DataHome | None":
    """Start the data home beside trainer `ctx` when its tag's data plane is
    home (ctx.data_plane); None otherwise. With bucket credentials it restores
    the window a previous home uploaded, so a trainer that moved here from a
    remote home finds it. When the controller says this home is remote
    (SCZ_HOME_UPLOADS), that restore is required, and the home then keeps the
    bucket able to resume it with uploads."""
    if ctx.data_plane != scheduler.DATA_PLANE_HOME:
        return None
    cfg = scheduler.SchedulerConfig(
        games_per_generation=params.games_per_generation, open_ahead=params.open_ahead
    )
    r2 = r2_from_env() if "R2_BUCKET" in os.environ else None
    uploads = os.environ.get("SCZ_HOME_UPLOADS") == "1"
    home = DataHome(paths, cfg, ctx.records_sink, r2, window=params.window, uploads=uploads)
    if r2 is not None:
        home.restore(required=uploads)
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
    """The data plane's threads beside a trainer (see the module docstring).
    `r2` is the bucket a remote home's generations are kept in, for `restore`
    and, with `uploads`, for the upload thread; None when there is none."""

    def __init__(
        self,
        paths: TagPaths,
        cfg: scheduler.SchedulerConfig,
        records_sink,
        r2=None,
        chunk_games: scheduler.ChunkGamesFn = scheduler._header_games,
        *,
        window: int = 0,
        uploads: bool = False,
    ):
        assert r2 is not None or not uploads, "uploading needs the bucket"
        self._paths = paths
        self._cfg = cfg
        self._records = records_sink
        self._r2 = r2
        self._chunk_games = chunk_games
        self._window = window
        self.uploads = uploads  # whether the bucket keeps the generations (see restore)
        self._pruned_below = 0  # bucket generations below this index are gone
        self._gate: str | None = None
        self._error: Exception | None = None
        # Nothing is published: uploads are the upload thread's. Finishing the
        # generators stays the controller's (tick_for_task).
        self._hooks = SchedulerHooks(
            paths=paths, gate=self._set_gate, finish=_no_finish, publish=None
        )
        loops = [self.schedule, self.upload] if uploads else [self.schedule]
        self._threads = [threading.Thread(target=self._run, args=(f,), daemon=True) for f in loops]

    def start(self):
        for t in self._threads:
            t.start()

    def check(self):
        """Raise a thread's failure, if one has stopped on one."""
        if self._error is not None:
            raise self._error

    def step(self):
        """One pass of every loop, in order: schedule, then upload."""
        self.schedule()
        if self.uploads:
            self.upload()

    def schedule(self):
        """Assign staged chunks to generations, then publish the gate and a
        heartbeat. Local disk only, apart from the one small record."""
        with _tree_lock(self._paths):
            scheduler.tick(self._paths, self._cfg, self._hooks, self._chunk_games)
        self._publish_state()

    def upload(self):
        """Upload complete generations and prune the bucket's old ones."""
        self._upload_complete()
        self._prune_bucket()

    def restore(self, required: bool = True):
        """Pull the uploaded generations from the window behind the trainer's
        cursor onward that this machine lacks. Run before `start`, after the
        cursor is restored. When `required`, a failure raises, as a trainer
        that resumed on a partial window would silently diverge; otherwise an
        unreachable bucket leaves the window to what is on disk."""
        cursor = lifecycle.read_train_state(self._paths).get("generation_index", 0)
        indices = self._bucket_generations()
        if indices is None and not required:
            return
        assert indices is not None, "listing the bucket's generations failed"
        for index in indices:
            gen_dir = self._paths.generation_dir(index)
            if index < cursor - self._window or lifecycle.is_complete(gen_dir):
                continue
            prefix = self._generation_prefix(gen_dir)
            if not _listed(self._r2, f"{prefix}/{lifecycle.MANIFEST_NAME}"):
                continue  # an upload that never finished
            got = rclone(self._r2, "copy", prefix, str(gen_dir), capture=True)
            assert got.returncode == 0, f"restoring {gen_dir.name} failed: {got.stderr}"
            lifecycle.mark_published(gen_dir)
            print(f"data home: restored {gen_dir.name} from the bucket")

    def _run(self, loop):
        while True:
            try:
                loop()
            except Exception as e:  # noqa: BLE001 -- re-raised by check(), not lost
                self._error = e
                return
            time.sleep(POLL_SECONDS)

    def _set_gate(self, role: str, reason: str | None):
        assert role == scheduler.GENERATE_ROLE, role
        self._gate = reason

    def _generation_prefix(self, gen_dir: Path) -> str:
        return bucket_path(self._r2, self._paths.task, self._paths.tag, "generations", gen_dir.name)

    def _bucket_generations(self) -> list[int] | None:
        """Indices of the generation directories in the bucket, or None when
        the listing fails."""
        root = bucket_path(self._r2, self._paths.task, self._paths.tag, "generations")
        listing = rclone(self._r2, "lsf", "--dirs-only", root, capture=True)
        if listing.returncode != 0:
            print(f"data home: listing bucket generations failed: {listing.stderr.strip()}")
            return None
        return sorted(
            int(name.rstrip("/").removeprefix("gen_"))
            for name in listing.stdout.split()
            if name.startswith("gen_")
        )

    def _upload_complete(self):
        """Upload every complete generation not yet published, oldest first,
        stopping at a failure for the next pass to retry."""
        for index in lifecycle.list_generation_indices(self._paths):
            gen_dir = self._paths.generation_dir(index)
            if lifecycle.is_complete(gen_dir) and not lifecycle.is_published(gen_dir):
                if not self._upload(gen_dir):
                    return
                lifecycle.mark_published(gen_dir)

    def _upload(self, gen_dir: Path) -> bool:
        prefix = self._generation_prefix(gen_dir)
        manifest = lifecycle.MANIFEST_NAME
        res = rclone(
            self._r2, "copy", "--size-only", "--exclude", manifest, str(gen_dir), prefix,
            capture=True,
        )  # fmt: skip
        if res.returncode == 0:
            res = rclone(
                self._r2, "copyto", str(gen_dir / manifest), f"{prefix}/{manifest}", capture=True
            )
        if res.returncode != 0:
            print(f"data home: uploading {gen_dir.name} failed: {res.stderr.strip()}")
        return res.returncode == 0

    def _prune_bucket(self):
        """Delete bucket generations older than the window behind the trainer's
        cursor, which no restore needs any more. A failure waits for the next
        cursor move."""
        if self._window <= 0:
            return  # an unbounded corpus: every generation stays
        cursor = lifecycle.read_train_state(self._paths).get("generation_index", 0)
        keep_from = cursor - self._window
        indices = self._bucket_generations() if keep_from > self._pruned_below else None
        if indices is None:
            return
        for index in indices:
            if self._pruned_below <= index < keep_from:
                gen_dir = self._paths.generation_dir(index)
                gone = rclone(self._r2, "purge", self._generation_prefix(gen_dir), capture=True)
                if gone.returncode != 0:
                    print(f"data home: pruning {gen_dir.name} failed: {gone.stderr.strip()}")
                    return
        self._pruned_below = keep_from

    def _publish_state(self):
        record = {"gate": self._gate, "heartbeat": time.time()}
        try:
            self._records.push_json(SCHEDULER_STATE_REL, record)
        except AssertionError as e:  # a bucket sink's failed upload; retried next pass
            print(f"data home: publishing the scheduler state failed: {e}")


def _no_finish(role: str):
    raise AssertionError("a data home never finishes a role; the controller does")


def _listed(r2, path: str) -> bool:
    """Whether the bucket holds the object at `path`."""
    return bool(rclone(r2, "lsf", path, capture=True).stdout.strip())
