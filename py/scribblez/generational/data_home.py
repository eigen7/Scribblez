"""A tag's data home: the generation data plane, run beside its trainer.

The machine a generational tag's trainer runs on holds the tag's whole data
plane: its staging, generations and ingest ledger. Generators on that machine
deliver into its staging dir by rename. Generators elsewhere deliver where
they run; the controller collects their chunks and, for a home on an ssh
machine, pushes them into its staging (WorkerManager._relay_staging). So every
chunk arrives the way a colocated generator's does. A thread of the trainer
(DataHome) does the rest, every POLL_SECONDS:

  - It runs the generation scheduler (scheduler.tick) over the local tree.
  - It publishes the scheduler's state (its gate on the generate role and a
    heartbeat) through the trainer's records sink as scheduler_state.json. The
    controller parks and releases generators from that record
    (scheduler.tick_for_task).

A data home on an ssh machine can vanish with its disk (a spot loss, a
released machine), so the controller keeps the durable copy: it pulls each
complete generation into its own tree and then marks it pulled in the home
(lifecycle.ACK_NAME). A remote home (SCZ_REMOTE_HOME) therefore never lets its
trainer evict a generation the controller has not pulled. A new home is
seeded from the controller's copy before its trainer starts, and a home the
trainer leaves is swept into it (WorkerManager._seed_home, _sweep_home).

The trainer's generation reads are the local sink's no-ops. A failure stops
the thread and is re-raised by `check`, which the training loop calls: the
runner fails rather than leave the generators parked behind a heartbeat that
has gone quiet.
"""

import fcntl
import os
import threading
import time
from contextlib import contextmanager

from scribblez.paths import SCHEDULER_STATE_REL, TagPaths
from scribblez.workloads.base import SchedulerHooks

from . import scheduler

POLL_SECONDS = 5

LOCK_NAME = "scheduler.lock"


def start_for(ctx, paths: TagPaths, params) -> "DataHome":
    """Start the data home beside trainer `ctx`. It is remote when the
    controller says so (SCZ_REMOTE_HOME): on an ssh machine, whose
    generations the controller pulls."""
    cfg = scheduler.SchedulerConfig(
        games_per_generation=params.games_per_generation, open_ahead=params.open_ahead
    )
    remote = os.environ.get("SCZ_REMOTE_HOME") == "1"
    home = DataHome(paths, cfg, ctx.records_sink, remote=remote)
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
    `remote`: whether the controller pulls this home's generations, so its
    trainer keeps each until it is pulled."""

    def __init__(
        self,
        paths: TagPaths,
        cfg: scheduler.SchedulerConfig,
        records_sink,
        chunk_games: scheduler.ChunkGamesFn = scheduler._header_games,
        *,
        remote: bool = False,
    ):
        self._paths = paths
        self._cfg = cfg
        self._records = records_sink
        self._chunk_games = chunk_games
        self.remote = remote
        self._gate: str | None = None
        self._error: Exception | None = None
        # Finishing the generators stays the controller's (tick_for_task).
        self._hooks = SchedulerHooks(paths=paths, gate=self._set_gate, finish=_no_finish)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def check(self):
        """Raise the thread's failure, if it has stopped on one."""
        if self._error is not None:
            raise self._error

    def step(self):
        """One pass: assign staged chunks to generations, then publish the gate
        and a heartbeat."""
        with _tree_lock(self._paths):
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

    def _publish_state(self):
        self._records.push_json(SCHEDULER_STATE_REL, {"gate": self._gate, "heartbeat": time.time()})


def _no_finish(role: str):
    raise AssertionError("a data home never finishes a role; the controller does")
