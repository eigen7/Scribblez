"""The generate role shared by the generational-training workloads
(position_eval, max_move_per_lane).

Generators know nothing about generations. Each cycle plays one .slog chunk of
self-play games into its own subdirectory of the worker's private work dir and
hands it to a background Deliverer, which moves it into the tag's staging area
(a rename for a local worker, an upload for a bucket-delivering one) while the
next cycle starts. The generation scheduler on the controller then assigns
staged chunks to generations (scribblez/generational/scheduler.py).

Delivery runs off the generation path because on a bucket-delivering worker an
upload (one rclone process per file) can cost a sizeable fraction of the time
it took to generate the chunk. Each cycle gets its own subdirectory so its
files never collide with a chunk still waiting to be delivered.

play_game always runs with seed 0, which makes the binary seed itself from
std::random_device. Any deterministic seed partition across a fleet would risk
two workers playing the same games, so the corpus is deliberately not
reproducible.

The work dir is wiped on start. A crash mid-cycle can leave a truncated .slog,
so leftovers are never delivered; a restart loses at most the in-flight chunk
and any chunks still queued for delivery.
"""

import queue
import shutil
import threading
import time
from pathlib import Path
from typing import NamedTuple

from scribblez.selfplay import hasty_player_spec, run_games
from scribblez.workloads.base import StatsSpec, WorkerContext
from scribblez.workloads.worker import WorkerStats, WorkerStopped

# The staging area under the tag's data/ dir (locally and in the bucket).
STAGING_DIR = "staging"

# Games per generator cycle. play_game writes one .slog per 1000 games (its
# kGamesPerFile), so one cycle delivers exactly one chunk.
GAMES_PER_CHUNK = 1000

GENERATOR_STATS = StatsSpec(unit="games", phases={"gen_s": "self-play", "upload_s": "deliver"})


def player_spec(params) -> str:
    # Only position_eval's params carry weirdbot_generation; the other workloads
    # sharing this role always play HastyBot.
    if getattr(params, "weirdbot_generation", False):
        return "--type=weirdbot"
    return hasty_player_spec(params.hasty_temperature, params.hasty_top_k, endgame=True)


def _deliver_chunk_dir(sink, worker_id: str, chunk_dir: Path) -> tuple[int, int]:
    """Deliver every .slog in `chunk_dir` to staging, with a -<worker_id> stem
    suffix for global uniqueness. Returns (chunks, bytes)."""
    chunks, nbytes = 0, 0
    for f in sorted(chunk_dir.glob("*.slog")):
        nbytes += sink.deliver(f, f"{STAGING_DIR}/{f.stem}-{worker_id}.slog")
        chunks += 1
    return chunks, nbytes


class DeliveryResult(NamedTuple):
    """One cycle's contribution to the stats sample, complete only once its
    chunk has finished delivering."""

    gen_seconds: float
    upload_seconds: float
    chunks: int
    nbytes: int


class Deliverer:
    """Delivers finished self-play chunks on a background thread.

    The thread drains a FIFO queue of submitted chunk directories: it delivers
    each directory's .slog files through the sink, removes the directory, and
    posts a DeliveryResult. A single thread keeps deliveries serialized and
    finishing in submission order, so results come back in cycle order.

    A delivery failure stops the thread and is re-raised from the next
    `collect()` or `drain()`, so it fails the runner instead of vanishing in
    the background.
    """

    def __init__(self, sink, worker_id: str):
        self._sink = sink
        self._worker_id = worker_id
        self._pending = queue.Queue()
        self._done = queue.Queue()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, chunk_dir: Path, gen_seconds: float):
        """Queue `chunk_dir` for delivery; returns immediately."""
        self._pending.put((chunk_dir, gen_seconds))

    def collect(self) -> list[DeliveryResult]:
        """Every delivery finished since the last call, in submission order.
        Non-blocking. Raises the background thread's failure, if it has hit
        one."""
        results = []
        while True:
            try:
                results.append(self._done.get_nowait())
            except queue.Empty:
                break
        if self._error is not None:
            raise self._error
        return results

    def drain(self) -> list[DeliveryResult]:
        """Block until every submitted chunk is delivered, then return what
        `collect` had not yet picked up. Call before the runner exits,
        including on SIGTERM, so nothing submitted is lost."""
        self._pending.put(None)
        self._thread.join()
        return self.collect()

    def _run(self):
        while True:
            item = self._pending.get()
            if item is None:
                return
            chunk_dir, gen_seconds = item
            try:
                t0 = time.monotonic()
                chunks, nbytes = _deliver_chunk_dir(self._sink, self._worker_id, chunk_dir)
                self._done.put(DeliveryResult(gen_seconds, time.monotonic() - t0, chunks, nbytes))
            except Exception as e:  # noqa: BLE001 -- re-raised by collect()/drain(), not lost
                self._error = e
                return
            finally:
                shutil.rmtree(chunk_dir, ignore_errors=True)


def _publish(stats: WorkerStats, results: list[DeliveryResult]):
    for r in results:
        stats.cycle_done(
            {"gen_s": r.gen_seconds, "upload_s": r.upload_seconds},
            units=r.chunks * GAMES_PER_CHUNK,
            nbytes=r.nbytes,
        )
        print(f"delivered chunk: {r.chunks} chunk(s) of {GAMES_PER_CHUNK} games staged")


def run_generate(ctx: WorkerContext) -> int:
    """The generate-role runner: one chunk per cycle, delivered to staging in
    the background."""
    p = ctx.params
    work_dir = ctx.tag_paths().work_dir(ctx.worker_id)
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    stats = WorkerStats(ctx)
    spec_str = player_spec(p)
    print(f"worker {ctx.worker_id} ({ctx.sink.kind}): generating tag '{ctx.tag}' with {p}")

    deliverer = Deliverer(ctx.sink, ctx.worker_id)
    cycle = 0
    try:
        while ctx.max_cycles == 0 or cycle < ctx.max_cycles:
            cycle += 1
            chunk_dir = work_dir / f"cycle-{cycle}"
            chunk_dir.mkdir()
            t0 = time.monotonic()
            rc = run_games(
                chunk_dir,
                num_games=GAMES_PER_CHUNK,
                threads=ctx.threads,
                player_spec=spec_str,
                random_opening_mean=p.random_opening_mean,
                face_up_leaves=p.face_up_leaves,
            )
            gen_seconds = time.monotonic() - t0
            if rc != 0:
                _publish(stats, deliverer.drain())
                return rc
            deliverer.submit(chunk_dir, gen_seconds)
            _publish(stats, deliverer.collect())
            print(f"cycle {cycle}: chunk queued for delivery")
    except WorkerStopped:
        _publish(stats, deliverer.drain())
        print("SIGTERM: drained pending deliveries; exiting (any in-flight chunk is discarded)")
        return 0
    _publish(stats, deliverer.drain())
    return 0


def fetch_deps(params):
    """Runtime data deps for HastyBot self-play: the engine's default lexicon
    and Macondo's strategy tables."""
    from cloud import worker_deps

    worker_deps.fetch_lexicon(worker_deps.DEFAULT_LEXICON)
    worker_deps.fetch_macondo_strategy(worker_deps.DEFAULT_LEXICON)
