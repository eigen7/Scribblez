"""The generate role shared by the generational-training workloads.

A generator is generation-agnostic: each cycle plays one whole .slog chunk of
HastyBot self-play games into its own subdirectory of the worker's private
work dir, then hands that directory to a background Deliverer, which delivers
it to the tag's staging area (a rename for local workers, an upload for cloud
ones) while the next cycle's generation starts immediately. The generation
scheduler on the controller host assigns staged chunks to generation
directories; generators never see generations.

Delivery is deliberately off the generation critical path: on a cloud worker,
uploading a chunk (an rclone process per file) can take as long as a
meaningful fraction of the time spent generating it, and none of that upload
time does useful work if it blocks the next chunk from starting. Each cycle
gets its own chunk subdirectory (rather than reusing one work dir) so the next
cycle's files never collide with a chunk still waiting on its delivery.

Chunks always run play_game with seed 0 (the binary draws from
std::random_device per chunk): a fleet splitting a generation under any
deterministic seed partition would duplicate games, so distributed corpus
reproducibility is deliberately not offered.

The work dir is wiped on start: a crash mid-cycle may leave a truncated .slog
(play_game buffers a batch and writes it in one shot), so leftovers are never
delivered -- a restart loses at most the in-flight chunk and whatever chunks
were still queued for delivery.
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
    # WeirdBot self-play (a diagnostic corpus) puts the leave-forcing bot on both
    # seats; every other run uses HastyBot. The flag lives only on the
    # position_eval params, so read it defensively for workloads that share this
    # generate role without it.
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
    """Delivers finished self-play chunks off the generation critical path.

    A single background thread drains a FIFO queue of (chunk directory, that
    cycle's generation time) pairs submitted by the main thread: it delivers
    each directory's .slog files through the sink, removes the directory, and
    posts a DeliveryResult. One thread draining one queue is what keeps
    deliveries serialized with each other and completing in submission order,
    so a caller collecting results never has to sort them back into cycle
    order itself.

    A delivery failure (the sink asserts) stops the thread and is re-raised
    from the next `collect()` or from `drain()` -- surfacing as the runner's
    failure rather than vanishing silently in the background thread.
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
        """Every delivery that has finished since the last call, in the order
        submitted. Non-blocking: empty when none has finished yet. Raises the
        background thread's failure, if it hit one delivering any chunk so
        far (including ones this call doesn't otherwise report)."""
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
        """Block until every chunk submitted so far has finished delivering,
        then return whatever `collect` had not yet picked up. Call before the
        runner exits (including on SIGTERM) so nothing submitted is lost."""
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
    """The generate-role runner: one chunk per cycle, delivered to staging off
    the generation path."""
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
