"""The generate role shared by the generational-training workloads.

A generator is generation-agnostic: each cycle plays one whole .slog chunk of
HastyBot self-play games in the worker's private work dir, then delivers it to
the tag's staging area (a rename for local workers, an upload for cloud ones).
The generation scheduler on the controller host assigns staged chunks to
generation directories; generators never see generations.

Chunks always run play_game with seed 0 (the binary draws from
std::random_device per chunk): a fleet splitting a generation under any
deterministic seed partition would duplicate games, so distributed corpus
reproducibility is deliberately not offered.

The work dir is wiped on start: a crash mid-cycle may leave a truncated .slog
(play_game buffers a batch and writes it in one shot), so leftovers are never
delivered -- a restart loses at most the in-flight chunk.
"""

import shutil
import time

from scribblez.selfplay import hasty_player_spec, run_games
from scribblez.workloads.base import StatsSpec, WorkerContext
from scribblez.workloads.worker import WorkerStats, WorkerStopped

# The staging area under the tag's data/ dir (locally and in the bucket).
STAGING_DIR = "staging"

GENERATOR_STATS = StatsSpec(unit="games", phases={"gen_s": "self-play", "upload_s": "deliver"})


def player_spec(params) -> str:
    return hasty_player_spec(
        params.hasty_temperature, params.hasty_top_k, params.hasty_temp_min_bag, endgame=True
    )


def _deliver_chunks(ctx: WorkerContext, work_dir) -> tuple[int, int]:
    """Deliver every completed chunk to staging, with a -<worker_id> stem
    suffix for global uniqueness. Returns (chunks, bytes)."""
    chunks, nbytes = 0, 0
    for f in sorted(work_dir.glob("*.slog")):
        nbytes += ctx.sink.deliver(f, f"{STAGING_DIR}/{f.stem}-{ctx.worker_id}.slog")
        chunks += 1
    return chunks, nbytes


def run_generate(ctx: WorkerContext) -> int:
    """The generate-role runner: one chunk per cycle, delivered to staging."""
    p = ctx.params
    work_dir = ctx.tag_paths().work_dir(ctx.worker_id)
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    stats = WorkerStats(ctx)
    spec_str = player_spec(p)
    print(f"worker {ctx.worker_id} ({ctx.sink.kind}): generating tag '{ctx.tag}' with {p}")

    cycle = 0
    try:
        while ctx.max_cycles == 0 or cycle < ctx.max_cycles:
            cycle += 1
            t0 = time.monotonic()
            rc = run_games(
                work_dir,
                num_games=p.games_per_chunk,
                games_per_file=p.games_per_chunk,
                threads=ctx.threads,
                player_spec=spec_str,
                random_opening_mean=p.random_opening_mean,
            )
            gen_seconds = time.monotonic() - t0
            if rc != 0:
                return rc
            t1 = time.monotonic()
            chunks, nbytes = _deliver_chunks(ctx, work_dir)
            stats.cycle_done(
                {"gen_s": gen_seconds, "upload_s": time.monotonic() - t1},
                units=chunks * p.games_per_chunk,
                nbytes=nbytes,
            )
            print(f"cycle {cycle}: {chunks} chunk(s) of {p.games_per_chunk} games staged")
    except WorkerStopped:
        print("SIGTERM: exiting (any in-flight chunk is discarded on next start)")
    return 0


def fetch_deps(params):
    """Runtime data deps for HastyBot self-play: the engine's default lexicon
    and Macondo's strategy tables."""
    from cloud import worker_deps

    worker_deps.fetch_lexicon(worker_deps.DEFAULT_LEXICON)
    worker_deps.fetch_macondo_strategy(worker_deps.DEFAULT_LEXICON)
