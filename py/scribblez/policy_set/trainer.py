"""The policy_set train role (docs/plans/deep_bot_v0.md, "Workload 2:
policy_set"). Until the policy-set model's forms arrive (step 8), it trains
nothing: it consumes generations and scores the static-equity baseline.

It hosts the tag's data home, the generation scheduler beside it, so its
generations form and its held-out set fills: about one chunk in
`holdout_every` is routed into the set (TagPaths.held_out_chunks_dir) until it
holds `holdout_chunks`. Per generation, once the cursor generation is
complete, it records the generation's pool statistics and the baseline's
metrics on the held-out pools (policy_set/metrics.py), advances its cursor by
the generation's pools, and evicts generations beyond the window.
"""

import time
from dataclasses import asdict
from pathlib import Path

from scribblez import params as params_mod
from scribblez.generational import data_home, lifecycle
from scribblez.generational.records import TrainRecorder
from scribblez.paths import TagPaths
from scribblez.policy_set import metrics, pools
from scribblez.position_eval.trainer import wait_for_generation
from scribblez.train_common import timed_print
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.worker import WorkerStats, WorkerStopped

# The baseline's record key prefix; each form's records carry its own.
BASELINE = "equity_"


def read_dir_pools(directory: Path, limit: int = 0) -> list[pools.Pool]:
    """The pools of the .pool files in `directory`, by file name, from the
    first `limit` files (0: all)."""
    files = sorted(directory.glob(f"*{pools.SUFFIX}"))
    if limit > 0:
        files = files[:limit]
    return [p for f in files for p in pools.read_pools(f)[1]]


def generation_record(paths: TagPaths, params, gen: int) -> tuple[dict, int]:
    """Generation `gen`'s record and its pool count: the generation's pool
    statistics, and the baseline on the held-out pools there are so far."""
    trained = read_dir_pools(paths.generation_dir(gen))
    held = read_dir_pools(paths.held_out_chunks_dir, params.holdout_chunks)
    record = {"epoch": gen, **metrics.pool_stats(trained, "")}
    if held:
        record.update(metrics.pool_stats(held, "eval_"))
        record.update(metrics.set_metrics(held, metrics.equity_top, BASELINE))
    return record, len(trained)


def run_generations(paths: TagPaths, params, recorder, home, sink, stats: WorkerStats):
    """The wait->score->advance loop, from the published cursor onward."""
    state = lifecycle.read_train_state(paths)
    gen, rows = state.get("generation_index", 0), state.get("rows_trained", 0)
    while not params_mod.reached(rows, params.max_rows):
        wait_for_generation(paths, gen, sink, home)
        t0 = time.time()
        record, n = generation_record(paths, params, gen)
        rows += n
        gen += 1
        lifecycle.write_train_state(paths, {"generation_index": gen, "rows_trained": rows})
        recorder.commit_generation(gen - 1, rows, record)
        held = record.get("eval_pools", 0)
        recall = record.get(f"{BASELINE}eval_recall_8", float("nan"))
        timed_print(f"[gen {gen - 1}] {n} pools; held-out {held}, equity recall@8 {recall:.3f}")
        stats.cycle_done({"eval_s": time.time() - t0}, units=n, nbytes=0)
        lifecycle.evict_beyond_window(paths, gen - 1, params.window, keep_unpulled=home.remote)
    timed_print(f"Stopped at {rows} pools (generation {gen}).")


def run(ctx: WorkerContext) -> int:
    """The train-role entry point."""
    params = ctx.params
    paths = ctx.tag_paths()
    paths.root.mkdir(parents=True, exist_ok=True)
    recorder = TrainRecorder(ctx.sink)
    recorder.publish_run(ctx.tag, asdict(params), 0, {}, {})
    home = data_home.start_for(
        ctx,
        paths,
        params,
        chunk_sidecar=pools.SUFFIX,
        holdout_every=params.holdout_every,
        holdout_chunks=params.holdout_chunks,
    )
    try:
        run_generations(paths, params, recorder, home, ctx.sink, WorkerStats(ctx))
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped.")
    return 0
