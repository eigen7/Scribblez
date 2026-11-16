"""The hidden_blind_spots generator (scribblez/workloads/hidden_blind_spots.py):
a cycle plays, pools, confirms and delivers a chunk, and under inferred
beliefs writes the viewer's opponent-rack panel for the positions it kept.

Apart from the workload module because the panel's beliefs come from the leave
inference model run under onnxruntime (leave_inference/beliefs.py). Every
worker imports the workload registry, and the engine worker image has no
onnxruntime.
"""

import shutil
import sys
import time
from pathlib import Path
from typing import NamedTuple

from scribblez import params as params_mod
from scribblez.blind_spots_beliefs import BeliefWriter
from scribblez.leave_inference.beliefs import LeaveBeliefs
from scribblez.workloads import policy_set
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.blind_spots import GCG_DIR
from scribblez.workloads.hidden_blind_spots import (
    FOUND_SUFFIX,
    confirm_chunk,
    deliver_chunk,
    play_chunk,
)
from scribblez.workloads.worker import WorkerStats, WorkerStopped


class CycleResult(NamedTuple):
    phases: dict[str, float]  # seconds, keyed as the role's StatsSpec phases
    found: int  # positions kept
    nbytes: int  # delivered


def write_beliefs(writer: BeliefWriter, chunk_dir: Path):
    """The viewer's beliefs for the confirmed chunk in `chunk_dir`."""
    (slog,) = chunk_dir.glob("*.slog")
    writer.write(slog.with_name(slog.stem + FOUND_SUFFIX), chunk_dir / GCG_DIR)


def run_cycle(
    ctx: WorkerContext,
    leaf: Path,
    model: Path | None,
    writer: BeliefWriter | None,
    chunk_dir: Path,
) -> CycleResult:
    """One cycle: play, pool, confirm and deliver a chunk in `chunk_dir`, with
    `writer`'s beliefs where the pools use `model`."""
    p = ctx.params
    t0 = time.monotonic()
    play_chunk(chunk_dir, ctx.worker_id, p, ctx.threads)
    t1 = time.monotonic()
    confirm_chunk(p, leaf, model, chunk_dir, ctx.threads)
    if writer is not None:
        write_beliefs(writer, chunk_dir)
    t2 = time.monotonic()
    found, nbytes = deliver_chunk(ctx.sink, chunk_dir)
    phases = {"gen_s": t1 - t0, "sim_s": t2 - t1, "upload_s": time.monotonic() - t2}
    return CycleResult(phases, found, nbytes)


def run(ctx: WorkerContext) -> int:
    """The generate-role runner: one chunk per cycle until max_cycles or
    SIGTERM. The work dir is wiped on start, since a stopped cycle's chunk is
    redone rather than resumed."""
    try:
        leaf, model = policy_set.resolve_sources(ctx)
    except (FileNotFoundError, params_mod.ParamsError) as e:
        print(f"error: hidden_blind_spots source: {e}", file=sys.stderr)
        return 1
    writer = None
    if model is not None:
        p = ctx.params
        writer = BeliefWriter(
            LeaveBeliefs(model), p.leave_inference_tag, p.leave_inference_generation
        )
    work_dir = ctx.tag_paths().work_dir(ctx.worker_id)
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    stats = WorkerStats(ctx)
    print(f"worker {ctx.worker_id} ({ctx.sink.kind}): collecting tag '{ctx.tag}' with {ctx.params}")
    cycle = 0
    try:
        while ctx.max_cycles == 0 or cycle < ctx.max_cycles:
            cycle += 1
            chunk_dir = work_dir / f"cycle-{cycle}"
            chunk_dir.mkdir()
            r = run_cycle(ctx, leaf, model, writer, chunk_dir)
            shutil.rmtree(chunk_dir)
            stats.cycle_done(r.phases, units=r.found, nbytes=r.nbytes)
            print(f"cycle {cycle}: {r.found} position(s) kept")
    except WorkerStopped:
        print("SIGTERM: the chunk in progress is discarded")
    return 0
