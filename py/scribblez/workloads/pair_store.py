"""The pair-producing generate role's shared machinery: the cycle loop and the
delivery of .slog + same-stem-sidecar pairs (.sobs for kill_test, .mset for
move_set_eval) to a tag's data store.

Both members of a pair get the same -<worker_id> stem suffix, so names stay
globally unique across workers while preserving the stem-based pair matching
downstream readers rely on. The sidecar is delivered before its .slog: a .slog
missing its sidecar reads as pending work downstream, while an orphaned
sidecar is inert -- so the store only ever presents complete pairs plus inert
leftovers.
"""

import time
from pathlib import Path

from scribblez.workloads.worker import WorkerStats, WorkerStopped


def deliver_pairs(
    sink, out_dir: Path, worker_id: str, sidecar_ext: str, dest_dir: str
) -> tuple[int, int, float]:
    """Deliver every complete .slog/sidecar pair in `out_dir` (not just the
    current cycle's -- a restarted worker flushes leftovers too) to the tag's
    `dest_dir` store. Returns (pairs, bytes, seconds)."""
    moved, nbytes, t0 = 0, 0, time.monotonic()
    for sidecar in sorted(out_dir.glob(f"*{sidecar_ext}")):
        slog = sidecar.with_suffix(".slog")
        for f in (sidecar, slog):
            nbytes += sink.deliver(f, f"{dest_dir}/{f.stem}-{worker_id}{f.suffix}")
        moved += 1
    return moved, nbytes, time.monotonic() - t0


def run_pair_generate(ctx, run_cycle, sidecar_ext: str, dest_dir: str) -> int:
    """The generate-role loop shared by the pair-producing workloads: flush any
    completed pairs a previous run left undelivered, then alternate
    `run_cycle(work_dir, params, threads) -> (returncode, phases)` with pair
    delivery until max_cycles or SIGTERM. `phases` is the cycle's per-phase
    timing sample (the role's StatsSpec keys), to which the delivery time is
    appended as `upload_s`; a nonzero cycle returncode ends the run with it."""
    work_dir = ctx.tag_paths().work_dir(ctx.worker_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    stats = WorkerStats(ctx)
    print(f"worker {ctx.worker_id} ({ctx.sink.kind}): generating tag '{ctx.tag}' with {ctx.params}")

    cycle = 0
    try:
        deliver_pairs(ctx.sink, work_dir, ctx.worker_id, sidecar_ext, dest_dir)
        while ctx.max_cycles == 0 or cycle < ctx.max_cycles:
            cycle += 1
            returncode, phases = run_cycle(work_dir, ctx.params, ctx.threads)
            if returncode != 0:
                return returncode
            moved, nbytes, secs = deliver_pairs(
                ctx.sink, work_dir, ctx.worker_id, sidecar_ext, dest_dir
            )
            stats.cycle_done({**phases, "upload_s": secs}, units=moved, nbytes=nbytes)
            print(f"cycle {cycle}: {moved} pair(s) delivered")
    except WorkerStopped:
        moved, _, _ = deliver_pairs(ctx.sink, work_dir, ctx.worker_id, sidecar_ext, dest_dir)
        print(f"SIGTERM: flushed {moved} completed pair(s); exiting")
    return 0


def count_pairs(store_dir: Path, sidecar_ext: str) -> int:
    """Pairs in a tag's store, by counting sidecars. A delivery interrupted
    between a pair's two members can leave an orphaned sidecar briefly counted
    here; it is inert to every consumer, so the count stays a progress reading
    rather than a completeness guarantee."""
    return sum(1 for _ in store_dir.glob(f"*{sidecar_ext}")) if store_dir.is_dir() else 0
