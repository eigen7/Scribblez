"""Pair stores: the shared generate loop and store readers of the workloads
whose unit of output is a .slog plus a same-stem sidecar (.sobs for kill_test,
.mset for move_set_eval, .mset plus .sobs for evidence_trajectories).

Every member of a pair gets the same -<worker_id> stem suffix on delivery, so
names are unique across workers and readers can still match pairs by stem.
Sidecars are delivered before their .slog. A .slog without its sidecar would
read downstream as pending work, while a sidecar without its .slog is inert,
so an interrupted delivery leaves only complete pairs plus inert leftovers.
"""

import time
import zlib
from pathlib import Path

from scribblez.workloads.worker import WorkerStats, WorkerStopped


def deliver_pairs(
    sink,
    out_dir: Path,
    worker_id: str,
    sidecar_ext: str,
    dest_dir: str,
    extra_sidecar_exts: tuple[str, ...] = (),
) -> tuple[int, int, float]:
    """Deliver every complete pair in `out_dir` to the tag's `dest_dir` store,
    including any a previous run left behind. A pair is complete when its
    `sidecar_ext` member exists. `extra_sidecar_exts` members go along when
    present, delivered first so they are never what a complete-looking pair is
    missing. Returns (pairs, bytes, seconds)."""
    moved, nbytes, t0 = 0, 0, time.monotonic()
    for sidecar in sorted(out_dir.glob(f"*{sidecar_ext}")):
        slog = sidecar.with_suffix(".slog")
        extras = [p for ext in extra_sidecar_exts if (p := sidecar.with_suffix(ext)).exists()]
        for f in (*extras, sidecar, slog):
            nbytes += sink.deliver(f, f"{dest_dir}/{f.stem}-{worker_id}{f.suffix}")
        moved += 1
    return moved, nbytes, time.monotonic() - t0


def run_pair_generate(
    ctx,
    run_cycle,
    sidecar_ext: str,
    dest_dir: str,
    target_pairs: int = 0,
    extra_sidecar_exts: tuple[str, ...] = (),
) -> int:
    """The generate-role loop shared by the pair-producing workloads.

    Flushes pairs a previous run left undelivered, then alternates
    `run_cycle(work_dir, params, threads) -> (returncode, phases)` with
    delivery until max_cycles, `target_pairs`, or SIGTERM. `phases` holds the
    cycle's timings keyed as in the role's StatsSpec; the delivery time is
    added as `upload_s`. A nonzero returncode ends the run with that code.

    `target_pairs` (0 = unbounded) is a size for the whole store, not a count
    for this worker. It is checked against the store as the sink sees it (the
    tag's data tree, or the bucket's listing of it), so a restarted worker
    resumes toward the same total and several workers on one tag stop together.
    """
    work_dir = ctx.tag_paths().work_dir(ctx.worker_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    stats = WorkerStats(ctx)

    def held() -> int:
        return ctx.sink.count_data_files(dest_dir, sidecar_ext)

    print(f"worker {ctx.worker_id} ({ctx.sink.kind}): generating tag '{ctx.tag}' with {ctx.params}")

    cycle = 0
    try:
        deliver_pairs(ctx.sink, work_dir, ctx.worker_id, sidecar_ext, dest_dir, extra_sidecar_exts)
        while ctx.max_cycles == 0 or cycle < ctx.max_cycles:
            if target_pairs and held() >= target_pairs:
                print(f"target of {target_pairs} pair(s) reached; exiting")
                return 0
            cycle += 1
            returncode, phases = run_cycle(work_dir, ctx.params, ctx.threads)
            if returncode != 0:
                return returncode
            moved, nbytes, secs = deliver_pairs(
                ctx.sink, work_dir, ctx.worker_id, sidecar_ext, dest_dir, extra_sidecar_exts
            )
            stats.cycle_done({**phases, "upload_s": secs}, units=moved, nbytes=nbytes)
            toward = f"/{target_pairs}" if target_pairs else ""
            in_store = f", {held()}{toward} in store" if target_pairs else ""
            print(f"cycle {cycle}: {moved} pair(s) delivered{in_store}")
    except WorkerStopped:
        moved, _, _ = deliver_pairs(
            ctx.sink, work_dir, ctx.worker_id, sidecar_ext, dest_dir, extra_sidecar_exts
        )
        print(f"SIGTERM: flushed {moved} completed pair(s); exiting")
    return 0


def complete_pairs(store_dir: str | Path, sidecar_ext: str) -> list[Path]:
    """The `sidecar_ext` files in a store whose .slog is present, sorted.
    Consumers need both halves (the sidecar, and the .slog replay the inputs
    are recomputed from), and a store can hold orphaned sidecars."""
    store_dir = Path(store_dir)
    return sorted(f for f in store_dir.glob(f"*{sidecar_ext}") if f.with_suffix(".slog").exists())


def count_pairs(store_dir: Path, sidecar_ext: str) -> int:
    """Pairs in a tag's store, counted by sidecar. An orphaned sidecar is
    counted too, so this is a progress reading, not a count of usable pairs
    (complete_pairs)."""
    return sum(1 for _ in store_dir.glob(f"*{sidecar_ext}")) if store_dir.is_dir() else 0


def split_pair_stems(stems: list[str], holdout_every: int) -> tuple[list[str], list[str]]:
    """(train, holdout) stems: about one in `holdout_every` is held out.

    The split is by whole pair because a position-level split leaks through
    the game prefixes positions share. Each stem's side is a hash of the stem,
    not its place in the sorted list: a trainer re-takes the split as the store
    grows, and with several generate workers interleaving deliveries, a
    list-position rule would move pairs between sides. A pair that changes
    sides is one that was trained on and then scored as held out.
    """
    ordered = sorted(stems)
    if holdout_every <= 0:
        return ordered, []
    held = [zlib.crc32(s.encode()) % holdout_every == 0 for s in ordered]
    train = [s for s, h in zip(ordered, held, strict=True) if not h]
    holdout = [s for s, h in zip(ordered, held, strict=True) if h]
    return train, holdout


# How long the store must go without a new sidecar before a tag with no
# target_pairs is taken to be done. A generate cycle (a self-play batch plus
# labeling) takes minutes, and a training pass over an early, small corpus is
# far quicker, so one pass without a delivery proves nothing.
QUIET_SECONDS = 900


class CorpusClock:
    """Decides when a tag's pair store has stopped growing: from then on a
    training pass covers the whole corpus and may spend the epoch budget.

    With `target_pairs` set, the store is final once it holds that many
    complete pairs and the current pass absorbed nothing new. The second
    condition matters when several generate workers run: one that was
    mid-cycle when another crossed the target still delivers its pairs, and
    starting the budget before they land would score the run's epochs against
    two different holdouts.

    Without a target, the store is final once its newest sidecar is older than
    QUIET_SECONDS. That is a property of the store, not of this trainer's
    history, so it answers the same on a fresh start, mid-run, or after a
    restart.
    """

    def __init__(self, store: Path, target_pairs: int, sidecar_ext: str):
        self._store = Path(store)
        self._target = target_pairs
        self._sidecar_ext = sidecar_ext

    def is_final(self, absorbed: int) -> bool:
        if self._target:
            held = len(complete_pairs(self._store, self._sidecar_ext))
            return not absorbed and held >= self._target
        return time.time() - self._last_delivery() >= QUIET_SECONDS

    def _last_delivery(self) -> float:
        """When the store was last written to, or 0.0 while it is empty."""
        if not self._store.is_dir():
            return 0.0
        files = self._store.glob(f"*{self._sidecar_ext}")
        return max((f.stat().st_mtime for f in files), default=0.0)
