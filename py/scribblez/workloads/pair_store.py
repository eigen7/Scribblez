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
    """Deliver every complete .slog/sidecar pair in `out_dir` (not just the
    current cycle's -- a restarted worker flushes leftovers too) to the tag's
    `dest_dir` store. A pair is complete when its `sidecar_ext` member exists;
    `extra_sidecar_exts` members ride along when present (delivered first, so
    they are never the missing member of an already-complete-looking pair).
    Returns (pairs, bytes, seconds)."""
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
    """The generate-role loop shared by the pair-producing workloads: flush any
    completed pairs a previous run left undelivered, then alternate
    `run_cycle(work_dir, params, threads) -> (returncode, phases)` with pair
    delivery until max_cycles, `target_pairs`, or SIGTERM. `phases` is the
    cycle's per-phase timing sample (the role's StatsSpec keys), to which the
    delivery time is appended as `upload_s`; a nonzero cycle returncode ends
    the run with it.

    `target_pairs` (0 = unbounded) is a size the store is grown to rather than a
    count this worker produces: it is read from the store, so restarting a
    worker resumes toward the same total instead of starting over, and several
    workers on one tag converge on it together. Only a role whose sink leaves
    pairs in the tag's own data tree can ask for it -- a worker uploading to a
    bucket cannot see the store to count it.
    """
    work_dir = ctx.tag_paths().work_dir(ctx.worker_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    store = ctx.tag_paths().data_dir / dest_dir
    stats = WorkerStats(ctx)
    print(f"worker {ctx.worker_id} ({ctx.sink.kind}): generating tag '{ctx.tag}' with {ctx.params}")

    cycle = 0
    try:
        deliver_pairs(ctx.sink, work_dir, ctx.worker_id, sidecar_ext, dest_dir, extra_sidecar_exts)
        while ctx.max_cycles == 0 or cycle < ctx.max_cycles:
            if target_pairs and count_pairs(store, sidecar_ext) >= target_pairs:
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
            held = f", {count_pairs(store, sidecar_ext)}{toward} in store" if target_pairs else ""
            print(f"cycle {cycle}: {moved} pair(s) delivered{held}")
    except WorkerStopped:
        moved, _, _ = deliver_pairs(
            ctx.sink, work_dir, ctx.worker_id, sidecar_ext, dest_dir, extra_sidecar_exts
        )
        print(f"SIGTERM: flushed {moved} completed pair(s); exiting")
    return 0


def complete_pairs(store_dir: str | Path, sidecar_ext: str) -> list[Path]:
    """The `sidecar_ext` files in a store whose companion .slog is present,
    sorted. Every consumer needs both halves -- the sidecar, and the replay
    the inputs are recomputed from -- and a store can hold an orphaned sidecar
    (see the module docstring)."""
    store_dir = Path(store_dir)
    return sorted(f for f in store_dir.glob(f"*{sidecar_ext}") if f.with_suffix(".slog").exists())


def count_pairs(store_dir: Path, sidecar_ext: str) -> int:
    """Pairs in a tag's store, by counting sidecars. A delivery interrupted
    between a pair's two members can leave an orphaned sidecar briefly counted
    here; it is inert to every consumer, so the count stays a progress reading
    rather than a completeness guarantee."""
    return sum(1 for _ in store_dir.glob(f"*{sidecar_ext}")) if store_dir.is_dir() else 0


def split_pair_stems(stems: list[str], holdout_every: int) -> tuple[list[str], list[str]]:
    """(train, holdout) stems: about one in `holdout_every` is held out.

    File-level (whole pairs) because position-level splits leak through shared
    game prefixes, and decided by a hash of the stem rather than by a position
    in the list -- like move_set_eval.sweep_pair, and for a sharper reason
    here. A trainer re-takes this split as the store grows, so an assignment
    that depended on where a stem sat in the sorted list would move pairs
    between the sides whenever one arrived out of order (two generate workers
    interleave their deliveries), and a pair that changed sides is a pair
    trained on and then scored as held out.
    """
    ordered = sorted(stems)
    if holdout_every <= 0:
        return ordered, []
    held = [zlib.crc32(s.encode()) % holdout_every == 0 for s in ordered]
    train = [s for s, h in zip(ordered, held, strict=True) if not h]
    holdout = [s for s, h in zip(ordered, held, strict=True) if h]
    return train, holdout


# How long the store must sit untouched before a tag that declared no
# generation size is taken to be done. A generation cycle is 200 self-play
# games plus labeling -- minutes -- and a training pass over an early, small
# corpus is far quicker, so a single quiet pass says only that the pass fell
# between two deliveries.
QUIET_SECONDS = 900


class CorpusClock:
    """Decides when a tag's pair store has stopped growing -- the point from
    which a training pass is over the whole corpus and may spend the epoch
    budget.

    A tag with a declared `target_pairs` is answered by the store reaching it,
    and by nothing having arrived on the pass that saw it: a second generate
    worker mid-cycle when the first crossed the target still has pairs to
    deliver, and counting the budget from before they land would score the
    run's epochs against two different holdouts.

    With no declared size there is no end to read, so growth is judged from
    when the store was last written: a corpus whose newest sidecar is older
    than QUIET_SECONDS has no generator behind it, and one being delivered
    into never is. That is a property of the store rather than of this
    worker's own history, so it reads the same on a fresh start, mid-run, and
    after a restart -- none of which have watched the generator from the
    beginning.

    Complete pairs are counted the way every other reader of the store counts
    them (complete_pairs): a sidecar whose .slog has not landed yet is not one
    a trainer can use, and delivery writes the sidecar first.
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
