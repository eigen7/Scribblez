"""The positions backfill: completing a tag's analyses of the known positions
(positions.py) after the manifest gains positions. Each generation's analysis
is made once, after its match, so a position added later is missing from
every analysis made before. The backfill analyzes what each one lacks under
that generation's own exports, newest generation first, and rewrites the
file with the entries already there unchanged.

It is the `backfill` role of deep_bot_leaf_eval and of deep_bot_generation,
which the operator adds, pauses and resumes in the dashboard, and whose
slot finishes once every analysis holds every position.
py/scripts/deep_bot_leaf_eval/backfill_positions.py runs the same runner on
chosen generations without the dashboard. The slot is local only: it reads
the tag's exports and analyses where the controller keeps them.
"""

import time
from collections.abc import Callable, Iterator
from pathlib import Path

from scribblez.deep_bot import known_positions
from scribblez.deep_bot.known_positions import Position
from scribblez.deep_bot_leaf_eval import positions as leaf_positions
from scribblez.paths import TagPaths
from scribblez.workloads.base import StatsSpec, WorkerContext
from scribblez.workloads.worker import WorkerStats, WorkerStopped

ROLE = "backfill"
STATS = StatsSpec(unit="positions", phases={"positions_s": "known positions"})

# A workload's exports of one analyzed generation: (paths, params,
# generation) -> (its leaf network, the leave inference model its analysis
# draws the opponent's leaves from).
Exports = Callable[[TagPaths, object, int], tuple[Path, Path]]


def backfill(
    paths: TagPaths,
    params,
    exports: Exports,
    generation: int,
    positions: list[Position],
    work: Path,
    threads: int,
) -> tuple[Path, int] | None:
    """Generation `generation`'s analysis with the `positions` it lacks
    added, written under `work`, and how many were added; None when it lacks
    none."""
    doc = leaf_positions.read(paths, generation)
    lacking = leaf_positions.missing(doc, positions)
    if not lacking:
        return None
    leaf, leave = exports(paths, params, generation)
    doc = leaf_positions.extend(doc, lacking, leaf, leave, work / "positions", threads)
    out = work / f"gen_{generation:06d}.json"
    leaf_positions.write(doc, out)
    return out, len(lacking)


def backfill_all(
    ctx: WorkerContext, exports: Exports, generations: list[int] | None
) -> Iterator[tuple[int, int, float]]:
    """Backfill every analyzed generation of the tag, or only `generations`,
    newest first, at most ctx.max_cycles of them (0: no limit), yielding
    (generation, positions added, seconds taken) after each one delivered."""
    paths = ctx.tag_paths()
    positions = known_positions.load()
    work = paths.work_dir(ctx.worker_id) / ROLE
    due = [
        g
        for g in reversed(leaf_positions.analyzed(paths))
        if generations is None or g in generations
    ]
    print(f"worker {ctx.worker_id}: checking {len(due)} analyses of tag '{ctx.tag}'")
    cycles = 0
    for g in due:
        if ctx.max_cycles and cycles >= ctx.max_cycles:
            return
        began = time.monotonic()
        result = backfill(paths, ctx.params, exports, g, positions, work, ctx.threads)
        if result is None:
            continue
        cycles += 1
        path, added = result
        ctx.sink.deliver(path, leaf_positions.rel(g))
        print(f"[generation {g}] added {added} positions")
        yield g, added, time.monotonic() - began


def run(ctx: WorkerContext, exports: Exports) -> int:
    """The backfill role's runner: every analyzed generation, each one a
    cycle of the slot's stats. Returns, so the slot finishes, once every
    analysis holds every position."""
    stats_rec = WorkerStats(ctx)
    try:
        for _, added, seconds in backfill_all(ctx, exports, None):
            stats_rec.cycle_done({"positions_s": seconds}, units=added, nbytes=0)
    except WorkerStopped:
        print("SIGTERM: exiting (an analysis in progress is redone on the next start)")
    return 0
