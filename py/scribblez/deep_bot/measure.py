"""The deep_bot measure role (docs/plans/deep_bot_generations.md, "The
matches"): each assigned release, as the generating agent, plays a match
against HastyBot and one against release 0, over the same seeds and mirrored
pairs for every release, and every game is delivered.

The controller (measure_dispatch.py) assigns releases and records results, so
this runner needs only its inbox and runs on a machine with neither the
database nor the tag's releases:

    data/match_inbox/<worker_id>/
        releases/release_<k>/...       a release to play (release.files)
        releases/release_000000/...    the opponent, kept
        assigned/release_<k>.json      play release k

An assignment arrives after the releases it needs. A finished one is renamed
with DONE_SUFFIX and stays until the controller has its results
(match_eval/dispatch.py explains why). An interrupted match is replayed from
the same seeds at the next start.
"""

import time
from pathlib import Path

from scribblez.deep_bot import LEAF, LEAVE, matches, release
from scribblez.paths import DONE_SUFFIX
from scribblez.workloads import deep_bot
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.worker import WorkerStats, WorkerStopped

# Inbox poll interval while idle (match_eval/runner.py's reasoning).
POLL_SECONDS = 1

ASSIGNED_DIR = "assigned"


def assigned_release(inbox: Path) -> int | None:
    """The release this slot is to play, the newest of several; None while
    idle."""
    tokens = sorted((inbox / ASSIGNED_DIR).glob("release_*.json"))
    return int(tokens[-1].stem.removeprefix("release_")) if tokens else None


def agent_spec(ctx: WorkerContext, number: int) -> str:
    """The generating agent on release `number` of the slot's inbox."""
    inbox = ctx.tag_paths().match_inbox_dir(ctx.worker_id)
    return deep_bot.player_spec(
        ctx.params,
        inbox / release.export_rel(number, LEAF),
        inbox / release.export_rel(number, LEAVE),
    )


def opponent_spec(ctx: WorkerContext, opponent: str) -> str:
    return matches.HASTYBOT_SPEC if opponent == matches.HASTYBOT else agent_spec(ctx, 0)


def measure_release(ctx: WorkerContext, number: int, stats_rec: WorkerStats):
    """Play and deliver each of release `number`'s matches."""
    work = ctx.tag_paths().work_dir(ctx.worker_id)
    work.mkdir(parents=True, exist_ok=True)
    p = ctx.params
    for opponent in matches.opponents_of(number):
        specs = (agent_spec(ctx, number), opponent_spec(ctx, opponent))
        record, slog = matches.play(
            number, opponent, specs, p.match_games, p.match_seed, ctx.threads, work
        )
        nbytes = matches.deliver(ctx.sink, record, slog, work)
        stats_rec.cycle_done(
            {"match_s": record["elapsed_s"]}, units=len(record["games"]), nbytes=nbytes
        )
        print(matches.describe(record))


def run(ctx: WorkerContext) -> int:
    """The measure role entry point."""
    inbox = ctx.tag_paths().match_inbox_dir(ctx.worker_id)
    stats_rec = WorkerStats(ctx)
    print(f"worker {ctx.worker_id}: measuring the releases of tag '{ctx.tag}'")
    cycles = 0
    try:
        while ctx.max_cycles == 0 or cycles < ctx.max_cycles:
            number = assigned_release(inbox)
            if number is None:
                time.sleep(POLL_SECONDS)
                continue
            cycles += 1
            measure_release(ctx, number, stats_rec)
            token = inbox / ASSIGNED_DIR / f"release_{number:06d}.json"
            token.rename(token.with_name(token.name + DONE_SUFFIX))
    except WorkerStopped:
        print("SIGTERM: exiting (an in-flight match is discarded and replayed on next start)")
    return 0
