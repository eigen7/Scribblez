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
        known-positions/...            the known positions, kept
        assigned/release_<k>.json      play release k

An assignment arrives after the releases it needs. A finished one is renamed
with DONE_SUFFIX and stays until the controller has its results
(match_eval/dispatch.py explains why). An interrupted match is replayed from
the same seeds at the next start.
"""

import json
import time
from pathlib import Path

from scribblez.deep_bot import LEAF, LEAVE, known_positions, matches, release
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


def agent_spec(ctx: WorkerContext, number: int, reply: str = "hasty") -> str:
    """The generating agent on release `number` of the slot's inbox, or the
    same agent under another `reply` model."""
    inbox = ctx.tag_paths().match_inbox_dir(ctx.worker_id)
    return deep_bot.player_spec(
        ctx.params,
        inbox / release.export_rel(number, LEAF),
        inbox / release.export_rel(number, LEAVE),
        reply,
    )


def take_picks(ctx: WorkerContext, number: int, work: Path) -> dict:
    """Release `number`'s move at each known position of the slot's inbox,
    under each reply model (matches.picks_record)."""
    inbox = ctx.tag_paths().match_inbox_dir(ctx.worker_id)
    positions = known_positions.load(inbox / matches.KNOWN_POSITIONS_DIR)
    picks = {}
    for reply in matches.PICK_REPLIES:
        asked = known_positions.ask(
            positions, agent_spec(ctx, number, reply), work / "picks", ctx.threads
        )
        picks[reply] = [
            {"id": p.id, "pick": a["pick"], "reading": p.read(a["pick"]), "seconds": a["seconds"]}
            for p, a in zip(positions, asked, strict=True)
        ]
    return matches.picks_record(number, picks)


def deliver_picks(ctx: WorkerContext, record: dict, work: Path) -> int:
    path = work / "picks.json"
    path.write_text(json.dumps(record) + "\n")
    return ctx.sink.deliver(path, matches.picks_rel(record["release"]))


def opponent_spec(ctx: WorkerContext, opponent: str) -> str:
    return matches.HASTYBOT_SPEC if opponent == matches.HASTYBOT else agent_spec(ctx, 0)


def measure_release(ctx: WorkerContext, number: int, stats_rec: WorkerStats):
    """Take and deliver release `number`'s picks at the known positions,
    then play and deliver each of its matches."""
    work = ctx.tag_paths().work_dir(ctx.worker_id)
    work.mkdir(parents=True, exist_ok=True)
    p = ctx.params
    t0 = time.monotonic()
    record = take_picks(ctx, number, work)
    nbytes = deliver_picks(ctx, record, work)
    stats_rec.cycle_done({"picks_s": time.monotonic() - t0}, units=0, nbytes=nbytes)
    print(f"[release {number}] picks at the known positions: {reference_counts(record)}")
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


def reference_counts(record: dict) -> str:
    """How many of a release's picks are the positions' reference moves, per
    reply model, in a line."""
    return ", ".join(
        f"{sum(r['reading'] == known_positions.REFERENCE for r in rows)} of {len(rows)} "
        f"under {reply}"
        for reply, rows in record["picks"].items()
    )


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
