"""The leaf_bot_generation measure role: a slot plays one assigned match at a
time, a release as DeepBot against HastyBot or against release 0, over the
same seeds and mirrored pairs for every release, and delivers each game's
result row. With its match against HastyBot it also delivers the release's
analysis of the known positions (deep_bot_leaf_eval/positions.py), which the
Positions tab shows.

The controller (measure_dispatch.py) assigns matches across the slots and
records results, so this runner needs only its inbox and runs on a machine
with neither the database nor the tag's releases:

    data/match_inbox/<worker_id>/
        releases/release_<k>/...                a release to play (LAYOUT.files)
        releases/release_000000/...             the seeds, kept
        assigned/release_<k>-<opponent>.json    play release k against opponent

An assignment arrives after the releases it needs. A finished one is renamed
with DONE_SUFFIX and stays until the controller has its result
(match_eval/dispatch.py explains why). An interrupted match is replayed from
the same seeds at the next start.
"""

import time
from pathlib import Path

from scribblez.deep_bot import LEAF, LEAVE, matches, release
from scribblez.deep_bot_leaf_eval import positions as leaf_positions
from scribblez.paths import DONE_SUFFIX
from scribblez.workloads import known_position_inputs
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.leaf_bot_generation import release_spec
from scribblez.workloads.worker import WorkerStats, WorkerStopped

# Inbox poll interval while idle (match_eval/runner.py's reasoning).
POLL_SECONDS = 1

ASSIGNED_DIR = "assigned"


def token_name(number: int, opponent: str) -> str:
    """The assignment of release `number`'s match against `opponent`."""
    return f"{matches.match_stem(number, opponent)}.json"


def parse_token(name: str) -> tuple[int, str]:
    """The (release, opponent) an assignment, played or not, stands for."""
    stem = name.removesuffix(DONE_SUFFIX).removesuffix(".json")
    number, opponent = stem.split("-", 1)
    return int(number.removeprefix("release_")), opponent


def assigned_match(inbox: Path) -> tuple[int, str] | None:
    """The match this slot is to play, the newest release's of several;
    None while idle."""
    tokens = sorted((inbox / ASSIGNED_DIR).glob("release_*.json"))
    return parse_token(tokens[-1].name) if tokens else None


def agent_spec(ctx: WorkerContext, number: int) -> str:
    """DeepBot on release `number` of the slot's inbox, as the matches play
    it."""
    p = ctx.params
    inbox = ctx.tag_paths().match_inbox_dir(ctx.worker_id)
    return release_spec(inbox, number, p.match_candidates, p.match_opp_leaves)


def opponent_spec(ctx: WorkerContext, opponent: str) -> str:
    return matches.HASTYBOT_SPEC if opponent == matches.HASTYBOT else agent_spec(ctx, 0)


def analyze_positions(ctx: WorkerContext, number: int, work: Path) -> Path:
    """Release `number`'s analysis of the known positions, written under
    `work`: DeepBot over every legal move, its opponent leaves drawn from the
    release's own leave inference model."""
    p = ctx.params
    inbox = ctx.tag_paths().match_inbox_dir(ctx.worker_id)
    doc = leaf_positions.analyze(
        known_position_inputs.load(ctx),
        inbox / release.export_rel(number, LEAF),
        inbox / release.export_rel(number, LEAVE),
        p.analysis_opp_leaves,
        p.match_candidates,
        work / "positions",
        ctx.threads,
    )
    path = work / f"positions_{number:06d}.json"
    leaf_positions.write({"generation": number, **doc}, path)
    return path


def play_assignment(ctx: WorkerContext, number: int, opponent: str, stats_rec: WorkerStats):
    """Play and deliver release `number`'s match against `opponent`, with
    the release's analysis of the known positions before the match against
    HastyBot, whose result the controller reads as the assignment done."""
    work = ctx.tag_paths().work_dir(ctx.worker_id)
    work.mkdir(parents=True, exist_ok=True)
    p = ctx.params
    specs = (agent_spec(ctx, number), opponent_spec(ctx, opponent))
    record, _ = matches.play(
        number, opponent, specs, p.match_games, p.match_seed, ctx.threads, work, keep_games=False
    )
    t0 = time.monotonic()
    nbytes = 0
    if opponent == matches.HASTYBOT:
        analysis = analyze_positions(ctx, number, work)
        nbytes += ctx.sink.deliver(analysis, leaf_positions.rel(number))
    positions_s = time.monotonic() - t0
    nbytes += matches.deliver(ctx.sink, record, None, work)
    stats_rec.cycle_done(
        {"match_s": record["elapsed_s"], "positions_s": positions_s},
        units=len(record["games"]),
        nbytes=nbytes,
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
            assigned = assigned_match(inbox)
            if assigned is None:
                time.sleep(POLL_SECONDS)
                continue
            cycles += 1
            play_assignment(ctx, *assigned, stats_rec)
            token = inbox / ASSIGNED_DIR / token_name(*assigned)
            token.rename(token.with_name(token.name + DONE_SUFFIX))
    except WorkerStopped:
        print("SIGTERM: exiting (an in-flight match is discarded and replayed on next start)")
    return 0
