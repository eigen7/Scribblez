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

import json
import shutil
import time
from pathlib import Path

from scribblez.deep_bot import LEAF, LEAVE, matches, release
from scribblez.match_eval import harness
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


def play_match(ctx: WorkerContext, number: int, opponent: str, work: Path) -> tuple[dict, Path]:
    """Release `number` against `opponent`: the match's record and its .slog,
    under `work`."""
    results = work / "results.jsonl"
    slog_dir = work / "games"
    shutil.rmtree(slog_dir, ignore_errors=True)
    t0 = time.monotonic()
    harness.play_round(
        agent_spec(ctx, number),
        opponent_spec(ctx, opponent),
        num_pairs=ctx.params.match_games // 2,
        threads=ctx.threads,
        seed=ctx.params.match_seed,
        results_file=results,
        slog_dir=slog_dir,
    )
    games = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    (slog,) = slog_dir.glob("*.slog")  # play_game writes up to 1000 games a file
    return matches.result_record(number, opponent, games, time.monotonic() - t0), slog


def deliver(ctx: WorkerContext, record: dict, slog: Path, work: Path) -> int:
    """Hand a finished match to the controller: its games, then the rows
    that announce them. Returns the bytes delivered."""
    number, opponent = record["release"], record["opponent"]
    rows = work / "result.json"
    rows.write_text(json.dumps(record) + "\n")
    nbytes = ctx.sink.deliver(slog, matches.games_rel(number, opponent))
    return nbytes + ctx.sink.deliver(rows, matches.result_rel(number, opponent))


def measure_release(ctx: WorkerContext, number: int, stats_rec: WorkerStats):
    """Play and deliver each of release `number`'s matches."""
    work = ctx.tag_paths().work_dir(ctx.worker_id)
    work.mkdir(parents=True, exist_ok=True)
    for opponent in matches.opponents_of(number):
        record, slog = play_match(ctx, number, opponent, work)
        nbytes = deliver(ctx, record, slog, work)
        summary = matches.estimate(record["games"])
        stats_rec.cycle_done(
            {"match_s": record["elapsed_s"]}, units=summary["games"], nbytes=nbytes
        )
        print(
            f"[release {number}] vs {opponent}: {summary['wins']}/{summary['draws']}/"
            f"{summary['losses']} W/D/L, score={summary['score']:.3f}"
            f"+-{summary['ci_half_width']:.3f} in {record['elapsed_s']:.0f}s"
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
