"""The match_arms role runner: plays a list of agent configurations ("arms")
against one opponent, one arm per cycle.

Each cycle takes the first arm without a match_arm row, plays its full pair
budget through the match harness, and writes the row the dashboard's Arms tab
reads. Every arm starts from the same base seed, so all arms face the same
deals pair for pair (common random numbers) and their scores compare directly.

Once every arm has a row the runner exits; when the dashboard respawns it, it
finds nothing to do and exits again. A stop mid-arm discards that arm's
partial results, and the next start replays it from the same seeds.
"""

import time

from scribblez import stats
from scribblez.dashboard import db
from scribblez.match_eval import harness
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.match_arms import Arm, parse_arms
from scribblez.workloads.worker import WorkerStats, WorkerStopped


def _pending_arm(conn, arms: list[Arm]) -> Arm | None:
    """The first arm with no row, in the experiment's declared order."""
    done = {r["arm"] for r in db.read_all_match_arms(conn)}
    for arm in arms:
        if arm.name not in done:
            return arm
    return None


def _play_arm(ctx: WorkerContext, arm: Arm) -> dict:
    """Play one arm's full pair budget in rounds; returns its match_arm record
    without elapsed_s, which the caller adds."""
    p = ctx.params
    results_file = ctx.tag_paths().work_dir(ctx.worker_id) / "match_results.jsonl"
    counts = [0] * 5
    wins = draws = losses = 0
    pairs_done = 0
    while pairs_done < p.pairs_per_arm:
        num_pairs = min(p.round_pairs, p.pairs_per_arm - pairs_done)
        round_result = harness.play_round(
            arm.player_spec,
            p.opponent,
            num_pairs=num_pairs,
            threads=ctx.threads,
            seed=p.seed + pairs_done,
            results_file=results_file,
            face_up_leaves=p.face_up_leaves,
        )
        for i, c in enumerate(stats.pair_score_counts(round_result.pair_scores)):
            counts[i] += c
        wins += round_result.wins
        draws += round_result.draws
        losses += round_result.losses
        pairs_done += num_pairs
        mean, _ = stats.mean_and_variance(counts)
        print(f"[{arm.name}] {pairs_done}/{p.pairs_per_arm} pairs: score={mean:.3f}")
    mean, ci = stats.score_confidence_interval(counts)
    return {
        "player_spec": arm.player_spec,
        "opponent": p.opponent,
        "games": wins + draws + losses,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "pair_counts": counts,
        "score": mean,
        "ci_half_width": ci,
    }


def run(ctx: WorkerContext) -> int:
    """The match_arms role entry point."""
    arms = parse_arms(ctx.params.arms)
    conn = db.connect(ctx.tag_paths().dashboard_db)
    stats_rec = WorkerStats(ctx)
    print(f"worker {ctx.worker_id}: {len(arms)} arm(s) vs '{ctx.params.opponent}'")

    cycles = 0
    try:
        while ctx.max_cycles == 0 or cycles < ctx.max_cycles:
            arm = _pending_arm(conn, arms)
            if arm is None:
                print("all arms measured; exiting")
                break
            cycles += 1
            t0 = time.monotonic()
            record = _play_arm(ctx, arm)
            record["elapsed_s"] = time.monotonic() - t0
            db.write_match_arm(conn, arm.name, record)
            stats_rec.cycle_done({"match_s": record["elapsed_s"]}, units=record["games"], nbytes=0)
            print(
                f"[{arm.name}] done: {record['wins']}/{record['draws']}/{record['losses']} "
                f"W/D/L, score={record['score']:.3f}+-{record['ci_half_width']:.3f} "
                f"in {record['elapsed_s']:.0f}s"
            )
    except WorkerStopped:
        print("SIGTERM: exiting (an in-flight arm is discarded and replayed on next start)")
    return 0
