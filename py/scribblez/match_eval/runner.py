"""The match_eval role: automated match play during training (roadmap A1).

A singleton worker beside the trainer that turns each exported checkpoint into
a match-play readout without blocking the training loop. Each cycle it picks
the newest eligible generation whose ONNX has no match row yet, plays paired
rounds against the configured opponent through the match harness, and lets the
sequential test (scribblez.stats.sprt) stop the match as soon as the pairs
settle it -- or the pair budget caps it. Results land in the tag's
dashboard.db (a match_eval row per generation, plus match_* metric scalars),
which is where the dashboard's win-rate and LLR curves read from.

A SIGTERM mid-match discards the partial match; the generation is still
row-less, so the next start replays it from the same fixed seeds.
"""

import time
from dataclasses import dataclass

from scribblez import stats
from scribblez.dashboard import db
from scribblez.match_eval import harness
from scribblez.stats import SprtResult
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.worker import WorkerStats, WorkerStopped

# How often to re-poll models/ when every exported generation already has its
# match row.
POLL_SECONDS = 10


def _exported_generations(paths) -> list[int]:
    return sorted(int(p.stem.rsplit("_", 1)[1]) for p in paths.onnx_dir.glob("model_epoch_*.onnx"))


def _evaluated_generations(conn) -> set[int]:
    return {r["epoch"] for r in conn.execute("SELECT epoch FROM match_eval")}


def _pending_generation(paths, conn, every: int) -> int | None:
    """The newest exported generation that is due a match and has none. Newest
    first keeps the readout tracking the training frontier; older stragglers
    backfill on later cycles."""
    done = _evaluated_generations(conn)
    pending = [g for g in _exported_generations(paths) if g % every == 0 and g not in done]
    return max(pending) if pending else None


def _model_player_spec(onnx_path) -> str:
    return f"--type=neural --model={onnx_path} --name=model"


def _storable_llr(result: SprtResult) -> float:
    """The LLR clamped to twice the decision bounds: a degenerate sweep has an
    infinite LLR, which SQLite stores but JSON cannot carry to the dashboard,
    and anything beyond the bounds means only 'decided' anyway."""
    return max(min(result.llr, 2.0 * result.upper), 2.0 * result.lower)


def _rows_trained_label(conn, gen: int) -> int:
    """The rows-clock the trainer recorded for this generation (the dashboard's
    alternate x-axis), 0 if the metrics row has not landed yet."""
    row = conn.execute(
        "SELECT value FROM metrics WHERE epoch = ? AND name = 'positions'", (gen,)
    ).fetchone()
    return int(row["value"]) if row is not None else 0


@dataclass(frozen=True)
class MatchOutcome:
    """One generation's finished match: the accumulated pentanomial pair
    counts, the per-game W/D/L behind them, and the sequential test's state."""

    pair_counts: list[int]
    wins: int
    draws: int
    losses: int
    sprt: SprtResult

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses


def _play_match(ctx: WorkerContext, gen: int) -> MatchOutcome:
    """Play one generation's match in SPRT-checked rounds."""
    p = ctx.params
    paths = ctx.tag_paths()
    results_file = paths.work_dir(ctx.worker_id) / "match_results.jsonl"
    counts = [0] * 5
    wins = draws = losses = 0
    pairs_done = 0
    result = stats.sprt(counts, p.match_p0, p.match_p1)
    while pairs_done < p.match_max_pairs:
        num_pairs = min(p.match_round_pairs, p.match_max_pairs - pairs_done)
        round_result = harness.play_round(
            _model_player_spec(paths.onnx_path(gen)),
            p.match_opponent,
            num_pairs=num_pairs,
            threads=ctx.threads,
            seed=p.match_seed + pairs_done,
            results_file=results_file,
            face_up_leaves=p.face_up_leaves,
        )
        for i, c in enumerate(stats.pair_score_counts(round_result.pair_scores)):
            counts[i] += c
        wins += round_result.wins
        draws += round_result.draws
        losses += round_result.losses
        pairs_done += num_pairs
        result = stats.sprt(counts, p.match_p0, p.match_p1)
        mean, _ = stats.mean_and_variance(counts)
        print(
            f"[gen {gen}] {pairs_done} pairs: score={mean:.3f} "
            f"llr={result.llr:+.2f} ({result.decision})"
        )
        if result.decision != "continue":
            break
    return MatchOutcome(counts, wins, draws, losses, result)


def run(ctx: WorkerContext) -> int:
    """The match_eval role runner: one generation's match per cycle."""
    p = ctx.params
    if p.match_every_generations <= 0:
        print("match_every_generations is 0: match eval is disabled for this tag")
        return 0
    paths = ctx.tag_paths()
    conn = db.connect(paths.dashboard_db)
    stats_rec = WorkerStats(ctx)
    print(f"worker {ctx.worker_id}: match eval for tag '{ctx.tag}' vs '{p.match_opponent}'")

    cycles = 0
    try:
        while ctx.max_cycles == 0 or cycles < ctx.max_cycles:
            gen = _pending_generation(paths, conn, p.match_every_generations)
            if gen is None:
                time.sleep(POLL_SECONDS)
                continue
            cycles += 1
            t0 = time.monotonic()
            outcome = _play_match(ctx, gen)
            elapsed = time.monotonic() - t0
            mean, ci = stats.score_confidence_interval(outcome.pair_counts)
            llr = _storable_llr(outcome.sprt)
            db.write_match_eval(
                conn,
                gen,
                {
                    "positions": _rows_trained_label(conn, gen),
                    "opponent": p.match_opponent,
                    "games": outcome.games,
                    "wins": outcome.wins,
                    "draws": outcome.draws,
                    "losses": outcome.losses,
                    "pair_counts": outcome.pair_counts,
                    "score": mean,
                    "ci_half_width": ci,
                    "llr": llr,
                    "llr_lower": outcome.sprt.lower,
                    "llr_upper": outcome.sprt.upper,
                    "decision": outcome.sprt.decision,
                    "elapsed_s": elapsed,
                },
            )
            db.write_metrics(
                conn, gen, {"match_score": mean, "match_llr": llr, "match_games": outcome.games}
            )
            stats_rec.cycle_done({"match_s": elapsed}, units=outcome.games, nbytes=0)
            print(
                f"[gen {gen}] done: {outcome.wins}/{outcome.draws}/{outcome.losses} W/D/L, "
                f"score={mean:.3f}+-{ci:.3f}, decision={outcome.sprt.decision} in {elapsed:.0f}s"
            )
    except WorkerStopped:
        print("SIGTERM: exiting (an in-flight match is discarded and replayed on next start)")
    return 0
