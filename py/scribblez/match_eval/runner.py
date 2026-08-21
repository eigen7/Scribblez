"""The match_eval role: automated match play during training (roadmap A1).

A worker beside the trainer -- on this machine or on another one over ssh --
that turns exported checkpoints into match-play readouts without blocking the
training loop. Each cycle plays the model its inbox holds against the
configured opponent through the match harness, over the same fixed number of
mirrored pairs for every generation, and delivers the outcome as one small
JSON.

What to play and what the result means are the controller's business, not the
worker's (match_eval/dispatch.py): it puts the model in the inbox and turns
the delivered file into the tag's dashboard.db rows. So this runner needs
nothing but its own directory -- which is what lets the role run on a machine
that has neither the database nor the exports, e.g. a second machine doing the
eval matches while this one trains.

An interrupted match (SIGTERM, a killed container) leaves the model where it
was, so the next start replays it from the same fixed seeds rather than losing
it or recording half of it.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from scribblez import stats
from scribblez.match_eval import harness
from scribblez.paths import DONE_SUFFIX, MATCH_RESULTS_DIR, ONNX_PREFIX, TagPaths
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.worker import WorkerStats, WorkerStopped

# How often to re-check the inbox when the controller has assigned nothing.
# Short because it is dead time on the eval machine at the worst moment: the
# next assignment lands just after a match ends, so a long poll would idle the
# GPU for half of it on every match. Listing a directory that holds at most a
# couple of files costs microseconds.
POLL_SECONDS = 1


def _assigned_model(paths: TagPaths, worker_id: str) -> Path | None:
    """The export this slot has been assigned, or None while it is idle. The
    newest wins if several are somehow there; the others are picked up on later
    cycles."""
    inbox = paths.match_inbox_dir(worker_id)
    models = sorted(inbox.glob(f"{ONNX_PREFIX}*.onnx"))
    return models[-1] if models else None


def _model_player_spec(onnx_path: Path) -> str:
    return f"--type=neural --model={onnx_path} --name=model"


@dataclass(frozen=True)
class MatchOutcome:
    """One generation's finished match: the pentanomial pair counts and the
    per-game W/D/L behind them."""

    pair_counts: list[int]
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses


def _play_match(ctx: WorkerContext, model: Path) -> MatchOutcome:
    """Play one generation's match: match_pairs mirrored pairs off the tag's
    base seed."""
    p = ctx.params
    result = harness.play_round(
        _model_player_spec(model),
        p.match_opponent,
        num_pairs=p.match_pairs,
        threads=ctx.threads,
        seed=p.match_seed,
        results_file=ctx.tag_paths().work_dir(ctx.worker_id) / "match_results.jsonl",
        face_up_leaves=p.face_up_leaves,
    )
    counts = stats.pair_score_counts(result.pair_scores)
    return MatchOutcome(counts, result.wins, result.draws, result.losses)


def match_record(ctx: WorkerContext, gen: int, outcome: MatchOutcome, elapsed: float) -> dict:
    """One finished match as the controller ingests it (dispatch.RESULT_FIELDS).
    The columns the controller fills in itself -- the rows-clock label -- are
    not here: they are read off a database this worker may not have."""
    mean, ci = stats.score_confidence_interval(outcome.pair_counts)
    return {
        "epoch": gen,
        "opponent": ctx.params.match_opponent,
        "games": outcome.games,
        "wins": outcome.wins,
        "draws": outcome.draws,
        "losses": outcome.losses,
        "pair_counts": outcome.pair_counts,
        "score": mean,
        "ci_half_width": ci,
        "elapsed_s": elapsed,
    }


def _deliver(ctx: WorkerContext, record: dict) -> int:
    """Hand one finished match to the controller. Returns the bytes delivered."""
    gen = record["epoch"]
    path = ctx.tag_paths().work_dir(ctx.worker_id) / f"gen_{gen:06d}-{ctx.worker_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return ctx.sink.deliver(path, f"{MATCH_RESULTS_DIR}/{path.name}")


def run(ctx: WorkerContext) -> int:
    """The match_eval role runner: one assigned generation's match per cycle."""
    paths = ctx.tag_paths()
    stats_rec = WorkerStats(ctx)
    print(
        f"worker {ctx.worker_id}: match eval for tag '{ctx.tag}' vs '{ctx.params.match_opponent}'"
    )
    if ctx.params.match_every_generations <= 0:
        # Said once rather than exiting: an exited worker is one the reconcile
        # pass respawns, so a disabled tag would become a restart loop. The
        # slot idles instead, and its log says why it will never do anything.
        print("match_every_generations is 0: match eval is disabled for this tag")

    cycles = 0
    try:
        while ctx.max_cycles == 0 or cycles < ctx.max_cycles:
            model = _assigned_model(paths, ctx.worker_id)
            if model is None:
                time.sleep(POLL_SECONDS)
                continue
            gen = paths.onnx_epoch(model)
            cycles += 1
            t0 = time.monotonic()
            outcome = _play_match(ctx, model)
            record = match_record(ctx, gen, outcome, time.monotonic() - t0)
            nbytes = _deliver(ctx, record)
            # Marked, not removed: the controller reads the inbox to decide
            # what to assign, and until it has the result in hand -- which for
            # a container is a collection away -- this generation must still
            # count as spoken for. It removes the marker once the result is
            # recorded (match_eval/dispatch.py).
            model.rename(model.with_name(model.name + DONE_SUFFIX))
            stats_rec.cycle_done(
                {"match_s": record["elapsed_s"]}, units=outcome.games, nbytes=nbytes
            )
            print(
                f"[gen {gen}] done: {outcome.wins}/{outcome.draws}/{outcome.losses} W/D/L, "
                f"score={record['score']:.3f}+-{record['ci_half_width']:.3f} "
                f"in {record['elapsed_s']:.0f}s"
            )
    except WorkerStopped:
        print("SIGTERM: exiting (an in-flight match is discarded and replayed on next start)")
    return 0
