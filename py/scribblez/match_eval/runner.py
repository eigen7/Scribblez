"""The match_eval role: match play against a fixed opponent during training.

The worker runs beside the trainer, locally or on another machine over ssh.
Each cycle it plays the model in its inbox against the configured opponent,
over the same seeds and number of mirrored pairs for every generation, and
delivers the outcome as a small JSON file.

The controller (dispatch.py) decides what to play and records the results, so
this runner needs only its own inbox and can run on a machine that has
neither the database nor the exports.

An interrupted match leaves the model in the inbox, so the next start replays
it from the same seeds.

How an export plays is the workload's: run() takes a Player, the export's
--player spec and the information condition the match is played under. The
default, model_player, is position_eval's and the evidence trainer's. A
workload may also deliver files of its own about each played export
(AfterMatch), before the result that marks the generation done.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from scribblez import stats
from scribblez.match_eval import harness
from scribblez.paths import DONE_SUFFIX, MATCH_RESULTS_DIR, ONNX_PREFIX, TagPaths
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.worker import WorkerStats, WorkerStopped

# Inbox poll interval while idle. Kept short because the next assignment
# usually lands just after a match ends, so polling latency is idle GPU time on
# every match; listing the tiny inbox is nearly free.
POLL_SECONDS = 1


def _assigned_model(paths: TagPaths, worker_id: str) -> Path | None:
    """The export assigned to this slot, or None while idle. If several are
    present, the newest goes first."""
    inbox = paths.match_inbox_dir(worker_id)
    models = sorted(inbox.glob(f"{ONNX_PREFIX}*.onnx"))
    return models[-1] if models else None


def _model_player_spec(onnx_path: Path) -> str:
    """The --player spec for an assigned position-evaluation export: the
    neural agent."""
    return f"--type=neural --model={onnx_path} --name=model"


# How an export plays: (ctx, the assigned export) -> (its --player spec,
# whether the match is played with face-up leaves).
Player = Callable[[WorkerContext, Path], tuple[str, bool]]
# What a workload delivers about a played export besides the match:
# (ctx, the export, its generation) -> [(a file, the path under data/ it is
# delivered to)].
AfterMatch = Callable[[WorkerContext, Path, int], list[tuple[Path, str]]]


def model_player(ctx: WorkerContext, model: Path) -> tuple[str, bool]:
    """A position-evaluation export (_model_player_spec), under the tag's own
    information condition."""
    return _model_player_spec(model), ctx.params.face_up_leaves


@dataclass(frozen=True)
class MatchOutcome:
    """One generation's match result: pair-score counts
    (stats.pair_score_counts) and per-game W/D/L."""

    pair_counts: list[int]
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses


def _play_match(ctx: WorkerContext, spec: str, face_up_leaves: bool) -> MatchOutcome:
    """Play `match_pairs` mirrored pairs of `spec` against the tag's opponent
    from the tag's fixed seed."""
    p = ctx.params
    result = harness.play_round(
        spec,
        p.match_opponent,
        num_pairs=p.match_pairs,
        threads=ctx.threads,
        seed=p.match_seed,
        results_file=ctx.tag_paths().work_dir(ctx.worker_id) / "match_results.jsonl",
        face_up_leaves=face_up_leaves,
    )
    counts = stats.pair_score_counts(result.pair_scores)
    return MatchOutcome(counts, result.wins, result.draws, result.losses)


def match_record(ctx: WorkerContext, gen: int, outcome: MatchOutcome, elapsed: float) -> dict:
    """A finished match in the form the controller ingests
    (dispatch.RESULT_FIELDS). Rows trained is omitted; the controller adds it
    from the database, which this worker may not have."""
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


def run(
    ctx: WorkerContext, player: Player = model_player, after_match: AfterMatch | None = None
) -> int:
    """The match_eval role entry point, the assigned exports playing as
    `player` says, and `after_match`'s files delivered with each result."""
    paths = ctx.tag_paths()
    stats_rec = WorkerStats(ctx)
    print(
        f"worker {ctx.worker_id}: match eval for tag '{ctx.tag}' vs '{ctx.params.match_opponent}'"
    )
    if ctx.params.match_every_generations <= 0:
        # Idle rather than exit: the reconcile pass respawns exited workers, so
        # exiting would become a restart loop.
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
            outcome = _play_match(ctx, *player(ctx, model))
            record = match_record(ctx, gen, outcome, time.monotonic() - t0)
            extras = after_match(ctx, model, gen) if after_match else []
            nbytes = sum(ctx.sink.deliver(path, rel) for path, rel in extras)
            nbytes += _deliver(ctx, record)
            # Mark rather than delete: the generation must stay assigned until
            # the controller has the result (see dispatch.py).
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
