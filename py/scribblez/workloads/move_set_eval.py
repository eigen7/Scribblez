"""The move-set-evaluation workload (docs/roadmap.md A2): distillation targets
for the student model, and the student's trainer.

A generate cycle plays a HastyBot self-play batch into a fresh .slog, runs
move_set_eval_target_generator over every .slog still missing its .mset
sidecar (so an interrupted run's backlog is picked up), and delivers every
complete pair to the tag's slogs/ store. It is the kill_test cycle with the
distillation target generator in place of the sim tool.

The teacher is a position_eval tag's export, named by `teacher_tag` and pinned
at task creation (`finalize`) to a concrete `teacher_generation`: the tag's
latest export when the param is left at -1. The generator stamps the teacher's
content hash into each .mset and MsetDataset refuses a corpus with mixed
hashes, so every worker must read the same model bytes. Pinning guarantees
that: position_eval exports are write-once, and a worker restarting after a
newer generation lands still resolves the pinned one.

The generate role runs on a GPU slot of either kind. A local slot reads the
teacher ONNX in place. A remote slot has no position_eval tag to read, so the
role declares the pinned export as an input (RoleSpec.inputs): the controller
stages a copy for it (in the bucket for a rented machine, in the container on
the operator's own), and run_generate finds it through base.resolve_input.

Every `sweep_every`-th pair is labeled in the generator's full-sweep mode
instead: every legal candidate of a few positions per game, capped. These
pairs are the held-out slice the student's ranking metrics (top-K recall,
teacher-value regret) are read on, because the
stratified ~15-candidate sample never shows the tail moves those metrics
exist to catch. The choice is a hash of the .slog stem (sweep_pair), so a
resumed cycle makes the same decision; the .mset header records it for
readers downstream.

The singleton train role (scribblez/move_set_eval/trainer.py) distills the
student over the pair store while the generator is still filling it, holding
its epoch budget until the store reaches `target_pairs`. A tag started with
one worker of each role therefore grows its corpus, trains on all of it, and
stops, unattended. Like the generator it runs on a GPU slot of either kind.

evidence_trajectories is a separate workload that reuses this one's labeling
step (workloads/mset_targets.py).
"""

import dataclasses
import functools
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from cloud.runtime_abi import RUNTIME_TORCH

from scribblez import params as params_mod
from scribblez.generational.optimizer_arms import OPTIMIZER_SCHEDULE_FREE, OPTIMIZERS
from scribblez.move_set_eval.targets import complete_pairs, partition_full_sweep
from scribblez.params import param
from scribblez.paths import POSITION_EVAL, TagPaths
from scribblez.selfplay import hasty_player_spec, run_games
from scribblez.trunk_arms import TRUNK_CONV, TRUNK_TRANSFORMER, TRUNKS
from scribblez.workloads import mset_targets, pair_store
from scribblez.workloads.base import (
    RoleSpec,
    StatsSpec,
    WorkerContext,
    WorkloadSpec,
    resolve_input,
)

# The tag's pair store, under the tag's data/ dir (locally and in the bucket).
SLOGS_DIR = "slogs"


# Parameter profiles (WorkloadSpec.profiles): one recipe per trunk, the same as
# position_eval's. Each overrides only some of the dataclass defaults below;
# the transformer profile adds gradient clipping.
PROFILES = {
    TRUNK_TRANSFORMER: {"trunk": TRUNK_TRANSFORMER, "grad_clip": 1.0},
    TRUNK_CONV: {"trunk": TRUNK_CONV},
}


@dataclass(frozen=True)
class MoveSetEvalParams:
    """A tag's parameters, frozen at task creation. The freeze is what keeps
    the corpus coherent: MsetDataset checks only that a corpus has one teacher
    hash and one information condition, so nothing else would catch workers
    sampling candidates under different quotas. Worker-level knobs (thread
    count) live on the slots.
    """

    teacher_tag: str = param(
        "",
        "name of the position_eval tag whose exported model is the teacher; required. One of "
        "its exports is pinned as the teacher at task creation (the latest, unless "
        "teacher_generation names one), so the tag must already have exported",
    )
    teacher_generation: int = param(
        -1,
        "which exported generation of teacher_tag to distill from (generations count from 0); "
        "-1 = its latest export at task creation. Resolved to a concrete generation then "
        "and frozen, so every worker and restart reads the same teacher",
    )
    games_per_batch: int = param(200, "self-play games per generation cycle")
    positions_per_game: int = param(0, "eligible turns targeted per game (0 = every eligible turn)")
    # The stratified candidate sample per position (mset_targets.StratifiedQuotas).
    quota_top: int = param(4, "candidates from the head of the equity ranking")
    quota_mid: int = param(4, "candidates sampled from the contention zone")
    quota_tail: int = param(4, "candidates sampled uniformly from the remaining ranks")
    quota_exchange: int = param(2, "exchange candidates")
    mid_rank_limit: int = param(32, "exclusive rank bound of the contention zone")
    # The full-sweep held-out slice. The ranking metrics must see every
    # candidate, which the stratified sample cannot show them.
    sweep_every: int = param(
        20,
        "label every Nth pair with a full sweep of each position's legal candidates instead "
        "of the stratified sample; swept pairs are held out, never trained on. 0 = none, "
        "and holdout_every reserves stratified pairs instead",
    )
    sweep_positions_per_game: int = param(
        2,
        "eligible turns swept per game in a full-sweep pair; a swept position costs "
        "~1000x a stratified one to label, so this is small on purpose",
    )
    sweep_candidate_cap: int = param(
        1500,
        "plays labeled per swept position, by static-equity rank (exchanges and the played "
        "move are kept regardless). It bounds two-blank racks, whose ~20k moves are mostly "
        "redundant blank designations, and leaves normal positions complete",
    )
    # Self-play condition (mirrors position_eval's generation params).
    hasty_temperature: float = param(0.0, "HastyBot softmax temperature (0 = greedy)")
    hasty_top_k: int = param(10, "HastyBot candidate count when the temperature is > 0")
    random_opening_mean: float = param(
        2.0,
        "open each game with K uniformly random plies (K ~ round(Exp(mean))); positions "
        "before the last random ply are ineligible, so targets stay agent-play only",
    )
    face_up_leaves: bool = param(
        True,
        "play the face-up-leaves variant (docs/roadmap.md) in self-play generation; the "
        "teacher must then be an open-leaves model (the generator refuses the mismatch), "
        "and each .mset records the condition so the student trains under it too",
    )
    target_pairs: int = param(
        600,
        "stop generating once the store holds this many pairs (0 = generate until paused). "
        "Reaching it also tells the trainer its corpus is final, so a tag with both workers "
        "started runs to completion unattended",
    )
    # Student training (the train role; scribblez/move_set_eval/trainer.py).
    train_epochs: int = param(
        20,
        "epochs over the finished corpus before the trainer stops (0 = run until paused). "
        "Passes taken while the store is still growing do not spend this budget, so it "
        "always buys passes over the whole corpus. With target_pairs = 0 the corpus counts "
        "as finished once no pair has arrived for 15 minutes",
    )
    warmup_pairs: int = param(
        100,
        "pairs the store must hold before training starts; below this the corpus is too "
        "small to learn from and the held-out slice too thin to read. The trainer also waits "
        "for the first held-out pair, so the gate metrics come from the full-sweep slice "
        "from the first pass. Reaching target_pairs ends both waits, so a smaller run "
        "still trains",
    )
    holdout_every: int = param(
        20,
        "fallback holdout for a corpus with no full-sweep pairs: hold out about one pair in N "
        "(whole pairs, chosen by stem hash) for the recall/rank metrics; 0 evaluates on the "
        "training pairs (a smoke check, not a held-out score). Ignored once the corpus has "
        "swept pairs, which are the holdout",
    )
    batch_positions: int = param(64, "positions per training batch")
    optimizer: str = param(
        OPTIMIZER_SCHEDULE_FREE,
        "optimizer arm (scribblez/generational/optim.py): 'wsd' is AdamW on a "
        "warmup-stable-decay schedule over rows trained; 'schedule_free' is AdamWScheduleFree, "
        "which needs no schedule or horizon and makes every pass's export deployable",
        choices=OPTIMIZERS,
    )
    lr: float = param(
        0.0,
        "learning rate: the schedule's peak under wsd, the constant rate under schedule_free; "
        "0 = the arm's own default",
    )
    lr_warmup_rows: int = param(
        15_000_000,
        "linear LR warmup length, in candidate moves trained (this trainer's rows-clock; "
        "~half a pass over the reference corpus of docs/move_set_eval_results.md)",
    )
    lr_cycle_rows: int = param(
        300_000_000,
        "period of the stable->decay->restart LR cycle, in candidate moves trained "
        "(~10 reference passes; the last fifth of each cycle decays); "
        "unused by schedule_free",
    )
    weight_decay: float = param(
        1e-4,
        "AdamW weight decay, applied to weight matrices and conv kernels only (not to norm "
        "gains, biases or the transformer's positional parameters)",
    )
    adam_beta2: float = param(
        0.999,
        "Adam's second-moment decay rate (beta2); 0.95 adapts faster to shifts in gradient "
        "scale and is the usual choice for transformers",
    )
    grad_clip: float = param(
        0.0, "clip each step's gradient to this global norm (clip_grad_norm_); 0 = no clipping"
    )
    num_blocks: int = param(10, "board-trunk residual blocks")
    trunk_channels: int = param(192, "board-trunk width")
    trunk: str = param(
        TRUNK_CONV,
        "board-trunk tower (scribblez/spatial_trunk.py): 'conv' is a residual conv tower; "
        "'transformer' is a KataGo-style nested-bottleneck transformer over the board cells "
        "plus 27 tile-supply register tokens (one per tile type, carrying its rack and "
        "unseen-pool counts, and the opponent's leave under face-up leaves), so the "
        "placement-plane readout can gate a square's cross-checks on whether the tiles that "
        "fit it are available",
        choices=TRUNKS,
    )
    transformer_mid_channels: int = param(
        192, "transformer trunk: width inside each nested-bottleneck block"
    )
    transformer_heads: int = param(
        6, "transformer trunk: attention heads per layer (head dim = mid channels / heads)"
    )
    transformer_ffn_channels: int = param(512, "transformer trunk: SwiGLU FFN hidden width")
    num_heads: int = param(
        4, "attention heads of the move-to-board cross-attention and the evidence fusion stage"
    )
    lambda_sd: float = param(0.004, "score-diff loss weight")
    lambda_planes: float = param(
        1.0, "placement-plane softmax-CE loss weight (the per-move readouts of roadmap item 1)"
    )
    huber_delta_mean: float = param(10.0, "Huber delta, score-diff mean head")
    huber_delta_std: float = param(10.0, "Huber delta, score-diff std head")


@dataclass(frozen=True)
class CycleResult:
    returncode: int
    gen_seconds: float  # self-play batch wall time
    mset_seconds: float  # target-generator wall time


def _teacher_paths(params: MoveSetEvalParams, mount_root=None) -> TagPaths:
    """The position_eval tag `params.teacher_tag` lives in."""
    return TagPaths(params.teacher_tag, POSITION_EVAL, *([mount_root] if mount_root else []))


def resolved_teacher_generation(params: MoveSetEvalParams, mount_root=None) -> int:
    """The concrete teacher generation for `params`: `teacher_generation`
    itself when it is >= 0, else (-1) the teacher tag's latest export. Raises
    ParamsError if teacher_tag is unset or the tag has no export."""
    if not params.teacher_tag:
        raise params_mod.ParamsError("teacher_tag is required")
    if params.teacher_generation < -1:
        raise params_mod.ParamsError(
            f"teacher_generation must be a generation index (>= 0) or -1 for the latest, "
            f"got {params.teacher_generation}"
        )
    if params.teacher_generation >= 0:
        return params.teacher_generation
    gens = _teacher_paths(params, mount_root).exported_generations()
    if not gens:
        raise params_mod.ParamsError(
            f"position_eval tag '{params.teacher_tag}' has no exported model to distill from"
        )
    return max(gens)


def teacher_onnx(params: MoveSetEvalParams, mount_root=None) -> Path:
    """The teacher ONNX in the position_eval tag's models/ dir."""
    return _teacher_paths(params, mount_root).onnx_path(
        resolved_teacher_generation(params, mount_root)
    )


def finalize(spec: WorkloadSpec, tag: str, params: MoveSetEvalParams) -> MoveSetEvalParams:
    """Pin the teacher to a concrete exported generation at task creation (see
    the module docstring). Fails here, where the operator sees it, if the
    named tag has no such export."""
    generation = resolved_teacher_generation(params)
    if not _teacher_paths(params).onnx_path(generation).is_file():
        raise params_mod.ParamsError(
            f"position_eval tag '{params.teacher_tag}' has no generation {generation} exported"
        )
    return dataclasses.replace(params, teacher_generation=generation)


def sweep_pair(stem: str, sweep_every: int) -> bool:
    """Whether the pair with this .slog stem is labeled as a full sweep.

    A hash of the stem rather than a counter: a resumed cycle relabels every
    .slog still missing its .mset, so the decision must be recoverable from the
    file name alone.
    """
    if sweep_every <= 0:
        return False
    return zlib.crc32(stem.encode()) % sweep_every == 0


def label_pending(pending: list[Path], params: MoveSetEvalParams, threads: int, model: str) -> int:
    """Label `pending` .slog files against the teacher ONNX at `model`, one
    generator run per selection mode."""
    stratified = [s for s in pending if not sweep_pair(s.stem, params.sweep_every)]
    swept = [s for s in pending if sweep_pair(s.stem, params.sweep_every)]
    rc = 0
    if stratified:
        rc = mset_targets.label_stratified(
            stratified,
            model,
            mset_targets.StratifiedQuotas.from_params(params),
            params.positions_per_game,
            threads,
        )
    if rc == 0 and swept:
        rc = mset_targets.label_full_sweep(
            swept,
            model,
            params.sweep_candidate_cap,
            params.sweep_positions_per_game,
            threads,
        )
    return rc


def run_one_cycle(
    out_dir: Path, params: MoveSetEvalParams, threads: int, model: str
) -> CycleResult:
    """One generation cycle into `out_dir`, labeling against the teacher ONNX at
    `model`, with per-phase wall times."""
    t0 = time.monotonic()
    rc = run_games(
        out_dir,
        num_games=params.games_per_batch,
        threads=threads,
        player_spec=hasty_player_spec(params.hasty_temperature, params.hasty_top_k, endgame=True),
        random_opening_mean=params.random_opening_mean,
        face_up_leaves=params.face_up_leaves,
    )
    gen_seconds = time.monotonic() - t0
    if rc != 0:
        print(f"play_game exited with code {rc}", file=sys.stderr)
        return CycleResult(rc, gen_seconds, 0.0)

    pending = sorted(s for s in out_dir.glob("*.slog") if not s.with_suffix(".mset").exists())
    if not pending:
        return CycleResult(0, gen_seconds, 0.0)
    t1 = time.monotonic()
    rc = label_pending(pending, params, threads, model)
    return CycleResult(rc, gen_seconds, time.monotonic() - t1)


def _cycle(model: str, work_dir: Path, params: MoveSetEvalParams, threads: int) -> tuple[int, dict]:
    """One cycle in the shared generate loop's (returncode, phases) shape;
    run_generate binds `model`."""
    r = run_one_cycle(work_dir, params, threads, model)
    return r.returncode, {"gen_s": r.gen_seconds, "mset_s": r.mset_seconds}


# The tag-relative name a remote slot finds its teacher under (RoleSpec.inputs).
TEACHER_INPUT = "inputs/teacher.onnx"


def inputs(params: MoveSetEvalParams) -> dict[str, Path]:
    """The generate role's one out-of-tag input: the pinned teacher export."""
    return {TEACHER_INPUT: teacher_onnx(params)}


def run_generate(ctx: WorkerContext) -> int:
    """The generate-role runner: the shared pair-store loop over run_one_cycle.
    The teacher path is resolved once, before the first cycle."""
    try:
        model = str(resolve_input(ctx, TEACHER_INPUT, teacher_onnx(ctx.params, ctx.mount_root)))
    except FileNotFoundError as e:
        print(f"error: teacher model: {e}", file=sys.stderr)
        return 1
    return pair_store.run_pair_generate(
        ctx,
        functools.partial(_cycle, model),
        ".mset",
        SLOGS_DIR,
        target_pairs=ctx.params.target_pairs,
    )


def progress(spec: WorkloadSpec, tag: str) -> list[tuple[str, object]]:
    return [("pairs", pair_store.count_pairs(spec.paths(tag).data_dir / SLOGS_DIR, ".mset"))]


def slog_dir(tag: str) -> Path:
    """The tag's pair store of .slog/.mset pairs: what MsetDataset takes as a
    data dir."""
    return SPEC.paths(tag).data_dir / SLOGS_DIR


def split_pairs(store: Path, holdout_every: int) -> tuple[list[Path], list[Path]]:
    """(train, holdout) .mset paths of a tag's complete pairs.

    When the corpus has full-sweep pairs, they are the holdout: they are the
    only pairs the ranking metrics mean anything on, and they already reserve
    whole games from training. `holdout_every` then goes unused, so the two
    reservations do not stack; it applies only to a corpus without swept
    pairs.
    """
    stratified, swept = partition_full_sweep(complete_pairs(store))
    if swept:
        return sorted(stratified), sorted(swept)
    train, holdout = pair_store.split_pair_stems([f.stem for f in stratified], holdout_every)
    return [store / f"{s}.mset" for s in train], [store / f"{s}.mset" for s in holdout]


SPEC = WorkloadSpec(
    name="move_set_eval",
    title="Generate move-set-eval targets",
    params_cls=MoveSetEvalParams,
    roles=(
        RoleSpec(
            name="generate",
            title="Generator (GPU)",
            runner="scribblez.workloads.move_set_eval:run_generate",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            inputs="scribblez.workloads.move_set_eval:inputs",
            gpu=True,
            stats=StatsSpec(
                unit="pairs",
                phases={
                    "gen_s": "self-play",
                    "mset_s": "targets",
                    "upload_s": "deliver",
                },
            ),
        ),
        RoleSpec(
            name="train",
            title="Student trainer (GPU)",
            runner="scribblez.move_set_eval.trainer:run",
            runtime=RUNTIME_TORCH,
            deps="scribblez.move_set_eval.trainer:fetch_train_deps",
            ingest="scribblez.generational.train_ingest:tick",
            singleton=True,
            gpu=True,
            stats=StatsSpec(unit="rows", phases={"train_s": "train", "eval_s": "eval"}),
        ),
    ),
    progress="scribblez.workloads.move_set_eval:progress",
    sync_data_dirs=(SLOGS_DIR,),
    finalize="scribblez.workloads.move_set_eval:finalize",
    profiles=PROFILES,
    default_profile=TRUNK_TRANSFORMER,
    # The required teacher first, then the run's shape: information condition,
    # epoch budget (a fixed horizon, unlike position_eval's open-ended run),
    # corpus size, and optimizer arm.
    primary_params=(
        "teacher_tag",
        "face_up_leaves",
        "train_epochs",
        "target_pairs",
        "optimizer",
    ),
)
