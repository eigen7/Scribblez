"""The deep_bot_leaf_eval workload: DeepBot's rollout-leaf value model
(deep_bot_leaf_eval/model.py), trained on HastyBot self-play, HastyBot being
the policy DeepBot's rollouts continue with.

It is generational, like position_eval: any number of generators deliver
hidden-leaves HastyBot self-play into staging, and the singleton trainer
trains one epoch per generation over a sliding window, hosting the tag's data
home. Self-play has no random openings, so every eligible turn's outcome
follows the policy alone.

The match_eval role (scribblez/match_eval) plays every
`match_every_generations`th export as LeafBot (engine agent/leaf_bot.h, the
leaf network valuing HastyBot's best moves with no search) against a fixed
opponent, its opponent leaves drawn from a leave inference model pinned at
task creation (`match_leave_inference_tag`, `match_leave_inference_generation`).
After each match the slot analyzes the known positions under the same export
(deep_bot_leaf_eval/positions.py), which the Positions tab shows.
"""

from dataclasses import dataclass, replace
from pathlib import Path

from cloud.runtime_abi import RUNTIME_TORCH

from scribblez import params as params_mod
from scribblez.deep_bot import known_positions
from scribblez.deep_bot_leaf_eval import positions as leaf_positions
from scribblez.generational.optimizer_arms import OPTIMIZER_SCHEDULE_FREE, OPTIMIZERS
from scribblez.generational.scheduler import TICK_FOR_TASK
from scribblez.match_eval import runner as match_runner
from scribblez.params import param
from scribblez.paths import MATCH_RESULTS_DIR, TagPaths
from scribblez.trunk_arms import NORMALIZE_SCALAR_HELP, TRUNK_CONV, TRUNK_TRANSFORMER, TRUNKS
from scribblez.workloads.base import (
    RoleSpec,
    SlotPlan,
    StatsSpec,
    WorkerContext,
    WorkloadSpec,
    resolve_input,
)
from scribblez.workloads.pinned_exports import export_generation, pinned_generation
from scribblez.workloads.position_eval import TRAINER_STATS
from scribblez.workloads.selfplay_gen import GENERATOR_STATS, STAGING_DIR, generate, hasty_spec

LEAVE_INFERENCE = "leave_inference"
# Where a remote match slot finds the pinned leave inference model, and the
# known positions it analyzes.
LEAVE_INPUT = "inputs/leave_inference.onnx"
POSITIONS_INPUT_DIR = "inputs/known-positions"


@dataclass(frozen=True)
class DeepBotLeafEvalParams:
    # Generation.
    games_per_generation: int = param(20000, "self-play games per generation")
    open_ahead: int = param(
        4, "generations kept open ahead of the trainer's cursor before generators are parked"
    )
    hasty_temperature: float = param(0.0, "HastyBot softmax temperature (0 = greedy)")
    hasty_top_k: int = param(10, "HastyBot candidate count when the temperature is > 0")
    # Held-out games, played by the trainer itself (leave_inference/holdout.py).
    holdout_games: int = param(2000, "held-out self-play games the trainer evaluates on")
    # Training window.
    window: int = param(4, "generations trained over (sliding window); <=0 keeps all")
    turns_per_game: int = param(
        5,
        "turns trained per game per generation; 0 = every eligible turn. A game has about 20 "
        "eligible turns and stays in the window for `window` generations, each taking the next "
        "turns of a fixed per-game order, so 5 x 4 trains each turn once: all the data, no reuse",
    )
    max_rows: int = param(
        -1, "stop the trainer after this many rows (-1 = run until paused)", end=True
    )
    # Optimization.
    batch_size: int = param(256, "minibatch size")
    optimizer: str = param(
        OPTIMIZER_SCHEDULE_FREE,
        "optimizer arm (scribblez/generational/optim.py): 'wsd', 'muon' or 'schedule_free'",
        choices=OPTIMIZERS,
    )
    lr: float = param(0.0, "learning rate; 0 = the arm's own default")
    lr_warmup_rows: int = param(200_000, "linear LR warmup length, in rows trained")
    lr_cycle_rows: int = param(
        2_000_000, "period of the WSD schedule's cycle, in rows trained; unused by schedule_free"
    )
    averaging_window_rows: int = param(
        20_000_000,
        "schedule_free: the deployed average covers at most the last this many rows trained "
        "(uniform before then); unused by wsd and muon",
    )
    weight_decay: float = param(1e-4, "AdamW weight decay on weight matrices and conv kernels")
    adam_beta2: float = param(0.999, "Adam's second-moment decay rate (beta2)")
    grad_clip: float = param(0.0, "clip each step's gradient to this global norm; 0 = no clipping")
    # Loss.
    lambda_wld: float = param(1.0, "WLD loss weight")
    lambda_sd: float = param(0.0002, "score-diff loss weight")
    huber_delta_mean: float = param(10.0, "Huber delta, score-diff mean head")
    huber_delta_std: float = param(10.0, "Huber delta, score-diff std head")
    lambda_ownership: float = param(
        0.1, "ownership loss weight (per-square cross-entropy over the empty squares)"
    )
    lambda_points: float = param(
        1.0, "square-points loss weight (per-square mean and variance, in units of 100 points)"
    )
    lambda_going_out: float = param(0.1, "going-out loss weight (three-way cross-entropy)")
    lambda_end_points: float = param(
        1.0, "end-of-game points loss weight (mean and variance, in units of 100 points)"
    )
    # Model (deep_bot_leaf_eval/model.py).
    trunk: str = param(
        TRUNK_TRANSFORMER,
        "'transformer': cell, tile-register and global tokens through plain attention layers; "
        "'conv': the residual conv tower, ~3x the transformer's TensorRT throughput at equal FLOPs",
        choices=TRUNKS,
    )
    join_planes: bool = param(
        True,
        "read the row's join planes: per seat (my leave, the opponent's, the pool) and play "
        "axis, how well its tiles fit each hook square's cross-check. Off measures what they give",
    )
    width: int = param(64, "trunk width (channels)")
    depth: int = param(4, "transformer layers, or conv residual blocks")
    num_heads: int = param(2, "transformer attention heads (head dim = width / heads)")
    normalize_scalar: bool = param(True, f"conv trunk: {NORMALIZE_SCALAR_HELP}")
    # Match eval (the match_eval role): LeafBot against a fixed opponent.
    match_every_generations: int = param(
        10, "match-eval cadence: play a match for every Nth exported generation; 0 disables"
    )
    match_opponent: str = param(
        "--type=hastybot-endgame", "the fixed opponent's --player spec, e.g. --type=sim"
    )
    match_pairs: int = param(
        5000, "mirrored game pairs per generation's match; every match plays all of them"
    )
    match_seed: int = param(
        1,
        "base game seed for matches, fixed per tag so every generation faces identical deals; "
        "must be nonzero",
    )
    match_leave_inference_tag: str = param(
        "",
        "the leave_inference tag whose model draws LeafBot's opponent leaves; required while "
        "matches are on",
    )
    match_leave_inference_generation: int = param(
        -1,
        "which generation of match_leave_inference_tag (-1: its latest export, pinned at task "
        "creation), the same for every match of the tag",
    )
    match_candidates: int = param(10, "LeafBot's candidates: HastyBot's best by static equity")
    match_opp_leaves: int = param(
        16, "the opponent's leaves LeafBot values each candidate over, drawn from the belief"
    )


# The trainer's peak GPU memory in GiB, measured 2026-10-23 on the RTX 5000 Ada
# as position_eval's figures are (workloads/position_eval.py): the process as
# nvidia-smi sees it, CUDA context included, over generations of training
# (compiled, bf16) and the held-out eval that follows each, join planes on.
#
#   transformer  1.76 GiB  (torch peak reserved 1.50)
#   conv         1.01 GiB  (torch peak reserved 0.71)
#
# Measured at the default architecture and batch size only (TRAINER_GPU_KEY);
# any other configuration has no figure, and the queue refuses to place it.
# The table holds each peak plus TRAINER_GPU_HEADROOM_GB.
TRAINER_GPU_HEADROOM_GB = 0.5
_MEASURED_GB = {TRUNK_TRANSFORMER: 1.76, TRUNK_CONV: 1.01}
# The params the measurements held fixed, at their defaults.
TRAINER_GPU_KEY = ("batch_size", "width", "depth", "num_heads")
# Match eval: a 1000-pair match of LeafBot (a conv 64x4 leaf, a leave_inference
# export) against HastyBot-endgame at 28 threads peaked at 1.77 GiB
# (2026-11-17), its two TensorRT plans already cached. On a machine without
# them each engine first builds its plan, whose scratch is capped at 1 GiB
# (NeuralNet's workspace_bytes); that part is a bound from the cap, not a
# measurement.
MATCH_EVAL_GPU_GB = 1.77 + 1.0 + TRAINER_GPU_HEADROOM_GB


def gpu_need(params, role: str) -> float | None:
    """The WorkloadSpec.gpu_need hook: GiB one slot of `role` needs."""
    if role == "generate":
        return 0.0
    if role == "match_eval":
        return MATCH_EVAL_GPU_GB
    defaults = DeepBotLeafEvalParams()
    if any(getattr(params, k) != getattr(defaults, k) for k in TRAINER_GPU_KEY):
        return None
    return _MEASURED_GB[params.trunk] + TRAINER_GPU_HEADROOM_GB


def layout(params, vcpus: int, generator_threads: int | None) -> list[SlotPlan]:
    """The WorkloadSpec.layout hook: a trainer and a generator per machine, the
    generator on every vCPU (see position_eval's layout), plus match eval when
    its cadence is on."""
    roles = ["train", "generate"] + (["match_eval"] if params.match_every_generations > 0 else [])
    threads = {"generate": generator_threads or vcpus}
    return [SlotPlan(r, threads.get(r), gpu_need(params, r)) for r in roles]


def _leave_paths(params, mount_root: Path) -> TagPaths:
    return TagPaths(params.match_leave_inference_tag, LEAVE_INFERENCE, mount_root)


def finalize(spec, paths: TagPaths, params):
    """Pin the matches' leave inference model to a concrete generation at task
    creation, failing here, where the operator sees it, if it has no export."""
    if params.match_every_generations <= 0:
        return params
    if not params.match_leave_inference_tag:
        raise params_mod.ParamsError(
            "matches need match_leave_inference_tag (or match_every_generations=0)"
        )
    leave = pinned_generation(
        _leave_paths(params, paths.mount_root),
        params.match_leave_inference_generation,
        "match_leave_inference",
    )
    return replace(params, match_leave_inference_generation=leave)


def leave_model_path(params, mount_root: Path) -> Path:
    """The matches' leave inference model on the controller's mount."""
    paths = _leave_paths(params, mount_root)
    return paths.onnx_path(
        export_generation(paths, params.match_leave_inference_generation, "match_leave_inference")
    )


def inputs(params, mount_root: Path) -> dict[str, Path]:
    """The match_eval role's out-of-tag inputs: the pinned leave inference
    model and the known positions."""
    out = {LEAVE_INPUT: leave_model_path(params, mount_root)}
    for f in known_positions.files():
        out[f"{POSITIONS_INPUT_DIR}/{f.name}"] = f
    return out


def _known_positions(ctx: WorkerContext) -> list[known_positions.Position]:
    """The known positions where the slot reads them: in place beside the
    controller, else from the copies staged under the tag."""
    staged = [
        resolve_input(ctx, f"{POSITIONS_INPUT_DIR}/{f.name}", f) for f in known_positions.files()
    ]
    return known_positions.load(staged[-1].parent)  # the manifest, last


def leafbot_player(ctx: WorkerContext, model: Path) -> tuple[str, bool]:
    """A leaf export as the match plays it (match_eval.runner.Player): LeafBot
    with the endgame solved, under hidden leaves, as the leaf trained."""
    p = ctx.params
    leave = resolve_input(ctx, LEAVE_INPUT, leave_model_path(p, ctx.mount_root))
    spec = (
        f"--type=leafbot-endgame --leaf-model={model} --leave-inference={leave} "
        f"--candidates={p.match_candidates} --opp-leaves={p.match_opp_leaves} --name=model"
    )
    return spec, False


def analyze_positions(ctx: WorkerContext, model: Path, gen: int) -> list[tuple[Path, str]]:
    """The match_eval.runner.AfterMatch of the slot: the export's analysis of
    the known positions, as the matches' LeafBot sees them."""
    p = ctx.params
    leave = resolve_input(ctx, LEAVE_INPUT, leave_model_path(p, ctx.mount_root))
    work = ctx.tag_paths().work_dir(ctx.worker_id) / "positions"
    doc = leaf_positions.analyze(
        _known_positions(ctx),
        model,
        leave,
        p.match_opp_leaves,
        p.match_candidates,
        work,
        ctx.threads,
    )
    path = work.parent / f"positions_gen_{gen:06d}.json"
    leaf_positions.write({"generation": gen, **doc}, path)
    return [(path, leaf_positions.rel(gen))]


def run_match(ctx: WorkerContext) -> int:
    """The match_eval role's runner: match_eval's, the exports playing as
    LeafBot, each one's analysis of the known positions delivered with its
    result."""
    return match_runner.run(ctx, leafbot_player, analyze_positions)


def run_generate(ctx: WorkerContext) -> int:
    """Hidden-leaves HastyBot self-play without random openings."""
    return generate(ctx, hasty_spec(ctx.params), face_up_leaves=False, random_opening_mean=0.0)


SPEC = WorkloadSpec(
    name="deep_bot_leaf_eval",
    title="Train DeepBot leaf eval",
    params_cls=DeepBotLeafEvalParams,
    roles=(
        RoleSpec(
            name="generate",
            title="Generator",
            runner="scribblez.workloads.deep_bot_leaf_eval:run_generate",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            stats=GENERATOR_STATS,
        ),
        RoleSpec(
            name="train",
            title="Trainer (GPU)",
            runner="scribblez.deep_bot_leaf_eval.trainer:run",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            runtime=RUNTIME_TORCH,
            ingest="scribblez.generational.train_ingest:tick",
            singleton=True,
            kinds=("local", "ssh"),
            gpu=True,
            stats=TRAINER_STATS,
        ),
        RoleSpec(
            name="match_eval",
            title="Match eval (GPU)",
            runner="scribblez.workloads.deep_bot_leaf_eval:run_match",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            dispatch="scribblez.match_eval.dispatch:tick",
            inputs="scribblez.workloads.deep_bot_leaf_eval:inputs",
            singleton=True,
            kinds=("local", "ssh"),
            gpu=True,
            stats=StatsSpec(unit="games", phases={"match_s": "match play"}),
        ),
    ),
    scheduler=TICK_FOR_TASK,
    progress="scribblez.generational.scheduler:progress",
    complete="scribblez.generational.scheduler:complete",
    pace_role="train",
    collected_dirs=(STAGING_DIR, MATCH_RESULTS_DIR, leaf_positions.POSITIONS_DIR),
    layout="scribblez.workloads.deep_bot_leaf_eval:layout",
    gpu_need="scribblez.workloads.deep_bot_leaf_eval:gpu_need",
    finalize="scribblez.workloads.deep_bot_leaf_eval:finalize",
)
