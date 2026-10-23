"""The deep_bot_leaf_eval workload: DeepBot's rollout-leaf value model
(deep_bot_leaf_eval/model.py), trained on HastyBot self-play, HastyBot being
the policy DeepBot's rollouts continue with.

It is generational, like position_eval: any number of generators deliver
hidden-leaves HastyBot self-play into staging, and the singleton trainer
trains one epoch per generation over a sliding window, hosting the tag's data
home. Self-play has no random openings, so every eligible turn's outcome
follows the policy alone.
"""

from dataclasses import dataclass

from cloud.runtime_abi import RUNTIME_TORCH

from scribblez.generational.optimizer_arms import OPTIMIZER_SCHEDULE_FREE, OPTIMIZERS
from scribblez.generational.scheduler import TICK_FOR_TASK
from scribblez.params import param
from scribblez.trunk_arms import TRUNK_TRANSFORMER, TRUNKS
from scribblez.workloads.base import RoleSpec, WorkerContext, WorkloadSpec
from scribblez.workloads.position_eval import TRAINER_STATS
from scribblez.workloads.selfplay_gen import GENERATOR_STATS, STAGING_DIR, generate, hasty_spec


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
    weight_decay: float = param(1e-4, "AdamW weight decay on weight matrices and conv kernels")
    adam_beta2: float = param(0.999, "Adam's second-moment decay rate (beta2)")
    grad_clip: float = param(0.0, "clip each step's gradient to this global norm; 0 = no clipping")
    # Loss.
    lambda_wld: float = param(1.0, "WLD loss weight")
    lambda_sd: float = param(0.0002, "score-diff loss weight")
    huber_delta_mean: float = param(10.0, "Huber delta, score-diff mean head")
    huber_delta_std: float = param(10.0, "Huber delta, score-diff std head")
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
    ),
    scheduler=TICK_FOR_TASK,
    progress="scribblez.generational.scheduler:progress",
    complete="scribblez.generational.scheduler:complete",
    pace_role="train",
    collected_dirs=(STAGING_DIR,),
)
