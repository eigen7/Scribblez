"""The leave inference workload (docs/plans/deep_bot_v0.md, "Workload 1:
leave_inference"): DeepBot's model of the opponent's leave, trained on the
true leaves replay recovers from HastyBot self-play, with the win table
refitted from the same games as a side product.

It is generational, like position_eval: any number of generators deliver
hidden-leaves HastyBot self-play into staging, and the singleton trainer
trains one epoch per generation over a sliding window, hosting the tag's data
home. Self-play has no random openings: a random ply in a row's history, or
as the move that set its target leave, is not the policy the model learns.
"""

from dataclasses import dataclass

from cloud.runtime_abi import RUNTIME_TORCH

from scribblez.generational.optimizer_arms import OPTIMIZER_SCHEDULE_FREE, OPTIMIZERS
from scribblez.generational.scheduler import TICK_FOR_TASK
from scribblez.leave_inference import forgone
from scribblez.leave_inference.heads import HEAD_TILTED, HEADS
from scribblez.params import param
from scribblez.workloads.base import RoleSpec, WorkerContext, WorkloadSpec
from scribblez.workloads.position_eval import TRAINER_STATS
from scribblez.workloads.selfplay_gen import (
    GENERATOR_STATS,
    STAGING_DIR,
    ChunkSidecar,
    generate,
    hasty_spec,
)


@dataclass(frozen=True)
class LeaveInferenceParams:
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
    # Model.
    public: bool = param(
        False,
        "infer the leave from public information alone: the input withholds the observer's "
        "rack and exchanges, while the loss still conditions on the observer's pool, as "
        "serving the model to any observer would. Not yet servable",
    )
    num_blocks: int = param(8, "conv tower residual blocks")
    trunk_channels: int = param(128, "conv tower width")
    head: str = param(
        HEAD_TILTED,
        "output head (leave_inference/model.py): 'tilted' is the design's 29 floats over the "
        "leave table; 'autoregressive' decodes the leave tile by tile, a Python-only comparison "
        "that measures what the compact head gives up",
        choices=HEADS,
    )


# The sidecar every chunk carries: the forgone plays its rows read.
FORGONE_SIDECAR = ChunkSidecar(forgone.SUFFIX, forgone.write_sidecars)


def run_generate(ctx: WorkerContext) -> int:
    """Hidden-leaves HastyBot self-play without random openings: the leave is
    what the model infers, from the policy's own moves. Each chunk carries its
    .forgone sidecars."""
    return generate(
        ctx,
        hasty_spec(ctx.params),
        face_up_leaves=False,
        random_opening_mean=0.0,
        sidecar=FORGONE_SIDECAR,
    )


SPEC = WorkloadSpec(
    name="leave_inference",
    title="Train leave inference",
    params_cls=LeaveInferenceParams,
    roles=(
        RoleSpec(
            name="generate",
            title="Generator",
            runner="scribblez.workloads.leave_inference:run_generate",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            stats=GENERATOR_STATS,
        ),
        RoleSpec(
            name="train",
            title="Trainer (GPU)",
            runner="scribblez.leave_inference.trainer:run",
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
