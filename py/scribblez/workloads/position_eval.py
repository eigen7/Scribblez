"""The position-evaluation training workload.

Three roles on one tag: any number of interchangeable generate workers
(local/cloud) producing self-play chunks into the tag's staging area, a
singleton local train worker consuming complete generations (sliding window,
one epoch per generation, per-checkpoint ONNX + dashboard metrics), and a
singleton match_eval worker turning exported checkpoints into match-play
readouts against a fixed opponent (scribblez/match_eval/runner.py). The
generation scheduler (scribblez/generational/scheduler.py) assigns staged
chunks to generation directories and paces the generator fleet against the
trainer's published cursor.

The match_eval slot may sit on another machine (kind "ssh"), which is how the
eval matches stop competing with training for this host's GPU. Its work is
assigned and its results ingested by the controller
(scribblez/match_eval/dispatch.py), so the slot needs neither the exports nor
the database -- only a GPU and the worker image.

Parameters here are the frozen task params: they define the corpus and the
model, so every worker on a tag must share them. Live operator knobs
(DataLoader workers, torch threads) are dashboard.db controls, and per-worker
resources (threads/vcpus) live on the worker slots.
"""

from dataclasses import dataclass

from scribblez.generational.optimizer_arms import OPTIMIZER_WSD, OPTIMIZERS
from scribblez.params import param
from scribblez.paths import MATCH_RESULTS_DIR
from scribblez.workloads.base import RoleSpec, StatsSpec, WorkloadSpec
from scribblez.workloads.selfplay_gen import GENERATOR_STATS, STAGING_DIR

TRAINER_STATS = StatsSpec(unit="rows", phases={"train_s": "train", "eval_s": "eval"})


@dataclass(frozen=True)
class PositionEvalParams:
    # Generation.
    games_per_generation: int = param(20000, "self-play games per generation")
    open_ahead: int = param(
        4, "generations kept open ahead of the trainer's cursor before generators are parked"
    )
    hasty_temperature: float = param(0.0, "HastyBot softmax temperature (0 = greedy)")
    hasty_top_k: int = param(10, "HastyBot candidate count when the temperature is > 0")
    random_opening_mean: float = param(
        2.0,
        "open each game with K uniformly-random plies (K ~ round(Exp(mean))) before the "
        "HastyBots take over, reaching off-policy states; 0 disables",
    )
    face_up_leaves: bool = param(
        False,
        "play the face-up-leaves variant (docs/roadmap.md) in self-play generation AND "
        "match eval, so the model trains and is measured under one information condition",
    )
    weirdbot_generation: bool = param(
        False,
        "generate the diagnostic WeirdBot corpus: put the leave-forcing WeirdBot on both "
        "self-play seats instead of HastyBot, making the opponent-leave-letter x cross-check "
        "conjunction the dominant training signal; off leaves generation unchanged",
    )
    # Match eval (the match_eval role; docs/roadmap.md A1).
    match_every_generations: int = param(
        5, "match-eval cadence: play a match for every Nth exported generation; 0 disables"
    )
    match_opponent: str = param(
        "--type=hastybot-endgame", "the fixed opponent's --player spec, e.g. --type=sim"
    )
    match_pairs: int = param(
        200, "mirrored game pairs per generation's match; every match plays all of them"
    )
    match_seed: int = param(
        1,
        "base game seed for matches; fixed per tag so every generation faces identical deals "
        "(must be nonzero)",
    )
    # Training window.
    window: int = param(4, "generations trained over (sliding window); <=0 keeps all")
    turns_per_game: int = param(1, "turns sampled per game per generation; 0 = every eligible turn")
    max_rows: int = param(0, "stop the trainer after this many rows (0 = run until paused)")
    # Optimization.
    batch_size: int = param(256, "minibatch size")
    optimizer: str = param(
        OPTIMIZER_WSD,
        "optimizer arm (scribblez/generational/optim.py): 'wsd' is AdamW on the rows-clock "
        "warmup-stable-decay schedule, 'schedule_free' is AdamWScheduleFree -- no schedule, "
        "no horizon to pick, every generation's export deployable",
        choices=OPTIMIZERS,
    )
    lr: float = param(
        0.0,
        "learning rate -- the peak the warmup-stable-decay schedule decays away from under the "
        "wsd arm, the constant the averaged iterate is taken around under schedule_free; "
        "0 = the arm's own default",
    )
    lr_warmup_rows: int = param(
        200_000, "linear LR warmup length, in positions trained (~2.5 default generations)"
    )
    lr_cycle_rows: int = param(
        2_000_000,
        "period of the stable->decay->restart LR cycle, in positions trained "
        "(~25 default generations; the last fifth of each cycle decays); "
        "unused by the schedule_free arm, which has no cycle",
    )
    weight_decay: float = param(1e-4, "AdamW weight decay")
    # Model.
    num_blocks: int = param(10, "residual blocks")
    trunk_channels: int = param(192, "trunk width")
    use_film: bool = param(
        False,
        "FiLM-style multiplicative conditioning at the trunk's scalar/global-context "
        "injection sites (scalars emit a per-channel gain alongside the additive bias); "
        "off is the additive-injection baseline",
    )
    contingent_features: bool = param(
        False,
        "encode the full input layout including the contingent-draw potential features; "
        "off trains the smaller ablation baseline",
    )
    # Loss.
    lambda_wld: float = param(1.0, "WLD (value) loss weight; drop to isolate other heads")
    lambda_sd: float = param(0.0002, "score-diff loss weight")
    lambda_next_placement: float = param(0.5, "marginal placement loss weight (opp and self)")
    lambda_win_placement: float = param(0.5, "win-placement conjunction loss weight (opp and self)")
    placement_pos_weight: float = param(
        1.0,
        "BCE pos_weight for the placement heads; >1 up-weights the sparse occupied "
        "cells (a diagnostic that trades calibration for magnitude)",
    )
    huber_delta_mean: float = param(10.0, "Huber delta, score-diff mean head")
    huber_delta_std: float = param(10.0, "Huber delta, score-diff std head")
    # Activation-magnitude restoring forces (docs/fp16_safe_serving.md).
    lambda_wld_z: float = param(1e-4, "z-loss weight on the WLD logits (squared logsumexp)")
    lambda_pool_act: float = param(
        1e-6, "mean-square penalty weight on the trunk's pooled-FC pre-activations"
    )


SPEC = WorkloadSpec(
    name="position_eval",
    title="Train position evaluation",
    params_cls=PositionEvalParams,
    roles=(
        RoleSpec(
            name="generate",
            title="Generator",
            runner="scribblez.workloads.selfplay_gen:run_generate",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            interruptible=True,
            stats=GENERATOR_STATS,
        ),
        RoleSpec(
            name="train",
            title="Trainer (GPU)",
            runner="scribblez.position_eval.trainer:run",
            singleton=True,
            kinds=("local",),
            gpu=True,
            stats=TRAINER_STATS,
        ),
        RoleSpec(
            name="match_eval",
            title="Match eval (GPU)",
            runner="scribblez.match_eval.runner:run",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            dispatch="scribblez.match_eval.dispatch:tick",
            singleton=True,
            kinds=("local", "ssh"),
            gpu=True,
            stats=StatsSpec(unit="games", phases={"match_s": "match play"}),
        ),
    ),
    scheduler="scribblez.generational.scheduler:tick_for_task",
    progress="scribblez.generational.scheduler:progress",
    sync_data_dirs=(STAGING_DIR,),
    local_data_dirs=(MATCH_RESULTS_DIR,),
)
