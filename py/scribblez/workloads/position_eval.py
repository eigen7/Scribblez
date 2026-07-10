"""The position-evaluation training workload.

Two roles on one tag: any number of interchangeable generate workers
(local/cloud) producing self-play chunks into the tag's staging area, and a
singleton local train worker consuming complete generations (sliding window,
reuse-derived epochs, per-checkpoint ONNX + dashboard metrics). The generation
scheduler (scribblez/generational/scheduler.py) assigns staged chunks to the
frozen test split and to generation directories, and paces the generator fleet
against the trainer's published cursor.

Parameters here are the frozen task params: they define the corpus and the
model, so every worker on a tag must share them. Live operator knobs (base
learning rate, DataLoader workers, torch threads) are dashboard.db controls,
and per-worker resources (threads/vcpus) live on the worker slots.
"""

from dataclasses import dataclass

from scribblez.params import param
from scribblez.workloads.base import RoleSpec, StatsSpec, WorkloadSpec
from scribblez.workloads.selfplay_gen import GENERATOR_STATS, STAGING_DIR

TRAINER_STATS = StatsSpec(unit="rows", phases={"train_s": "train", "eval_s": "eval"})


@dataclass(frozen=True)
class PositionEvalParams:
    # Generation.
    games_per_generation: int = param(20000, "self-play games per generation")
    test_games: int = param(2000, "held-out test games, generated once before gen 0; 0 disables")
    games_per_chunk: int = param(1000, "games per generator cycle (= per .slog chunk)")
    open_ahead: int = param(
        1, "generations kept open ahead of the trainer's cursor before generators are parked"
    )
    hasty_temperature: float = param(0.0, "HastyBot softmax temperature (0 = greedy)")
    hasty_top_k: int = param(10, "HastyBot candidate count when the temperature is > 0")
    hasty_temp_min_bag: int = param(0, "sample only on turns with at least this many bag tiles")
    random_opening_mean: float = param(
        2.0,
        "open each game with K uniformly-random plies (K ~ round(Exp(mean))) before the "
        "HastyBots take over, reaching off-policy states; 0 disables",
    )
    # Training window / reuse.
    window: int = param(4, "generations trained over (sliding window); <=0 keeps all")
    turns_per_game: int = param(1, "turns sampled per game per epoch; 0 = every eligible turn")
    reuse_per_position: float = param(
        2.0, "target gradient passes per unique position; derives epochs-per-generation"
    )
    epochs_per_generation: int = param(
        0, "explicit epochs-per-generation, overriding reuse_per_position; 0 derives it"
    )
    max_rows: int = param(0, "stop the trainer after this many rows (0 = run until paused)")
    # Optimization.
    batch_size: int = param(256, "minibatch size")
    lr: float = param(1e-3, "initial base learning rate (seeds the live base_lr control)")
    warmup_rows: int = param(0, "linear LR warmup over the first this-many rows")
    weight_decay: float = param(1e-4, "AdamW weight decay")
    # Model.
    num_blocks: int = param(10, "residual blocks")
    trunk_channels: int = param(192, "trunk width")
    contingent_features: bool = param(
        False,
        "encode the full input layout including the contingent-draw potential features; "
        "off trains the smaller ablation baseline",
    )
    lexicon_module: str = param(
        "none", "compiled-lexicon tool to attach to the trunk (see lexical_tool/modules.py)"
    )
    # Loss.
    lambda_sd: float = param(0.004, "score-diff loss weight")
    lambda_next_placement: float = param(0.5, "marginal placement loss weight (opp and self)")
    lambda_win_placement: float = param(0.5, "win-placement conjunction loss weight (opp and self)")
    huber_delta_mean: float = param(10.0, "Huber delta, score-diff mean head")
    huber_delta_std: float = param(10.0, "Huber delta, score-diff std head")
    # Per-checkpoint evaluation.
    eval_dataset: str = param(
        "", "GCG dataset for the Positions tab's per-checkpoint eval; empty = the committed default"
    )
    quality_dataset: str = param(
        "", "large GCG dataset for the aggregate quality curves; empty = the committed default"
    )
    no_eval: bool = param(False, "disable the per-checkpoint Positions-tab evaluation")
    no_quality: bool = param(False, "disable the per-checkpoint aggregate quality evaluation")


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
    ),
    scheduler="scribblez.generational.scheduler:tick_for_task",
    progress="scribblez.generational.scheduler:progress",
    sync_data_dirs=(STAGING_DIR,),
)
