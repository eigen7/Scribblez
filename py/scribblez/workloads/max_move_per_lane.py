"""The max-move-per-lane training workload (the representation probe).

Same two-role shape as position_eval -- interchangeable generate workers plus
a singleton local trainer -- with the probe's own model/loss parameters (its
eval is train accuracy plus the frozen lane-analysis GCG dataset).
"""

from dataclasses import dataclass

from scribblez.params import param
from scribblez.workloads.base import RoleSpec, WorkloadSpec
from scribblez.workloads.position_eval import TRAINER_STATS
from scribblez.workloads.selfplay_gen import GENERATOR_STATS, STAGING_DIR


@dataclass(frozen=True)
class MaxMovePerLaneParams:
    # Generation.
    games_per_generation: int = param(20000, "self-play games per generation")
    open_ahead: int = param(
        4, "generations kept open ahead of the trainer's cursor before generators are parked"
    )
    hasty_temperature: float = param(0.0, "HastyBot softmax temperature (0 = greedy)")
    hasty_top_k: int = param(10, "HastyBot candidate count when the temperature is > 0")
    random_opening_mean: float = param(0.0, "random-opening plies per game (0 disables)")
    face_up_leaves: bool = param(
        True, "play the face-up-leaves variant (docs/roadmap.md) in self-play generation"
    )
    # Training window.
    window: int = param(4, "generations trained over (sliding window); <=0 keeps all")
    turns_per_game: int = param(1, "turns sampled per game per generation; 0 = every turn")
    max_rows: int = param(0, "stop the trainer after this many rows (0 = run until paused)")
    # Optimization.
    batch_size: int = param(256, "minibatch size")
    lr: float = param(1e-3, "peak learning rate of the warmup-stable-decay schedule")
    lr_warmup_rows: int = param(
        200_000, "linear LR warmup length, in positions trained (~2.5 default generations)"
    )
    lr_cycle_rows: int = param(
        2_000_000,
        "period of the stable->decay->restart LR cycle, in positions trained "
        "(~25 default generations; the last fifth of each cycle decays)",
    )
    weight_decay: float = param(1e-4, "AdamW weight decay")
    # Model.
    trunk_channels: int = param(128, "CNN trunk width")
    num_blocks: int = param(8, "trunk residual blocks")
    lane_layers: int = param(4, "lane transformer layers")
    lane_heads: int = param(4, "lane transformer attention heads")
    ffn_mult: int = param(4, "lane transformer FFN width multiple")
    rack_tokens: int = param(4, "rack tokens prepended per lane")
    lexicon_module: str = param(
        "none", "compiled-lexicon tool to plug in (see lexical_tool/modules.py)"
    )
    # Loss.
    lambda_cdf: float = param(1.0, "score-CDF (CRPS) loss weight")
    lambda_occ: float = param(100.0, "occupancy (move) loss weight")
    lambda_has_move: float = param(1.0, "has-move loss weight")
    # Per-checkpoint evaluation.
    lane_eval_dataset: str = param(
        "", "GCG dataset for the Lane-analysis tab's eval; empty = the committed default"
    )
    no_lane_eval: bool = param(False, "disable the per-checkpoint lane-analysis evaluation")


SPEC = WorkloadSpec(
    name="max_move_per_lane",
    title="Train max-move-per-lane probe",
    params_cls=MaxMovePerLaneParams,
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
            runner="scribblez.max_move_per_lane.trainer:run",
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
