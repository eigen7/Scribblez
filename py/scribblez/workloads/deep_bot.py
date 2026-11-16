"""The deep_bot workload's params (docs/plans/deep_bot_generations.md): one
tag that advances DeepBot's generations. The trainer is
scribblez/deep_bot/trainer.py. The generator, the measure role and the
workload's registration are not built yet, so no tag of it can be created.
"""

from dataclasses import dataclass

from scribblez.params import ParamsError, param


@dataclass(frozen=True)
class DeepBotParams:
    # Generation.
    games_per_generation: int = param(2000, "self-play games per generation")
    open_ahead: int = param(
        1, "generations kept open ahead of the trainer's cursor before generators are parked"
    )
    pool_positions_per_game: int = param(
        20, "turns of each game whose pool (the agent's screen) the chunk keeps for the policy"
    )
    # The seeds: tags of the three standalone workloads, whose rolling
    # checkpoints the tag starts from.
    seed_leave_tag: str = param("", "leave_inference tag the leave inference model starts from")
    seed_leaf_tag: str = param("", "deep_bot_leaf_eval tag the leaf network starts from")
    seed_policy_tag: str = param("", "policy_set tag the policy starts from")
    # Training windows, in generations ending at the cursor.
    leave_window: int = param(2, "generations the leave inference model trains over")
    leave_turns_per_game: int = param(
        10, "leave inference: turns trained per game per generation; 0 = every eligible turn"
    )
    leaf_window: int = param(2, "generations the leaf network trains over")
    leaf_turns_per_game: int = param(
        10, "leaf network: turns trained per game per generation; 0 = every eligible turn"
    )
    policy_window: int = param(2, "generations the policy trains over")
    # The rows-clocks, in generations of this workload's rows: the seeds' are
    # sized for HastyBot's data rate.
    warmup_generations: int = param(2, "linear LR warmup length, in generations")
    averaging_generations: int = param(
        8, "schedule_free: the deployed average covers at most the last this many generations"
    )
    leave_lr: float = param(0.0, "leave inference learning rate; 0 = its optimizer arm's default")
    leaf_lr: float = param(0.0, "leaf network learning rate; 0 = its optimizer arm's default")
    policy_lr: float = param(0.0, "policy learning rate; 0 = its optimizer arm's default")
    keep_checkpoint_every: int = param(
        10, "keep every this many releases' checkpoint, as a seed for a later tag; 0 = none"
    )
    max_generations: int = param(
        -1, "stop the trainer after this many generations (-1 = run until paused)", end=True
    )

    def __post_init__(self):
        for name in ("leave_window", "leaf_window", "policy_window"):
            if getattr(self, name) < 1:
                raise ParamsError(f"{name} must be >= 1, got {getattr(self, name)}")
