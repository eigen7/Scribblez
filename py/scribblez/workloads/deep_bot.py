"""The deep_bot workload (docs/plans/deep_bot_generations.md): one tag that
advances DeepBot's generations.

Creating the tag copies its three seeds under it (`finalize`,
scribblez/deep_bot/seeds.py). The singleton trainer
(scribblez/deep_bot/trainer.py) hosts the data home, trains the leave
inference model, the leaf network and the policy on each completed
generation, and publishes them together as a release
(scribblez/deep_bot/release.py).

A generator cycle takes the tag's newest release and plays a chunk of
hidden-leaves self-play, `deepbot-endgame` under the `hasty` reply in both
seats on the release's leaf network and leave inference model. It then writes
the chunk's .forgone sidecar and labels `pool_positions_per_game` turns of
each game with policy_set_pool_tool on the same two models, into the chunk's
.pool. The tool stands in until self-play writes the pools of its own turns.

Both roles run on the controller's machine for now: a generator reads the
releases from the tag in place, and nothing carries one to a remote slot yet.
The measure role is not built.
"""

import functools
from dataclasses import dataclass
from pathlib import Path

from cloud.runtime_abi import RUNTIME_TORCH

from scribblez import params as params_mod
from scribblez.deep_bot import LEAF, LEAVE, release, seeds
from scribblez.generational import lifecycle
from scribblez.generational.scheduler import TICK_FOR_TASK
from scribblez.leave_inference import forgone
from scribblez.params import ParamsError, param
from scribblez.paths import TagPaths
from scribblez.policy_set import POOL_SUFFIX
from scribblez.workloads.base import RoleSpec, StatsSpec, WorkerContext, WorkloadSpec
from scribblez.workloads.policy_set import PoolLabelParams, write_pools
from scribblez.workloads.selfplay_gen import (
    GENERATOR_STATS,
    STAGING_DIR,
    Cycle,
    generate_cycles,
)

# The sidecars every chunk carries beside its .slog: the forgone plays the
# leave inference rows read, and the pools the policy trains on.
CHUNK_SIDECARS = (forgone.SUFFIX, POOL_SUFFIX)


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
    games_per_chunk: int = param(
        50, "games per generator cycle; a generator takes up a new release at its next cycle"
    )
    # The agent's budget in self-play, at DeepBot's defaults.
    screen_rollouts: int = param(
        100, "rollouts per legal move in the agent's screen, and per candidate of a pool"
    )
    rollouts: int = param(1000, "rollouts per candidate the screen keeps")
    blank_prune_points: int = param(
        40,
        "the screen and the pools drop a blank move scoring this many points or more below the "
        "greedy play; the policy seed's pools must have used the same",
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


def player_spec(params: DeepBotParams, leaf: Path, leave: Path) -> str:
    """The --player spec of the generating agent on a release's leaf network
    and leave inference model."""
    return (
        f"--type=deepbot-endgame --reply hasty --leaf-model {leaf} --leave-inference {leave} "
        f"--screen-rollouts {params.screen_rollouts} --rollouts {params.rollouts} "
        f"--blank-prune-points {params.blank_prune_points}"
    )


def pool_params(params: DeepBotParams) -> PoolLabelParams:
    """The pool tool's labeling, the agent's screen at the turns it samples."""
    return PoolLabelParams(
        positions_per_game=params.pool_positions_per_game,
        rollouts=params.screen_rollouts,
        blank_prune_points=params.blank_prune_points,
    )


def release_cycle(paths: TagPaths, params: DeepBotParams) -> Cycle:
    """A cycle on the tag's newest release. Raises RuntimeError before
    release 0, which the trainer publishes before its data home's first
    heartbeat releases the generators."""
    number = release.newest(paths)
    if number is None:
        raise RuntimeError(f"no release under {release.releases_dir(paths)}; start the trainer")
    leaf = release.export_path(paths, number, LEAF)
    leave = release.export_path(paths, number, LEAVE)
    print(f"playing release {number}")
    return Cycle(
        player_spec(params, leaf, leave),
        functools.partial(write_pools, pool_params(params), leaf, leave),
    )


def run_generate(ctx: WorkerContext) -> int:
    """The generate-role runner."""
    return generate_cycles(
        ctx,
        functools.partial(release_cycle, ctx.tag_paths(), ctx.params),
        CHUNK_SIDECARS,
        face_up_leaves=False,
        random_opening_mean=0.0,
        games_per_chunk=ctx.params.games_per_chunk,
    )


def finalize(spec, paths: TagPaths, params: DeepBotParams) -> DeepBotParams:
    """Copy the seeds under the new tag, failing here, where the operator
    sees it, if a seed tag has no checkpoint."""
    seeds.install(paths, params)
    return params


def complete(spec, paths: TagPaths, params: DeepBotParams) -> bool:
    """The WorkloadSpec.complete entry: whether the trainer's published
    cursor has reached `max_generations`."""
    generation = lifecycle.read_train_state(paths).get("generation_index", 0)
    return params_mod.reached(generation, params.max_generations)


def progress(spec, paths: TagPaths, params: DeepBotParams) -> list[tuple[str, object]]:
    """The newest release and the games it has trained on."""
    out: list[tuple[str, object]] = []
    newest = release.newest(paths)
    if newest is not None:
        out.append(("release", newest))
    state = lifecycle.read_train_state(paths)
    if state:
        out.append(("games", state.get("rows_trained", 0)))
    return out


SPEC = WorkloadSpec(
    name="deep_bot",
    title="Advance DeepBot's generations",
    params_cls=DeepBotParams,
    roles=(
        RoleSpec(
            name="generate",
            title="Generator (GPU)",
            runner="scribblez.workloads.deep_bot:run_generate",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            # Local until a release reaches a remote slot.
            kinds=("local",),
            gpu=True,
            stats=GENERATOR_STATS,
        ),
        RoleSpec(
            name="train",
            title="Trainer (GPU)",
            runner="scribblez.deep_bot.trainer:run",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            runtime=RUNTIME_TORCH,
            ingest="scribblez.generational.train_ingest:tick",
            singleton=True,
            # Local until the seeds reach a remote trainer as role inputs.
            kinds=("local",),
            gpu=True,
            stats=StatsSpec(
                unit="games",
                phases={"eval_s": "eval", "train_s": "train", "upload_s": "upload"},
                background=frozenset({"upload_s"}),  # the trainer's OutputDeliverer thread
            ),
        ),
    ),
    scheduler=TICK_FOR_TASK,
    progress="scribblez.workloads.deep_bot:progress",
    complete="scribblez.workloads.deep_bot:complete",
    finalize="scribblez.workloads.deep_bot:finalize",
    pace_role="train",
    collected_dirs=(STAGING_DIR,),
    primary_params=(
        "seed_leave_tag",
        "seed_leaf_tag",
        "seed_policy_tag",
        "games_per_generation",
    ),
)
