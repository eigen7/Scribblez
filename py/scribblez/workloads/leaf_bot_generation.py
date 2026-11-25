"""The leaf_bot_generation workload: one generation step of DeepBot, taken by
hand. A tag starts from a leave inference model and a leaf network (its
seeds), plays DeepBot on them against itself, and trains both models on
those games, warm-started from the seeds. The next step is a new tag seeded
from this one's checkpoint, once the operator judges this one done.

Unlike deep_bot, the games' policy never changes within a tag: every
generator plays release 0, the seeds' exports, for the tag's whole life, so
the models chase a fixed target, as the standalone workloads' do under
HastyBot. Creating the tag copies its seeds under it (`finalize`): either a
previous tag of this workload (`seed_tag`), whose rolling checkpoint holds
both models, or a leave_inference and a deep_bot_leaf_eval tag.

The singleton trainer (scribblez/leaf_bot_generation/trainer.py) hosts the
data home, scores each generation under its newest release and release 0,
trains both models on it, and publishes them together as the next release
(deep_bot/release.py).

The measure role's slots, any number of them, play each due release as
DeepBot against HastyBot and against release 0, and analyze the known
positions under it for the Positions tab, one match to a slot at a time
(scribblez/leaf_bot_generation/measure_dispatch.py). The backfill role adds
the positions the manifest gains later to the analyses already made
(workloads/positions_backfill.py).

Every role but the backfill runs on the controller's machine or on one
reached over ssh, as deep_bot's do.
"""

import functools
from dataclasses import dataclass
from pathlib import Path

from cloud.runtime_abi import RUNTIME_TORCH

from scribblez.deep_bot import LEAF, LEAVE, release, seeds
from scribblez.deep_bot_leaf_eval import positions as leaf_positions
from scribblez.generational.scheduler import TICK_FOR_TASK
from scribblez.leaf_bot_generation import KINDS, LAYOUT, WORKLOAD
from scribblez.params import ParamsError, param
from scribblez.paths import MATCH_RESULTS_DIR, TagPaths
from scribblez.workloads import known_position_inputs, positions_backfill
from scribblez.workloads.base import (
    DeliveredSet,
    RoleSpec,
    StatsSpec,
    WorkerContext,
    WorkloadSpec,
    await_push,
)
from scribblez.workloads.leave_inference import FORGONE_SIDECAR
from scribblez.workloads.selfplay_gen import GENERATOR_STATS, STAGING_DIR, generate


@dataclass(frozen=True)
class LeafBotGenerationParams:
    # The seeds.
    seed_tag: str = param(
        "",
        "leaf_bot_generation tag whose rolling checkpoint seeds both models: the previous "
        "generation. Empty: seed from seed_leave_tag and seed_leaf_tag",
    )
    seed_leave_tag: str = param(
        "", "leave_inference tag the leave inference model starts from, when seed_tag is empty"
    )
    seed_leaf_tag: str = param(
        "", "deep_bot_leaf_eval tag the leaf network starts from, when seed_tag is empty"
    )
    # Self-play: DeepBot on release 0, the seeds' exports, in both seats.
    games_per_generation: int = param(20000, "self-play games per generation")
    open_ahead: int = param(
        4, "generations kept open ahead of the trainer's cursor before generators are parked"
    )
    selfplay_candidates: int = param(
        10, "the self-play DeepBot's candidates: HastyBot's best by static equity"
    )
    selfplay_opp_leaves: int = param(
        16,
        "the opponent's leaves the self-play DeepBot values each candidate over, drawn from the "
        "belief",
    )
    # Training windows, in generations ending at the cursor. 4 x 5 trains each
    # turn of a game about once, as the standalone workloads' recipes do.
    leave_window: int = param(4, "generations the leave inference model trains over")
    leave_turns_per_game: int = param(
        5, "leave inference: turns trained per game per generation; 0 = every eligible turn"
    )
    leaf_window: int = param(4, "generations the leaf network trains over")
    leaf_turns_per_game: int = param(
        5, "leaf network: turns trained per game per generation; 0 = every eligible turn"
    )
    # The rows-clocks, in generations of this workload's rows. At the defaults
    # a generation is ~400K rows a model: the warmup ~2x the standalone
    # recipes' 200K rows, the average their 20M.
    warmup_generations: int = param(1, "linear LR warmup length, in generations")
    averaging_generations: int = param(
        50, "schedule_free: the deployed average covers at most the last this many generations"
    )
    leave_lr: float = param(0.0, "leave inference learning rate; 0 = its optimizer arm's default")
    leaf_lr: float = param(0.0, "leaf network learning rate; 0 = its optimizer arm's default")
    # The measure role's matches and analyses.
    match_every_generations: int = param(
        10, "measure every Nth release: matches against HastyBot and release 0; 0 = none"
    )
    match_games: int = param(
        10000, "games of each match, mirrored in pairs over the same seeds for every release"
    )
    match_seed: int = param(1, "the matches' first game seed (not 0)")
    match_candidates: int = param(
        10, "the matches' DeepBot's candidates: HastyBot's best by static equity"
    )
    match_opp_leaves: int = param(
        16, "the opponent's leaves the matches' DeepBot values each candidate over"
    )
    analysis_opp_leaves: int = param(
        128, "the opponent's leaves the Positions tab's analysis values each move over"
    )
    max_generations: int = param(
        -1, "stop the trainer after this many generations (-1 = run until paused)", end=True
    )

    def __post_init__(self):
        if self.seed_tag and (self.seed_leave_tag or self.seed_leaf_tag):
            raise ParamsError("give seed_tag or seed_leave_tag and seed_leaf_tag, not both")
        for name in ("leave_window", "leaf_window"):
            if getattr(self, name) < 1:
                raise ParamsError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.match_games < 2 or self.match_games % 2:
            raise ParamsError(f"match_games must be even and >= 2, got {self.match_games}")
        if self.match_seed == 0:
            raise ParamsError("match_seed must not be 0, which play_game reads as a random seed")

    @property
    def window(self) -> int:
        """The generations behind the cursor the tag keeps: the longer of the
        models' windows. The controller reads it for its copy of a remote
        trainer's generations."""
        return max(self.leave_window, self.leaf_window)


def deepbot_spec(leaf: Path, leave: Path, candidates: int, opp_leaves: int) -> str:
    """The --player spec of DeepBot, endgame solved, on a release's leaf
    network and leave inference model."""
    return (
        f"--type=deepbot-endgame --leaf-model={leaf} --leave-inference={leave} "
        f"--candidates={candidates} --opp-leaves={opp_leaves}"
    )


def release_spec(root: Path, number: int, candidates: int, opp_leaves: int) -> str:
    """DeepBot on release `number` of the releases laid out under `root`: a
    tag's, or a measure slot's inbox."""
    leaf, leave = (root / release.export_rel(number, kind) for kind in (LEAF, LEAVE))
    return deepbot_spec(leaf, leave, candidates, opp_leaves)


def _seed_release(paths: TagPaths) -> bool | None:
    return True if release.exists(paths, 0) else None


def run_generate(ctx: WorkerContext) -> int:
    """Hidden-leaves DeepBot self-play on release 0 without random openings,
    each chunk with the .forgone sidecars the leave inference rows read. The
    trainer publishes release 0 before its data home releases the
    generators; a remote generator may start before the controller's push
    and waits for it."""
    paths, p = ctx.tag_paths(), ctx.params
    why = f"no release 0 under {release.releases_dir(paths)}; start the trainer"
    await_push(ctx, functools.partial(_seed_release, paths), why)
    spec = release_spec(paths.root, 0, p.selfplay_candidates, p.selfplay_opp_leaves)
    return generate(
        ctx, spec, face_up_leaves=False, random_opening_mean=0.0, sidecar=FORGONE_SIDECAR
    )


def seed_release_files(paths: TagPaths, params) -> list[str]:
    """The generate role's RoleSpec.tag_files: release 0, once it exists."""
    return LAYOUT.files(0) if release.exists(paths, 0) else []


def seed_files(paths: TagPaths, params) -> list[str]:
    """The train role's RoleSpec.tag_files: the two seeds."""
    return seeds.files(paths, KINDS)


def seed_sources(params, mount_root: Path) -> dict[str, Path]:
    """Each model's seed: the previous generation's rolling checkpoint, which
    holds both, or the standalone tags'. Raises ParamsError if a tag is
    missing or has no checkpoint."""
    if params.seed_tag:
        return dict.fromkeys(KINDS, seeds.checkpoint_of(params.seed_tag, WORKLOAD, mount_root))
    return {kind: seeds.source(params, kind, mount_root) for kind in KINDS}


def finalize(spec, paths: TagPaths, params: LeafBotGenerationParams) -> LeafBotGenerationParams:
    """Copy the seeds under the new tag, failing here, where the operator
    sees it and before anything is copied, if a seed is missing."""
    seeds.install(paths, seed_sources(params, paths.mount_root))
    return params


def backfill_exports(paths: TagPaths, params, number: int) -> tuple[Path, Path]:
    """The positions_backfill.Exports of release `number`: its leaf network and leave
    inference model, which its analysis drew the opponent's leaves from."""
    return release.export_path(paths, number, LEAF), release.export_path(paths, number, LEAVE)


def belief_model(paths: TagPaths, params) -> tuple[Path, str]:
    """The leave inference model the Positions tab charts the opponent's rack
    under, and its label: release 0's, which the self-play plays."""
    return release.export_path(paths, 0, LEAVE), "release 0, the self-play's"


def run_backfill(ctx: WorkerContext) -> int:
    return positions_backfill.run(ctx, backfill_exports)


def inputs(params, mount_root: Path) -> dict[str, Path]:
    """The measure role's out-of-tag inputs: the known positions."""
    return known_position_inputs.inputs()


SPEC = WorkloadSpec(
    name=WORKLOAD,
    title="Advance DeepBot a generation",
    params_cls=LeafBotGenerationParams,
    roles=(
        RoleSpec(
            name="generate",
            title="Generator (GPU)",
            runner="scribblez.workloads.leaf_bot_generation:run_generate",
            tag_files="scribblez.workloads.leaf_bot_generation:seed_release_files",
            gpu=True,
            stats=GENERATOR_STATS,
        ),
        RoleSpec(
            name="train",
            title="Trainer (GPU)",
            runner="scribblez.leaf_bot_generation.trainer:run",
            runtime=RUNTIME_TORCH,
            ingest="scribblez.generational.train_ingest:tick",
            singleton=True,
            tag_files="scribblez.workloads.leaf_bot_generation:seed_files",
            delivered_sets=(
                DeliveredSet(
                    outbox=release.OUTBOX,
                    marker=release.MANIFEST,
                    install="scribblez.deep_bot.release:install",
                ),
            ),
            gpu=True,
            stats=StatsSpec(
                unit="games",
                phases={"eval_s": "eval", "train_s": "train", "upload_s": "upload"},
                background=frozenset({"upload_s"}),  # the trainer's OutputDeliverer thread
            ),
        ),
        RoleSpec(
            name="measure",
            title="Matches (GPU)",
            runner="scribblez.leaf_bot_generation.measure:run",
            dispatch="scribblez.leaf_bot_generation.measure_dispatch:tick",
            inputs="scribblez.workloads.leaf_bot_generation:inputs",
            gpu=True,
            stats=StatsSpec(
                unit="games", phases={"match_s": "match", "positions_s": "known positions"}
            ),
        ),
        RoleSpec(
            name=positions_backfill.ROLE,
            title="Positions backfill (GPU)",
            runner="scribblez.workloads.leaf_bot_generation:run_backfill",
            singleton=True,
            kinds=("local",),
            gpu=True,
            stats=positions_backfill.STATS,
        ),
    ),
    scheduler=TICK_FOR_TASK,
    progress="scribblez.workloads.deep_bot:progress",
    complete="scribblez.workloads.deep_bot:complete",
    finalize="scribblez.workloads.leaf_bot_generation:finalize",
    pace_role="train",
    collected_dirs=(STAGING_DIR, MATCH_RESULTS_DIR, leaf_positions.POSITIONS_DIR),
    primary_params=(
        "seed_tag",
        "seed_leave_tag",
        "seed_leaf_tag",
        "games_per_generation",
        "selfplay_candidates",
    ),
)
