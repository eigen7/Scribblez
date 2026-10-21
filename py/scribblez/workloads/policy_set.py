"""The policy_set workload (docs/plans/deep_bot_v0.md, "Workload 2: policy_set"):
labeled pools for the policy-set model, and its trainer
(scribblez/policy_set/trainer.py), which until the model's forms arrive scores
the static-equity baseline on held-out pools.

It is generational, like leave_inference: any number of generators deliver
surveyed chunks into staging, and the singleton trainer hosts the tag's data
home, where the scheduler forms generations of `games_per_generation` games
(`positions_per_game` pools each) and routes held-out chunks.

A generator cycle plays a chunk of hidden-leaves HastyBot self-play (no random
openings: the pools are positions of the policy's own play), then surveys
`positions_per_game` turns of each game with sim_candidate_survey_tool's pool
recipe: every candidate of the pool simmed `rollouts` times, raced, each
rollout stopped at the win table after `horizon_plies`. The chunk's .pool
(sim/pool_labels.h's labels per candidate) is delivered as the .slog's
sidecar.

The source: `leave_inference_tag` names a leave_inference tag, and
`leave_inference_generation` one of its generations, pinned at task creation
(`finalize`). That generation's win table stops every rollout. With
`leave_beliefs = inferred` its exported model also draws each rollout's
opponent leave, for which the survey reads each .slog's .forgone sidecar,
written first; with `uniform` the opponent's whole rack is drawn uniformly.
Both files reach remote generators as role inputs (`inputs`).
"""

import functools
import subprocess
import sys
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

from cloud.runtime_abi import RUNTIME_TORCH

from scribblez import params as params_mod
from scribblez.generational.scheduler import TICK_FOR_TASK
from scribblez.leave_inference import forgone
from scribblez.params import param
from scribblez.paths import ENGINE_DIR, TagPaths
from scribblez.policy_set import POOL_SUFFIX
from scribblez.workloads.base import (
    RoleSpec,
    StatsSpec,
    WorkerContext,
    WorkloadSpec,
    resolve_input,
)
from scribblez.workloads.selfplay_gen import (
    GENERATOR_STATS,
    STAGING_DIR,
    ChunkSidecar,
    generate,
    hasty_spec,
)

SURVEY_TOOL = ENGINE_DIR / "sim_candidate_survey_tool"
LEAVE_INFERENCE = "leave_inference"
BELIEFS_INFERRED = "inferred"
BELIEFS_UNIFORM = "uniform"
LEAVE_BELIEFS = (BELIEFS_INFERRED, BELIEFS_UNIFORM)
# The role inputs: where a remote generator finds the pinned files.
MODEL_INPUT = "inputs/leave_inference.onnx"
WIN_TABLE_INPUT = "inputs/win_table.csv"


@dataclass(frozen=True)
class PolicySetParams:
    # The source.
    leave_inference_tag: str = param(
        "",
        "the leave_inference tag whose win table stops the rollouts and, with "
        "leave_beliefs=inferred, whose model draws the opponents' leaves",
    )
    leave_inference_generation: int = param(
        -1,
        "which generation of leave_inference_tag (-1: its latest, pinned at task creation)",
    )
    leave_beliefs: str = param(
        BELIEFS_INFERRED,
        "opponents' leaves: 'inferred' (the leave inference model's belief) or 'uniform'",
        choices=LEAVE_BELIEFS,
    )
    # The survey.
    horizon_plies: int = param(3, "plies after the candidate before the win table values a rollout")
    rollouts: int = param(100, "rollouts per candidate (racing stops clearly beaten ones early)")
    race: bool = param(True, "stop a candidate once it is clearly beaten")
    positions_per_game: int = param(2, "turns of each game surveyed")
    blank_prune_points: int = param(
        40, "drop a blank move scoring this many points or more below the greedy play"
    )
    # Self-play.
    games_per_chunk: int = param(
        50, "games per generator cycle; each is surveyed before the chunk is delivered"
    )
    hasty_temperature: float = param(0.0, "HastyBot softmax temperature (0 = greedy)")
    hasty_top_k: int = param(10, "HastyBot candidate count when the temperature is > 0")
    # Generations and held-out pools.
    games_per_generation: int = param(1000, "games per generation (positions_per_game pools each)")
    open_ahead: int = param(
        4, "generations kept open ahead of the trainer's cursor before generators are parked"
    )
    window: int = param(10, "generations kept behind the trainer's cursor; <=0 keeps all")
    holdout_every: int = param(
        20, "route about one chunk in this many into the held-out set instead of a generation"
    )
    holdout_chunks: int = param(10, "chunks in the held-out set; later held-out chunks are dropped")
    max_rows: int = param(
        -1, "stop the trainer after this many pools (-1 = run until paused)", end=True
    )


def _source_paths(params: PolicySetParams, mount_root: Path) -> TagPaths:
    return TagPaths(params.leave_inference_tag, LEAVE_INFERENCE, mount_root)


def source_generations(params: PolicySetParams, mount_root: Path) -> list[int]:
    """The source tag's generations that can label pools, ascending: those
    with a win table, and under inferred beliefs an exported model too."""
    paths = _source_paths(params, mount_root)
    gens = set(paths.win_table_generations())
    if params.leave_beliefs == BELIEFS_INFERRED:
        gens &= set(paths.exported_generations())
    return sorted(gens)


def resolved_source_generation(params: PolicySetParams, mount_root: Path) -> int:
    """`leave_inference_generation`, or (-1) the source tag's latest usable
    generation. Raises ParamsError if the tag is unset or has none."""
    if not params.leave_inference_tag:
        raise params_mod.ParamsError("leave_inference_tag is required")
    if params.leave_inference_generation < -1:
        raise params_mod.ParamsError(
            "leave_inference_generation must be a generation index (>= 0) or -1 for the latest, "
            f"got {params.leave_inference_generation}"
        )
    if params.leave_inference_generation >= 0:
        return params.leave_inference_generation
    gens = source_generations(params, mount_root)
    if not gens:
        raise params_mod.ParamsError(
            f"leave_inference tag '{params.leave_inference_tag}' has no generation with "
            f"{'a model and ' if params.leave_beliefs == BELIEFS_INFERRED else ''}a win table"
        )
    return gens[-1]


def win_table_path(params: PolicySetParams, mount_root: Path) -> Path:
    return _source_paths(params, mount_root).win_table_path(
        resolved_source_generation(params, mount_root)
    )


def model_path(params: PolicySetParams, mount_root: Path) -> Path:
    return _source_paths(params, mount_root).onnx_path(
        resolved_source_generation(params, mount_root)
    )


def finalize(spec, paths: TagPaths, params: PolicySetParams) -> PolicySetParams:
    """Pin the source to a concrete generation at task creation, failing here,
    where the operator sees it, if that generation cannot label pools."""
    generation = resolved_source_generation(params, paths.mount_root)
    if generation not in source_generations(params, paths.mount_root):
        raise params_mod.ParamsError(
            f"leave_inference tag '{params.leave_inference_tag}' generation {generation} has "
            f"no {'model or ' if params.leave_beliefs == BELIEFS_INFERRED else ''}win table"
        )
    return replace(params, leave_inference_generation=generation)


def inputs(params: PolicySetParams, mount_root: Path) -> dict[str, Path]:
    """The generate role's out-of-tag inputs: the pinned win table and, under
    inferred beliefs, the pinned model."""
    out = {WIN_TABLE_INPUT: win_table_path(params, mount_root)}
    if params.leave_beliefs == BELIEFS_INFERRED:
        out[MODEL_INPUT] = model_path(params, mount_root)
    return out


def survey_command(
    params: PolicySetParams,
    slog_dir: Path,
    table: Path,
    model: Path | None,
    seed: int,
    threads: int,
) -> list[str]:
    """sim_candidate_survey_tool's pool recipe over `slog_dir`, writing a .pool
    beside each .slog."""
    cmd = [
        str(SURVEY_TOOL), "--slog-dir", str(slog_dir), "--recipe", "pool",
        "--positions-per-game", str(params.positions_per_game),
        "--rollouts", str(params.rollouts), "--race", str(params.race).lower(),
        "--win-table", str(table), "--horizon-plies", str(params.horizon_plies),
        "--blank-prune-points", str(params.blank_prune_points),
        "--seed", str(seed), "--threads", str(threads),
    ]  # fmt: skip
    if model is not None:
        cmd += ["--leave-inference", str(model)]
    return cmd


def write_pools(
    params: PolicySetParams, table: Path, model: Path | None, chunk_dir: Path, threads: int
):
    """Survey the chunk in `chunk_dir`: its .forgone sidecar first when the
    model draws the leaves, then its .pool. The survey seed is the chunk
    stem's hash, so chunks sample their turns and rollouts independently.
    Raises subprocess.CalledProcessError if a tool fails."""
    if model is not None:
        forgone.write_sidecars(chunk_dir, threads)
    (slog,) = chunk_dir.glob("*.slog")
    seed = zlib.crc32(slog.stem.encode())
    subprocess.run(survey_command(params, chunk_dir, table, model, seed, threads), check=True)


def run_generate(ctx: WorkerContext) -> int:
    """The generate-role runner. The pinned inputs are resolved once, before
    the first cycle."""
    p = ctx.params
    try:
        table = resolve_input(ctx, WIN_TABLE_INPUT, win_table_path(p, ctx.mount_root))
        model = None
        if p.leave_beliefs == BELIEFS_INFERRED:
            model = resolve_input(ctx, MODEL_INPUT, model_path(p, ctx.mount_root))
    except FileNotFoundError as e:
        print(f"error: policy_set source: {e}", file=sys.stderr)
        return 1
    return generate(
        ctx,
        hasty_spec(p),
        face_up_leaves=False,
        random_opening_mean=0.0,
        sidecar=ChunkSidecar(POOL_SUFFIX, functools.partial(write_pools, p, table, model)),
        games_per_chunk=p.games_per_chunk,
    )


SPEC = WorkloadSpec(
    name="policy_set",
    title="Label policy-set pools",
    params_cls=PolicySetParams,
    roles=(
        RoleSpec(
            name="generate",
            title="Generator (GPU)",
            runner="scribblez.workloads.policy_set:run_generate",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            inputs="scribblez.workloads.policy_set:inputs",
            # The leave inference model runs through TensorRT; GPU machines
            # only, whatever the beliefs.
            gpu=True,
            stats=GENERATOR_STATS,
        ),
        RoleSpec(
            name="train",
            title="Trainer",
            runner="scribblez.policy_set.trainer:run",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            runtime=RUNTIME_TORCH,
            ingest="scribblez.generational.train_ingest:tick",
            singleton=True,
            kinds=("local", "ssh"),
            stats=StatsSpec(unit="pools", phases={"eval_s": "eval"}),
        ),
    ),
    scheduler=TICK_FOR_TASK,
    progress="scribblez.generational.scheduler:progress",
    complete="scribblez.generational.scheduler:complete",
    finalize="scribblez.workloads.policy_set:finalize",
    pace_role="train",
    collected_dirs=(STAGING_DIR,),
    primary_params=("leave_inference_tag", "leave_inference_generation", "leave_beliefs"),
)
