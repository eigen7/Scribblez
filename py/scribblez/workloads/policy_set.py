"""The policy_set workload (docs/plans/deep_bot_v0.md, "Workload 2: policy_set"):
labeled pools for the policy-set model, and its trainer
(scribblez/policy_set/trainer.py).

It is generational, like leave_inference: any number of generators deliver
labeled chunks into staging, and the singleton trainer hosts the tag's data
home, where the scheduler forms generations of `games_per_generation` games
(`positions_per_game` pools each) and routes held-out chunks.

A generator cycle plays a chunk of hidden-leaves HastyBot self-play (no random
openings: the pools are positions of the policy's own play), then labels
`positions_per_game` turns of each game with policy_set_pool_tool on the GPU:
every candidate of the pool simmed up to `rollouts` times, raced in rounds,
each rollout the candidate, two HastyBot replies and the leaf network. The
chunk's .pool (sim/pool_labels.h's labels per candidate) is delivered as the
.slog's sidecar.

The sources, pinned at task creation (`finalize`): `leaf_eval_tag` and
`leaf_eval_generation` name a deep_bot_leaf_eval generation, whose exported
network values every rollout's end state; with `leave_beliefs = inferred`,
`leave_inference_tag` and `leave_inference_generation` name a leave_inference
generation, whose exported model seats each rollout's opponent leave, for
which the tool reads each .slog's .forgone sidecar, written first. With
`uniform` the opponent's whole rack is drawn uniformly. The pinned files reach
remote generators as role inputs (`inputs`).
"""

import functools
import subprocess
import sys
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

from cloud.runtime_abi import RUNTIME_TORCH

from scribblez import params as params_mod
from scribblez.generational.optimizer_arms import OPTIMIZER_SCHEDULE_FREE, OPTIMIZERS
from scribblez.generational.scheduler import TICK_FOR_TASK
from scribblez.leave_inference import forgone
from scribblez.params import param
from scribblez.paths import ENGINE_DIR, TagPaths
from scribblez.policy_set import FORM_NAMES, POOL_SUFFIX
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

POOL_TOOL = ENGINE_DIR / "policy_set_pool_tool"
LEAVE_INFERENCE = "leave_inference"
LEAF_EVAL = "deep_bot_leaf_eval"
BELIEFS_INFERRED = "inferred"
BELIEFS_UNIFORM = "uniform"
LEAVE_BELIEFS = (BELIEFS_INFERRED, BELIEFS_UNIFORM)
# The role inputs: where a remote generator finds the pinned files.
MODEL_INPUT = "inputs/leave_inference.onnx"
LEAF_INPUT = "inputs/leaf_eval.onnx"
# write_pools' leaf for tests: the tool's CPU reference with its synthetic
# leaf, since the test suites hide the GPU.
LEAF_SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class PoolLabelParams:
    """What labels a chunk's pools: the pinned sources, the labeling and the
    self-play it labels. Shared with hidden_blind_spots, whose generators label
    pools the same way before confirming blind spots in them."""

    # The sources.
    leaf_eval_tag: str = param(
        "", "the deep_bot_leaf_eval tag whose exported network values every rollout's end state"
    )
    leaf_eval_generation: int = param(
        -1, "which generation of leaf_eval_tag (-1: its latest export, pinned at task creation)"
    )
    leave_inference_tag: str = param(
        "",
        "with leave_beliefs=inferred, the leave_inference tag whose model seats the opponents' "
        "leaves",
    )
    leave_inference_generation: int = param(
        -1,
        "which generation of leave_inference_tag (-1: its latest export, pinned at task creation)",
    )
    leave_beliefs: str = param(
        BELIEFS_INFERRED,
        "opponents' leaves: 'inferred' (the leave inference model's belief) or 'uniform'",
        choices=LEAVE_BELIEFS,
    )
    # The labels.
    rollouts: int = param(100, "rollouts per candidate (racing stops clearly beaten ones early)")
    race: bool = param(True, "stop a candidate once it is clearly beaten, checked between rounds")
    positions_per_game: int = param(2, "turns of each game labeled")
    blank_prune_points: int = param(
        40, "drop a blank move scoring this many points or more below the greedy play"
    )
    # Self-play.
    games_per_chunk: int = param(
        50, "games per generator cycle; each is labeled before the chunk is delivered"
    )
    hasty_temperature: float = param(0.0, "HastyBot softmax temperature (0 = greedy)")
    hasty_top_k: int = param(10, "HastyBot candidate count when the temperature is > 0")


@dataclass(frozen=True)
class PolicySetParams(PoolLabelParams):
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
    # The model.
    # The forms to train, side by side; at least one.
    train_slot: bool = param(False, "train the slot form (a one-shot slot decoder)")
    train_greedy: bool = param(False, "train the greedy form (a gain head over a growing set)")
    train_policy: bool = param(True, "train the policy form (a softmax over the candidates)")
    trunk_channels: int = param(128, "board trunk and candidate vector width")
    num_blocks: int = param(8, "board trunk residual blocks")
    num_heads: int = param(4, "attention heads")
    slots: int = param(16, "slot form: query slots, the largest set it builds")
    slot_layers: int = param(2, "slot form: decoder layers")
    greedy_rows: int = param(
        4, "greedy form: chosen sets drawn per pool per batch (subset assembly's rows)"
    )
    # Optimization.
    batch_size: int = param(8, "pools per batch, each with all its candidates")
    optimizer: str = param(
        OPTIMIZER_SCHEDULE_FREE,
        "optimizer arm (scribblez/generational/optim.py): 'wsd', 'muon' or 'schedule_free'",
        choices=OPTIMIZERS,
    )
    lr: float = param(0.0, "learning rate; 0 = the arm's own default")
    lr_warmup_rows: int = param(20_000, "linear LR warmup length, in pools trained")
    lr_cycle_rows: int = param(
        200_000, "period of the WSD schedule's cycle, in pools trained; unused by schedule_free"
    )
    averaging_window_rows: int = param(
        2_000_000,
        "schedule_free: the deployed average covers at most the last this many pools trained "
        "(uniform before then); unused by wsd and muon",
    )
    weight_decay: float = param(1e-4, "AdamW weight decay on weight matrices and conv kernels")
    adam_beta2: float = param(0.999, "Adam's second-moment decay rate (beta2)")
    grad_clip: float = param(1.0, "clip each step's gradient to this global norm; 0 = no clipping")

    def __post_init__(self):
        if not any(getattr(self, f"train_{name}") for name in FORM_NAMES):
            raise params_mod.ParamsError(
                "no form to train: set one of " + ", ".join(f"train_{n}" for n in FORM_NAMES)
            )


def _leave_paths(params: PoolLabelParams, mount_root: Path) -> TagPaths:
    return TagPaths(params.leave_inference_tag, LEAVE_INFERENCE, mount_root)


def _leaf_paths(params: PoolLabelParams, mount_root: Path) -> TagPaths:
    return TagPaths(params.leaf_eval_tag, LEAF_EVAL, mount_root)


def _generation(paths: TagPaths, generation: int, name: str) -> int:
    """`generation` of the source at `paths`, or (-1) its latest export.
    Raises ParamsError if the tag is unset, the index invalid, or (-1) the tag
    has no export. A generation already pinned is taken as is: a remote worker's
    mount has none of the source's exports, only the staged input."""
    if not paths.tag:
        raise params_mod.ParamsError(f"{name}_tag is required")
    if generation < -1:
        raise params_mod.ParamsError(
            f"{name}_generation must be a generation index (>= 0) or -1 for the latest, "
            f"got {generation}"
        )
    if generation >= 0:
        return generation
    exported = paths.exported_generations()
    if not exported:
        raise params_mod.ParamsError(
            f"{paths.task} tag '{paths.tag}' has no export{_export_hint(paths)}"
        )
    return exported[-1]


def _pinned(paths: TagPaths, generation: int, name: str) -> int:
    """_generation, failing also if that generation has no export."""
    generation = _generation(paths, generation, name)
    if generation not in paths.exported_generations():
        raise params_mod.ParamsError(
            f"{paths.task} tag '{paths.tag}' generation {generation} has no export"
            f"{_export_hint(paths)}"
        )
    return generation


def _export_hint(paths: TagPaths) -> str:
    """For a leaf eval tag: the command that exports its latest generation."""
    if paths.task != LEAF_EVAL:
        return ""
    return (
        f"; python -m scribblez.deep_bot_leaf_eval.onnx_export --tag {paths.tag} exports "
        "its latest generation"
    )


def leaf_path(params: PoolLabelParams, mount_root: Path) -> Path:
    paths = _leaf_paths(params, mount_root)
    return paths.onnx_path(_generation(paths, params.leaf_eval_generation, "leaf_eval"))


def model_path(params: PoolLabelParams, mount_root: Path) -> Path:
    paths = _leave_paths(params, mount_root)
    return paths.onnx_path(_generation(paths, params.leave_inference_generation, "leave_inference"))


def finalize(spec, paths: TagPaths, params: PoolLabelParams) -> PoolLabelParams:
    """Pin the sources to concrete generations at task creation, failing here,
    where the operator sees it, if a pinned generation has no export."""
    mount = paths.mount_root
    leaf = _pinned(_leaf_paths(params, mount), params.leaf_eval_generation, "leaf_eval")
    params = replace(params, leaf_eval_generation=leaf)
    if params.leave_beliefs != BELIEFS_INFERRED:
        return params
    leave = _pinned(
        _leave_paths(params, mount), params.leave_inference_generation, "leave_inference"
    )
    return replace(params, leave_inference_generation=leave)


def inputs(params: PoolLabelParams, mount_root: Path) -> dict[str, Path]:
    """The generate role's out-of-tag inputs: the pinned leaf network and,
    under inferred beliefs, the pinned leave inference model."""
    out = {LEAF_INPUT: leaf_path(params, mount_root)}
    if params.leave_beliefs == BELIEFS_INFERRED:
        out[MODEL_INPUT] = model_path(params, mount_root)
    return out


def pool_command(
    params: PoolLabelParams,
    slog_dir: Path,
    leaf: Path | str,
    model: Path | None,
    seed: int,
    threads: int,
) -> list[str]:
    """policy_set_pool_tool over `slog_dir`, writing a .pool beside each .slog;
    with `leaf` LEAF_SYNTHETIC, on the CPU reference (tests)."""
    cmd = [
        str(POOL_TOOL), "--slog-dir", str(slog_dir), "--leaf-model", str(leaf),
        "--positions-per-game", str(params.positions_per_game),
        "--rollouts", str(params.rollouts), "--race", str(params.race).lower(),
        "--blank-prune-points", str(params.blank_prune_points),
        "--seed", str(seed), "--threads", str(threads),
    ]  # fmt: skip
    if leaf == LEAF_SYNTHETIC:
        cmd += ["--backend", "cpu"]
    if model is not None:
        cmd += ["--leave-inference", str(model)]
    return cmd


def write_pools(
    params: PoolLabelParams, leaf: Path | str, model: Path | None, chunk_dir: Path, threads: int
):
    """Label the chunk in `chunk_dir`: its .forgone sidecar first when the
    model seats the leaves, then its .pool. The seed is the chunk stem's hash,
    so chunks sample their turns and rollouts independently. Raises
    subprocess.CalledProcessError if a tool fails."""
    if model is not None:
        forgone.write_sidecars(chunk_dir, threads)
    (slog,) = chunk_dir.glob("*.slog")
    seed = zlib.crc32(slog.stem.encode())
    subprocess.run(pool_command(params, chunk_dir, leaf, model, seed, threads), check=True)


def resolve_sources(ctx: WorkerContext) -> tuple[Path, Path | None]:
    """The pinned (leaf network, leave inference model or None) for this
    worker. Raises FileNotFoundError or ParamsError if one is missing."""
    p = ctx.params
    leaf = resolve_input(ctx, LEAF_INPUT, leaf_path(p, ctx.mount_root))
    model = None
    if p.leave_beliefs == BELIEFS_INFERRED:
        model = resolve_input(ctx, MODEL_INPUT, model_path(p, ctx.mount_root))
    return leaf, model


def run_generate(ctx: WorkerContext) -> int:
    """The generate-role runner. The pinned inputs are resolved once, before
    the first cycle."""
    p = ctx.params
    try:
        leaf, model = resolve_sources(ctx)
    except (FileNotFoundError, params_mod.ParamsError) as e:
        print(f"error: policy_set source: {e}", file=sys.stderr)
        return 1
    return generate(
        ctx,
        hasty_spec(p),
        face_up_leaves=False,
        random_opening_mean=0.0,
        sidecar=ChunkSidecar(POOL_SUFFIX, functools.partial(write_pools, p, leaf, model)),
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
            # The rollouts, the leaf network and the leave inference model
            # all run on the GPU.
            gpu=True,
            stats=GENERATOR_STATS,
        ),
        RoleSpec(
            name="train",
            title="Trainer (GPU)",
            runner="scribblez.policy_set.trainer:run",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            runtime=RUNTIME_TORCH,
            ingest="scribblez.generational.train_ingest:tick",
            singleton=True,
            kinds=("local", "ssh"),
            gpu=True,
            stats=StatsSpec(
                unit="pools",
                phases={"train_s": "train", "eval_s": "eval", "upload_s": "upload"},
                background=frozenset({"upload_s"}),  # the trainer's OutputDeliverer thread
            ),
        ),
    ),
    scheduler=TICK_FOR_TASK,
    progress="scribblez.generational.scheduler:progress",
    complete="scribblez.generational.scheduler:complete",
    finalize="scribblez.workloads.policy_set:finalize",
    pace_role="train",
    collected_dirs=(STAGING_DIR,),
    primary_params=(
        "leaf_eval_tag",
        "leaf_eval_generation",
        "leave_inference_tag",
        "leave_inference_generation",
        "leave_beliefs",
    ),
)
