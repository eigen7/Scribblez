"""The leaf_eval_hasty_rollouts workload: each legal move of every known
position played out to the game's end by hastybot-endgame
(deep_bot_leaf_eval/hasty_rollouts.py), the column the Positions tab compares
a leaf network's values with.

Its one role takes each known position in turn: LeafBot over the monte-carlo
leaf plays every move out `playouts` times, each against its own opponent
leave drawn from the pinned leave inference model, and the slot delivers
data/hasty_rollouts/<gcg stem>.json. A restarted slot skips the positions it already
delivered. The leave inference model runs on TensorRT, so the slot needs a
GPU, though the playouts are all CPU. Once every position is in, the slot
idles until the controller, which sees the collected files, finishes it.
py/scripts/deep_bot_leaf_eval/copy_hasty_rollouts.py then copies the tag's
rollouts into the repo.
"""

import time
from dataclasses import dataclass, replace
from pathlib import Path

from scribblez import params as params_mod
from scribblez.deep_bot import known_positions
from scribblez.deep_bot_leaf_eval import hasty_rollouts
from scribblez.deep_bot_leaf_eval import positions as leaf_positions
from scribblez.params import param
from scribblez.paths import TagPaths
from scribblez.workloads import known_position_inputs
from scribblez.workloads.base import (
    RoleSpec,
    StatsSpec,
    WorkerContext,
    WorkloadSpec,
    resolve_input,
)
from scribblez.workloads.pinned_exports import export_generation, pinned_generation
from scribblez.workloads.worker import WorkerStats, WorkerStopped

LEAVE_INFERENCE = "leave_inference"
# Where a remote slot finds the pinned leave inference model.
LEAVE_INPUT = "inputs/leave_inference.onnx"
ROLE = "rollouts"
# How often a slot with nothing left to compute looks for its stop.
IDLE_SECONDS = 60


@dataclass(frozen=True)
class LeafEvalHastyRolloutsParams:
    leave_inference_tag: str = param(
        "",
        "the leave_inference tag whose model draws the opponent's leaves: the one the "
        "deep_bot_leaf_eval tag's match_leave_inference_tag names, for the same belief",
    )
    leave_inference_generation: int = param(
        -1, "which generation of leave_inference_tag (-1: its latest export, pinned at creation)"
    )
    playouts: int = param(
        1024, "playouts per move to the game's end, each against its own opponent leave drawn"
    )
    # About 12 thread-seconds a move at the default (12 ms a playout, measured
    # 2026-11-19 with monte_carlo_sim_tool over the known positions), so the
    # known positions' 14,191 legal moves cost about 47 thread-hours, and
    # their best 600 each about 28. Every move by default, as the Positions
    # tab's analysis values every move.
    candidates: int = param(
        0,
        "static equity's best N moves of each position that are played out (0: every legal move). "
        "600 covers every reference move; H(IN)S alone has about 5,300",
    )


def _leave_paths(params, mount_root: Path) -> TagPaths:
    return TagPaths(params.leave_inference_tag, LEAVE_INFERENCE, mount_root)


def finalize(spec, paths: TagPaths, params):
    """Pin the leave inference model to a concrete generation at task creation."""
    if params.playouts < 1 or params.candidates < 0:
        raise params_mod.ParamsError("playouts must be positive, candidates >= 0")
    leave = pinned_generation(
        _leave_paths(params, paths.mount_root), params.leave_inference_generation, LEAVE_INFERENCE
    )
    return replace(params, leave_inference_generation=leave)


def leave_model_path(params, mount_root: Path) -> Path:
    """The pinned leave inference model on the controller's mount."""
    paths = _leave_paths(params, mount_root)
    return paths.onnx_path(
        export_generation(paths, params.leave_inference_generation, LEAVE_INFERENCE)
    )


def inputs(params, mount_root: Path) -> dict[str, Path]:
    """The role's out-of-tag inputs: the leave inference model and the known
    positions."""
    return {LEAVE_INPUT: leave_model_path(params, mount_root), **known_position_inputs.inputs()}


def run_settings(params) -> dict:
    return hasty_rollouts.run_settings(
        params.leave_inference_tag,
        params.leave_inference_generation,
        params.playouts,
        params.candidates,
    )


def _deliver(ctx: WorkerContext, doc: dict, position: known_positions.Position) -> int:
    path = ctx.tag_paths().work_dir(ctx.worker_id) / f"{position.gcg.stem}.json"
    leaf_positions.write(doc, path)
    return ctx.sink.deliver(path, hasty_rollouts.rel(position))


def run_rollouts(ctx: WorkerContext) -> int:
    """The role's runner: each known position the slot has not delivered, then
    idle until stopped."""
    leave = resolve_input(ctx, LEAVE_INPUT, leave_model_path(ctx.params, ctx.mount_root))
    data_dir = ctx.tag_paths().data_dir
    work = ctx.tag_paths().work_dir(ctx.worker_id)
    stats = WorkerStats(ctx)
    try:
        for p in known_position_inputs.load(ctx):
            if (data_dir / hasty_rollouts.rel(p)).is_file():
                continue
            run = run_settings(ctx.params)
            doc = hasty_rollouts.compute(p, leave, run, work / "ask", ctx.threads)
            nbytes = _deliver(ctx, doc, p)
            stats.cycle_done({"sim_s": doc["seconds"]}, units=1, nbytes=nbytes)
            print(f"{p.id}: {len(doc['values'])} moves in {doc['seconds']} s", flush=True)
        print("every position is computed; idle until the controller finishes the slot")
        while True:
            time.sleep(IDLE_SECONDS)
    except WorkerStopped:
        return 0


def computed(paths: TagPaths) -> int:
    """How many of the known positions the tag holds the rollouts of."""
    return sum((paths.data_dir / hasty_rollouts.rel(p)).is_file() for p in known_positions.load())


def complete(spec: WorkloadSpec, paths: TagPaths, params) -> bool:
    """The WorkloadSpec.complete entry: every known position is played out."""
    return computed(paths) == len(known_positions.load())


def tick(spec: WorkloadSpec, task, hooks):
    """The scheduler entry: finish the slot once every position's rollouts are
    collected."""
    if complete(spec, hooks.paths, params_mod.validate(spec.params_cls, task.params)):
        hooks.finish(ROLE)


def progress(spec: WorkloadSpec, paths: TagPaths, params) -> list[tuple[str, object]]:
    return [("positions computed", f"{computed(paths)} / {len(known_positions.load())}")]


SPEC = WorkloadSpec(
    name="leaf_eval_hasty_rollouts",
    title="Hasty rollouts of the known positions",
    params_cls=LeafEvalHastyRolloutsParams,
    roles=(
        RoleSpec(
            name=ROLE,
            title="Rollouts (GPU)",
            runner="scribblez.workloads.leaf_eval_hasty_rollouts:run_rollouts",
            inputs="scribblez.workloads.leaf_eval_hasty_rollouts:inputs",
            singleton=True,
            gpu=True,
            stats=StatsSpec(unit="positions", phases={"sim_s": "playouts"}),
        ),
    ),
    scheduler="scribblez.workloads.leaf_eval_hasty_rollouts:tick",
    progress="scribblez.workloads.leaf_eval_hasty_rollouts:progress",
    complete="scribblez.workloads.leaf_eval_hasty_rollouts:complete",
    finalize="scribblez.workloads.leaf_eval_hasty_rollouts:finalize",
    primary_params=("leave_inference_tag", "leave_inference_generation", "playouts", "candidates"),
    collected_dirs=(hasty_rollouts.ROLLOUTS_DIR,),
)
