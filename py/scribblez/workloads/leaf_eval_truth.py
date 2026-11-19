"""The leaf_eval_truth workload: what a perfect DeepBot leaf network would
value each legal move of every known position at (deep_bot_leaf_eval/truth.py),
the column the deep_bot_leaf_eval Positions tab compares the network's values
with.

Its one role takes each known position in turn: LeafBot over the monte-carlo
leaf plays every move out `playouts` times, each against its own opponent
leave drawn from the pinned leave inference model, and the slot delivers
data/truth/<gcg stem>.json. A restarted slot skips the positions it already
delivered. The leave inference model runs on TensorRT, so the slot needs a
GPU, though the playouts are all CPU. Once every position is in, the slot
idles until the controller, which sees the collected files, finishes it.
py/scripts/deep_bot_leaf_eval/copy_truth.py then copies the tag's truth into
the repo.
"""

import time
from dataclasses import dataclass, replace
from pathlib import Path

from scribblez import params as params_mod
from scribblez.deep_bot import known_positions
from scribblez.deep_bot_leaf_eval import positions as leaf_positions
from scribblez.deep_bot_leaf_eval import truth
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
ROLE = "truth"
# How often a slot with nothing left to compute looks for its stop.
IDLE_SECONDS = 60


@dataclass(frozen=True)
class LeafEvalTruthParams:
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
    # known positions' 28,835 legal moves cost about 98 thread-hours, 70% of
    # them THAN's, and their best 600 each about 23. Every move by default, as
    # the Positions tab's analysis values every move.
    candidates: int = param(
        0,
        "static equity's best N moves of each position that get a truth (0: every legal move). "
        "600 covers every reference move; THAN alone has about 20,000",
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
    return truth.run_settings(
        params.leave_inference_tag,
        params.leave_inference_generation,
        params.playouts,
        params.candidates,
    )


def _deliver(ctx: WorkerContext, doc: dict, position: known_positions.Position) -> int:
    path = ctx.tag_paths().work_dir(ctx.worker_id) / f"{position.gcg.stem}.json"
    leaf_positions.write(doc, path)
    return ctx.sink.deliver(path, truth.rel(position))


def run_truth(ctx: WorkerContext) -> int:
    """The role's runner: each known position the slot has not delivered, then
    idle until stopped."""
    leave = resolve_input(ctx, LEAVE_INPUT, leave_model_path(ctx.params, ctx.mount_root))
    data_dir = ctx.tag_paths().data_dir
    work = ctx.tag_paths().work_dir(ctx.worker_id)
    stats = WorkerStats(ctx)
    try:
        for p in known_position_inputs.load(ctx):
            if (data_dir / truth.rel(p)).is_file():
                continue
            doc = truth.compute(p, leave, run_settings(ctx.params), work / "ask", ctx.threads)
            nbytes = _deliver(ctx, doc, p)
            stats.cycle_done({"sim_s": doc["seconds"]}, units=1, nbytes=nbytes)
            print(f"{p.id}: {len(doc['values'])} moves in {doc['seconds']} s", flush=True)
        print("every position is computed; idle until the controller finishes the slot")
        while True:
            time.sleep(IDLE_SECONDS)
    except WorkerStopped:
        return 0


def computed(paths: TagPaths) -> int:
    """How many of the known positions the tag holds a truth for."""
    return sum((paths.data_dir / truth.rel(p)).is_file() for p in known_positions.load())


def complete(spec: WorkloadSpec, paths: TagPaths, params) -> bool:
    """The WorkloadSpec.complete entry: every known position has its truth."""
    return computed(paths) == len(known_positions.load())


def tick(spec: WorkloadSpec, task, hooks):
    """The scheduler entry: finish the slot once every truth is collected."""
    if complete(spec, hooks.paths, params_mod.validate(spec.params_cls, task.params)):
        hooks.finish(ROLE)


def progress(spec: WorkloadSpec, paths: TagPaths, params) -> list[tuple[str, object]]:
    return [("positions computed", f"{computed(paths)} / {len(known_positions.load())}")]


SPEC = WorkloadSpec(
    name="leaf_eval_truth",
    title="Leaf eval truth for the known positions",
    params_cls=LeafEvalTruthParams,
    roles=(
        RoleSpec(
            name=ROLE,
            title="Truth (GPU)",
            runner="scribblez.workloads.leaf_eval_truth:run_truth",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            inputs="scribblez.workloads.leaf_eval_truth:inputs",
            singleton=True,
            gpu=True,
            stats=StatsSpec(unit="positions", phases={"sim_s": "playouts"}),
        ),
    ),
    scheduler="scribblez.workloads.leaf_eval_truth:tick",
    progress="scribblez.workloads.leaf_eval_truth:progress",
    complete="scribblez.workloads.leaf_eval_truth:complete",
    finalize="scribblez.workloads.leaf_eval_truth:finalize",
    primary_params=("leave_inference_tag", "leave_inference_generation", "playouts", "candidates"),
    collected_dirs=(truth.TRUTH_DIR,),
)
