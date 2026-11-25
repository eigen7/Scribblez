"""The leaf_bot_generation train role: one process over the leave inference
model and the leaf network, hosting the tag's data home.

It starts both units from the tag's seeds (deep_bot/units.py) and publishes
them as release 0, which every generator plays. Per generation `g`, once
self-play has completed it:

1. it scores the newest release's models and release 0's on generation `g`,
   which none of them has trained on;
2. it trains each unit one epoch over its own window ending at `g`;
3. it publishes release `g + 1`, the two exports then the manifest, then the
   rolling checkpoint of both models and optimizers, the cursor, and last the
   generation's record;
4. it evicts the generations older than the longer window.

Record keys carry their model's name (`leave_`, `leaf_`), and release 0's
scores the further prefix `r0_`. The two training losses are `loss_<model>`,
the names the dashboard's Loss tab stacks. The rolling checkpoint is what a
later tag of this workload seeds from (deep_bot/unit_set.py).
"""

import functools
import os
import sys
import time
from dataclasses import asdict, dataclass

import torch

from scribblez import params as params_mod
from scribblez.deep_bot import LEAF, LEAVE, release, unit_set, units
from scribblez.deep_bot.seeds import seed_path
from scribblez.generational import data_home, lifecycle, state_pair
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import CpuController, default_controls
from scribblez.generational.records import TrainRecorder, read_controls
from scribblez.leaf_bot_generation import KINDS, LAYOUT
from scribblez.leave_inference import forgone
from scribblez.paths import TagPaths
from scribblez.position_eval.trainer import OutputDeliverer, ensure_window, wait_for_generation
from scribblez.train_common import timed_print
from scribblez.workloads.base import WorkerContext, await_tag_file
from scribblez.workloads.worker import WorkerStats, WorkerStopped

# Release 0's record key prefix.
BASELINE = "r0_"


@dataclass
class LeafBotGenerationState(GenerationalState):
    """The cursor: `generation_index` is the next generation to train and the
    newest release's number; `rows_trained` counts the games trained on, the
    dashboard's clock; each model keeps its own rows-clock for its
    optimizer's schedule."""

    leave_rows: int = 0
    leaf_rows: int = 0


def evaluate_generation(by_kind: dict, gen_dir, cpu) -> dict:
    """Each model's metrics on the games of `gen_dir`, under its name."""
    return {
        **unit_set.prefixed(f"{LEAVE}_", by_kind[LEAVE].evaluate(gen_dir, cpu)),
        **unit_set.prefixed(f"{LEAF}_", by_kind[LEAF].evaluate(gen_dir, cpu)),
    }


def export_provenance(kind: str, tag: str, number: int) -> dict:
    """What release `number`'s export of `kind` records: the tag, and the
    release as its generation (nn/onnx_metadata.h)."""
    return {"tag": tag, "generation": number}


def publish_release(by_kind: dict, paths: TagPaths, tag: str, number: int, rows: dict):
    """Write release `number`: the two exports, then the manifest."""
    provenance = functools.partial(export_provenance, tag=tag, number=number)
    unit_set.export(by_kind, paths, number, provenance)
    LAYOUT.write_manifest(paths, number, tag, rows)


def _deliver_release(remote: bool, paths: TagPaths, number: int):
    """Hand release `number` to the controller, from a trainer on an ssh
    machine. A local trainer's is already in the controller's tag."""
    if remote:
        LAYOUT.deliver(paths, number)


def _deliver_generation(sink, remote, paths, gen, checkpoint_snap, state_snap, recorder, staged):
    """One generation's deliveries: the release it made, the state pair, and
    last the record."""
    _deliver_release(remote, paths, gen + 1)
    state_pair.deliver(sink, checkpoint_snap, state_snap, gen)
    recorder.deliver_staged(staged)


def games_in(gen_dir) -> int:
    return int(lifecycle.read_manifest(gen_dir)["committed_games"])


def _publish_generation(by_kind, recorder, paths, params, state, gen: int, record: dict, ctx):
    """Advance the cursor past generation `gen`, publish the release it made
    and the checkpoint, and queue the generation's deliveries."""
    state.generation_index = gen + 1
    state.rows_trained += games_in(paths.generation_dir(gen))
    publish_release(by_kind, paths, ctx["tag"], gen + 1, unit_set.unit_rows(by_kind, state))
    unit_set.save(paths, by_kind, params, state)
    lifecycle.write_train_state(paths, asdict(state))
    timed_print(f"[gen {gen}] games={state.rows_trained}; release {gen + 1} published")
    staged = recorder.stage_generation(gen, state.rows_trained, record)
    ctx["deliverer"].submit(
        f"generation {gen}",
        functools.partial(
            _deliver_generation,
            ctx["sink"],
            ctx["remote"],
            paths,
            gen,
            state_pair.snapshot(paths.rolling_checkpoint, gen),
            state_pair.snapshot(paths.train_state_path, gen),
            recorder,
            staged,
        ),
    )


def run_one_generation(by_kind, baseline, recorder, paths, params, state, ctx):
    """Score the cursor generation, train on it, and publish the release it
    makes."""
    gen = state.generation_index
    gen_dir = paths.generation_dir(gen)
    t0 = time.time()
    record = {"epoch": gen, **evaluate_generation(by_kind, gen_dir, ctx["cpu"])}
    record |= unit_set.prefixed(BASELINE, evaluate_generation(baseline, gen_dir, ctx["cpu"]))
    t_train = time.time()
    record |= unit_set.train(by_kind, paths, params, state, ctx["cpu"])
    sys.stdout.write("\n")
    t_publish = time.time()
    _publish_generation(by_kind, recorder, paths, params, state, gen, record, ctx)
    finished = ctx["deliverer"].collect()
    ctx["stats"].cycle_done(
        {
            "eval_s": t_train - t0,
            "train_s": t_publish - t_train,
            "upload_s": finished[-1][1] if finished else 0.0,
        },
        units=games_in(gen_dir),
        nbytes=0,
    )


def run_generations(by_kind, baseline, recorder, paths, params, state, ctx):
    """The wait->score->train->publish loop, from the resumed cursor onward."""
    for kind, unit in by_kind.items():
        unit.start(recorder, getattr(state, f"{kind}_rows"))
    ctx["cpu"] = CpuController(recorder, ctx["read_controls"])
    while not params_mod.reached(state.generation_index, params.max_generations):
        ctx["cpu"].refresh(state.rows_trained)
        wait_for_generation(paths, state.generation_index, ctx["sink"], ctx["data_home"])
        run_one_generation(by_kind, baseline, recorder, paths, params, state, ctx)
        lifecycle.evict_beyond_window(
            paths, state.generation_index - 1, params.window, keep_unpulled=ctx["data_home"].remote
        )
    timed_print(f"Stopped at generation {state.generation_index}.")


def ensure_newest_release(by_kind, paths: TagPaths, tag: str, remote: bool, state):
    """Publish the units as the cursor's release unless it is there: release
    0, the seeds', which every generation is played on, or after a restore
    onto a new machine the release the checkpoint holds."""
    number = state.generation_index
    if release.newest(paths) == number:
        return
    publish_release(by_kind, paths, tag, number, unit_set.unit_rows(by_kind, state))
    _deliver_release(remote, paths, number)
    timed_print(f"release {number} published")


def run(ctx: WorkerContext) -> int:
    """The train-role entry point."""
    params = ctx.params
    paths = ctx.tag_paths()
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(os.environ.get("SCZ_DEVICE", "cuda"))
    print(f"Tag root: {paths.root}\nDevice: {device}")

    for kind in KINDS:
        await_tag_file(ctx, seed_path(paths, kind))
    by_kind = units.build_units(paths, params, device, KINDS)
    # Release 0, kept loaded and never trained: each generation is scored
    # under it too, so a chart can tell the models' progress from the games'.
    baseline = units.build_units(paths, params, device, KINDS)
    n_params = sum(unit.n_params for unit in by_kind.values())
    print(f"Models: {n_params:,} parameters")
    recorder = TrainRecorder(ctx.sink)
    recorder.publish_run(
        ctx.tag,
        asdict(params),
        n_params,
        {f"loss_{kind}": 1.0 for kind in KINDS},
        default_controls(),
    )
    run_ctx = {
        "tag": ctx.tag,
        "sink": ctx.sink,
        "remote": ctx.kind != "local",
        "read_controls": functools.partial(read_controls, ctx.sink),
        "stats": WorkerStats(ctx),
        "deliverer": OutputDeliverer(),
    }
    state_pair.install_seed(paths)
    state = unit_set.resume(by_kind, paths, device, LeafBotGenerationState)
    # Release 0 before the data home's first heartbeat releases the generators.
    ensure_newest_release(by_kind, paths, ctx.tag, run_ctx["remote"], state)
    ensure_window(paths, ctx.sink, state.generation_index, params.window)
    lifecycle.write_train_state(paths, asdict(state))
    run_ctx["data_home"] = data_home.start_for(ctx, paths, params, chunk_sidecars=(forgone.SUFFIX,))
    try:
        run_generations(by_kind, baseline, recorder, paths, params, state, run_ctx)
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped; the last published release is checkpointed.")
    finally:
        run_ctx["deliverer"].drain()
    return 0
