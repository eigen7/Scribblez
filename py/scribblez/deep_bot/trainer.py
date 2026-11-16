"""The deep_bot train role (docs/plans/deep_bot_generations.md, "The
trainer"): one process over the leave inference model, the leaf network and
the policy, hosting the tag's data home.

It starts the three units from the tag's seeds (units.py) and publishes them
as release 0. Per generation `g`, once self-play has completed it:

1. it scores the newest release's models and release 0's on generation `g`,
   which none of them has trained on;
2. it trains each unit one epoch over its own window ending at `g`;
3. it publishes release `g + 1` (release.py): the three exports, then the
   manifest, then one checkpoint of the three models and optimizers, the
   cursor, and last the generation's record;
4. it evicts the generations older than the longest window.

Record keys carry their model's name (`leave_`, `leaf_`; the policy's
metrics already name their form and the equity baseline), and release 0's
scores the further prefix `r0_`.
"""

import functools
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn

from scribblez import params as params_mod
from scribblez.deep_bot import KINDS, LEAF, LEAVE, POLICY, release, units
from scribblez.generational import checkpoint, data_home, lifecycle, state_pair
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import CpuController, default_controls
from scribblez.generational.records import TrainRecorder, read_controls
from scribblez.leave_inference import forgone
from scribblez.paths import TagPaths
from scribblez.policy_set import dataset, pools
from scribblez.position_eval.trainer import OutputDeliverer, ensure_window, wait_for_generation
from scribblez.train_common import timed_print
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.worker import WorkerStats, WorkerStopped

# Release 0's record key prefix.
BASELINE = "r0_"
# The sidecars every chunk carries beside its .slog: the forgone plays the
# leave inference rows read, and the pools the policy trains on.
CHUNK_SIDECARS = (forgone.SUFFIX, pools.SUFFIX)


@dataclass
class DeepBotState(GenerationalState):
    """The cursor: `generation_index` is the next generation to train and the
    newest release's number; `rows_trained` counts the games trained on, the
    dashboard's clock; each model keeps its own rows-clock for its
    optimizer's schedule."""

    leave_rows: int = 0
    leaf_rows: int = 0
    policy_rows: int = 0


class OptimizerSet:
    """The units' optimizers as the rolling checkpoint saves and loads them."""

    def __init__(self, by_kind: dict):
        self._units = by_kind

    def state_dict(self) -> dict:
        return {kind: unit.optimizer.state_dict() for kind, unit in self._units.items()}

    def load_state_dict(self, saved: dict):
        for kind, unit in self._units.items():
            unit.optimizer.load_state_dict(saved[kind])


def models(by_kind: dict) -> nn.ModuleDict:
    """The units' models as the rolling checkpoint saves and loads them."""
    return nn.ModuleDict({kind: unit.model for kind, unit in by_kind.items()})


def unit_rows(state: DeepBotState) -> dict[str, int]:
    return {kind: getattr(state, f"{kind}_rows") for kind in KINDS}


def prefixed(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def evaluate_generation(by_kind: dict, gen_dir, cpu) -> dict:
    """Each model's metrics on the games of `gen_dir`: the leave inference
    model's and the leaf network's under their names, and the policy's, whose
    keys name their form and the equity baseline already."""
    return {
        **prefixed(f"{LEAVE}_", by_kind[LEAVE].evaluate(gen_dir, cpu)),
        **prefixed(f"{LEAF}_", by_kind[LEAF].evaluate(gen_dir, cpu)),
        **by_kind[POLICY].evaluate(dataset.read_items([gen_dir])),
    }


def train_unit(kind: str, unit, window, gen: int, rows: int, cpu):
    """`unit`'s epoch over `window`; the policy's reads no CPU controls."""
    if kind == POLICY:
        return unit.train(window, gen, rows)
    return unit.train(window, gen, rows, cpu)


def epoch_loss(kind: str, result) -> float:
    return result.losses["total"] if kind == LEAF else result.loss


def train_generation(by_kind: dict, paths: TagPaths, params, state: DeepBotState, cpu) -> dict:
    """Train each unit one epoch over its window ending at the cursor
    generation, advancing its rows-clock. Returns the record's training keys."""
    gen = state.generation_index
    record = {}
    for kind, unit in by_kind.items():
        window = lifecycle.window_dirs(paths, gen, units.window(params, kind))
        result, elapsed = train_unit(kind, unit, window, gen, getattr(state, f"{kind}_rows"), cpu)
        setattr(state, f"{kind}_rows", result.rows_trained)
        record |= {
            f"{kind}_loss": epoch_loss(kind, result),
            f"{kind}_lr": unit.optim_arm.current,
            f"{kind}_train_s": elapsed,
            f"{kind}_rows": result.rows_trained,
        }
    by_kind[POLICY].check_finite(f"after generation {gen}")
    return record


def publish_release(by_kind: dict, paths: TagPaths, tag: str, number: int, rows: dict) -> list:
    """Write release `number`: the three exports, then its manifest. Returns
    the files, the manifest last."""
    release.release_dir(paths, number).mkdir(parents=True, exist_ok=True)
    files = []
    for kind, unit in by_kind.items():
        path = release.export_path(paths, number, kind)
        if unit.export(path, {"tag": tag, "release": number}) is False:
            raise RuntimeError(f"the {kind} seed trains nothing a release can serve")
        files.append(path)
    files.append(release.write_manifest(paths, number, tag, rows))
    return files


def save_checkpoint(by_kind: dict, paths: TagPaths, params, state: DeepBotState):
    """The rolling checkpoint of the three models and optimizers, and every
    `keep_checkpoint_every`th release's copy of it."""
    config = {
        "params": asdict(params),
        "units": {kind: asdict(unit.params) for kind, unit in by_kind.items()},
    }
    checkpoint.save(paths, models(by_kind), OptimizerSet(by_kind), state, config)
    every = params.keep_checkpoint_every
    if every > 0 and state.generation_index % every == 0:
        kept = paths.checkpoint_path(state.generation_index)
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths.rolling_checkpoint, kept)


def resume(by_kind: dict, paths: TagPaths, device) -> DeepBotState:
    """Load the rolling checkpoint into the units, if there is one, and
    return its cursor; else the units stay at their seeds."""
    ckpt, state = checkpoint.load(paths, device, DeepBotState)
    if ckpt is None:
        return state
    models(by_kind).load_state_dict(ckpt["model_state_dict"])
    OptimizerSet(by_kind).load_state_dict(ckpt["optimizer_state_dict"])
    timed_print(f"Resuming at generation {state.generation_index}")
    return state


def _deliver_release(sink, paths: TagPaths, files: list):
    for path in files:
        sink.deliver_output(path, str(path.relative_to(paths.root)), keep=True)


def _deliver_generation(sink, paths, gen, files, checkpoint_snap, state_snap, recorder, staged):
    """One generation's deliveries: the release, the state pair, and last the
    record."""
    _deliver_release(sink, paths, files)
    state_pair.deliver(sink, checkpoint_snap, state_snap, gen)
    recorder.deliver_staged(staged)


def games_in(gen_dir) -> int:
    return int(lifecycle.read_manifest(gen_dir)["committed_games"])


def run_one_generation(by_kind, baseline, recorder, paths, params, state, ctx):
    """Score the cursor generation, train on it, and publish the release it
    makes."""
    gen = state.generation_index
    gen_dir = paths.generation_dir(gen)
    t0 = time.time()
    record = {"epoch": gen}
    record |= evaluate_generation(by_kind, gen_dir, ctx["cpu"])
    record |= prefixed(BASELINE, evaluate_generation(baseline, gen_dir, ctx["cpu"]))
    t_train = time.time()
    record |= train_generation(by_kind, paths, params, state, ctx["cpu"])
    sys.stdout.write("\n")
    t_publish = time.time()
    state.generation_index = gen + 1
    state.rows_trained += games_in(gen_dir)
    files = publish_release(by_kind, paths, ctx["tag"], gen + 1, unit_rows(state))
    save_checkpoint(by_kind, paths, params, state)
    lifecycle.write_train_state(paths, asdict(state))
    timed_print(f"[gen {gen}] games={state.rows_trained}; release {gen + 1} published")
    staged = recorder.stage_generation(gen, state.rows_trained, record)
    ctx["deliverer"].submit(
        f"generation {gen}",
        functools.partial(
            _deliver_generation,
            ctx["sink"],
            paths,
            gen,
            files,
            state_pair.snapshot(paths.rolling_checkpoint, gen),
            state_pair.snapshot(paths.train_state_path, gen),
            recorder,
            staged,
        ),
    )
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
    longest = max(units.window(params, kind) for kind in KINDS)
    while not params_mod.reached(state.generation_index, params.max_generations):
        ctx["cpu"].refresh(state.rows_trained)
        wait_for_generation(paths, state.generation_index, ctx["sink"], ctx["data_home"])
        run_one_generation(by_kind, baseline, recorder, paths, params, state, ctx)
        lifecycle.evict_beyond_window(
            paths, state.generation_index - 1, longest, keep_unpulled=ctx["data_home"].remote
        )
    timed_print(f"Stopped at generation {state.generation_index}.")


def ensure_newest_release(by_kind: dict, paths: TagPaths, tag: str, sink, state: DeepBotState):
    """Publish the units as the cursor's release unless it is there: release
    0, the seeds', which the first generation is played on, or after a
    restore onto a new machine the release the checkpoint holds."""
    number = state.generation_index
    if release.newest(paths) == number:
        return
    _deliver_release(sink, paths, publish_release(by_kind, paths, tag, number, unit_rows(state)))
    timed_print(f"release {number} published")


def run(ctx: WorkerContext) -> int:
    """The train-role entry point."""
    params = ctx.params
    paths = ctx.tag_paths()
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(os.environ.get("SCZ_DEVICE", "cuda"))
    print(f"Tag root: {paths.root}\nDevice: {device}")

    by_kind = units.build_units(paths, params, device)
    # Release 0, kept loaded and never trained: each generation is scored
    # under it too, so a chart can tell the models' progress from the games'
    # drift.
    baseline = units.build_units(paths, params, device)
    n_params = sum(unit.n_params for unit in by_kind.values())
    print(f"Models: {n_params:,} parameters")
    recorder = TrainRecorder(ctx.sink)
    recorder.publish_run(
        ctx.tag,
        asdict(params),
        n_params,
        {f"{kind}_loss": 1.0 for kind in KINDS},
        default_controls(),
    )
    run_ctx = {
        "tag": ctx.tag,
        "sink": ctx.sink,
        "read_controls": functools.partial(read_controls, ctx.sink),
        "stats": WorkerStats(ctx),
        "deliverer": OutputDeliverer(),
    }
    state_pair.install_seed(paths)
    state = resume(by_kind, paths, device)
    ensure_newest_release(by_kind, paths, ctx.tag, ctx.sink, state)
    longest = max(units.window(params, kind) for kind in KINDS)
    ensure_window(paths, ctx.sink, state.generation_index, longest)
    lifecycle.write_train_state(paths, asdict(state))
    run_ctx["data_home"] = data_home.start_for(ctx, paths, params, chunk_sidecars=CHUNK_SIDECARS)
    try:
        run_generations(by_kind, baseline, recorder, paths, params, state, run_ctx)
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped; the last published release is checkpointed.")
    finally:
        run_ctx["deliverer"].drain()
    return 0
