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
scores the further prefix `r0_`. The three training losses are `loss_<model>`,
the names the dashboard's Loss tab stacks.
"""

import functools
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from scribblez import params as params_mod
from scribblez.deep_bot import (
    KINDS,
    LEAF,
    LEAVE,
    POLICY,
    belief_by_rank,
    kept_checkpoints,
    release,
    units,
)
from scribblez.deep_bot.seeds import seed_path
from scribblez.ffi import read_file_header, struct_dtype
from scribblez.generational import checkpoint, data_home, lifecycle, state_pair
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import CpuController, default_controls
from scribblez.generational.records import TrainRecorder, read_controls
from scribblez.paths import TagPaths
from scribblez.policy_set import POOL_SUFFIX, dataset
from scribblez.policy_set.onnx_export import LEAVE_GENERATION_KEY, LEAVE_TAG_KEY
from scribblez.position_eval.trainer import OutputDeliverer, ensure_window, wait_for_generation
from scribblez.train_common import timed_print
from scribblez.workloads.base import WorkerContext, await_tag_file
from scribblez.workloads.deep_bot import CHUNK_SIDECARS
from scribblez.workloads.worker import WorkerStats, WorkerStopped

# Release 0's record key prefix.
BASELINE = "r0_"


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


def newest_release_share(gen_dir, newest: int) -> float:
    """The share of `gen_dir`'s games the agent played on release `newest`,
    the newest while the generation filled. The rest came from generators
    that had not yet taken it up (the plan's "Pacing"). A chunk's .pool
    header names the leaf network its games ran on, whose generation is the
    release's number."""
    header = struct_dtype("PoolFileHeader")
    games = on_newest = 0
    for slog in sorted(gen_dir.glob("*.slog")):
        count = read_file_header(slog)[0]
        sources = np.fromfile(slog.with_suffix(POOL_SUFFIX), dtype=header, count=1)[0]
        games += count
        on_newest += count if sources["leaf_model"]["generation"] == newest else 0
    return on_newest / games


def evaluate_generation(by_kind: dict, gen_dir, cpu) -> dict:
    """Each model's metrics on the games of `gen_dir`: the leave inference
    model's, with its score by the rank of the move played, and the leaf
    network's under their names, and the policy's, whose keys name their form
    and the equity baseline already."""
    return {
        **prefixed(f"{LEAVE}_", by_kind[LEAVE].evaluate(gen_dir, cpu)),
        **prefixed(f"{LEAVE}_", belief_by_rank.evaluate(by_kind[LEAVE], gen_dir)),
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
            f"loss_{kind}": epoch_loss(kind, result),
            f"{kind}_lr": unit.optim_arm.current,
            f"{kind}_train_s": elapsed,
            f"{kind}_rows": result.rows_trained,
        }
    by_kind[POLICY].check_finite(f"after generation {gen}")
    return record


def export_provenance(kind: str, tag: str, number: int) -> dict:
    """What release `number`'s export of `kind` records: the tag, and the
    release as its generation, the entry the engine reads a model's identity
    from (nn/onnx_metadata.h), so a chunk's pool header names the release
    that labeled it. The policy also names the leave inference model it is
    served beside, its own release's, in place of its seed tag's."""
    out = {"tag": tag, "generation": number}
    if kind == POLICY:
        out |= {LEAVE_TAG_KEY: tag, LEAVE_GENERATION_KEY: number}
    return out


def publish_release(by_kind: dict, paths: TagPaths, tag: str, number: int, rows: dict):
    """Write release `number`: the three exports, then its manifest."""
    release.release_dir(paths, number).mkdir(parents=True, exist_ok=True)
    for kind, unit in by_kind.items():
        path = release.export_path(paths, number, kind)
        if unit.export(path, export_provenance(kind, tag, number)) is False:
            raise RuntimeError(f"the {kind} seed trains nothing a release can serve")
    release.write_manifest(paths, number, tag, rows)


def save_checkpoint(by_kind: dict, paths: TagPaths, params, state: DeepBotState, remote: bool):
    """The rolling checkpoint of the three models and optimizers, and every
    `keep_checkpoint_every`th release's copy of it."""
    config = {
        "params": asdict(params),
        "units": {kind: asdict(unit.params) for kind, unit in by_kind.items()},
    }
    checkpoint.save(paths, models(by_kind), OptimizerSet(by_kind), state, config)
    every = params.keep_checkpoint_every
    if every > 0 and state.generation_index % every == 0:
        kept_checkpoints.keep(paths, state.generation_index, remote)


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


def _deliver_release(remote: bool, paths: TagPaths, number: int):
    """Hand release `number` to the controller, from a trainer on an ssh
    machine. A local trainer's is already in the controller's tag."""
    if remote:
        release.deliver(paths, number)


def _deliver_generation(sink, remote, paths, gen, checkpoint_snap, state_snap, recorder, staged):
    """One generation's deliveries: the release it made, the state pair, and
    last the record."""
    _deliver_release(remote, paths, gen + 1)
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
    record = {"epoch": gen, "newest_release_share": newest_release_share(gen_dir, gen)}
    record |= evaluate_generation(by_kind, gen_dir, ctx["cpu"])
    record |= prefixed(BASELINE, evaluate_generation(baseline, gen_dir, ctx["cpu"]))
    t_train = time.time()
    record |= train_generation(by_kind, paths, params, state, ctx["cpu"])
    sys.stdout.write("\n")
    t_publish = time.time()
    state.generation_index = gen + 1
    state.rows_trained += games_in(gen_dir)
    publish_release(by_kind, paths, ctx["tag"], gen + 1, unit_rows(state))
    save_checkpoint(by_kind, paths, params, state, ctx["remote"])
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


def ensure_newest_release(by_kind, paths: TagPaths, tag: str, remote: bool, state: DeepBotState):
    """Publish the units as the cursor's release unless it is there: release
    0, the seeds', which the first generation is played on, or after a
    restore onto a new machine the release the checkpoint holds."""
    number = state.generation_index
    if release.newest(paths) == number:
        return
    publish_release(by_kind, paths, tag, number, unit_rows(state))
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
    state = resume(by_kind, paths, device)
    ensure_newest_release(by_kind, paths, ctx.tag, run_ctx["remote"], state)
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
