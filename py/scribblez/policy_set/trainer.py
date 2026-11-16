"""The policy_set train role (docs/plans/deep_bot_v0.md, "Workload 2:
policy_set"): the policy-set model's forms, trained on the labeled pools,
with the static-equity baseline scored beside them.

It hosts the tag's data home, the generation scheduler beside it, so its
generations form and its held-out set fills: about one chunk in
`holdout_every` is routed into the set (TagPaths.held_out_chunks_dir) until it
holds `holdout_chunks`. Per generation, once the cursor generation is
complete, it trains each form one epoch over the window's pools, evaluates
the forms and the baseline on the held-out pools (policy_set/metrics.py),
checkpoints, exports the policy form for PolicyBot (onnx_export.py), and
delivers the export, the checkpoint, the cursor and last the record.
Each form's eval keys carry its name as a prefix, the baseline's "equity_";
its training losses are "loss_<form>_<N>", the Loss tab's naming.

The checkpoint config records the policy-set format version and the board-row
and move encoding versions; a checkpoint written under others is refused
rather than resumed.

The models, their optimizer and their epoch are the unit's (unit.py), which
the deep_bot trainer drives too; this module is the loop around one.
"""

import functools
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from scribblez import params as params_mod
from scribblez.generational import checkpoint, data_home, lifecycle, state_pair
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.records import TrainRecorder
from scribblez.paths import TagPaths
from scribblez.policy_set import FORM_GREEDY, FORM_POLICY, FORM_SLOT, dataset, forms, metrics, pools
from scribblez.policy_set.unit import BASELINE, VERSION_KEYS, PolicySetUnit
from scribblez.position_eval.trainer import OutputDeliverer, wait_for_generation
from scribblez.train_common import timed_print
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.policy_set import chunk_sidecars
from scribblez.workloads.worker import WorkerStats, WorkerStopped

# The policy form's loss terms (policy.form_losses).
POLICY_TERMS = (
    "ce",
    "value_mean",
    "value_var",
    "ownership",
    "square_points_mean",
    "square_points_var",
    "going_out",
    "end_points_mean",
    "end_points_var",
)


def check_resumable(paths: TagPaths, config: dict):
    """Refuse a rolling checkpoint written under other versions."""
    path = paths.rolling_checkpoint
    if not path.exists():
        return
    saved = torch.load(path, map_location="cpu", weights_only=False)["config"]
    for key in VERSION_KEYS:
        if saved.get(key) != config[key]:
            raise RuntimeError(
                f"{path} has {key}={saved.get(key)}, but this run has {config[key]}; "
                "start a new tag (or check out the code it was trained with)"
            )


def held_out_items(paths: TagPaths, params) -> list[dataset.PoolItem]:
    """The held-out set's pools, from its first `holdout_chunks` files."""
    files = sorted(paths.held_out_chunks_dir.glob(f"*{pools.SUFFIX}"))[: params.holdout_chunks]
    return [
        dataset.PoolItem(f.with_suffix(".slog"), p) for f in files for p in pools.read_pools(f)[1]
    ]


def loss_weights(params) -> dict[str, float]:
    """The training loss's terms, each weighted 1, as the Loss tab stacks
    them."""
    keys = []
    if FORM_SLOT in forms.selected(params):
        keys += [f"loss_slot_{n}" for n in metrics.SET_SIZES if n <= params.slots]
    if FORM_GREEDY in forms.selected(params):
        keys.append("loss_greedy_gain")
    if FORM_POLICY in forms.selected(params):
        keys += [f"loss_policy_{term}" for term in POLICY_TERMS]
    return dict.fromkeys(keys, 1.0)


def export_generation(paths: TagPaths, gen: int, unit: PolicySetUnit, tag: str) -> Path | None:
    """Export generation `gen`'s policy form for PolicyBot, if it trains one;
    the export's path."""
    path = paths.onnx_path(gen)
    return path if unit.export(path, {"tag": tag, "generation": gen}) else None


def _deliver_generation(sink, paths, gen, export, checkpoint_snap, state_snap, recorder, staged):
    """One generation's deliveries, the record last."""
    if export is not None:
        sink.deliver_output(export, str(export.relative_to(paths.root)))
    state_pair.deliver(sink, checkpoint_snap, state_snap, gen)
    recorder.deliver_staged(staged)


def train_one_generation(unit, recorder, paths, params, state, ctx):
    """Train one epoch over the window ending at the cursor generation, then
    evaluate, checkpoint and queue the generation's deliveries."""
    gen = state.generation_index
    window = lifecycle.window_dirs(paths, gen, params.window)
    rows_before = state.rows_trained
    result, elapsed = unit.train(window, gen, state.rows_trained)
    state.rows_trained = result.rows_trained
    state.generation_index = gen + 1
    t_eval = time.time()
    record = {
        "epoch": gen,
        **unit.train_metrics(result),
        "lr": unit.optim_arm.current,
        "elapsed_s": elapsed,
        **unit.optim_arm.metrics(),
        **unit.window_stats(),
        **unit.evaluate(held_out_items(paths, params)),
    }
    recalls = " ".join(
        f"{name}={record.get(f'{name}_eval_recall_8', float('nan')):.3f}"
        for name in (*unit.models, BASELINE.rstrip("_"))
    )
    timed_print(f"[gen {gen}] pools={state.rows_trained} loss={result.loss:.4f} recall@8 {recalls}")
    unit.check_finite(f"after generation {gen}")
    export = export_generation(paths, gen, unit, ctx["tag"])
    checkpoint.save(paths, unit.models, unit.optimizer, state, ctx["config"])
    lifecycle.write_train_state(paths, asdict(state))
    staged = recorder.stage_generation(gen, state.rows_trained, record)
    ctx["deliverer"].submit(
        f"generation {gen}",
        functools.partial(
            _deliver_generation,
            ctx["sink"],
            paths,
            gen,
            export,
            state_pair.snapshot(paths.rolling_checkpoint, gen),
            state_pair.snapshot(paths.train_state_path, gen),
            recorder,
            staged,
        ),
    )
    finished = ctx["deliverer"].collect()
    ctx["stats"].cycle_done(
        {
            "train_s": elapsed,
            "eval_s": time.time() - t_eval,
            "upload_s": finished[-1][1] if finished else 0.0,
        },
        units=state.rows_trained - rows_before,
        nbytes=0,
    )


def run_generations(unit, recorder, paths, params, state, ctx):
    """The wait->train->advance loop, from the resumed cursor onward."""
    unit.start(recorder, state.rows_trained)
    while not params_mod.reached(state.rows_trained, params.max_rows):
        wait_for_generation(paths, state.generation_index, ctx["sink"], ctx["data_home"])
        train_one_generation(unit, recorder, paths, params, state, ctx)
        lifecycle.evict_beyond_window(
            paths, state.generation_index - 1, params.window, keep_unpulled=ctx["data_home"].remote
        )
    timed_print(f"Stopped at {state.rows_trained} pools (generation {state.generation_index}).")


def run(ctx: WorkerContext) -> int:
    """The train-role entry point."""
    params = ctx.params
    paths = ctx.tag_paths()
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(os.environ.get("SCZ_DEVICE", "cuda"))
    print(f"Tag root: {paths.root}\nDevice: {device}")

    unit = PolicySetUnit(params, device)
    print(f"Models ({', '.join(unit.models)}): {unit.n_params:,} parameters")
    config = unit.config()
    recorder = TrainRecorder(ctx.sink)
    recorder.publish_run(
        ctx.tag,
        asdict(params),
        unit.n_params,
        loss_weights(params),
        {},
    )
    run_ctx = {
        "config": config,
        "tag": ctx.tag,
        "sink": ctx.sink,
        "stats": WorkerStats(ctx),
        "deliverer": OutputDeliverer(),
    }
    state_pair.install_seed(paths)
    # After the seed, which may be what puts a rolling checkpoint in place.
    check_resumable(paths, config)
    state: GenerationalState = checkpoint.resume(paths, unit.models, unit.optimizer, device)
    lifecycle.write_train_state(paths, asdict(state))
    run_ctx["data_home"] = data_home.start_for(
        ctx,
        paths,
        params,
        chunk_sidecars=chunk_sidecars(params),
        holdout_every=params.holdout_every,
        holdout_chunks=params.holdout_chunks,
    )
    try:
        run_generations(unit, recorder, paths, params, state, run_ctx)
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped; last completed generation is checkpointed.")
    finally:
        run_ctx["deliverer"].drain()
    return 0
