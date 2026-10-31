"""The policy_set train role (docs/plans/deep_bot_v0.md, "Workload 2:
policy_set"): the policy-set model's forms, trained on the labeled pools,
with the static-equity baseline scored beside them.

It hosts the tag's data home, the generation scheduler beside it, so its
generations form and its held-out set fills: about one chunk in
`holdout_every` is routed into the set (TagPaths.held_out_chunks_dir) until it
holds `holdout_chunks`. Per generation, once the cursor generation is
complete, it trains each form one epoch over the window's pools, evaluates
the forms and the baseline on the held-out pools (policy_set/metrics.py),
checkpoints, and delivers the checkpoint, the cursor and last the record.
Each form's eval keys carry its name as a prefix, the baseline's "equity_";
its training losses are "loss_<form>_<N>", the Loss tab's naming.

The checkpoint config records the policy-set format version and the board-row
and move encoding versions; a checkpoint written under others is refused
rather than resumed.
"""

import functools
import os
import sys
import time
from dataclasses import asdict

import torch
from torch import nn

from scribblez import params as params_mod
from scribblez.ffi import format_layout, task_input_shapes
from scribblez.generational import checkpoint, data_home, lifecycle, state_pair
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import progress_line
from scribblez.generational.finite_guard import check_parameters
from scribblez.generational.optim import build_optim_arm, build_optimizer
from scribblez.generational.records import TrainRecorder
from scribblez.move_set_eval.moves import move_encoding_version
from scribblez.paths import TagPaths
from scribblez.policy_set import (
    FORM_GREEDY,
    FORM_POLICY,
    FORM_SLOT,
    dataset,
    forms,
    metrics,
    pools,
    train_loop,
)
from scribblez.policy_set.model import PolicySetEncoder, trunk_scalar
from scribblez.position_eval.trainer import OutputDeliverer, wait_for_generation
from scribblez.train_common import timed_print
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.worker import WorkerStats, WorkerStopped

# The baseline's record key prefix; each form's records carry its own.
BASELINE = "equity_"
# The checkpoint-config entries a resume must match.
VERSION_KEYS = ("policy_set_format_version", "move_encoding_version", "input_encoding_version")
# Batches the schedule-free arm recomputes BatchNorm statistics over before
# evaluating and checkpointing (optim.ScheduleFreeArm.eval_mode).
BN_RECALIBRATION_BATCHES = 16


def versions() -> dict[str, int]:
    """This code's values of VERSION_KEYS."""
    constants = format_layout()["constants"]
    return {
        "policy_set_format_version": constants["policy_set"]["format_version"],
        "move_encoding_version": move_encoding_version(),
        "input_encoding_version": constants["input_encoding_version"],
    }


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


def build_encoder(params) -> PolicySetEncoder:
    shapes = {s.name: s.dims for s in task_input_shapes()}
    return PolicySetEncoder(
        shapes["input_spatial"][0],
        shapes["input_scalar"][0],
        params.trunk_channels,
        params.num_blocks,
        params.num_heads,
        params.normalize_scalar,
    )


def build_models(params) -> nn.ModuleDict:
    """The forms `params` selects (forms.selected), each on its own encoder."""
    return nn.ModuleDict(
        {
            name: forms.FORMS[name].build(build_encoder(params), params)
            for name in forms.selected(params)
        }
    )


def _trunk_forward(models: nn.ModuleDict, spatial, scalar):
    """The passes that reach the models' BatchNorm layers: each form's trunk,
    on the trunk's scalar input (model.trunk_scalar)."""
    for model in models.values():
        model.encoder.trunk(spatial, scalar)


def recalibration_batches(batcher, items, params, gen, device):
    """(spatial, scalar) pairs for BatchNorm recalibration, drawn under a seed
    distinct from the epoch's."""
    order = dataset.batch_order(len(items), params.batch_size, seed=gen * 1000003 + 1)
    for idx in order[:BN_RECALIBRATION_BATCHES]:
        b = batcher.build([items[i] for i in idx])
        yield (
            b["input_spatial"].to(device),
            trunk_scalar(b["input_scalar"], b["opp_rack"]).to(device),
        )


def held_out_items(paths: TagPaths, params) -> list[dataset.PoolItem]:
    """The held-out set's pools, from its first `holdout_chunks` files."""
    files = sorted(paths.held_out_chunks_dir.glob(f"*{pools.SUFFIX}"))[: params.holdout_chunks]
    return [
        dataset.PoolItem(f.with_suffix(".slog"), p) for f in files for p in pools.read_pools(f)[1]
    ]


def eval_record(models, batcher, held: list[dataset.PoolItem], params, device) -> dict:
    """The held-out pools' statistics, the baseline's metrics, and each
    form's metrics and set-building time."""
    if not held:
        return {}
    pool_list = [it.pool for it in held]
    return {
        **metrics.pool_stats(pool_list, "eval_"),
        **metrics.set_metrics(pool_list, metrics.equity_top, BASELINE),
        **train_loop.evaluate(models, batcher, held, device, params),
        **train_loop.set_build_ms(models, batcher, held, device),
    }


def loss_weights(params) -> dict[str, float]:
    """The training loss's terms, each weighted 1, as the Loss tab stacks
    them."""
    keys = []
    if FORM_SLOT in forms.selected(params):
        keys += [f"loss_slot_{n}" for n in metrics.SET_SIZES if n <= params.slots]
    if FORM_GREEDY in forms.selected(params):
        keys.append("loss_greedy_gain")
    if FORM_POLICY in forms.selected(params):
        keys.append("loss_policy_ce")
    return dict.fromkeys(keys, 1.0)


def _deliver_generation(sink, gen, checkpoint_snap, state_snap, recorder, staged):
    """One generation's deliveries, the record last."""
    state_pair.deliver(sink, checkpoint_snap, state_snap, gen)
    recorder.deliver_staged(staged)


@torch.no_grad()
def parameter_norm(model: nn.Module) -> float:
    """The global L2 norm of `model`'s parameters: a slow divergence shows as
    its growth well before anything goes non-finite."""
    return float(torch.stack([p.float().norm() for p in model.parameters()]).norm())


def train_one_generation(models, optimizer, optim_arm, recorder, paths, params, state, ctx):
    """Train one epoch over the window ending at the cursor generation, then
    evaluate, checkpoint and queue the generation's deliveries."""
    gen = state.generation_index
    window = lifecycle.window_dirs(paths, gen, params.window)
    items = dataset.read_items(window)
    timed_print(f"generation {gen}: window {[d.name for d in window]} ({len(items)} pools)")
    device, batcher = ctx["device"], ctx["batcher"]
    t0 = time.time()
    rows_before = state.rows_trained
    optim_arm.train_mode()
    result = train_loop.run_epoch(
        models,
        optimizer,
        batcher,
        items,
        device,
        params,
        seed=gen * 1000003,
        lr_fn=optim_arm.lr_fn,
        rows_trained=state.rows_trained,
        on_batch=functools.partial(progress_line, gen),
    )
    state.rows_trained = result.rows_trained
    state.generation_index = gen + 1
    elapsed = time.time() - t0
    sys.stdout.write("\n")
    optim_arm.eval_mode(
        models, recalibration_batches(batcher, items, params, gen, device), _trunk_forward
    )
    t_eval = time.time()
    record = {
        "epoch": gen,
        "loss": result.loss,
        **{f"loss_{key}": v for key, v in result.terms.items()},
        **{f"grad_norm_{name}": v for name, v in result.grad_norms.items()},
        **{f"param_norm_{name}": parameter_norm(model) for name, model in models.items()},
        "lr": optim_arm.current,
        "elapsed_s": elapsed,
        **optim_arm.metrics(),
        **metrics.pool_stats([it.pool for it in items], ""),
        **eval_record(models, batcher, held_out_items(paths, params), params, device),
    }
    recalls = " ".join(
        f"{name}={record.get(f'{name}_eval_recall_8', float('nan')):.3f}"
        for name in (*models, BASELINE.rstrip("_"))
    )
    timed_print(f"[gen {gen}] pools={state.rows_trained} loss={result.loss:.4f} recall@8 {recalls}")
    check_parameters(models.items(), f"after generation {gen}")
    checkpoint.save(paths, models, optimizer, state, ctx["config"])
    lifecycle.write_train_state(paths, asdict(state))
    staged = recorder.stage_generation(gen, state.rows_trained, record)
    ctx["deliverer"].submit(
        f"generation {gen}",
        functools.partial(
            _deliver_generation,
            ctx["sink"],
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
            "train_s": elapsed,
            "eval_s": time.time() - t_eval,
            "upload_s": finished[-1][1] if finished else 0.0,
        },
        units=state.rows_trained - rows_before,
        nbytes=0,
    )


def run_generations(models, optimizer, recorder, paths, params, state, ctx):
    """The wait->train->advance loop, from the resumed cursor onward."""
    optim_arm = build_optim_arm(recorder, params, optimizer, state.rows_trained)
    while not params_mod.reached(state.rows_trained, params.max_rows):
        wait_for_generation(paths, state.generation_index, ctx["sink"], ctx["data_home"])
        train_one_generation(models, optimizer, optim_arm, recorder, paths, params, state, ctx)
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

    models = build_models(params).to(device)
    n_params = sum(p.numel() for p in models.parameters())
    print(f"Models ({', '.join(models)}): {n_params:,} parameters")
    optimizer = build_optimizer(models, params)
    config = {**asdict(params), **versions()}
    recorder = TrainRecorder(ctx.sink)
    recorder.publish_run(
        ctx.tag,
        asdict(params),
        n_params,
        loss_weights(params),
        {},
    )
    run_ctx = {
        "config": config,
        "sink": ctx.sink,
        "device": device,
        "batcher": dataset.PoolBatcher(),
        "stats": WorkerStats(ctx),
        "deliverer": OutputDeliverer(),
    }
    state_pair.install_seed(paths)
    # After the seed, which may be what puts a rolling checkpoint in place.
    check_resumable(paths, config)
    state: GenerationalState = checkpoint.resume(paths, models, optimizer, device)
    lifecycle.write_train_state(paths, asdict(state))
    run_ctx["data_home"] = data_home.start_for(
        ctx,
        paths,
        params,
        chunk_sidecar=pools.SUFFIX,
        holdout_every=params.holdout_every,
        holdout_chunks=params.holdout_chunks,
    )
    try:
        run_generations(models, optimizer, recorder, paths, params, state, run_ctx)
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped; last completed generation is checkpointed.")
    finally:
        run_ctx["deliverer"].drain()
    return 0
