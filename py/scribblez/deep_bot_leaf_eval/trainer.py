"""The deep_bot_leaf_eval train role: position_eval's consume-and-train loop
(position_eval/trainer.py describes it) for the leaf eval model.

Per generation it waits for self-play to complete the cursor generation,
trains one epoch over the window, fits a win table from the window, evaluates
the model and both win tables (the fitted one and Macondo's) on the held-out
games, checkpoints, and delivers the table, the checkpoint, the cursor and
last the record. There is no export yet: serving is a later step.

The held-out games are leave inference's (leave_inference/holdout.py),
played by the trainer itself at a fixed seed. The rows' format version and
the hash of the leave values they read are recorded in the checkpoint config,
and a checkpoint written under others is refused rather than resumed.
"""

import functools
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from scribblez import params as params_mod
from scribblez.dataset import SlogDataset
from scribblez.deep_bot_leaf_eval.evaluation import evaluate
from scribblez.deep_bot_leaf_eval.model import LAYOUT, LeafEvalModel, LeafEvalShape
from scribblez.ffi import LeaveValueArrays
from scribblez.generational import checkpoint, data_home, lifecycle, state_pair
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import CpuController, default_controls, progress_line
from scribblez.generational.optim import build_optim_arm, build_optimizer
from scribblez.generational.records import TrainRecorder, read_controls
from scribblez.leave_inference import holdout, win_table
from scribblez.leave_inference.trainer import FaceUpGuard
from scribblez.paths import TagPaths
from scribblez.position_eval.train_loop import LossConfig, run_epoch
from scribblez.position_eval.trainer import (
    OutputDeliverer,
    ensure_window,
    recalibration_batches,
    wait_for_generation,
)
from scribblez.train_common import timed_print
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.selfplay_gen import hasty_spec
from scribblez.workloads.worker import WorkerStats, WorkerStopped

TASK = "deep_bot_leaf_eval"
EVAL_BATCH = 1024
FORMAT_KEY = "leaf_eval_format_version"
LEAVE_VALUES_KEY = "leave_values"


def model_shape(params) -> LeafEvalShape:
    return LeafEvalShape(
        trunk=params.trunk,
        join_planes=params.join_planes,
        width=params.width,
        depth=params.depth,
        num_heads=params.num_heads,
    )


def loss_config(params) -> LossConfig:
    """position_eval's loss config, without its placement heads."""
    return LossConfig(
        lambda_wld=params.lambda_wld,
        lambda_sd=params.lambda_sd,
        lambda_next_placement=0.0,
        lambda_win_placement=0.0,
        huber_delta_mean=params.huber_delta_mean,
        huber_delta_std=params.huber_delta_std,
    )


def check_resumable(paths: TagPaths, config: dict):
    """Refuse a rolling checkpoint written under another row format or against
    other leave values."""
    path = paths.rolling_checkpoint
    if not path.exists():
        return
    saved = torch.load(path, map_location="cpu", weights_only=False)["config"]
    for key in (FORMAT_KEY, LEAVE_VALUES_KEY):
        if saved.get(key) != config[key]:
            raise RuntimeError(
                f"{path} has {key}={saved.get(key)}, but this run has {config[key]}; "
                "start a new tag (or check out the code it was trained with)"
            )


def _rows_left(params, state: GenerationalState) -> bool:
    return not params_mod.reached(state.rows_trained, params.max_rows)


def _rel(paths: TagPaths, path: Path) -> str:
    return str(path.relative_to(paths.root))


def _holdout_batches(held: holdout.HoldoutSet, cpu):
    ds = SlogDataset(held.dir, task=TASK, apply_symmetry=False, num_workers=cpu.dataloader_workers)
    return ds.iter_batches(EVAL_BATCH, seed=0)


def fit_win_table(paths: TagPaths, params, gen: int, window: list[Path], tag: str) -> Path:
    """Fit generation `gen`'s win table from the window and write it."""
    samples = win_table.read_samples(sorted(p for d in window for p in d.glob("*.slog")))
    provenance = {
        "tag": tag,
        "generation": gen,
        "policy": hasty_spec(params),
        "games": samples.games,
        "positions": len(samples),
    }
    path = paths.win_table_path(gen)
    win_table.write(path, win_table.fit(samples), provenance)
    return path


def _deliver_generation(sink, paths, gen, checkpoint_snap, state_snap, recorder, staged):
    """One generation's deliveries, the record last."""
    table = paths.win_table_path(gen)
    sink.deliver_output(table, _rel(paths, table), keep=True)
    state_pair.deliver(sink, checkpoint_snap, state_snap, gen)
    recorder.deliver_staged(staged)


def _record(state, gen, result, elapsed, optim_arm) -> dict:
    losses = result.losses
    return {
        "epoch": gen,
        "positions": state.rows_trained,
        "loss": losses["total"],
        "loss_wld": losses["wld"],
        "loss_score_diff": losses["score_diff"],
        "loss_score_diff_mean": losses["score_diff_mean"],
        "loss_score_diff_std": losses["score_diff_std"],
        "wld_acc": result.wld_acc,
        "lr": optim_arm.current,
        **result.grad_norm,
        "elapsed_s": elapsed,
        **optim_arm.metrics(),
    }


def _checkpoint_and_eval(
    model, optimizer, recorder, paths, device, params, state, gen, result, elapsed, window, ctx
):
    """Fit the win table, evaluate, checkpoint, publish the cursor, and queue
    the generation's deliveries. Returns the seconds spent."""
    sys.stdout.write("\n")
    t0 = time.time()
    record = _record(state, gen, result, elapsed, ctx["optim_arm"])
    table = fit_win_table(paths, params, gen, window, ctx["tag"])
    batches = _holdout_batches(ctx["holdout"], ctx["cpu"])
    record.update(evaluate(model, batches, device, table, win_table.MACONDO_WIN_TABLE))
    timed_print(
        f"[gen {gen}] rows={state.rows_trained} loss={record['loss']:.4f} "
        f"logloss model={record['eval_model_logloss']:.4f} "
        f"wintable={record['eval_wintable_logloss']:.4f} "
        f"macondo={record['eval_wintable_macondo_logloss']:.4f}"
    )
    checkpoint.save(paths, model, optimizer, state, ctx["config"])
    lifecycle.write_train_state(paths, asdict(state))
    staged = recorder.stage_generation(gen, state.rows_trained, record)
    ctx["deliverer"].submit(
        f"generation {gen}",
        functools.partial(
            _deliver_generation,
            ctx["sink"],
            paths,
            gen,
            state_pair.snapshot(paths.rolling_checkpoint, gen),
            state_pair.snapshot(paths.train_state_path, gen),
            recorder,
            staged,
        ),
    )
    return time.time() - t0


def train_one_generation(
    model, train_model, optimizer, recorder, paths, device, params, state, ctx
):
    """Train one epoch over the window ending at the cursor generation, then
    evaluate and checkpoint under that generation's index."""
    gen = state.generation_index
    window = lifecycle.window_dirs(paths, gen, params.window)
    ctx["face_up_guard"].check(window)
    cpu, optim_arm = ctx["cpu"], ctx["optim_arm"]
    ds = SlogDataset(window, task=TASK, apply_symmetry=True, num_workers=cpu.dataloader_workers)
    timed_print(
        f"generation {gen}: window {[d.name for d in window]} "
        f"({ds.num_games} games, {ds.num_samples} rows)"
    )
    # drop_last keeps one static shape for the compiled forward.
    batches = ds.iter_batches(
        params.batch_size,
        seed=gen * 1000003,
        turns_per_game=params.turns_per_game,
        epoch_index=gen,
        drop_last=True,
    )
    t0 = time.time()
    rows_before = state.rows_trained
    optim_arm.train_mode()
    result = run_epoch(
        train_model,
        optimizer,
        batches,
        device,
        ctx["loss_cfg"],
        lr_fn=optim_arm.lr_fn,
        rows_trained=state.rows_trained,
        on_batch=functools.partial(progress_line, gen),
        grad_clip=params.grad_clip,
    )
    state.rows_trained = result.rows_trained
    state.generation_index = gen + 1
    elapsed = time.time() - t0
    optim_arm.eval_mode(model, recalibration_batches(ds, params, gen, device))
    eval_seconds = _checkpoint_and_eval(
        model, optimizer, recorder, paths, device, params, state, gen, result, elapsed, window, ctx
    )
    optim_arm.train_mode()
    finished = ctx["deliverer"].collect()
    ctx["stats"].cycle_done(
        {
            "train_s": elapsed,
            "eval_s": eval_seconds,
            "upload_s": finished[-1][1] if finished else 0.0,
        },
        units=state.rows_trained - rows_before,
        nbytes=0,
    )


def run_generational_training(
    model, train_model, optimizer, recorder, paths, device, params, state, ctx
):
    """The wait->train->advance loop, from the resumed cursor onward."""
    ctx["optim_arm"] = build_optim_arm(recorder, params, optimizer, state.rows_trained)
    ctx["cpu"] = CpuController(recorder, ctx["read_controls"])
    ctx["loss_cfg"] = loss_config(params)
    while _rows_left(params, state):
        ctx["cpu"].refresh(state.rows_trained)
        wait_for_generation(paths, state.generation_index, ctx["sink"], ctx["data_home"])
        train_one_generation(
            model, train_model, optimizer, recorder, paths, device, params, state, ctx
        )
        evicted = lifecycle.evict_beyond_window(
            paths, state.generation_index - 1, params.window, keep_unpulled=ctx["data_home"].remote
        )
        if evicted:
            timed_print(f"evicted generations {evicted} (window={params.window})")
    timed_print(f"Stopped at {state.rows_trained} rows (generation {state.generation_index}).")


def run(ctx: WorkerContext) -> int:
    """The train-role entry point."""
    params = ctx.params
    paths = ctx.tag_paths()
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(os.environ.get("SCZ_DEVICE", "cuda"))
    print(f"Tag root: {paths.root}\nDevice: {device}")

    held = holdout.ensure(paths.holdout_dir, params.holdout_games, holdout.SEED, hasty_spec(params))
    timed_print(f"held-out set {held.content_hash} ({len(held.slogs)} files)")

    model = LeafEvalModel(model_shape(params)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    optimizer = build_optimizer(model, params)
    torch.set_float32_matmul_precision("high")
    train_model = torch.compile(model)

    config = {
        **asdict(params),
        FORMAT_KEY: LAYOUT["format_version"],
        LEAVE_VALUES_KEY: LeaveValueArrays().content_hash,
        "holdout_hash": held.content_hash,
    }
    recorder = TrainRecorder(ctx.sink)
    recorder.publish_run(
        ctx.tag,
        asdict(params),
        n_params,
        {"loss_wld": params.lambda_wld, "loss_score_diff": params.lambda_sd},
        default_controls(),
    )
    run_ctx = {
        "tag": ctx.tag,
        "config": config,
        "sink": ctx.sink,
        "read_controls": functools.partial(read_controls, ctx.sink),
        "holdout": held,
        "face_up_guard": FaceUpGuard(),
        "stats": WorkerStats(ctx),
        "deliverer": OutputDeliverer(),
    }

    state_pair.install_seed(paths)
    # After the seed, which may be what puts a rolling checkpoint in place.
    check_resumable(paths, config)
    state = checkpoint.resume(paths, model, optimizer, device)
    ensure_window(paths, ctx.sink, state.generation_index, params.window)
    lifecycle.write_train_state(paths, asdict(state))
    run_ctx["data_home"] = data_home.start_for(ctx, paths, params)
    try:
        run_generational_training(
            model, train_model, optimizer, recorder, paths, device, params, state, run_ctx
        )
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped; last completed epoch is checkpointed.")
    finally:
        run_ctx["deliverer"].drain()
    return 0
