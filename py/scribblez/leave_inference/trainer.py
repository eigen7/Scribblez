"""The leave_inference train role (docs/plans/subtree_bot_v0.md, "Workload 1:
leave_inference"): position_eval's consume-and-train loop for the leave
inference model, with the win table as its side product.

Per generation it waits for self-play to complete the cursor generation,
trains one epoch over the window, evaluates on the held-out games, refits the
win table from the window, exports the model for the engine (the tilted head
only), checkpoints, and delivers the table, the export, the checkpoint, the
cursor and last the record.

At startup it builds the leave table (in the engine, through the FFI) and
plays the held-out games, both on its own machine. The table's identity, the
hash of the leave values it was built from, is recorded in the checkpoint
config with the leave inference format version, and a checkpoint trained
against other leave values or another format is refused rather than resumed.

The model, its optimizer and its epoch are the unit's (unit.py), which the
deep_bot trainer drives too; this module is the loop around one.
"""

import functools
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from scribblez import params as params_mod
from scribblez.generational import checkpoint, data_home, lifecycle, state_pair
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import CpuController, default_controls
from scribblez.generational.records import TrainRecorder, read_controls
from scribblez.leave_inference import forgone, holdout, win_table
from scribblez.leave_inference.onnx_export import FORMAT_KEY, TABLE_KEY
from scribblez.leave_inference.unit import LeaveInferenceUnit
from scribblez.paths import TagPaths
from scribblez.position_eval.trainer import OutputDeliverer, ensure_window, wait_for_generation
from scribblez.train_common import timed_print
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.selfplay_gen import hasty_spec
from scribblez.workloads.worker import WorkerStats, WorkerStopped


def _rel(paths: TagPaths, path: Path) -> str:
    return str(path.relative_to(paths.root))


def check_resumable(paths: TagPaths, config: dict):
    """Refuse a rolling checkpoint written under another leave inference
    format or against other leave values."""
    path = paths.rolling_checkpoint
    if not path.exists():
        return
    saved = torch.load(path, map_location="cpu", weights_only=False)["config"]
    for key in (FORMAT_KEY, TABLE_KEY):
        if saved.get(key) != config[key]:
            raise RuntimeError(
                f"{path} has {key}={saved.get(key)}, but this run has {config[key]}; "
                "start a new tag (or check out the code it was trained with)"
            )


def _rows_left(params, state: GenerationalState) -> bool:
    return not params_mod.reached(state.rows_trained, params.max_rows)


def refit_win_table(paths: TagPaths, params, gen: int, window: list[Path], ctx) -> dict:
    """Fit generation `gen`'s win table from the window, write it, and return
    its held-out log loss beside Macondo's table's."""
    samples = win_table.read_samples(sorted(p for d in window for p in d.glob("*.slog")))
    grid = win_table.fit(samples)
    provenance = {
        "tag": ctx["tag"],
        "generation": gen,
        "policy": hasty_spec(params),
        "games": samples.games,
        "positions": len(samples),
    }
    path = paths.win_table_path(gen)
    win_table.write(path, grid, provenance)
    return {
        "eval_wintable_logloss": win_table.log_loss(path, ctx["holdout_win_samples"]),
        "eval_wintable_logloss_macondo": ctx["macondo_logloss"],
    }


def export_generation(paths: TagPaths, params, gen: int, unit: LeaveInferenceUnit, tag: str):
    """Export generation `gen`'s model for the engine, if its head is served."""
    provenance = {"tag": tag, "generation": gen, "policy": hasty_spec(params)}
    unit.export(paths.onnx_path(gen), provenance)


def _deliver_generation(sink, paths, gen, checkpoint_snap, state_snap, recorder, staged):
    """One generation's deliveries, the record last."""
    win = paths.win_table_path(gen)
    sink.deliver_output(win, _rel(paths, win), keep=True)
    model = paths.onnx_path(gen)
    if model.exists():
        sink.deliver_output(model, _rel(paths, model))
    state_pair.deliver(sink, checkpoint_snap, state_snap, gen)
    recorder.deliver_staged(staged)


def _checkpoint_and_eval(unit, recorder, paths, params, state, gen, result, elapsed, window, ctx):
    """Evaluate, refit the win table, export the model, checkpoint, publish
    the cursor, and queue the generation's deliveries. Returns the seconds
    spent."""
    sys.stdout.write("\n")
    t0 = time.time()
    record = {
        "epoch": gen,
        "loss": result.loss,
        "train_nll_vs_uniform": result.loss_vs_uniform,
        "valid_frac": result.valid_frac,
        "grad_norm_mean": result.grad_norm,
        "elapsed_s": elapsed,
        "lr": unit.optim_arm.current,
        **unit.optim_arm.metrics(),
    }
    record.update(unit.evaluate(ctx["holdout"].dir, ctx["cpu"]))
    record.update(refit_win_table(paths, params, gen, window, ctx))
    timed_print(
        f"[gen {gen}] rows={state.rows_trained} loss={result.loss:.4f} "
        f"eval_nll={record['eval_nll']:.4f} vs_uniform={record['eval_nll_vs_uniform']:+.4f} "
        f"hidden_gap={record['eval_hidden_gap']:+.4f} "
        f"wintable={record['eval_wintable_logloss']:.4f} "
        f"(macondo {record['eval_wintable_logloss_macondo']:.4f})"
    )
    export_generation(paths, params, gen, unit, ctx["tag"])
    checkpoint.save(paths, unit.model, unit.optimizer, state, ctx["config"])
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


def train_one_generation(unit, recorder, paths, params, state, ctx):
    """Train one epoch over the window ending at the cursor generation, then
    evaluate and checkpoint under that generation's index."""
    gen = state.generation_index
    window = lifecycle.window_dirs(paths, gen, params.window)
    rows_before = state.rows_trained
    result, elapsed = unit.train(window, gen, state.rows_trained, ctx["cpu"])
    state.rows_trained = result.rows_trained
    state.generation_index = gen + 1
    eval_seconds = _checkpoint_and_eval(
        unit, recorder, paths, params, state, gen, result, elapsed, window, ctx
    )
    unit.resume_training()
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


def run_generational_training(unit, recorder, paths, params, state, ctx):
    """The wait->train->advance loop, from the resumed cursor onward."""
    unit.start(recorder, state.rows_trained)
    ctx["cpu"] = CpuController(recorder, ctx["read_controls"])
    while _rows_left(params, state):
        ctx["cpu"].refresh(state.rows_trained)
        wait_for_generation(paths, state.generation_index, ctx["sink"], ctx["data_home"])
        train_one_generation(unit, recorder, paths, params, state, ctx)
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

    unit = LeaveInferenceUnit(params, device)
    held = holdout.ensure(paths.holdout_dir, params.holdout_games, holdout.SEED, hasty_spec(params))
    timed_print(
        f"leave values {unit.table.klv_hash}; held-out set {held.content_hash} "
        f"({len(held.slogs)} files)"
    )
    print(f"Model: {unit.n_params:,} parameters")

    config = {**asdict(params), **unit.versions(), "holdout_hash": held.content_hash}
    recorder = TrainRecorder(ctx.sink)
    recorder.publish_run(ctx.tag, asdict(params), unit.n_params, {"loss": 1.0}, default_controls())

    held_win_samples = win_table.read_samples(held.slogs)
    run_ctx = {
        "tag": ctx.tag,
        "config": config,
        "sink": ctx.sink,
        "read_controls": functools.partial(read_controls, ctx.sink),
        "holdout": held,
        "holdout_win_samples": held_win_samples,
        # The fixed reference each generation's table is read against.
        "macondo_logloss": win_table.log_loss(
            win_table.MACONDO_WIN_TABLE, held_win_samples, macondo_format=True
        ),
        "stats": WorkerStats(ctx),
        "deliverer": OutputDeliverer(),
    }

    state_pair.install_seed(paths)
    # After the seed, which may be what puts a rolling checkpoint in place.
    check_resumable(paths, config)
    state = checkpoint.resume(paths, unit.model, unit.optimizer, device)
    ensure_window(paths, ctx.sink, state.generation_index, params.window)
    lifecycle.write_train_state(paths, asdict(state))
    run_ctx["data_home"] = data_home.start_for(ctx, paths, params, chunk_sidecars=(forgone.SUFFIX,))
    try:
        run_generational_training(unit, recorder, paths, params, state, run_ctx)
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped; last completed epoch is checkpointed.")
    finally:
        run_ctx["deliverer"].drain()
    return 0
