"""The leave_inference train role (docs/plans/deep_bot_v0.md, "Workload 1:
leave_inference"): position_eval's consume-and-train loop for the leave
inference model, with the win table as its side product.

Per generation it waits for self-play to complete the cursor generation,
trains one epoch over the window, evaluates on the held-out games, refits the
win table from the window, checkpoints, and delivers the table, the
checkpoint, the cursor and last the record.

At startup it builds the leave table (in the engine, through the FFI) and
plays the held-out games, both on its own machine. The table's identity, the
hash of the leave values it was built from, is recorded in the checkpoint
config with the leave inference format version, and a checkpoint trained
against other leave values or another format is refused rather than resumed.
"""

import functools
import itertools
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from scribblez import params as params_mod
from scribblez.dataset import SlogDataset
from scribblez.ffi import LeaveValueArrays, format_layout, struct_dtype, task_input_shapes
from scribblez.generational import checkpoint, data_home, lifecycle, state_pair
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import CpuController, default_controls, progress_line
from scribblez.generational.optim import build_optim_arm, build_optimizer
from scribblez.generational.records import TrainRecorder, read_controls
from scribblez.leave_inference import holdout, loss, win_table
from scribblez.leave_inference.model import LeaveInferenceModel
from scribblez.leave_inference.train_loop import LAYOUT, evaluate, run_epoch
from scribblez.paths import TagPaths
from scribblez.position_eval.trainer import OutputDeliverer, ensure_window, wait_for_generation
from scribblez.train_common import timed_print
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.selfplay_gen import hasty_spec
from scribblez.workloads.worker import WorkerStats, WorkerStopped

TASK = "leave_inference"
EVAL_BATCH = 1024
# Batches the schedule-free arm recomputes BatchNorm statistics over, as in
# position_eval/trainer.py.
BN_RECALIBRATION_BATCHES = 32
FACE_UP_FLAG = format_layout()["constants"]["slog"]["flag_face_up_leaves"]
# The checkpoint-config entries a resume must match.
FORMAT_KEY = "leave_inference_format_version"
TABLE_KEY = "leave_values"


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


class FaceUpGuard:
    """Refuses face-up-leaves corpora: their leaves are public, so there is
    nothing to infer. Each file is checked once."""

    def __init__(self):
        self._checked: set[Path] = set()
        self._header = struct_dtype("SlogFileHeader")

    def check(self, dirs: list[Path]):
        for path in (p for d in dirs for p in d.glob("*.slog")):
            if path in self._checked:
                continue
            flags = int(np.fromfile(path, self._header, count=1)[0]["flags"])
            if flags & FACE_UP_FLAG:
                raise RuntimeError(
                    f"{path} is a face-up-leaves corpus; leave inference needs hidden leaves"
                )
            self._checked.add(path)


def _rows_left(params, state: GenerationalState) -> bool:
    return not params_mod.reached(state.rows_trained, params.max_rows)


def _recalibration_batches(ds, params, gen, device):
    batches = ds.iter_batches(params.batch_size, seed=gen * 1000003 + 1, epoch_index=gen)
    for batch in itertools.islice(batches, BN_RECALIBRATION_BATCHES):
        yield batch["input_spatial"].to(device), batch["input_scalar"].to(device)


def _holdout_batches(held: holdout.HoldoutSet, cpu):
    ds = SlogDataset(
        held.dir,
        task=TASK,
        post_move=False,
        apply_symmetry=False,
        num_workers=cpu.dataloader_workers,
    )
    return ds.iter_batches(EVAL_BATCH, seed=0)


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


def _deliver_generation(sink, paths, gen, checkpoint_snap, state_snap, recorder, staged):
    """One generation's deliveries, the record last."""
    win = paths.win_table_path(gen)
    sink.deliver_output(win, _rel(paths, win), keep=True)
    state_pair.deliver(sink, checkpoint_snap, state_snap, gen)
    recorder.deliver_staged(staged)


def _checkpoint_and_eval(
    model,
    optimizer,
    recorder,
    paths,
    device,
    params,
    state,
    gen,
    result,
    window,
    optim_arm,
    cpu,
    ctx,
):
    """Evaluate, refit the win table, checkpoint, publish the cursor, and
    queue the generation's deliveries. Returns the seconds spent."""
    sys.stdout.write("\n")
    t0 = time.time()
    record = {
        "epoch": gen,
        "positions": state.rows_trained,
        "loss": result.loss,
        "train_nll_vs_uniform": result.loss_vs_uniform,
        "valid_frac": result.valid_frac,
        "grad_norm_mean": result.grad_norm,
        "elapsed_s": ctx["epoch_seconds"],
        "lr": optim_arm.current,
        **optim_arm.metrics(),
    }
    record.update(evaluate(model, ctx["table"], _holdout_batches(ctx["holdout"], cpu), device))
    record.update(refit_win_table(paths, params, gen, window, ctx))
    timed_print(
        f"[gen {gen}] rows={state.rows_trained} loss={result.loss:.4f} "
        f"eval_nll={record['eval_nll']:.4f} vs_uniform={record['eval_nll_vs_uniform']:+.4f} "
        f"wintable={record['eval_wintable_logloss']:.4f} "
        f"(macondo {record['eval_wintable_logloss_macondo']:.4f})"
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
    model, train_model, optimizer, recorder, paths, device, params, state, optim_arm, cpu, ctx
):
    """Train one epoch over the window ending at the cursor generation, then
    evaluate and checkpoint under that generation's index."""
    gen = state.generation_index
    window = lifecycle.window_dirs(paths, gen, params.window)
    ctx["face_up_guard"].check(window)
    ds = SlogDataset(
        window, task=TASK, post_move=False, apply_symmetry=True, num_workers=cpu.dataloader_workers
    )
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
        ctx["table"],
        batches,
        device,
        lr_fn=optim_arm.lr_fn,
        rows_trained=state.rows_trained,
        grad_clip=params.grad_clip,
        on_batch=functools.partial(progress_line, gen),
    )
    state.rows_trained = result.rows_trained
    state.generation_index = gen + 1
    elapsed = time.time() - t0
    ctx["epoch_seconds"] = elapsed
    optim_arm.eval_mode(model, _recalibration_batches(ds, params, gen, device))
    eval_seconds = _checkpoint_and_eval(
        model,
        optimizer,
        recorder,
        paths,
        device,
        params,
        state,
        gen,
        result,
        window,
        optim_arm,
        cpu,
        ctx,
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
    optim_arm = build_optim_arm(recorder, params, optimizer, state.rows_trained)
    cpu = CpuController(recorder, ctx["read_controls"])
    while _rows_left(params, state):
        cpu.refresh(state.rows_trained)
        wait_for_generation(paths, state.generation_index, ctx["sink"], ctx["data_home"])
        train_one_generation(
            model,
            train_model,
            optimizer,
            recorder,
            paths,
            device,
            params,
            state,
            optim_arm,
            cpu,
            ctx,
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

    table = loss.TableTensors(LeaveValueArrays(), device)
    held = holdout.ensure(
        paths.holdout_dir, params.holdout_games, params.holdout_seed, hasty_spec(params)
    )
    timed_print(
        f"leave values {table.klv_hash}; held-out set {held.content_hash} ({len(held.slogs)} files)"
    )

    in_shapes = {s.name: s.dims for s in task_input_shapes(TASK)}
    model = LeaveInferenceModel(
        spatial_planes=in_shapes["input_spatial"][0],
        scalar_size=in_shapes["input_scalar"][0],
        trunk_channels=params.trunk_channels,
        num_blocks=params.num_blocks,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    optimizer = build_optimizer(model, params)
    # TF32 for the trunk; the tilted-prior pass pins its own precision.
    torch.set_float32_matmul_precision("high")
    train_model = torch.compile(model)

    config = {
        **asdict(params),
        FORMAT_KEY: LAYOUT["format_version"],
        TABLE_KEY: table.klv_hash,
        "holdout_hash": held.content_hash,
    }
    check_resumable(paths, config)

    recorder = TrainRecorder(ctx.sink)
    recorder.publish_run(ctx.tag, asdict(params), n_params, {"loss": 1.0}, default_controls())

    held_win_samples = win_table.read_samples(held.slogs)
    run_ctx = {
        "tag": ctx.tag,
        "config": config,
        "sink": ctx.sink,
        "read_controls": functools.partial(read_controls, ctx.sink),
        "table": table,
        "holdout": held,
        "holdout_win_samples": held_win_samples,
        # The fixed reference each generation's table is read against.
        "macondo_logloss": win_table.log_loss(
            win_table.MACONDO_WIN_TABLE, held_win_samples, macondo_format=True
        ),
        "face_up_guard": FaceUpGuard(),
        "stats": WorkerStats(ctx),
        "deliverer": OutputDeliverer(),
    }

    state_pair.install_seed(paths)
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
