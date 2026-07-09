"""The position-evaluation train role: an open-ended consume->train loop.

The trainer is a pure consumer. It never generates games: the generation
scheduler (scribblez/generational/scheduler.py) fills generation directories
from generator workers' staged chunks, and the trainer

  1. waits until its cursor generation is complete on disk,
  2. trains a few epochs over a sliding window of the most recent complete
     generations -- reusing each position across epochs and, with
     turns_per_game, sampling several turns per game,
  3. advances, evicting generations older than the window, and publishes its
     cursor (train_state.json) so the scheduler can pace the generator fleet.

Everything is keyed on cumulative rows trained (the rows-clock): the dashboard
x-axis, the warmup learning rate, and the restart cursor. A single rolling
model.pt holds resume state, so pausing and restarting the worker continues
exactly where it left off; SIGTERM stops at the next batch boundary, losing at
most the current (uncheckpointed) epoch. The base learning rate and the CPU
thread pools (DataLoader workers, torch intra-op threads) are live controls in
the per-tag dashboard.db, adopted at the next epoch / generation.

Runs as the singleton `train` worker of the position_eval workload (launched
by the worker entrypoint with SCZ_ROLE=train), or directly via the
scripts/position_eval/train.py CLI for headless debugging.
"""

import functools
import os
import sys
import time
from dataclasses import asdict

import torch

from scribblez.dashboard import db
from scribblez.dataset import SlogDataset
from scribblez.ffi import get_input_shapes, set_contingent_features
from scribblez.generational import checkpoint, lifecycle
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import (
    CONTROL_BASE_LR,
    CpuController,
    LrController,
    init_controls,
    progress_line,
)
from scribblez.generational.lr import effective_lr
from scribblez.generational.reuse import effective_reuse, epochs_for_reuse
from scribblez.lexical_tool.modules import LexiconArgs
from scribblez.paths import TagPaths
from scribblez.position_eval import analysis as position_eval_analysis
from scribblez.position_eval.model import PositionEvalModel
from scribblez.position_eval.onnx_export import export_onnx
from scribblez.position_eval.train_loop import LossConfig, run_epoch
from scribblez.train_common import timed_print
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.worker import WorkerStats, WorkerStopped

# How often the trainer re-checks the manifests while waiting for its cursor
# generation to complete.
POLL_SECONDS = 5


def _publish_train_state(paths: TagPaths, state: GenerationalState):
    lifecycle.write_train_state(paths, asdict(state))


def wait_for_generation(paths: TagPaths, index: int):
    """Block until generation `index` is complete on disk. The GPU idling here
    is the generation-is-the-bottleneck signal, visible in the Stats tab."""
    announced = False
    while not lifecycle.is_complete(paths.generation_dir(index)):
        if not announced:
            timed_print(f"waiting for generation {index} to complete ...")
            announced = True
        time.sleep(POLL_SECONDS)
    if announced:
        timed_print(f"generation {index} is complete")


def _rows_left(params, state: GenerationalState) -> bool:
    return params.max_rows == 0 or state.rows_trained < params.max_rows


def _checkpoint_and_eval(
    model, optimizer, conn, paths, device, params, state, result, elapsed, ctx
):
    """Record this epoch's metrics + eval (keyed on the checkpoint index, with the
    rows-clock stored as `positions`), save the rolling checkpoint, export ONNX,
    and publish the cursor."""
    sys.stdout.write("\n")
    avg = result.losses
    ci = state.checkpoint_index
    base_lr = db.read_control(conn, CONTROL_BASE_LR, default=params.lr)
    lr_now = effective_lr(base_lr, state.rows_trained, params.warmup_rows)
    timed_print(
        f"[gen {state.generation_index} epoch {state.epoch_in_generation}] ckpt {ci} "
        f"rows={state.rows_trained} loss={avg['total']:.4f} wld_acc={result.wld_acc:.4f} "
        f"lr={lr_now:.2e} {elapsed:.1f}s"
    )
    record = {
        "epoch": ci,
        "positions": state.rows_trained,
        "generation": state.generation_index,
        "loss": avg["total"],
        "loss_wld": avg["wld"],
        "loss_score_diff": avg["score_diff"],
        "loss_score_diff_mean": avg["score_diff_mean"],
        "loss_score_diff_std": avg["score_diff_std"],
        "loss_opp_next_placement": avg["opp_next_placement"],
        "loss_self_next_placement": avg["self_next_placement"],
        "loss_opp_win_placement": avg["opp_win_placement"],
        "loss_self_win_placement": avg["self_win_placement"],
        "wld_acc": result.wld_acc,
        "lr": lr_now,
        "elapsed_s": elapsed,
    }
    # Aggregate model-vs-Monte-Carlo quality over the large held-out dataset,
    # folded into the same metrics record so the Loss tab plots it alongside the
    # training curves.
    t_eval = time.time()
    if ctx["position_eval_quality"] is not None:
        record.update(eval_position_eval_quality(model, ctx["position_eval_quality"], device))
        timed_print(
            f"  quality: win_mae={record['eval_win_mae']:.4f} "
            f"sd_mean_mae={record['eval_sd_mean_mae']:.1f}"
        )
    db.write_metrics(conn, ci, record)
    if ctx["position_eval"] is not None:
        eval_position_eval(model, ctx["position_eval"], device, conn, ci, state.rows_trained)
    checkpoint.save(paths, model, optimizer, state, ctx["config"])
    _publish_train_state(paths, state)
    onnx_path = paths.onnx_path(ci)
    export_onnx(
        model,
        onnx_path,
        ctx["spatial_planes"],
        ctx["scalar_size"],
        contingent_features=params.contingent_features,
    )
    return time.time() - t_eval


def epochs_this_generation(params, ds) -> int:
    """The epoch count to run over the current window: epochs_per_generation when
    set explicitly, otherwise derived from reuse_per_position and the window's
    measured average eligible turns per game."""
    if params.epochs_per_generation > 0:
        return params.epochs_per_generation
    avg_eligible = ds.num_samples / max(ds.num_games, 1)
    return epochs_for_reuse(
        params.reuse_per_position, params.window, params.turns_per_game, avg_eligible
    )


def train_one_generation(
    model, optimizer, conn, paths, device, params, state, loss_cfg, lr_controller, cpu, ctx
) -> int:
    """Train `epochs` passes over the current window, resuming at
    state.epoch_in_generation and checkpointing after each epoch. Returns the
    target epoch count (so the caller can tell a max_rows early stop from a
    completed generation)."""
    window = lifecycle.window_dirs(paths, state.generation_index, params.window)
    ds = SlogDataset(
        window, post_move=True, apply_symmetry=True, num_workers=cpu.dataloader_workers
    )
    epochs = epochs_this_generation(params, ds)
    avg_eligible = ds.num_samples / max(ds.num_games, 1)
    reuse = effective_reuse(epochs, params.window, params.turns_per_game, avg_eligible)
    timed_print(
        f"generation {state.generation_index}: window {[d.name for d in window]} "
        f"({ds.num_games} games, {ds.num_samples} rows, {avg_eligible:.1f} eligible turns/game); "
        f"{epochs} epochs/gen -> ~{reuse:.2f} passes/position (turns/game {params.turns_per_game})"
    )
    while state.epoch_in_generation < epochs and _rows_left(params, state):
        e = state.epoch_in_generation
        # The global epoch index (monotonic across the whole run) seeds the epoch
        # shuffle and the per-game turn rotation, so every pass shuffles differently
        # and draws distinct turns. checkpoint_index counts completed epochs, so it
        # is exactly that index and survives restarts.
        global_epoch = state.checkpoint_index
        batches = ds.iter_batches(
            params.batch_size,
            seed=global_epoch * 1000003,
            turns_per_game=params.turns_per_game,
            epoch_index=global_epoch,
        )
        t0 = time.time()
        rows_before = state.rows_trained
        result = run_epoch(
            model,
            optimizer,
            batches,
            device,
            loss_cfg,
            lr_fn=lr_controller.epoch_lr_fn(state.rows_trained),
            rows_trained=state.rows_trained,
            on_batch=functools.partial(progress_line, state.generation_index, e),
        )
        state.rows_trained = result.rows_trained
        state.epoch_in_generation = e + 1
        state.checkpoint_index += 1
        elapsed = time.time() - t0
        eval_seconds = _checkpoint_and_eval(
            model, optimizer, conn, paths, device, params, state, result, elapsed, ctx
        )
        if ctx["stats"] is not None:
            ctx["stats"].cycle_done(
                {"train_s": elapsed, "eval_s": eval_seconds},
                units=state.rows_trained - rows_before,
                nbytes=0,
            )
    return epochs


def run_generational_training(model, optimizer, conn, paths, device, params, state, ctx):
    """The wait->train->advance loop, from the resumed cursor onward."""
    loss_cfg = LossConfig.from_args(params)
    lr_controller = LrController(conn, params.lr, params.warmup_rows)
    cpu = CpuController(conn)
    while _rows_left(params, state):
        cpu.refresh(state.rows_trained)
        wait_for_generation(paths, state.generation_index)
        epochs = train_one_generation(
            model, optimizer, conn, paths, device, params, state, loss_cfg, lr_controller, cpu, ctx
        )
        if state.epoch_in_generation < epochs:
            break  # stopped mid-generation by max_rows; resume here next run
        evicted = lifecycle.evict_beyond_window(paths, state.generation_index, params.window)
        if evicted:
            timed_print(f"evicted generations {evicted} (window={params.window})")
        state.generation_index += 1
        state.epoch_in_generation = 0
        checkpoint.save(paths, model, optimizer, state, ctx["config"])
        _publish_train_state(paths, state)
    timed_print(f"Stopped at {state.rows_trained} rows (generation {state.generation_index}).")


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------


def load_position_eval(params, spatial_planes: int) -> dict | None:
    """Build the frozen position-evaluation input batch once (or None if disabled / the
    dataset is empty / the lexicon is unavailable). At each checkpoint the model is
    run over it and the predictions are written to the dashboard DB, where the
    Positions tab pairs them with the Monte-Carlo ground truth."""
    if params.no_eval:
        return None
    dataset = params.eval_dataset or str(position_eval_analysis.DEFAULT_DATASET)
    try:
        names, inputs = position_eval_analysis.load_inputs(dataset)
    except Exception as e:  # missing lexicon / unreadable dataset
        timed_print(f"position-evaluation eval disabled: {e}")
        return None
    if not names:
        timed_print(f"position-evaluation eval disabled: no GCG positions in {dataset}")
        return None
    timed_print(f"position-evaluation eval: {len(names)} positions from {dataset}")
    return {"inputs": inputs, "spatial_planes": spatial_planes}


def eval_position_eval(model, position_eval: dict, device, conn, generation: int, positions: int):
    """Run the model over the frozen position-evaluation set and store this checkpoint's
    per-position predictions (WLD probabilities + score-delta mean/std)."""
    model.eval()
    preds = position_eval_analysis.predict(
        model, position_eval["inputs"], position_eval["spatial_planes"], device
    )
    db.write_position_eval_preds(conn, generation, positions, preds)


def load_position_eval_quality(params, spatial_planes: int) -> dict | None:
    """Build the large-dataset quality-eval batch and its Monte-Carlo ground truth once
    (or None if disabled / the dataset or its ground truth is unavailable). At each
    checkpoint the model is run over it and aggregate quality scalars are recorded for
    the Loss tab."""
    if params.no_quality:
        return None
    dataset = params.quality_dataset or str(position_eval_analysis.LARGE_DATASET)
    try:
        names, inputs = position_eval_analysis.load_inputs(dataset)
        gt = position_eval_analysis.load_ground_truth(dataset, names)
    except Exception as e:  # missing lexicon / dataset / ground truth
        timed_print(f"position-evaluation quality eval disabled: {e}")
        return None
    if not names:
        timed_print(f"position-evaluation quality eval disabled: no positions in {dataset}")
        return None
    timed_print(f"position-evaluation quality eval: {len(names)} positions from {dataset}")
    return {"inputs": inputs, "spatial_planes": spatial_planes, "gt": gt}


def eval_position_eval_quality(model, quality_eval: dict, device) -> dict:
    """Run the model over the large quality set and return the aggregate
    model-vs-Monte-Carlo quality scalars (win-equity/WLD + score-delta mean/std MAE)."""
    model.eval()
    preds = position_eval_analysis.predict(
        model, quality_eval["inputs"], quality_eval["spatial_planes"], device
    )
    return position_eval_analysis.quality_metrics(preds, quality_eval["gt"])


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def run(ctx: WorkerContext) -> int:
    """The train-role runner (invoked by the worker entrypoint; also the
    substance of the scripts/position_eval/train.py CLI)."""
    params = ctx.params
    # Pick the experiment arm before any engine call: it is baked into the
    # process-wide FFI session, whose reported input shapes -- and therefore the
    # model, the ONNX export, and the eval batches -- all follow it.
    set_contingent_features(params.contingent_features)

    paths = ctx.tag_paths()
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(os.environ.get("SCZ_DEVICE", "cuda"))
    print(f"Tag root: {paths.root}\nDevice: {device}")

    in_shapes = {s.name: s.dims for s in get_input_shapes()}
    spatial_planes = in_shapes["input_spatial"][0]
    scalar_size = in_shapes["input_scalar"][0]
    lex = LexiconArgs(module=params.lexicon_module)
    lexicon_module = lex.build(channels=params.trunk_channels)
    if lexicon_module is not None:
        print(f"Lexicon module: {lex.module}")
    model = PositionEvalModel(
        spatial_planes=spatial_planes,
        scalar_size=scalar_size,
        num_blocks=params.num_blocks,
        trunk_channels=params.trunk_channels,
        lexicon_module=lexicon_module,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )

    conn = db.connect(paths.dashboard_db)
    db.write_meta(conn, ctx.tag, asdict(params), n_params)
    db.write_loss_weights(
        conn,
        {
            "loss_wld": 1.0,
            "loss_score_diff": params.lambda_sd,
            "loss_opp_next_placement": params.lambda_next_placement,
            "loss_self_next_placement": params.lambda_next_placement,
            "loss_opp_win_placement": params.lambda_win_placement,
            "loss_self_win_placement": params.lambda_win_placement,
        },
    )
    init_controls(conn, params.lr)

    run_ctx = {
        "config": asdict(params),
        "spatial_planes": spatial_planes,
        "scalar_size": scalar_size,
        "position_eval": load_position_eval(params, spatial_planes),
        "position_eval_quality": load_position_eval_quality(params, spatial_planes),
        "stats": WorkerStats(ctx),
    }

    state = checkpoint.resume(paths, model, optimizer, device)
    _publish_train_state(paths, state)
    try:
        run_generational_training(model, optimizer, conn, paths, device, params, state, run_ctx)
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped; last completed epoch is checkpointed.")
    return 0
