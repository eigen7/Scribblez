"""The max-move-per-lane train role: an open-ended consume->train loop.

Sibling to the position-evaluation trainer (scribblez/position_eval/trainer.py).
It runs the same consumer lifecycle -- wait for the generation scheduler to
complete the cursor generation, train a few epochs over a sliding window of
the most recent generations (reusing each position, and with turns_per_game
sampling several turns per game), then advance, evict, and publish the cursor
-- but for the "highest-scoring move per lane" representation probe. The
per-lane labels are recomputed by replaying each .slog game and enumerating
legal moves at the position, so every turn of a game (including endgame turns)
is a training row.

There is no held-out probe/calibration eval or ONNX export -- this is a
representation-learning probe. Eval is the per-lane train accuracy, recorded
alongside the losses, plus a per-checkpoint lane-analysis pass over a frozen
GCG dataset for the dashboard's Lane-analysis tab.

Runs as the singleton `train` worker of the max_move_per_lane workload, or
directly via the scripts/max_move_per_lane/train.py CLI.
"""

import functools
import os
import sys
import time
from dataclasses import asdict

import torch

from scribblez import lane_analysis
from scribblez.dashboard import db
from scribblez.dataset import SlogDataset
from scribblez.ffi import get_max_move_per_lane_input_shapes
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
from scribblez.max_move_per_lane.model import MaxMovePerLaneModel
from scribblez.max_move_per_lane.train_loop import LossConfig, run_epoch
from scribblez.position_eval.trainer import wait_for_generation
from scribblez.train_common import timed_print
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.worker import WorkerStats, WorkerStopped


def _rows_left(params, state: GenerationalState) -> bool:
    return params.max_rows == 0 or state.rows_trained < params.max_rows


def _checkpoint_and_eval(
    model, optimizer, conn, paths, device, params, state, result, elapsed, ctx
):
    """Record this epoch's metrics + lane accuracies (keyed on the checkpoint
    index, with the rows-clock stored as `positions`), run the lane-analysis eval,
    save the rolling checkpoint, and publish the cursor."""
    sys.stdout.write("\n")
    avg = result.losses
    ci = state.checkpoint_index
    base_lr = db.read_control(conn, CONTROL_BASE_LR, default=params.lr)
    lr_now = effective_lr(base_lr, state.rows_trained, params.warmup_rows)
    timed_print(
        f"[gen {state.generation_index} epoch {state.epoch_in_generation}] ckpt {ci} "
        f"rows={state.rows_trained} loss={avg['total']:.4f} move_acc={result.accs['move_acc']:.4f} "
        f"score_acc={result.accs['score_acc']:.4f} lr={lr_now:.2e} {elapsed:.1f}s"
    )
    record = {
        "epoch": ci,
        "positions": state.rows_trained,
        "generation": state.generation_index,
        "loss": avg["total"],
        "loss_score_pdf": avg["score_pdf"],
        "loss_score_cdf": avg["score_cdf"],
        "loss_move": avg["move"],
        "loss_has_move": avg["has_move"],
        **result.accs,
        "lr": lr_now,
        "elapsed_s": elapsed,
    }
    db.write_metrics(conn, ci, record)
    t_eval = time.time()
    if ctx["lane_eval"] is not None:
        eval_lane_analysis(model, ctx["lane_eval"], device, conn, ci, state.rows_trained)
    checkpoint.save(paths, model, optimizer, state, ctx["config"])
    lifecycle.write_train_state(paths, asdict(state))
    return time.time() - t_eval


def epochs_this_generation(params, ds) -> int:
    """The epoch count to run over the current window: epochs_per_generation when
    set explicitly, otherwise derived from reuse_per_position and the window's
    measured average turns per game."""
    if params.epochs_per_generation > 0:
        return params.epochs_per_generation
    avg_turns = ds.num_samples / max(ds.num_games, 1)
    return epochs_for_reuse(
        params.reuse_per_position, params.window, params.turns_per_game, avg_turns
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
        window, task="max_move_per_lane", apply_symmetry=True, num_workers=cpu.dataloader_workers
    )
    epochs = epochs_this_generation(params, ds)
    avg_turns = ds.num_samples / max(ds.num_games, 1)
    reuse = effective_reuse(epochs, params.window, params.turns_per_game, avg_turns)
    timed_print(
        f"generation {state.generation_index}: window {[d.name for d in window]} "
        f"({ds.num_games} games, {ds.num_samples} rows, {avg_turns:.1f} turns/game); "
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
        lifecycle.write_train_state(paths, asdict(state))
    timed_print(f"Stopped at {state.rows_trained} rows (generation {state.generation_index}).")


def load_lane_eval(params, spatial_planes: int) -> dict | None:
    """Build the frozen lane-analysis input batch once (or None if disabled / the
    dataset is empty / the lexicon is unavailable). At each checkpoint the model is
    run over it and the predictions are written to the dashboard DB."""
    if params.no_lane_eval:
        return None
    dataset = params.lane_eval_dataset or str(lane_analysis.DEFAULT_DATASET)
    try:
        names, inputs = lane_analysis.load_inputs(dataset)
    except Exception as e:  # missing lexicon / unreadable dataset
        timed_print(f"lane-analysis eval disabled: {e}")
        return None
    if not names:
        timed_print(f"lane-analysis eval disabled: no GCG positions in {dataset}")
        return None
    timed_print(f"lane-analysis eval: {len(names)} positions from {dataset}")
    return {"inputs": inputs, "spatial_planes": spatial_planes}


def eval_lane_analysis(model, lane_eval: dict, device, conn, checkpoint_index: int, positions: int):
    """Run the model over the frozen lane-analysis set and store this checkpoint's
    per-(position, lane) predictions for the dashboard."""
    model.eval()
    preds = lane_analysis.predict(model, lane_eval["inputs"], lane_eval["spatial_planes"], device)
    db.write_lane_preds(conn, checkpoint_index, positions, preds)


def run(ctx: WorkerContext) -> int:
    """The train-role runner (invoked by the worker entrypoint; also the
    substance of the scripts/max_move_per_lane/train.py CLI)."""
    params = ctx.params
    paths = ctx.tag_paths()
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(os.environ.get("SCZ_DEVICE", "cuda"))
    print(f"Tag root: {paths.root}\nDevice: {device}")

    in_shapes = {s.name: s.dims for s in get_max_move_per_lane_input_shapes()}
    spatial_planes = in_shapes["input_spatial"][0]
    scalar_size = in_shapes["input_scalar"][0]
    lex = LexiconArgs(module=params.lexicon_module)
    lexicon_module = lex.build(channels=params.trunk_channels)
    if lexicon_module is not None:
        print(f"Lexicon module: {lex.module}")
    lane_ffn_mult = lex.lane_ffn_mult(lexicon_module is not None)
    if lane_ffn_mult is not None:
        print(f"Lane FFN width multiple shrunk to {lane_ffn_mult} (replace mode).")
    model = MaxMovePerLaneModel(
        spatial_planes=spatial_planes,
        scalar_size=scalar_size,
        trunk_channels=params.trunk_channels,
        num_blocks=params.num_blocks,
        lane_layers=params.lane_layers,
        lane_heads=params.lane_heads,
        ffn_mult=params.ffn_mult,
        n_rack_tokens=params.rack_tokens,
        lexicon_module=lexicon_module,
        lane_ffn_mult=lane_ffn_mult,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )

    conn = db.connect(paths.dashboard_db)
    db.write_meta(conn, ctx.tag, asdict(params), n_params)
    # Coefficients of each loss term in compute_loss's total (PDF has weight 1),
    # so the dashboard can stack the weighted contributions.
    db.write_loss_weights(
        conn,
        {
            "loss_score_pdf": 1.0,
            "loss_score_cdf": params.lambda_cdf,
            "loss_move": params.lambda_occ,
            "loss_has_move": params.lambda_has_move,
        },
    )
    init_controls(conn, params.lr)

    run_ctx = {
        "config": asdict(params),
        "lane_eval": load_lane_eval(params, spatial_planes),
        "stats": WorkerStats(ctx),
    }

    state = checkpoint.resume(paths, model, optimizer, device)
    lifecycle.write_train_state(paths, asdict(state))
    try:
        run_generational_training(model, optimizer, conn, paths, device, params, state, run_ctx)
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped; last completed epoch is checkpointed.")
    return 0
