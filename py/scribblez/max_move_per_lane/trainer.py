"""The max-move-per-lane train role: an open-ended consume->train loop.

Sibling to the position-evaluation trainer (scribblez/position_eval/trainer.py).
It runs the same consumer lifecycle -- wait for the generation scheduler to
complete the cursor generation, train one epoch over a sliding window of the
most recent generations (turns_per_game turns sampled per game), then advance,
evict, and publish the cursor -- but for the "highest-scoring move per lane"
representation probe. The
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
    model, optimizer, conn, paths, device, params, state, gen, result, elapsed, ctx
):
    """Record the trained generation's metrics + lane accuracies (keyed on the
    generation index `gen`, with the rows-clock stored as `positions`), run the
    lane-analysis eval, save the rolling checkpoint, and publish the cursor."""
    sys.stdout.write("\n")
    avg = result.losses
    ci = gen
    base_lr = db.read_control(conn, CONTROL_BASE_LR, default=params.lr)
    lr_now = effective_lr(base_lr, state.rows_trained, params.warmup_rows)
    timed_print(
        f"[gen {gen}] rows={state.rows_trained} loss={avg['total']:.4f} "
        f"move_acc={result.accs['move_acc']:.4f} "
        f"score_acc={result.accs['score_acc']:.4f} lr={lr_now:.2e} {elapsed:.1f}s"
    )
    record = {
        "epoch": ci,
        "positions": state.rows_trained,
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


def train_one_generation(
    model, optimizer, conn, paths, device, params, state, loss_cfg, lr_controller, cpu, ctx
):
    """Train one epoch over the window ending at the cursor generation, then
    checkpoint under that generation's index and advance the cursor."""
    gen = state.generation_index
    window = lifecycle.window_dirs(paths, gen, params.window)
    ds = SlogDataset(
        window, task="max_move_per_lane", apply_symmetry=True, num_workers=cpu.dataloader_workers
    )
    timed_print(
        f"generation {gen}: window {[d.name for d in window]} "
        f"({ds.num_games} games, {ds.num_samples} rows)"
    )
    # The generation index seeds the shuffle and the per-game turn rotation, so
    # each of the `window` passes a game gets over its residency shuffles
    # differently and draws distinct turns.
    batches = ds.iter_batches(
        params.batch_size,
        seed=gen * 1000003,
        turns_per_game=params.turns_per_game,
        epoch_index=gen,
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
        on_batch=functools.partial(progress_line, gen),
    )
    state.rows_trained = result.rows_trained
    state.generation_index = gen + 1
    elapsed = time.time() - t0
    eval_seconds = _checkpoint_and_eval(
        model, optimizer, conn, paths, device, params, state, gen, result, elapsed, ctx
    )
    if ctx["stats"] is not None:
        ctx["stats"].cycle_done(
            {"train_s": elapsed, "eval_s": eval_seconds},
            units=state.rows_trained - rows_before,
            nbytes=0,
        )


def run_generational_training(model, optimizer, conn, paths, device, params, state, ctx):
    """The wait->train->advance loop, from the resumed cursor onward."""
    loss_cfg = LossConfig.from_args(params)
    lr_controller = LrController(conn, params.lr, params.warmup_rows)
    cpu = CpuController(conn)
    while _rows_left(params, state):
        cpu.refresh(state.rows_trained)
        wait_for_generation(paths, state.generation_index)
        train_one_generation(
            model, optimizer, conn, paths, device, params, state, loss_cfg, lr_controller, cpu, ctx
        )
        evicted = lifecycle.evict_beyond_window(paths, state.generation_index - 1, params.window)
        if evicted:
            timed_print(f"evicted generations {evicted} (window={params.window})")
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


def eval_lane_analysis(model, lane_eval: dict, device, conn, generation: int, positions: int):
    """Run the model over the frozen lane-analysis set and store this checkpoint's
    per-(position, lane) predictions for the dashboard."""
    model.eval()
    preds = lane_analysis.predict(model, lane_eval["inputs"], lane_eval["spatial_planes"], device)
    db.write_lane_preds(conn, generation, positions, preds)


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
