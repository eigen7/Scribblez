"""Train role of the max_move_per_lane workload: an open-ended consume-and-train loop.

It follows the same generation lifecycle as the position-eval trainer
(position_eval/trainer.py), reusing its window and resume helpers. For each
generation it waits for self-play to finish it, trains one epoch over a sliding
window of recent generations, then checkpoints, advances the cursor and evicts
generations that left the window. Labels are recomputed by replaying each .slog
game and enumerating legal moves, so every turn, endgame included, can be a
training row.

This model is a representation-learning probe, so there is no held-out eval or
ONNX export. Its metrics are the per-lane training accuracies, plus predictions
on a fixed GCG position set for the dashboard's Lane-analysis tab. Both leave
through the worker's sink as the generation's record (generational/records.py);
the trainer never writes dashboard.db itself.

Runs as the workload's singleton `train` worker, or directly via
scripts/max_move_per_lane/train.py.
"""

import functools
import os
import sys
import time
from dataclasses import asdict

import torch

from scribblez import lane_analysis
from scribblez.dataset import SlogDataset
from scribblez.ffi import get_max_move_per_lane_input_shapes
from scribblez.generational import checkpoint, lifecycle
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import (
    CpuController,
    WsdLrController,
    WsdSchedule,
    default_controls,
    progress_line,
)
from scribblez.generational.records import TrainRecorder, read_controls
from scribblez.lexical_tool.modules import LexiconArgs
from scribblez.max_move_per_lane.model import MaxMovePerLaneModel
from scribblez.max_move_per_lane.train_loop import LossConfig, run_epoch
from scribblez.position_eval.trainer import ensure_window, restore_from_sink, wait_for_generation
from scribblez.train_common import timed_print
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.worker import WorkerStats, WorkerStopped


def _rows_left(params, state: GenerationalState) -> bool:
    return params.max_rows == 0 or state.rows_trained < params.max_rows


def _checkpoint_and_eval(
    model, optimizer, recorder, paths, device, params, state, gen, result, elapsed, lr_now, ctx
):
    """Run the lane-analysis eval, save the checkpoint and train state, then
    deliver the generation's record. Returns the seconds this took.

    The record goes last because its arrival is what tells the dashboard the
    generation is complete; everything it refers to must already be delivered."""
    sys.stdout.write("\n")
    avg = result.losses
    ci = gen
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
    t_eval = time.time()
    preds = None
    if ctx["lane_eval"] is not None:
        preds = {"lane_pred": eval_lane_analysis(model, ctx["lane_eval"], device)}
    checkpoint.save(paths, model, optimizer, state, ctx["config"])
    ctx["sink"].deliver_output(paths.rolling_checkpoint, "checkpoints/model.pt", keep=True)
    lifecycle.write_train_state(paths, asdict(state))
    ctx["sink"].deliver_output(paths.train_state_path, "train_state.json", keep=True)
    recorder.commit_generation(ci, state.rows_trained, record, preds)
    return time.time() - t_eval


def train_one_generation(
    model, optimizer, recorder, paths, device, params, state, loss_cfg, lr_controller, cpu, ctx
):
    """Train one epoch over the window ending at the cursor generation, then
    checkpoint and advance the cursor."""
    gen = state.generation_index
    window = lifecycle.window_dirs(paths, gen, params.window)
    ds = SlogDataset(
        window, task="max_move_per_lane", apply_symmetry=True, num_workers=cpu.dataloader_workers
    )
    timed_print(
        f"generation {gen}: window {[d.name for d in window]} "
        f"({ds.num_games} games, {ds.num_samples} rows)"
    )
    # Seed the shuffle and the per-game turn choice with the generation index. A
    # game stays in the window for `window` generations, and this makes each of
    # those passes shuffle differently and sample different turns.
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
        lr_fn=lr_controller.lr_fn,
        rows_trained=state.rows_trained,
        on_batch=functools.partial(progress_line, gen),
    )
    state.rows_trained = result.rows_trained
    state.generation_index = gen + 1
    elapsed = time.time() - t0
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
        elapsed,
        lr_controller.current,
        ctx,
    )
    if ctx["stats"] is not None:
        ctx["stats"].cycle_done(
            {"train_s": elapsed, "eval_s": eval_seconds},
            units=state.rows_trained - rows_before,
            nbytes=0,
        )


def run_generational_training(model, optimizer, recorder, paths, device, params, state, ctx):
    """Wait, train, advance and evict, from the resumed cursor until max_rows."""
    loss_cfg = LossConfig.from_args(params)
    lr_controller = WsdLrController(recorder, WsdSchedule.from_params(params), state.rows_trained)
    cpu = CpuController(recorder, ctx["read_controls"])
    while _rows_left(params, state):
        cpu.refresh(state.rows_trained)
        wait_for_generation(paths, state.generation_index, ctx["sink"])
        train_one_generation(
            model,
            optimizer,
            recorder,
            paths,
            device,
            params,
            state,
            loss_cfg,
            lr_controller,
            cpu,
            ctx,
        )
        evicted = lifecycle.evict_beyond_window(paths, state.generation_index - 1, params.window)
        if evicted:
            timed_print(f"evicted generations {evicted} (window={params.window})")
    timed_print(f"Stopped at {state.rows_trained} rows (generation {state.generation_index}).")


def load_lane_eval(params, spatial_planes: int) -> dict | None:
    """Encode the lane-analysis positions once, for eval at every checkpoint.

    Returns None, which disables the eval, if it is turned off, the dataset is
    empty, or it cannot be loaded (for example, the lexicon is missing)."""
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


def eval_lane_analysis(model, lane_eval: dict, device) -> dict:
    """Per-(position, lane) predictions on the lane-analysis set, for the generation's record."""
    model.eval()
    return lane_analysis.predict(model, lane_eval["inputs"], lane_eval["spatial_planes"], device)


def run(ctx: WorkerContext) -> int:
    """Entry point of the train role, called by the worker and by the train.py CLI."""
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

    recorder = TrainRecorder(ctx.sink)
    # Each loss term's weight in compute_loss's total, so the dashboard can stack
    # the weighted contributions.
    recorder.publish_run(
        ctx.tag,
        asdict(params),
        n_params,
        {
            "loss_score_pdf": 1.0,
            "loss_score_cdf": params.lambda_cdf,
            "loss_move": params.lambda_occ,
            "loss_has_move": params.lambda_has_move,
        },
        default_controls(),
    )

    run_ctx = {
        "config": asdict(params),
        "sink": ctx.sink,
        "read_controls": functools.partial(read_controls, ctx.sink),
        "lane_eval": load_lane_eval(params, spatial_planes),
        "stats": WorkerStats(ctx),
    }

    restore_from_sink(paths, ctx.sink)
    state = checkpoint.resume(paths, model, optimizer, device)
    ensure_window(paths, ctx.sink, state.generation_index, params.window)
    lifecycle.write_train_state(paths, asdict(state))
    try:
        run_generational_training(model, optimizer, recorder, paths, device, params, state, run_ctx)
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped; last completed epoch is checkpointed.")
    return 0
