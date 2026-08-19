"""The position-evaluation train role: an open-ended consume->train loop.

The trainer is a pure consumer. It never generates games: the generation
scheduler (scribblez/generational/scheduler.py) fills generation directories
from generator workers' staged chunks, and the trainer

  1. waits until its cursor generation is complete on disk,
  2. trains one epoch over a sliding window of the most recent complete
     generations (turns_per_game turns sampled per game), checkpointing under
     the generation's index,
  3. advances, evicting generations older than the window, and publishes its
     cursor (train_state.json) so the scheduler can pace the generator fleet.

One epoch per generation keeps data reuse low by construction -- a game is
trained on `window` times over its residency, once per generation it is part
of the window -- and makes epoch and generation the same clock. Everything
else is keyed on cumulative rows trained (the rows-clock): the dashboard
x-axis and the restart cursor. A single rolling model.pt holds resume state,
so pausing and restarting the worker continues exactly where it left off;
SIGTERM stops at the next batch boundary, losing at most the current
(uncheckpointed) generation. The optimizer and its learning-rate policy are
the run's `optimizer` arm (generational/optim.py); the CPU thread pools
(DataLoader workers, torch intra-op threads) are live controls in the per-tag
dashboard.db, adopted at the next generation.

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
from scribblez.ffi import (
    get_input_shapes,
    session_input_arm,
    set_contingent_features,
    set_opp_leave_input,
)
from scribblez.generational import checkpoint, lifecycle
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import CpuController, init_controls, progress_line
from scribblez.generational.optim import build_optim_arm, build_optimizer
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
    model, optimizer, conn, paths, device, params, state, gen, result, elapsed, optim_arm, ctx
):
    """Export ONNX, record the trained generation's metrics + eval (keyed on
    the generation index `gen`, with the rows-clock stored as `positions`),
    save the rolling checkpoint, and publish the cursor.

    ONNX export runs first because the eval step below is what makes this
    generation visible to the dashboard's Positions tab (it writes the row
    `db.read_position_eval_generations` reads): a dashboard request landing
    between that write and a later export would see the generation listed but
    find no ONNX file yet, so the placement-overlay prediction would come back
    null."""
    sys.stdout.write("\n")
    avg = result.losses
    lr_now = optim_arm.current
    ci = gen
    timed_print(
        f"[gen {gen}] rows={state.rows_trained} loss={avg['total']:.4f} "
        f"wld_acc={result.wld_acc:.4f} lr={lr_now:.2e} {elapsed:.1f}s"
    )
    record = {
        "epoch": ci,
        "positions": state.rows_trained,
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
        # Whatever else the arm wants on the record -- for a schedule-free run
        # the averaging weight, which is what anneals there in place of the rate.
        **optim_arm.metrics(),
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
    export_onnx(
        model,
        paths.onnx_path(ci),
        ctx["spatial_planes"],
        ctx["scalar_size"],
        contingent_features=params.contingent_features,
        opp_leave_input=params.face_up_leaves,
    )
    db.write_metrics(conn, ci, record)
    if ctx["position_eval"] is not None:
        eval_position_eval(model, ctx["position_eval"], device, conn, ci, state.rows_trained)
    checkpoint.save(paths, model, optimizer, state, ctx["config"])
    _publish_train_state(paths, state)
    return time.time() - t_eval


def train_one_generation(
    model, optimizer, conn, paths, device, params, state, loss_cfg, optim_arm, cpu, ctx
):
    """Train one epoch over the window ending at the cursor generation, then
    checkpoint under that generation's index and advance the cursor.

    The optimizer arm is switched to training weights around the epoch and to
    deployable ones around the checkpoint, so everything the checkpoint step
    reads -- quality eval, ONNX export, saved state -- sees the same weights a
    schedule-free run would deploy (a no-op under the WSD arm)."""
    gen = state.generation_index
    window = lifecycle.window_dirs(paths, gen, params.window)
    ds = SlogDataset(
        window, post_move=True, apply_symmetry=True, num_workers=cpu.dataloader_workers
    )
    timed_print(
        f"generation {gen}: window {[d.name for d in window]} "
        f"({ds.num_games} games, {ds.num_samples} eligible rows)"
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
    optim_arm.train_mode()
    result = run_epoch(
        model,
        optimizer,
        batches,
        device,
        loss_cfg,
        lr_fn=optim_arm.lr_fn,
        rows_trained=state.rows_trained,
        on_batch=functools.partial(progress_line, gen),
    )
    state.rows_trained = result.rows_trained
    state.generation_index = gen + 1
    elapsed = time.time() - t0
    optim_arm.eval_mode()
    eval_seconds = _checkpoint_and_eval(
        model,
        optimizer,
        conn,
        paths,
        device,
        params,
        state,
        gen,
        result,
        elapsed,
        optim_arm,
        ctx,
    )
    optim_arm.train_mode()
    if ctx["stats"] is not None:
        ctx["stats"].cycle_done(
            {"train_s": elapsed, "eval_s": eval_seconds},
            units=state.rows_trained - rows_before,
            nbytes=0,
        )


def run_generational_training(model, optimizer, conn, paths, device, params, state, ctx):
    """The wait->train->advance loop, from the resumed cursor onward."""
    loss_cfg = LossConfig.from_args(params)
    optim_arm = build_optim_arm(conn, params, optimizer, state.rows_trained)
    cpu = CpuController(conn)
    while _rows_left(params, state):
        cpu.refresh(state.rows_trained)
        wait_for_generation(paths, state.generation_index)
        train_one_generation(
            model, optimizer, conn, paths, device, params, state, loss_cfg, optim_arm, cpu, ctx
        )
        evicted = lifecycle.evict_beyond_window(paths, state.generation_index - 1, params.window)
        if evicted:
            timed_print(f"evicted generations {evicted} (window={params.window})")
    timed_print(f"Stopped at {state.rows_trained} rows (generation {state.generation_index}).")


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------


def load_position_eval(spatial_planes: int) -> dict | None:
    """Build the frozen position-evaluation input batch once (or None if the
    dataset is empty / the lexicon is unavailable). At each checkpoint the model is
    run over it and the predictions are written to the dashboard DB, where the
    Positions tab pairs them with the Monte-Carlo ground truth."""
    dataset = str(position_eval_analysis.DEFAULT_DATASET)
    try:
        names, inputs = position_eval_analysis.load_inputs(dataset, session_input_arm())
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


def load_position_eval_quality(spatial_planes: int) -> dict | None:
    """Build the large-dataset quality-eval batch and its Monte-Carlo ground truth once
    (or None if the dataset or its ground truth is unavailable). At each checkpoint
    the model is run over it and aggregate quality scalars are recorded for the
    Loss tab."""
    dataset = str(position_eval_analysis.LARGE_DATASET)
    try:
        names, inputs = position_eval_analysis.load_inputs(dataset, session_input_arm())
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
    # Pick the experiment arms before any engine call: they are baked into the
    # process-wide FFI session, whose reported input shapes -- and therefore the
    # model, the ONNX export, and the eval batches -- all follow them. A
    # face-up-leaves run trains on the opponent-leave input block, matching the
    # information condition its self-play games are generated under.
    set_contingent_features(params.contingent_features)
    set_opp_leave_input(params.face_up_leaves)

    paths = ctx.tag_paths()
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(os.environ.get("SCZ_DEVICE", "cuda"))
    print(f"Tag root: {paths.root}\nDevice: {device}")

    in_shapes = {s.name: s.dims for s in get_input_shapes()}
    spatial_planes = in_shapes["input_spatial"][0]
    scalar_size = in_shapes["input_scalar"][0]
    model = PositionEvalModel(
        spatial_planes=spatial_planes,
        scalar_size=scalar_size,
        num_blocks=params.num_blocks,
        trunk_channels=params.trunk_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    optimizer = build_optimizer(model, params)

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
    init_controls(conn)

    run_ctx = {
        "config": asdict(params),
        "spatial_planes": spatial_planes,
        "scalar_size": scalar_size,
        "position_eval": load_position_eval(spatial_planes),
        "position_eval_quality": load_position_eval_quality(spatial_planes),
        "stats": WorkerStats(ctx),
    }

    state = checkpoint.resume(paths, model, optimizer, device)
    _publish_train_state(paths, state)
    try:
        run_generational_training(model, optimizer, conn, paths, device, params, state, run_ctx)
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped; last completed epoch is checkpointed.")
    return 0
