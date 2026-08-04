"""The move-set-eval train role: distill the student over a tag's pair store.

Sibling to the position_eval / max_move_per_lane trainers, but a fixed-corpus
loop rather than a generational consumer: the corpus is the tag's slogs/ pair
store (grown by the generate role, usually paused during training), split at
file level into train and held-out pairs -- the held-out side being the store's
full-sweep pairs, the only ones the A3 gate metrics mean anything on -- and
trained for `train_epochs`
epochs. Each epoch records losses and the held-out top-K recall / Spearman
metrics to the tag's dashboard DB (the Loss tab's curves), publishes a stats
sample (the Stats tab), and saves the rolling checkpoint -- pausing and
restarting the worker resumes at the next epoch. The base learning rate is a
live control (Controls tab), adopted at the next epoch. The generational
consume->train lifecycle (docs/generational_teacher.md) replaces this loop
when it lands.

Runs as the singleton `train` worker of the move_set_eval workload (launched
by the worker entrypoint with SCZ_ROLE=train); scripts/move_set_eval/train.py
remains the headless CLI for ad-hoc runs outside any tag.
"""

import functools
import os
import time
from dataclasses import asdict

import torch

from scribblez.dashboard import db
from scribblez.ffi import set_contingent_features, set_opp_leave_input
from scribblez.generational import checkpoint
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import (
    CONTROL_BASE_LR,
    LrController,
    init_controls,
    progress_line,
)
from scribblez.move_set_eval.dataset import MsetDataset
from scribblez.move_set_eval.eval import eval_slice_line, evaluate
from scribblez.move_set_eval.model import MoveSetEvalModel
from scribblez.move_set_eval.train_loop import LossConfig, run_epoch
from scribblez.train_common import timed_print
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.move_set_eval import SLOGS_DIR, split_pairs
from scribblez.workloads.worker import WorkerStats, WorkerStopped


def load_datasets(paths, params) -> tuple[MsetDataset, MsetDataset]:
    """(train, holdout) datasets from the tag's pair store, split at file level
    (scribblez.workloads.move_set_eval.split_pairs). With no held-out pairs at
    all the metrics run on the training pairs (a smoke check); both datasets
    must agree on the teacher."""
    store = paths.data_dir / SLOGS_DIR
    train_files, holdout_files = split_pairs(store, params.holdout_every)
    if not train_files:
        raise FileNotFoundError(f"no complete .slog/.mset training pairs in {store}")
    train_ds = MsetDataset(mset_files=train_files)
    if not holdout_files:
        timed_print("no held-out pairs; metrics are on-train")
        return train_ds, train_ds
    holdout_ds = MsetDataset(mset_files=holdout_files)
    assert holdout_ds.model_hash == train_ds.model_hash, (
        "train/holdout pairs disagree on the teacher hash"
    )
    return train_ds, holdout_ds


def epochs_left(params, state: GenerationalState) -> bool:
    """The epoch counter rides GenerationalState.generation_index, so the
    rolling checkpoint's resume machinery is reused verbatim."""
    return params.train_epochs == 0 or state.generation_index < params.train_epochs


def train_one_epoch(model, optimizer, conn, paths, device, params, state, ctx):
    """One pass over the training pairs, then held-out metrics, the dashboard
    metric record, the rolling checkpoint, and a stats sample."""
    epoch = state.generation_index
    batches = ctx["train_ds"].iter_batches(params.batch_positions, seed=0, epoch_index=epoch)
    t0 = time.time()
    rows_before = state.rows_trained
    result = run_epoch(
        model,
        optimizer,
        batches,
        device,
        ctx["loss_cfg"],
        lr_fn=ctx["lr_controller"].epoch_lr_fn(state.rows_trained),
        rows_trained=state.rows_trained,
        on_batch=functools.partial(progress_line, epoch),
    )
    state.rows_trained = result.rows_trained
    state.generation_index = epoch + 1
    train_s = time.time() - t0

    t1 = time.time()
    metrics = evaluate(model, ctx["holdout_ds"], device, positions_per_batch=params.batch_positions)
    eval_s = time.time() - t1

    avg = result.losses
    lr_now = db.read_control(conn, CONTROL_BASE_LR, default=params.lr)
    recall = " ".join(f"r@{k}={metrics[f'recall@{k}']:.3f}" for k in (1, 3, 5))
    timed_print(
        f"[epoch {epoch}] rows={state.rows_trained} loss={avg['total']:.4f} "
        f"{recall} spearman={metrics['spearman']:.3f} "
        f"regret@1={metrics['regret@1']:.4f} (incumbent r@1="
        f"{metrics['recall@1_baseline']:.3f} regret@1={metrics['regret@1_baseline']:.4f}) "
        f"lr={lr_now:.2e} {train_s:.1f}s"
    )
    # recall@K / Spearman (and their incumbent baselines, flat reference
    # lines) land on the Loss tab's Accuracy panel, which plots every *_acc
    # series; the regrets have their own Training-tab figure (plots.py).
    record = {
        "epoch": epoch,
        "positions": state.rows_trained,
        "loss": avg["total"],
        "loss_wld": avg["wld"],
        "loss_score_diff": avg["score_diff"],
        "spearman_acc": metrics["spearman"],
        "spearman_baseline_acc": metrics["spearman_baseline"],
        "lr": lr_now,
        "elapsed_s": train_s,
    }
    for k in (1, 3, 5):
        record[f"recall{k}_acc"] = metrics[f"recall@{k}"]
        record[f"recall{k}_baseline_acc"] = metrics[f"recall@{k}_baseline"]
        record[f"regret{k}"] = metrics[f"regret@{k}"]
        record[f"regret{k}_baseline"] = metrics[f"regret@{k}_baseline"]
    db.write_metrics(conn, epoch, record)
    checkpoint.save(paths, model, optimizer, state, ctx["config"])
    ctx["stats"].cycle_done(
        {"train_s": train_s, "eval_s": eval_s},
        units=state.rows_trained - rows_before,
        nbytes=0,
    )


def run(ctx: WorkerContext) -> int:
    """The train-role runner (invoked by the worker entrypoint)."""
    params = ctx.params
    paths = ctx.tag_paths()
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(os.environ.get("SCZ_DEVICE", "cuda"))
    print(f"Tag root: {paths.root}\nDevice: {device}")

    set_contingent_features(params.contingent_features)
    train_ds, holdout_ds = load_datasets(paths, params)
    # The board input arm must carry the opponent-leave block iff the targets
    # were generated under the open-leaves condition.
    if train_ds.open_leaves:
        set_opp_leave_input(True)
    print(
        f"train: {train_ds.num_positions} positions / {train_ds.num_candidates} candidates; "
        f"eval: {holdout_ds.num_positions} positions / {holdout_ds.num_candidates} candidates "
        f"(open_leaves={train_ds.open_leaves})"
    )
    print(eval_slice_line(holdout_ds))

    model = MoveSetEvalModel(
        spatial_planes=train_ds.spatial_planes,
        scalar_size=train_ds.scalar_size,
        trunk_channels=params.trunk_channels,
        num_blocks=params.num_blocks,
        num_heads=params.num_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )

    conn = db.connect(paths.dashboard_db)
    db.write_meta(conn, ctx.tag, asdict(params), n_params)
    # Coefficients of each loss term in the optimized total (WLD has weight 1),
    # so the dashboard can stack the weighted contributions.
    db.write_loss_weights(conn, {"loss_wld": 1.0, "loss_score_diff": params.lambda_sd})
    init_controls(conn, params.lr)

    run_ctx = {
        "config": asdict(params),
        "train_ds": train_ds,
        "holdout_ds": holdout_ds,
        "loss_cfg": LossConfig.from_args(params),
        "lr_controller": LrController(conn, params.lr),
        "stats": WorkerStats(ctx),
    }

    state = checkpoint.resume(paths, model, optimizer, device)
    try:
        while epochs_left(params, state):
            train_one_epoch(model, optimizer, conn, paths, device, params, state, run_ctx)
        timed_print(
            f"Training complete: {state.generation_index} epochs, "
            f"{state.rows_trained} rows. Pause the worker (raising train_epochs "
            "needs a new tag; params are frozen)."
        )
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped; last completed epoch is checkpointed.")
    return 0
