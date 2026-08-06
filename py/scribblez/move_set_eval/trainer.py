"""The move-set-eval train role: distill the student over a tag's pair store.

Sibling to the position_eval / max_move_per_lane trainers, but a growing-corpus
loop rather than a generational consumer: the corpus is the tag's slogs/ pair
store, split at file level into train and held-out pairs -- the held-out side
being the store's full-sweep pairs, the only ones the A3 gate metrics mean
anything on.

The generate role writes that store while this runs, so the loop is built to
keep up with it rather than to snapshot it. It waits for the store to be worth
training on (`warmup_pairs`, and a full-sweep pair to read metrics against),
absorbs whatever arrived before each pass, and spends its `train_epochs` budget
only on passes over a corpus that has stopped growing -- so the budget always
buys passes over the whole corpus, however much of it existed when the worker
started. Both sides of the split grow, so the held-out slice freezes when the
generator stops and every budgeted epoch is scored against the same one. A tag
with a `target_pairs` generation size therefore runs to completion unattended:
start a worker of each type and both stop on their own.

Each pass records losses and the held-out top-K recall / Spearman metrics to
the tag's dashboard DB (the Loss
tab's curves), publishes a stats sample (the Stats tab), and saves the rolling
checkpoint -- pausing and restarting the worker resumes at the next pass. The
base learning rate is a live control (Controls tab), adopted at the next
pass. The generational consume->train lifecycle
(docs/generational_teacher.md) replaces this loop when it lands.

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
from scribblez.ffi import set_contingent_features
from scribblez.generational import checkpoint
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import (
    CONTROL_BASE_LR,
    LrController,
    init_controls,
    progress_line,
)
from scribblez.move_set_eval.dataset import MsetDataset, adopt_information_condition
from scribblez.move_set_eval.eval import eval_slice_line, evaluate
from scribblez.move_set_eval.model import MoveSetEvalModel
from scribblez.move_set_eval.targets import complete_pairs, partition_full_sweep
from scribblez.move_set_eval.train_loop import LossConfig, run_epoch
from scribblez.train_common import timed_print
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.move_set_eval import SLOGS_DIR, split_pairs
from scribblez.workloads.pair_store import count_pairs
from scribblez.workloads.worker import WorkerStats, WorkerStopped

POLL_SECONDS = 30


def store_is_ready(store, params) -> tuple[bool, str]:
    """Whether the pair store holds enough to start training, and why not.

    Two conditions, both about not locking the run into a corpus the generator
    has barely started: enough pairs to be worth a pass, and -- when the tag
    sweeps at all -- at least one swept pair, so the holdout is the full-sweep
    slice the A3 gate is read on rather than the stratified fallback that would
    otherwise be frozen in for good.
    """
    pairs = complete_pairs(store) if store.is_dir() else []
    if len(pairs) < params.warmup_pairs:
        return False, f"{len(pairs)}/{params.warmup_pairs} pairs"
    if params.sweep_every:
        _, swept = partition_full_sweep(pairs)
        if not swept:
            return False, f"{len(pairs)} pairs, no full-sweep pair yet"
    return True, ""


def wait_for_store(ctx, store, params):
    """Block until the store is ready to train on (store_is_ready), reporting
    progress. A worker started alongside its generator lands here rather than
    dying on an empty store or snapshotting a corpus minutes old."""
    while True:
        ready, why = store_is_ready(store, params)
        if ready:
            return
        timed_print(f"waiting for the pair store: {why}")
        time.sleep(POLL_SECONDS)  # SIGTERM raises WorkerStopped through this


def load_datasets(paths, params) -> tuple[MsetDataset, MsetDataset]:
    """(train, holdout) datasets from the tag's pair store, split at file level
    (scribblez.workloads.move_set_eval.split_pairs). With no held-out pairs at
    all the metrics run on the training pairs (a smoke check); both datasets
    must agree on the teacher. The corpus's information condition is adopted
    here because it must precede the first dataset."""
    store = paths.data_dir / SLOGS_DIR
    train_files, holdout_files = split_pairs(store, params.holdout_every)
    if not train_files:
        raise FileNotFoundError(f"no complete .slog/.mset training pairs in {store}")
    adopt_information_condition(train_files)
    train_ds = MsetDataset(mset_files=train_files)
    if not holdout_files:
        timed_print("no held-out pairs; metrics are on-train")
        return train_ds, train_ds
    holdout_ds = MsetDataset(mset_files=holdout_files)
    assert holdout_ds.model_hash == train_ds.model_hash, (
        "train/holdout pairs disagree on the teacher hash"
    )
    return train_ds, holdout_ds


def absorb_new_pairs(paths, params, train_ds: MsetDataset, holdout_ds: MsetDataset) -> int:
    """Ingest every pair delivered since the last pass, into whichever side the
    file-level split assigns it to, and return how many positions arrived.

    Both sides grow, so once the generator stops the holdout stops with it --
    which is what makes every budgeted epoch's metrics comparable without any
    freezing step. A holdout that is the training set (the no-held-out-pairs
    smoke case) is grown once, not twice.
    """
    train_files, holdout_files = split_pairs(paths.data_dir / SLOGS_DIR, params.holdout_every)
    added = train_ds.absorb(sorted(set(train_files) - set(train_ds.files)))
    if holdout_ds is not train_ds:
        added += holdout_ds.absorb(sorted(set(holdout_files) - set(holdout_ds.files)))
    return added


def corpus_is_final(paths, params, absorbed: int) -> bool:
    """Whether the pair store has stopped growing -- the point from which an
    epoch is a pass over the whole corpus and may spend the epoch budget.

    A tag with a `target_pairs` generation size answers this outright, which is
    immune to how the generator's pace happens to line up with an epoch's
    length. Without one there is no declared end, so the trainer falls back to
    the observation that nothing new arrived this pass.
    """
    if params.target_pairs:
        return count_pairs(paths.data_dir / SLOGS_DIR, ".mset") >= params.target_pairs
    return absorbed == 0


def epochs_left(params, state: GenerationalState) -> bool:
    """Whether the epoch budget has passes left. It is spent by
    GenerationalState.settled_epochs rather than by the pass counter, so passes
    taken while the generator is still delivering do not consume it."""
    return params.train_epochs == 0 or state.settled_epochs < params.train_epochs


def train_one_epoch(model, optimizer, conn, paths, device, params, state, ctx, settled: bool):
    """One pass over the training pairs, then held-out metrics, the dashboard
    metric record, the rolling checkpoint, and a stats sample. `settled` says
    whether the corpus was final for this pass, which is what decides if the
    pass spends the epoch budget."""
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
    state.settled_epochs += int(settled)
    train_s = time.time() - t0

    t1 = time.time()
    metrics = evaluate(model, ctx["holdout_ds"], device, positions_per_batch=params.batch_positions)
    eval_s = time.time() - t1

    avg = result.losses
    lr_now = db.read_control(conn, CONTROL_BASE_LR, default=params.lr)
    recall = " ".join(f"r@{k}={metrics[f'recall@{k}']:.3f}" for k in (1, 3, 5))
    # Whether the pass counted, so the log says which of the two phases the run
    # is in without the reader having to infer it from the corpus size.
    budget = (
        f"{state.settled_epochs}/{params.train_epochs}"
        if settled
        else f"corpus still growing, {ctx['train_ds'].num_positions} positions"
    )
    timed_print(
        f"[pass {epoch}] rows={state.rows_trained} loss={avg['total']:.4f} "
        f"{recall} spearman={metrics['spearman']:.3f} "
        f"regret@1={metrics['regret@1']:.4f} (incumbent r@1="
        f"{metrics['recall@1_baseline']:.3f} regret@1={metrics['regret@1_baseline']:.4f}) "
        f"lr={lr_now:.2e} {train_s:.1f}s [{budget}]"
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
    wait_for_store(ctx, paths.data_dir / SLOGS_DIR, params)
    train_ds, holdout_ds = load_datasets(paths, params)
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
            # Take up whatever the generator delivered during the last pass
            # before deciding whether this one is over a finished corpus.
            absorbed = absorb_new_pairs(paths, params, train_ds, holdout_ds)
            settled = corpus_is_final(paths, params, absorbed)
            train_one_epoch(model, optimizer, conn, paths, device, params, state, run_ctx, settled)
        timed_print(
            f"Training complete: {state.settled_epochs} epochs over the finished corpus "
            f"({state.generation_index} passes, {state.rows_trained} rows, "
            f"{train_ds.num_positions} positions). Pause the worker (raising train_epochs "
            "needs a new tag; params are frozen)."
        )
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped; last completed epoch is checkpointed.")
    return 0
