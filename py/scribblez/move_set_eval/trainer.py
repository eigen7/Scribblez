"""The move-set-eval train role: distill the student over a tag's pair store.

Sibling to the position_eval / max_move_per_lane trainers, but a growing-corpus
loop rather than a generational consumer: the corpus is the tag's slogs/ pair
store, split at file level into train and held-out pairs -- the held-out side
being the store's full-sweep pairs, the only ones the A3 gate metrics mean
anything on.

The generate role writes that store while this runs, so the loop keeps pace
with a corpus being written underneath it instead of snapshotting one -- which
is what lets a tag with a worker of each type run to completion unattended.
store_is_ready, absorb_new_pairs and CorpusClock below are that pacing, and
carry its reasoning.

Each pass records losses and the held-out top-K recall / Spearman metrics to
the tag's dashboard DB (the Loss tab's curves), publishes a stats sample (the
Stats tab), and saves the rolling checkpoint -- pausing and restarting the
worker resumes at the next pass. The base learning rate is a live control
(Controls tab), adopted at the next pass. The generational consume->train
lifecycle (docs/generational_teacher.md) replaces this loop when it lands.

Runs as the singleton `train` worker of the move_set_eval workload (launched
by the worker entrypoint with SCZ_ROLE=train); scripts/move_set_eval/train.py
remains the headless CLI for ad-hoc runs outside any tag.
"""

import functools
import os
import time
from dataclasses import asdict, dataclass

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


@dataclass
class MsetTrainState(GenerationalState):
    """The generational cursor plus this loop's epoch-budget clock.

    settled_epochs: passes that ran over a corpus which had stopped growing.
        The budget is spent from this rather than from the pass counter, so a
        worker started while its generator is still delivering does not retire
        having only seen the corpus's first minutes.
    """

    settled_epochs: int = 0


def store_is_ready(store, params) -> tuple[bool, str]:
    """Whether the pair store holds enough to start training, and why not.

    Two conditions, both about not locking the run into a corpus the generator
    has barely started: enough pairs to be worth a pass, and -- when the tag
    sweeps at all -- at least one swept pair, so the holdout is the full-sweep
    slice the A3 gate is read on rather than the stratified fallback that would
    otherwise be frozen in for good.

    A store that has reached the tag's generation size overrides both: nothing
    more is coming, so waiting on either could only wait forever. That is what
    keeps a small `target_pairs` -- below `warmup_pairs`, or too small to have
    drawn a swept pair -- a short run rather than a hung one.
    """
    pairs = complete_pairs(store) if store.is_dir() else []
    if params.target_pairs and len(pairs) >= params.target_pairs:
        return True, ""
    needed = max(1, params.warmup_pairs)
    if len(pairs) < needed:
        return False, f"{len(pairs)}/{needed} pairs"
    if params.sweep_every:
        _, swept = partition_full_sweep(pairs)
        if not swept:
            return False, f"{len(pairs)} pairs, no full-sweep pair yet"
    return True, ""


def wait_for_store(store, params):
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


# How long the store must sit untouched before a tag that declared no
# generation size is taken to be done. A generation cycle is 200 self-play
# games plus teacher labeling -- minutes -- and a training pass over an early,
# small corpus is far quicker, so a single quiet pass says only that the pass
# fell between two deliveries.
QUIET_SECONDS = 900


class CorpusClock:
    """Decides when the tag's pair store has stopped growing -- the point from
    which a pass is over the whole corpus and may spend the epoch budget.

    A tag with a declared `target_pairs` is answered by the store reaching it,
    and by nothing having arrived on the pass that saw it: a second generate
    worker mid-cycle when the first crossed the target still has pairs to
    deliver, and counting the budget from before they land would score the
    run's epochs against two different holdouts.

    With no declared size there is no end to read, so growth has to be
    inferred. A store this worker has never seen grow is static -- an existing
    corpus being trained on, with no generator behind it -- and is final at
    once. One that has grown must then stay quiet for QUIET_SECONDS, which a
    trainer outrunning its generator cannot satisfy in the gap between two
    deliveries.
    """

    def __init__(self, store, params):
        self._store = store
        self._params = params
        self._last_growth = None

    def is_final(self, absorbed: int) -> bool:
        if absorbed:
            self._last_growth = time.monotonic()
        if self._params.target_pairs:
            held = count_pairs(self._store, ".mset")
            return not absorbed and held >= self._params.target_pairs
        if self._last_growth is None:
            return True
        return time.monotonic() - self._last_growth >= QUIET_SECONDS


def epochs_left(params, state: MsetTrainState) -> bool:
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
        # Whether this pass's metrics are comparable with the rest: a pass over
        # a still-growing corpus is scored against a holdout that is itself
        # still growing, so its curve is not on the same footing as the
        # budgeted ones (plots.py charts only loss*/`*_acc`, so this rides in
        # the metrics table for a reader rather than onto the Loss tab).
        "settled": int(settled),
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
    wait_for_store(paths.data_dir / SLOGS_DIR, params)
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

    state = checkpoint.resume(paths, model, optimizer, device, state_cls=MsetTrainState)
    try:
        clock = CorpusClock(paths.data_dir / SLOGS_DIR, params)
        while epochs_left(params, state):
            # Take up whatever the generator delivered during the last pass
            # before deciding whether this one is over a finished corpus.
            absorbed = absorb_new_pairs(paths, params, train_ds, holdout_ds)
            settled = clock.is_final(absorbed)
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
