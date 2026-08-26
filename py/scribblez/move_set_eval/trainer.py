"""The move-set-eval train role: distill the student over a tag's pair store.

Sibling to the position_eval / max_move_per_lane trainers, but a growing-corpus
loop rather than a generational consumer: the corpus is the tag's slogs/ pair
store, split at file level into train and held-out pairs -- the held-out side
being the store's full-sweep pairs, the only ones the A3 gate metrics mean
anything on.

The generate role writes that store while this runs, so the loop keeps pace
with a corpus being written underneath it instead of snapshotting one -- which
is what lets a tag with a worker of each type run to completion unattended.
store_is_ready, absorb_new_pairs and pair_store.CorpusClock are that pacing, and
carry its reasoning.

Each pass records losses and the held-out top-K recall / Spearman metrics to
the tag's dashboard DB (the Loss tab's curves), publishes a stats sample (the
Stats tab), and saves the rolling checkpoint -- pausing and restarting the
worker resumes at the next pass. The learning rate follows the shared
rows-clock WSD schedule (generational/controls.py). The generational consume->train
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
from scribblez.generational import checkpoint
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import (
    WsdLrController,
    WsdSchedule,
    init_controls,
    progress_line,
)
from scribblez.move_set_eval.dataset import MsetDataset, adopt_information_condition
from scribblez.move_set_eval.eval import eval_slice_line, evaluate
from scribblez.move_set_eval.model import MoveSetEvalModel
from scribblez.move_set_eval.moves import move_encoding_version
from scribblez.move_set_eval.onnx_export import (
    FP16_PROBE_POSITIONS,
    export_onnx,
    fp16_probe_feeds_from_batch,
)
from scribblez.move_set_eval.targets import complete_pairs, read_mset_flags
from scribblez.move_set_eval.train_loop import LossConfig, run_epoch
from scribblez.train_common import timed_print
from scribblez.workloads import pair_store
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.move_set_eval import SLOGS_DIR, split_pairs
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
    has barely started: enough pairs to be worth a pass, and -- for a tag that
    asks for a holdout at all -- a split that actually yields one. Starting
    without a holdout aliases it to the training set for the whole run, and
    starting a sweeping tag before its first swept pair would read the gate
    metrics off the stratified fallback instead of the full-sweep slice.

    A store that has reached the tag's generation size overrides both: nothing
    more is coming, so waiting on either could only wait forever. That is what
    keeps a small `target_pairs` -- below `warmup_pairs`, or too small to have
    drawn a holdout pair -- a short run rather than a hung one.
    """
    pairs = complete_pairs(store) if store.is_dir() else []
    if params.target_pairs and len(pairs) >= params.target_pairs:
        return True, ""
    needed = max(1, params.warmup_pairs)
    if len(pairs) < needed:
        return False, f"{len(pairs)}/{needed} pairs"
    if (params.sweep_every or params.holdout_every) and not split_pairs(
        store, params.holdout_every
    )[1]:
        return False, f"{len(pairs)} pairs, no held-out pair yet"
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

    A pair's side is fixed the first time it is seen. The store's split can
    change shape underneath a running trainer -- the first swept pair to land
    turns every stratified pair into a training pair -- and a file that moved
    would be one trained on and then scored as held out.
    """
    train_files, holdout_files = split_pairs(paths.data_dir / SLOGS_DIR, params.holdout_every)
    seen = set(train_ds.files) | set(holdout_ds.files)
    added = train_ds.absorb(_ingestible(train_files, seen, train_ds))
    if holdout_ds is not train_ds:
        added += holdout_ds.absorb(_ingestible(holdout_files, seen, holdout_ds))
    return added


def _ingestible(files, seen: set, ds: MsetDataset) -> list:
    """The files `ds` can take: ones neither side already holds, whose header
    matches the corpus it holds. A file matching neither side's header sits out
    the run rather than crashing it -- a swept pair delivered to a tag whose
    trainer started before any swept pair existed has no side to join, since a
    dataset cannot mix header flags."""
    fresh = sorted(f for f in files if f not in seen)
    usable = [f for f in fresh if read_mset_flags(f) == ds.flags]
    for skipped in set(fresh) - set(usable):
        timed_print(f"{skipped.name}: header does not match its side's corpus; left out of the run")
    return usable


def corpus_clock(store, params) -> pair_store.CorpusClock:
    """The tag's corpus clock over its .mset pair store."""
    return pair_store.CorpusClock(store, params.target_pairs, ".mset")


def epochs_left(params, state: MsetTrainState) -> bool:
    """Whether the epoch budget has passes left -- spent by
    state.settled_epochs, whose reasoning is on MsetTrainState."""
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
        lr_fn=ctx["lr_controller"].lr_fn,
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
    lr_now = ctx["lr_controller"].current
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
        "loss_wld_z": avg["wld_z"],
        "loss_pool_act": avg["pool_act"],
        "loss_score_diff": avg["score_diff"],
        "loss_planes": avg["planes"],
        "spearman_acc": metrics["spearman"],
        "spearman_baseline_acc": metrics["spearman_baseline"],
        "lr": lr_now,
        "elapsed_s": train_s,
    }
    # The exchange-slice series (docs/roadmap.md A4 exchange analysis): named
    # without the _acc suffix, so they land on their own Training-tab figures
    # (plots.MSET_QUALITY) instead of crowding the Loss tab's Accuracy panel.
    # Plane-readout quality on the holdout, when it carries plane targets
    # (the stratified fallback holdout; the full-sweep slice does not).
    if "plane_bce" in metrics:
        record["plane_bce"] = metrics["plane_bce"]
    record["exch_rank_regret"] = metrics["exch_rank_regret"]
    record["exch_rank_regret_baseline"] = metrics["exch_rank_regret_baseline"]
    record["positions_with_exchanges"] = metrics["positions_with_exchanges"]
    for k in (1, 3, 5):
        record[f"recall{k}_acc"] = metrics[f"recall@{k}"]
        record[f"recall{k}_baseline_acc"] = metrics[f"recall@{k}_baseline"]
        record[f"regret{k}"] = metrics[f"regret@{k}"]
        record[f"regret{k}_baseline"] = metrics[f"regret@{k}_baseline"]
        record[f"exch_retention{k}"] = metrics[f"exch_retention@{k}"]
        record[f"exch_retention{k}_baseline"] = metrics[f"exch_retention@{k}_baseline"]
    # Per-pass ONNX beside the rolling checkpoint, under the pass index the
    # metrics record uses -- the artifact the engine runtime loads and any
    # match-eval consumer keys on. The config's recorded arm/version stamp the
    # metadata, so the export can never claim an encoding its rows didn't use.
    # Exported (and FP16-gated) before the metrics write so the recorded pass
    # always has its ONNX, and so the record carries the gate's peak.
    cfg = ctx["config"]
    fp16_peak = export_onnx(
        model,
        paths.onnx_path(epoch),
        cfg["spatial_planes"],
        cfg["scalar_size"],
        opp_leave_input=cfg["open_leaves"],
        move_encoding_version=cfg["move_encoding_version"],
        probe_feeds=ctx["fp16_probe"],
    )
    if fp16_peak is not None:
        record["fp16_probe_peak"] = fp16_peak
        timed_print(f"  fp16 probe: peak |activation| {fp16_peak:.0f}")
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
    db.write_loss_weights(
        conn,
        {
            "loss_wld": 1.0,
            "loss_score_diff": params.lambda_sd,
            "loss_planes": params.lambda_planes,
            "loss_wld_z": params.lambda_wld_z,
            "loss_pool_act": params.lambda_pool_act,
        },
    )
    init_controls(conn)

    # Beyond the frozen task params, the checkpoint config records what the
    # model was actually built against -- the adopted information-condition
    # arm, the input widths, and the engine's move-encoding version -- so a
    # standalone exporter can reconstruct and stamp the model without the
    # corpus, and a checkpoint can never silently meet a mismatched encoder.
    run_ctx = {
        "config": {
            **asdict(params),
            "open_leaves": train_ds.open_leaves,
            "spatial_planes": train_ds.spatial_planes,
            "scalar_size": train_ds.scalar_size,
            "move_encoding_version": move_encoding_version(),
        },
        "train_ds": train_ds,
        "holdout_ds": holdout_ds,
        "loss_cfg": LossConfig.from_args(params),
        "stats": WorkerStats(ctx),
        # The export gate's probe feeds (docs/fp16_safe_serving.md), built once
        # from a deterministic holdout batch.
        "fp16_probe": fp16_probe_feeds_from_batch(
            next(holdout_ds.iter_batches(FP16_PROBE_POSITIONS, seed=0))
        ),
    }

    state = checkpoint.resume(paths, model, optimizer, device, state_cls=MsetTrainState)
    run_ctx["lr_controller"] = WsdLrController(
        conn, WsdSchedule.from_params(params), state.rows_trained
    )
    try:
        clock = corpus_clock(paths.data_dir / SLOGS_DIR, params)
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
