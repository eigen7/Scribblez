"""The move_set_eval workload's train role: distill the student over a tag's
pair store.

Runs as the singleton `train` worker (SCZ_ROLE=train);
py/scripts/move_set_eval/train.py is the CLI for ad-hoc runs outside a tag.

Unlike the generational position_eval trainer, this trains over one growing
corpus: the tag's slogs/ pair store, split at file level into training pairs
and held-out pairs (the full-sweep pairs, on which the recall/regret metrics
mean something). The generate role writes the store while this runs, so the
loop keeps pace with it instead of snapshotting it; that is what lets a tag
with one worker of each role run to completion unattended. store_is_ready,
absorb_new_pairs and pair_store.CorpusClock implement the pacing and document
its reasoning. docs/plans/generational_teacher.md plans to replace this loop
with the generational lifecycle.

Each pass writes losses and held-out metrics to the tag's dashboard DB,
publishes a stats sample, exports ONNX and saves the rolling checkpoint, so a
paused worker resumes at the next pass. The optimizer and learning-rate policy
come from the run's `optimizer` arm (generational/optim.py), as in
position_eval.

All reads and writes go through the worker's sink (cloud/sinks.py): the store
is pulled through it before every look, and exports, checkpoints, records and
stats are delivered through it. A local worker and one on a rented GPU run the
same code with different sinks (docs/cloud_compute.md).
"""

import functools
import itertools
import os
import time
from dataclasses import asdict, dataclass

import torch

from scribblez.generational import checkpoint
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import progress_line
from scribblez.generational.optim import build_optim_arm, build_optimizer
from scribblez.generational.records import TrainRecorder
from scribblez.move_set_eval.dataset import MsetDataset, adopt_information_condition
from scribblez.move_set_eval.eval import eval_slice_line, evaluate
from scribblez.move_set_eval.model import MoveSetEvalModel
from scribblez.move_set_eval.moves import move_encoding_version
from scribblez.move_set_eval.onnx_export import export_onnx
from scribblez.move_set_eval.targets import complete_pairs, read_mset_flags
from scribblez.move_set_eval.train_loop import LossConfig, run_epoch
from scribblez.spatial_trunk import transformer_config
from scribblez.train_common import timed_print
from scribblez.workloads import pair_store
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.move_set_eval import SLOGS_DIR, split_pairs
from scribblez.workloads.worker import WorkerStats, WorkerStopped

POLL_SECONDS = 30

# Export retention (prune_exports): the newest KEEP_LAST_EXPORTS plus every
# KEEP_EVERY_EXPORT-th. Kept whole, a run's per-pass exports outgrow its corpus
# (6k passes x 40 MB = 240 GB). Pruning is safe because only the dashboard's
# listing reads exports in place; a tag that depends on one (the evidence
# workload's proposer) copies it into its own root first
# (mset_targets.pin_model).
KEEP_LAST_EXPORTS = 10
KEEP_EVERY_EXPORT = 100


@dataclass
class MsetTrainState(GenerationalState):
    """The generational cursor plus this loop's epoch-budget clock.

    settled_epochs: passes over a corpus that had stopped growing. The epoch
        budget counts these rather than all passes, so a worker started
        alongside its generator does not finish having seen only the corpus's
        first minutes.
    """

    settled_epochs: int = 0


def fetch_train_deps(params):
    """Fetch the default lexicon, which the FFI session loads to decode the
    pair store's rows. The pairs themselves come through the sink."""
    from cloud import worker_deps

    worker_deps.fetch_lexicon(worker_deps.DEFAULT_LEXICON)


def store_is_ready(store, params) -> tuple[bool, str]:
    """Whether the pair store holds enough to start training, and if not, why.

    Both conditions keep the run from locking into a barely started corpus:
    at least `warmup_pairs` pairs, and, for a tag that wants a holdout, a split
    that yields one. Starting without a holdout makes the training set stand
    in for it for the whole run (load_datasets), and starting a sweeping tag
    before its first swept pair would score the run on stratified pairs.

    A store that has reached `target_pairs` is ready regardless, since nothing
    more is coming; this keeps a small run from waiting forever.
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


def pull_store(store, sink):
    """Pull pairs that remote generators delivered into the local store (a
    no-op for a local worker, whose store is the controller's)."""
    store.mkdir(parents=True, exist_ok=True)
    sink.fetch_data_files(SLOGS_DIR, store)


def wait_for_store(store, params, sink=None):
    """Block until store_is_ready, reporting progress and pulling through
    `sink` (if given) before each look."""
    while True:
        if sink is not None:
            pull_store(store, sink)
        ready, why = store_is_ready(store, params)
        if ready:
            return
        timed_print(f"waiting for the pair store: {why}")
        time.sleep(POLL_SECONDS)  # SIGTERM raises WorkerStopped through this


def load_datasets(paths, params) -> tuple[MsetDataset, MsetDataset]:
    """(train, holdout) datasets from the tag's pair store, split at file level
    by split_pairs. With no held-out pairs the holdout is the training set
    itself (a smoke check)."""
    store = paths.data_dir / SLOGS_DIR
    train_files, holdout_files = split_pairs(store, params.holdout_every)
    if not train_files:
        raise FileNotFoundError(
            f"no complete .slog/.mset training pairs in {store} (a finished run retires its "
            "training pairs; a new run needs a new tag)"
        )
    adopt_information_condition(train_files)
    train_ds = MsetDataset(mset_files=train_files)
    require_training_rows(train_ds)
    if not holdout_files:
        timed_print("no held-out pairs; metrics are on-train")
        return train_ds, train_ds
    holdout_ds = MsetDataset(mset_files=holdout_files)
    assert holdout_ds.model_hash == train_ds.model_hash, (
        "train/holdout pairs disagree on the teacher hash"
    )
    return train_ds, holdout_ds


def require_training_rows(train_ds: MsetDataset):
    """Fail if every candidate was dropped for non-finite targets. That means
    the teacher overflowed its serving precision; left alone, every pass would
    export and checkpoint an untrained model."""
    if train_ds.num_candidates == 0:
        raise RuntimeError(
            f"no finite teacher targets in {len(train_ds.files)} training pair(s) "
            f"({train_ds.dropped_candidates} candidates dropped): the teacher's readouts are "
            "non-finite, so its serving precision overflows (docs/plans/fp16_safe_serving.md). "
            "Regenerate the corpus; nothing here is trainable."
        )


def prune_exports(
    paths, sink, epoch: int, keep_last: int = KEEP_LAST_EXPORTS, keep_every: int = KEEP_EVERY_EXPORT
):
    """After pass `epoch`'s export, delete the export that just left the
    `keep_last` window unless it is a multiple of `keep_every`. Pruning one
    export per pass needs no listing of what is kept, which a
    bucket-delivering trainer has no local copy of. The sink removes both the
    bucket copy and the local one."""
    stale = epoch - keep_last
    if stale < 0 or (keep_every > 0 and stale % keep_every == 0):
        return
    sink.remove_output(f"models/{paths.onnx_path(stale).name}")


def deliver_pass(paths, sink, epoch: int):
    """Hand pass `epoch`'s export and the rolling checkpoint to the sink. A
    bucket sink uploads both and deletes the local export; the checkpoint
    stays local for resume."""
    export = paths.onnx_path(epoch)
    sink.deliver_output(export, f"models/{export.name}")
    sink.deliver_output(paths.rolling_checkpoint, "checkpoints/model.pt", keep=True)


def restore_checkpoint(paths, sink):
    """Fetch the rolling checkpoint through the sink if this machine has none
    (a fresh rented machine)."""
    if paths.rolling_checkpoint.exists():
        return
    if sink.fetch_file("checkpoints/model.pt", paths.rolling_checkpoint):
        timed_print(f"restored the rolling checkpoint through the {sink.kind} sink")


def retire_training_pairs(train_ds: MsetDataset, sink) -> int:
    """Delete the finished run's training pairs (.mset and .slog) through the
    sink, returning the count.

    Params are frozen, so a finished run cannot be extended and nothing reads
    these pairs again; they are the bulk of a tag's disk footprint. Held-out
    pairs stay for later re-evaluation of the exports. Going through the sink
    also deletes the bucket copies, which later syncs would otherwise pull
    back."""
    sink.remove_outputs(
        [
            f"data/{SLOGS_DIR}/{path.name}"
            for mset in train_ds.files
            for path in (mset, mset.with_suffix(".slog"))
        ]
    )
    return len(train_ds.files)


def absorb_new_pairs(
    paths, params, train_ds: MsetDataset, holdout_ds: MsetDataset, sink=None
) -> int:
    """Ingest every pair delivered since the last pass into the side the
    file-level split assigns it, returning the number of positions added.

    Both sides grow until the generator stops, so the holdout is fixed for
    every budgeted (settled) epoch and their metrics are comparable.

    A pair's side is fixed the first time it is seen. The split can change
    under a running trainer (the first swept pair to land turns every
    stratified pair into a training pair), and a file that switched sides
    would be trained on and then scored as held out.
    """
    if sink is not None:
        pull_store(paths.data_dir / SLOGS_DIR, sink)
    train_files, holdout_files = split_pairs(paths.data_dir / SLOGS_DIR, params.holdout_every)
    seen = set(train_ds.files) | set(holdout_ds.files)
    added = train_ds.absorb(_ingestible(train_files, seen, train_ds))
    if holdout_ds is not train_ds:
        added += holdout_ds.absorb(_ingestible(holdout_files, seen, holdout_ds))
    return added


def _ingestible(files, seen: set, ds: MsetDataset) -> list:
    """Files neither side holds yet whose header flags match `ds`. A file that
    matches no side is skipped with a log line rather than crashing the run:
    e.g. a swept pair arriving after the trainer started with a stratified
    holdout has no side to join."""
    fresh = sorted(f for f in files if f not in seen)
    usable = [f for f in fresh if read_mset_flags(f) == ds.flags]
    for skipped in set(fresh) - set(usable):
        timed_print(f"{skipped.name}: header does not match its side's corpus; left out of the run")
    return usable


def corpus_clock(store, params) -> pair_store.CorpusClock:
    """The tag's corpus clock over its .mset pair store."""
    return pair_store.CorpusClock(store, params.target_pairs, ".mset")


def epochs_left(params, state: MsetTrainState) -> bool:
    """Whether the epoch budget (0 = unlimited) has passes left; see
    MsetTrainState.settled_epochs."""
    return params.train_epochs == 0 or state.settled_epochs < params.train_epochs


# Batches the schedule-free arm recomputes BatchNorm statistics over before
# evaluation and export (optim.ScheduleFreeArm.eval_mode). As in position_eval,
# a short forward-only prefix of a fresh pass suffices.
BN_RECALIBRATION_BATCHES = 32


def _rows_per_step(train_ds: MsetDataset, params) -> float:
    """Mean candidate moves per optimizer step. This trainer's rows clock
    counts candidate moves, and build_optimizer uses this ratio to convert
    lr_warmup_rows into the step count AdamWScheduleFree expects."""
    return params.batch_positions * train_ds.num_candidates / max(train_ds.num_positions, 1)


def _encode_board_only(model, spatial, scalar):
    """BatchNorm-recalibration forward. All BatchNorm layers are in the board
    trunk, so the board encode suffices and needs no candidate set."""
    model.encode_board(spatial, scalar)


def _recalibration_batches(train_ds: MsetDataset, params, epoch: int, device):
    """Board inputs for BatchNorm recalibration, drawn from the training pairs
    under a seed distinct from the epoch's. Lazy, so an arm that ignores them
    decodes nothing."""
    batches = train_ds.iter_batches(
        params.batch_positions, seed=epoch * 1000003 + 1, epoch_index=epoch
    )
    for batch in itertools.islice(batches, BN_RECALIBRATION_BATCHES):
        yield batch["input_spatial"].to(device), batch["input_scalar"].to(device)


def train_one_epoch(model, optimizer, recorder, paths, device, params, state, ctx, settled: bool):
    """One pass over the training pairs, then held-out metrics, the dashboard
    record, ONNX export, the rolling checkpoint and a stats sample. `settled`
    (the corpus had stopped growing) decides whether the pass spends the epoch
    budget.

    The optimizer arm is in eval mode for everything after training, so a
    schedule-free run scores, exports and saves the averaged weights it would
    deploy."""
    optim_arm = ctx["optim_arm"]
    epoch = state.generation_index
    batches = ctx["train_ds"].iter_batches(params.batch_positions, seed=0, epoch_index=epoch)
    t0 = time.time()
    rows_before = state.rows_trained
    optim_arm.train_mode()
    result = run_epoch(
        model,
        optimizer,
        batches,
        device,
        ctx["loss_cfg"],
        lr_fn=optim_arm.lr_fn,
        rows_trained=state.rows_trained,
        on_batch=functools.partial(progress_line, epoch),
        grad_clip=params.grad_clip,
    )
    if result.n_batches == 0:
        # load_datasets refuses an empty corpus, so this is a bug; fail rather
        # than record and export an untrained pass.
        raise RuntimeError(f"pass {epoch} saw no training rows")
    state.rows_trained = result.rows_trained
    state.generation_index = epoch + 1
    state.settled_epochs += int(settled)
    train_s = time.time() - t0
    optim_arm.eval_mode(
        model,
        _recalibration_batches(ctx["train_ds"], params, epoch, device),
        forward_fn=_encode_board_only,
    )

    t1 = time.time()
    metrics = evaluate(model, ctx["holdout_ds"], device, positions_per_batch=params.batch_positions)
    eval_s = time.time() - t1

    avg = result.losses
    lr_now = optim_arm.current
    recall = " ".join(f"r@{k}={metrics[f'recall@{k}']:.3f}" for k in (1, 3, 5))
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
    # The dashboard plots every *_acc series on the Loss tab's Accuracy panel;
    # the other quality series have their own figures (plots.MSET_QUALITY).
    record = {
        "epoch": epoch,
        "positions": state.rows_trained,
        # Unsettled passes are scored against a still-growing holdout, so their
        # metrics are not comparable with the settled ones.
        "settled": int(settled),
        "loss": avg["total"],
        "loss_wld": avg["wld"],
        "loss_score_diff": avg["score_diff"],
        "loss_planes": avg["planes"],
        "spearman_acc": metrics["spearman"],
        "spearman_baseline_acc": metrics["spearman_baseline"],
        "lr": lr_now,
        "elapsed_s": train_s,
        # Arm-specific series, e.g. the schedule-free averaging weight.
        **optim_arm.metrics(),
    }
    if "plane_ce" in metrics:
        record["plane_ce"] = metrics["plane_ce"]
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
    # The per-pass ONNX is keyed by the same pass index as the metrics record.
    # Its metadata is stamped from the config, i.e. the encoding the training
    # rows actually used. Export and checkpoint precede the record, so a
    # recorded pass always has its ONNX and its resume point.
    cfg = ctx["config"]
    export_onnx(
        model,
        paths.onnx_path(epoch),
        cfg["spatial_planes"],
        cfg["scalar_size"],
        opp_leave_input=cfg["open_leaves"],
        move_encoding_version=cfg["move_encoding_version"],
    )
    checkpoint.save(paths, model, optimizer, state, ctx["config"])
    deliver_pass(paths, ctx["sink"], epoch)
    prune_exports(paths, ctx["sink"], epoch)
    recorder.commit_generation(epoch, state.rows_trained, record)
    ctx["stats"].cycle_done(
        {"train_s": train_s, "eval_s": eval_s},
        units=state.rows_trained - rows_before,
        nbytes=0,
    )
    optim_arm.train_mode()


def publish_config(recorder, tag: str, params, model_params: int = 0):
    """Publish the frozen params (Info tab) and each loss term's weight in the
    total (so the Loss tab can stack weighted contributions). `model_params`
    is 0 until the model is built, after which this is called again."""
    recorder.publish_run(
        tag,
        asdict(params),
        model_params,
        {
            "loss_wld": 1.0,
            "loss_score_diff": params.lambda_sd,
            "loss_planes": params.lambda_planes,
        },
        {},
    )


def run(ctx: WorkerContext) -> int:
    """The train-role entrypoint."""
    params = ctx.params
    paths = ctx.tag_paths()
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(os.environ.get("SCZ_DEVICE", "cuda"))
    print(f"Tag root: {paths.root}\nDevice: {device}")

    # Before the warmup wait, so the Info tab is populated while the store fills.
    recorder = TrainRecorder(ctx.sink)
    publish_config(recorder, ctx.tag, params)

    # A finished run has retired its training pairs, so check the checkpoint
    # first rather than wait on a store that will never refill.
    restore_checkpoint(paths, ctx.sink)
    if not epochs_left(params, checkpoint.peek_state(paths, state_cls=MsetTrainState)):
        timed_print("Training complete (the epoch budget was spent in an earlier session).")
        return 0
    wait_for_store(paths.data_dir / SLOGS_DIR, params, ctx.sink)
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
        transformer=transformer_config(asdict(params)),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    publish_config(recorder, ctx.tag, params, n_params)  # re-stamp with the parameter count
    optimizer = build_optimizer(model, params, _rows_per_step(train_ds, params))

    # The checkpoint config also records what the model was built against
    # (information condition, input widths, move-encoding version), so a
    # standalone exporter can rebuild and stamp the model without the corpus
    # and a checkpoint cannot silently meet a mismatched encoder.
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
        "sink": ctx.sink,
    }

    state = checkpoint.resume(paths, model, optimizer, device, state_cls=MsetTrainState)
    run_ctx["optim_arm"] = build_optim_arm(recorder, params, optimizer, state.rows_trained)
    try:
        clock = corpus_clock(paths.data_dir / SLOGS_DIR, params)
        while epochs_left(params, state):
            # Absorb before asking whether the corpus is final.
            absorbed = absorb_new_pairs(paths, params, train_ds, holdout_ds, ctx.sink)
            settled = clock.is_final(absorbed)
            train_one_epoch(
                model, optimizer, recorder, paths, device, params, state, run_ctx, settled
            )
        timed_print(
            f"Training complete: {state.settled_epochs} epochs over the finished corpus "
            f"({state.generation_index} passes, {state.rows_trained} rows, "
            f"{train_ds.num_positions} positions). Pause the worker (raising train_epochs "
            "needs a new tag; params are frozen)."
        )
        if holdout_ds is not train_ds:
            n = retire_training_pairs(train_ds, ctx.sink)
            timed_print(f"Retired {n} training pair(s); the held-out pairs remain in the store.")
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped; last completed epoch is checkpointed.")
    return 0
