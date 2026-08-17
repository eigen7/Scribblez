"""The evidence_trajectories workload's train role: the fusion stage and the
proves-best head, trained on the tag's trajectory pair store over a frozen
student backbone.

The model is the move set evaluation student named by `student_checkpoint`
(a move_set_eval tag's rolling checkpoint: weights plus the config it was
built against), with its trunk, move encoder and distillation heads frozen;
EvidenceFusion and the proves-best head are what learn (model.freeze_backbone).
The student's information-condition arm must be the corpus's -- the trainer
refuses a hidden-leaves student on an open-leaves corpus and vice versa.

The loop is the mset trainer's growing-corpus loop: wait for the store, take
up new pairs each pass into a file-level split, spend the epoch budget only on
passes over a finished corpus (pair_store.CorpusClock), record metrics to the
dashboard DB, checkpoint. Every pass also writes its own checkpoint under
checkpoints/model_epoch_NNNN.pt (model weights + config): the evidence path
has no ONNX export yet (roadmap item 5), so the torch checkpoint is what the
dashboard's trajectory pane loads per generation.
"""

from __future__ import annotations

import functools
import os
import time
from dataclasses import asdict, dataclass

import torch

from scribblez.dashboard import db
from scribblez.evidence.dataset import (
    TrajectoryDataset,
    adopt_information_condition,
    complete_pairs,
)
from scribblez.evidence.train_loop import LossConfig, evaluate, run_epoch
from scribblez.ffi import move_encoding_version, set_contingent_features
from scribblez.generational import checkpoint
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import (
    WsdLrController,
    WsdSchedule,
    init_controls,
    progress_line,
)
from scribblez.move_set_eval.model import MoveSetEvalModel
from scribblez.train_common import timed_print
from scribblez.workloads import pair_store
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.evidence_trajectories import SLOGS_DIR, max_evidence
from scribblez.workloads.worker import WorkerStats, WorkerStopped

POLL_SECONDS = 30


@dataclass
class EvidenceTrainState(GenerationalState):
    """The generational cursor plus the epoch-budget clock (see the mset
    trainer's MsetTrainState for why the budget is spent from settled passes)."""

    settled_epochs: int = 0


def split_pairs(store, holdout_every: int) -> tuple[list, list]:
    """(train, holdout) .sobs paths of the store's complete pairs, split at
    file level by stem hash (pair_store.split_pair_stems)."""
    train, holdout = pair_store.split_pair_stems(
        [f.stem for f in complete_pairs(store)], holdout_every
    )
    return [store / f"{s}.sobs" for s in train], [store / f"{s}.sobs" for s in holdout]


def store_is_ready(store, params) -> tuple[bool, str]:
    """Enough pairs to be worth a pass, and a held-out pair when one is asked
    for; a store at the tag's target size overrides both (nothing more comes)."""
    pairs = complete_pairs(store) if store.is_dir() else []
    if params.target_pairs and len(pairs) >= params.target_pairs:
        return True, ""
    needed = max(1, params.warmup_pairs)
    if len(pairs) < needed:
        return False, f"{len(pairs)}/{needed} pairs"
    if params.holdout_every and not split_pairs(store, params.holdout_every)[1]:
        return False, f"{len(pairs)} pairs, no held-out pair yet"
    return True, ""


def wait_for_store(store, params):
    while True:
        ready, why = store_is_ready(store, params)
        if ready:
            return
        timed_print(f"waiting for the pair store: {why}")
        time.sleep(POLL_SECONDS)  # SIGTERM raises WorkerStopped through this


def load_datasets(store, params) -> tuple[TrajectoryDataset, TrajectoryDataset]:
    train_files, holdout_files = split_pairs(store, params.holdout_every)
    if not train_files:
        raise FileNotFoundError(f"no complete .slog/.sobs training pairs in {store}")
    adopt_information_condition(train_files)
    train_ds = TrajectoryDataset(train_files)
    if not holdout_files:
        timed_print("no held-out pairs; metrics are on-train")
        return train_ds, train_ds
    holdout_ds = TrajectoryDataset(holdout_files)
    assert holdout_ds.proposer_hash == train_ds.proposer_hash, (
        "train/holdout pairs disagree on the proposer hash"
    )
    return train_ds, holdout_ds


def absorb_new_pairs(store, params, train_ds, holdout_ds) -> int:
    """Ingest every pair delivered since the last pass into the side the split
    assigns it (a pair's side is fixed the first time it is seen)."""
    train_files, holdout_files = split_pairs(store, params.holdout_every)
    seen = set(train_ds.files) | set(holdout_ds.files)
    added = train_ds.absorb(sorted(f for f in train_files if f not in seen))
    if holdout_ds is not train_ds:
        added += holdout_ds.absorb(sorted(f for f in holdout_files if f not in seen))
    return added


def epochs_left(params, state: EvidenceTrainState) -> bool:
    return params.train_epochs == 0 or state.settled_epochs < params.train_epochs


# What the model needs from the student's config to be rebuilt without it.
_STUDENT_CONFIG_KEYS = (
    "spatial_planes",
    "scalar_size",
    "trunk_channels",
    "num_blocks",
    "num_heads",
    "contingent_features",
    "open_leaves",
    "move_encoding_version",
)


def load_student(path: str, device) -> tuple[MoveSetEvalModel, dict]:
    """The frozen-backbone model initialized from a move_set_eval rolling
    checkpoint, and that checkpoint's config (the arch and encoding arm the
    student was built against, which this model inherits)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MoveSetEvalModel(
        spatial_planes=cfg["spatial_planes"],
        scalar_size=cfg["scalar_size"],
        trunk_channels=cfg["trunk_channels"],
        num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"],
    )
    model.load_student(ckpt["model_state_dict"])
    model.freeze_backbone()
    return model.to(device), cfg


def save_epoch_checkpoint(paths, model, epoch: int, config: dict):
    """The per-pass checkpoint the trajectory pane loads: weights + config."""
    path = paths.checkpoint_path(epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, path)


def _metrics_record(epoch: int, state, settled: bool, losses: dict, m: dict, lr: float) -> dict:
    """The dashboard row: losses, and the plain-vs-conditioned metrics as
    *_acc series (the Loss tab's Accuracy panel) with the rest in the table."""
    record = {
        "epoch": epoch,
        "positions": state.rows_trained,
        "settled": int(settled),
        "loss": losses["total"],
        "loss_wld": losses["wld"],
        "loss_score_diff": losses["score_diff"],
        "loss_gain": losses["gain"],
        "lr": lr,
    }
    for k in ("cond_wld_ce", "plain_wld_ce", "cond_wld_ce_ev", "plain_wld_ce_ev"):
        if k in m:
            record[k] = m[k]
    for k in ("cond_value_mae", "plain_value_mae", "gain_mae", "gain_mae_ev"):
        if k in m:
            record[k] = m[k]
    for k in ("gain_hit", "gain_hit_baseline", "gain_hit_ev", "gain_hit_ev_baseline"):
        if k in m:
            record[f"{k}_acc"] = m[k]
    record["exact_p0_maxdiff"] = m["exact_p0_maxdiff"]
    record["eval_rows"] = m["rows"]
    record["eval_rows_ev"] = m["rows_ev"]
    return record


def train_one_epoch(model, optimizer, conn, paths, device, params, state, ctx, settled: bool):
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
        ctx["max_e"],
        lr_fn=ctx["lr_controller"].lr_fn,
        rows_trained=state.rows_trained,
        on_batch=functools.partial(progress_line, epoch),
    )
    state.rows_trained = result.rows_trained
    state.generation_index = epoch + 1
    state.settled_epochs += int(settled)
    train_s = time.time() - t0

    t1 = time.time()
    m = evaluate(model, ctx["holdout_ds"], device, params.batch_positions, ctx["max_e"])
    eval_s = time.time() - t1
    lr_now = ctx["lr_controller"].current
    budget = (
        f"{state.settled_epochs}/{params.train_epochs}"
        if settled
        else f"corpus still growing, {ctx['train_ds'].num_positions} positions"
    )
    timed_print(
        f"[pass {epoch}] rows={state.rows_trained} loss={result.losses['total']:.4f} "
        f"wld_ce cond={m.get('cond_wld_ce', float('nan')):.4f} "
        f"plain={m.get('plain_wld_ce', float('nan')):.4f} "
        f"(ev rows: cond={m.get('cond_wld_ce_ev', float('nan')):.4f} "
        f"plain={m.get('plain_wld_ce_ev', float('nan')):.4f}) "
        f"gain_mae={m.get('gain_mae', float('nan')):.4f} "
        f"gain_hit={m.get('gain_hit', float('nan')):.3f} "
        f"(base {m.get('gain_hit_baseline', float('nan')):.3f}) "
        f"p0_maxdiff={m['exact_p0_maxdiff']:.1e} lr={lr_now:.2e} {train_s:.1f}s [{budget}]"
    )
    db.write_metrics(conn, epoch, _metrics_record(epoch, state, settled, result.losses, m, lr_now))
    checkpoint.save(paths, model, optimizer, state, ctx["config"])
    save_epoch_checkpoint(paths, model, epoch, ctx["config"])
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
    if not params.student_checkpoint or not os.path.isfile(params.student_checkpoint):
        print(f"error: student_checkpoint {params.student_checkpoint!r} is not a readable file")
        return 1

    model, student_cfg = load_student(params.student_checkpoint, device)
    # The move rows this trainer encodes (candidates and evidence tokens) must
    # be the rows the frozen student learned; a version bump in the engine
    # would otherwise train the fusion against embeddings of the wrong layout.
    if student_cfg["move_encoding_version"] != move_encoding_version():
        print(
            f"error: the student was trained under move encoding version "
            f"{student_cfg['move_encoding_version']} but the engine encodes version "
            f"{move_encoding_version()}"
        )
        return 1
    set_contingent_features(student_cfg["contingent_features"])
    store = paths.data_dir / SLOGS_DIR
    wait_for_store(store, params)
    train_ds, holdout_ds = load_datasets(store, params)
    if train_ds.open_leaves != student_cfg["open_leaves"]:
        print(
            f"error: the corpus was simmed with open_leaves={train_ds.open_leaves} but the "
            f"student was trained with open_leaves={student_cfg['open_leaves']}"
        )
        return 1
    max_e = max_evidence(params)
    if train_ds.max_trajectory > max_e:
        print(
            f"error: a trajectory holds {train_ds.max_trajectory} candidates, more than the "
            f"recipe's {max_e} (1 + proposals_max + 1): the corpus was not simmed with this "
            "tag's recipe"
        )
        return 1
    print(
        f"train: {train_ds.num_positions} positions / {train_ds.num_candidates} simmed candidates; "
        f"eval: {holdout_ds.num_positions} positions / {holdout_ds.num_candidates} candidates "
        f"(open_leaves={train_ds.open_leaves}, proposer {train_ds.proposer_hash[:12]})"
    )
    trainable = model.evidence_parameters()
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_total:,} parameters, {n_train:,} trainable (fusion + proves-best head)")
    optimizer = torch.optim.AdamW(trainable, lr=params.lr, weight_decay=params.weight_decay)

    conn = db.connect(paths.dashboard_db)
    db.write_meta(conn, ctx.tag, asdict(params), n_train)
    db.write_loss_weights(
        conn,
        {"loss_wld": 1.0, "loss_score_diff": params.lambda_sd, "loss_gain": params.lambda_gain},
    )
    init_controls(conn)
    run_ctx = {
        "config": {
            **asdict(params),
            "student": {k: student_cfg[k] for k in _STUDENT_CONFIG_KEYS},
            "open_leaves": train_ds.open_leaves,
            "proposer_hash": train_ds.proposer_hash,
        },
        "train_ds": train_ds,
        "holdout_ds": holdout_ds,
        "loss_cfg": LossConfig.from_args(params),
        "max_e": max_e,
        "stats": WorkerStats(ctx),
    }

    state = checkpoint.resume(paths, model, optimizer, device, state_cls=EvidenceTrainState)
    run_ctx["lr_controller"] = WsdLrController(
        conn, WsdSchedule.from_params(params), state.rows_trained
    )
    try:
        clock = pair_store.CorpusClock(store, params.target_pairs, ".sobs")
        while epochs_left(params, state):
            absorbed = absorb_new_pairs(store, params, train_ds, holdout_ds)
            settled = clock.is_final(absorbed)
            train_one_epoch(model, optimizer, conn, paths, device, params, state, run_ctx, settled)
        timed_print(
            f"Training complete: {state.settled_epochs} epochs over the finished corpus "
            f"({state.generation_index} passes, {state.rows_trained} rows). Pause the worker."
        )
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped; last completed epoch is checkpointed.")
    return 0
