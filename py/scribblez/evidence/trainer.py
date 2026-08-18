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
from scribblez.evidence.checkpoints import STUDENT_CONFIG_KEYS, EvidenceCheckpoint, load_student
from scribblez.evidence.dataset import (
    TrajectoryDataset,
    adopt_information_condition,
    complete_pairs,
)
from scribblez.evidence.train_loop import LossConfig, evaluate, run_epoch
from scribblez.evidence.trajectory_view import DecisionAnalysis, position_set_metrics
from scribblez.ffi import move_encoding_version, set_contingent_features
from scribblez.generational import checkpoint
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import (
    WsdLrController,
    WsdSchedule,
    init_controls,
    progress_line,
)
from scribblez.sim_evidence.position_sets import DEFAULT_SET, POSITIONS_ROOT, ensure_sobs, set_gcgs
from scribblez.sim_evidence.sobs import read_sobs
from scribblez.train_common import timed_print
from scribblez.workloads import pair_store
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.evidence_trajectories import SLOGS_DIR, max_evidence, recipe_of
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


class PositionSetProbe:
    """The hand-maintained position set (positions/NWL23/<DEFAULT_SET>) as a
    per-pass readout: its .gcg positions with trajectory sidecars simmed under
    this tag's proposer + recipe (the same sidecars the dashboard's
    Trajectories tab shows), re-scored by the current model at every evidence
    prefix (trajectory_view.position_set_metrics). Empty when the set is
    absent or its sidecars cannot be generated -- the readout is optional and
    never fails training."""

    def __init__(self, params, student_cfg: dict, threads: int):
        self.student_cfg = student_cfg
        self.max_e = max_evidence(params)
        self.positions: list[tuple[str, object]] = []  # (gcg text, SobsPosition)
        set_dir = POSITIONS_ROOT / DEFAULT_SET
        if not set_gcgs(set_dir):
            return
        try:
            sobs = ensure_sobs(set_dir, params.proposer_model, recipe_of(params), threads)
        except Exception as e:  # noqa: BLE001 -- an optional readout
            timed_print(f"position-set metric disabled: {e}")
            return
        for gcg in set_gcgs(set_dir):
            self.positions.append((gcg.read_text(), read_sobs(sobs[gcg.stem])[0]))
        timed_print(f"position-set metric: {len(self.positions)} positions from {DEFAULT_SET}")

    def metrics(self, model, device) -> dict[str, float]:
        if not self.positions:
            return {}
        ckpt = EvidenceCheckpoint(model, self.student_cfg, trained=True)
        try:
            analyses = [
                DecisionAnalysis(ckpt, text, sobs, self.max_e, device)
                for text, sobs in self.positions
            ]
        except ValueError as e:  # a sidecar that does not match the position
            timed_print(f"position-set metric skipped: {e}")
            return {}
        return position_set_metrics(analyses)


def save_epoch_checkpoint(paths, model, epoch: int, config: dict):
    """The per-pass checkpoint the trajectory pane loads: weights + config."""
    path = paths.checkpoint_path(epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, path)


# Skipped (non-finite) batches tolerated per pass before the run is stopped:
# an isolated bad batch is survivable, a stream of them is a diverged model.
MAX_SKIPPED_PER_PASS = 10


def check_finite(model, result):
    """Stop the run -- before anything is checkpointed -- when the pass
    diverged: non-finite parameters, or more skipped batches than an isolated
    incident. The rolling checkpoint then holds the last good pass, and the
    worker exits non-zero instead of logging NaN passes to the budget."""
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(f"diverged: non-finite parameters {bad[:4]}")
    if result.skipped > MAX_SKIPPED_PER_PASS:
        raise RuntimeError(f"diverged: {result.skipped} batches with a non-finite loss this pass")


def _metrics_record(
    epoch: int, state, settled: bool, losses: dict, m: dict, lr: float, skipped: int
) -> dict:
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
        "skipped_batches": skipped,
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
    record.update({k: v for k, v in m.items() if k.startswith("posset_")})
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
    check_finite(model, result)

    t1 = time.time()
    m = evaluate(model, ctx["holdout_ds"], device, params.batch_positions, ctx["max_e"])
    m.update(ctx["posset"].metrics(model, device))
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
        f"p0_maxdiff={m['exact_p0_maxdiff']:.1e} lr={lr_now:.2e} "
        f"skipped={result.skipped} {train_s:.1f}s [{budget}]"
    )
    record = _metrics_record(epoch, state, settled, result.losses, m, lr_now, result.skipped)
    db.write_metrics(conn, epoch, record)
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
            "student": {k: student_cfg[k] for k in STUDENT_CONFIG_KEYS},
            "open_leaves": train_ds.open_leaves,
            "proposer_hash": train_ds.proposer_hash,
        },
        "train_ds": train_ds,
        "holdout_ds": holdout_ds,
        "loss_cfg": LossConfig.from_args(params),
        "max_e": max_e,
        "stats": WorkerStats(ctx),
        "posset": PositionSetProbe(params, student_cfg, threads=ctx.threads),
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
    except RuntimeError as e:
        timed_print(f"{e}; the rolling checkpoint holds the last finite pass. Exiting.")
        return 1
    return 0
