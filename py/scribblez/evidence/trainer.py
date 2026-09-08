"""The evidence_trajectories workload's train role: the fusion stage and the
proves-best head, trained on the tag's trajectory pair store over the
student backbone -- frozen, or jointly trained under a distillation anchor.

The model is the move set evaluation student named by `student_checkpoint`
(a move_set_eval tag's rolling checkpoint: weights plus the config it was
built against). Frozen mode (the default) holds its trunk,
move encoder and distillation heads at the checkpoint; EvidenceFusion and the
proves-best head are what learn (model.freeze_backbone). Unfrozen mode trains
everything: each step is joint over a trajectory batch (the sim-outcome loss
on the conditioned pass) and a batch of the same games' .mset teacher labels
(the ordinary student objective on the plain pass, which anchors it), the
backbone at `backbone_lr_mult` times the evidence path's rate. The plain
student then changes each pass, so it is exported per pass as ONNX -- what a
later generation's proposer is taken from -- and the frozen student's held-out
sim soft-CE is reported as the flat reference the moving plain pass is read
against (StudentReference). The student's information-condition arm must be
the corpus's -- the trainer refuses a hidden-leaves student on an open-leaves
corpus and vice versa.

The loop is the mset trainer's growing-corpus loop: wait for the store, take
up new pairs each pass into a file-level split shared by the .sobs and .mset
sides (a stem is train or held-out on both), spend the epoch budget only on
passes over a finished corpus (pair_store.CorpusClock), record metrics to the
dashboard DB, checkpoint. Every pass also writes its own checkpoint under
checkpoints/model_epoch_NNNN.pt (model weights + config): the evidence path
has no ONNX export yet (roadmap item 3), so the torch checkpoint is what the
dashboard's trajectory pane loads per generation.
"""

from __future__ import annotations

import functools
import os
import time
from dataclasses import asdict, dataclass

import torch

from scribblez.evidence.checkpoints import STUDENT_CONFIG_KEYS, EvidenceCheckpoint, load_student
from scribblez.evidence.dataset import (
    TrajectoryDataset,
    adopt_information_condition,
    complete_pairs,
    trajectory_positions,
)
from scribblez.evidence.train_loop import (
    DISTILL_LOSS_KEYS,
    Distillation,
    LossConfig,
    evaluate,
    run_epoch,
)
from scribblez.evidence.trajectory_view import DecisionAnalysis, position_set_metrics
from scribblez.ffi import move_encoding_version
from scribblez.generational import checkpoint
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import (
    WsdLrController,
    WsdSchedule,
    default_controls,
    progress_line,
)
from scribblez.generational.records import TrainRecorder
from scribblez.move_set_eval import eval as mset_eval
from scribblez.move_set_eval import train_loop as mset_train_loop
from scribblez.move_set_eval.dataset import MsetDataset
from scribblez.move_set_eval.onnx_export import export_onnx
from scribblez.sim_evidence.position_sets import DEFAULT_SET, POSITIONS_ROOT, ensure_sobs, set_gcgs
from scribblez.sim_evidence.sobs import read_sobs
from scribblez.train_common import timed_print
from scribblez.workloads import pair_store
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.evidence_trajectories import (
    SLOGS_DIR,
    max_evidence_width,
    max_pool_width,
    recipe_of,
)
from scribblez.workloads.worker import WorkerStats, WorkerStopped

POLL_SECONDS = 30


@dataclass
class EvidenceTrainState(GenerationalState):
    """The generational cursor plus the epoch-budget clock (see the mset
    trainer's MsetTrainState for why the budget is spent from settled passes)."""

    settled_epochs: int = 0


def split_pairs(store, holdout_every: int, ext: str = ".sobs") -> tuple[list, list]:
    """(train, holdout) `ext` sidecar paths of the store's complete pairs,
    split at file level by stem hash (pair_store.split_pair_stems). The split
    is of stems, so the .sobs side (the trajectory rows) and the .mset side
    (the distillation rows) hold a game on the same side; a stem is listed
    for `ext` only where that sidecar exists."""
    train, holdout = pair_store.split_pair_stems(
        [f.stem for f in complete_pairs(store)], holdout_every
    )
    return _sidecars(store, train, ext), _sidecars(store, holdout, ext)


def _sidecars(store, stems: list[str], ext: str) -> list:
    return [p for s in stems if (p := store / f"{s}{ext}").is_file()]


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


def _load_split(store, params, ext: str, build, hash_attr: str) -> tuple:
    """(train, holdout) datasets over one sidecar side of the split -- `build`
    makes a dataset from a file list; both sides must agree on `hash_attr`
    (the proposer or teacher). With no held-out files the training set stands
    in (metrics are then on-train)."""
    train_files, holdout_files = split_pairs(store, params.holdout_every, ext)
    if not train_files:
        raise FileNotFoundError(f"no complete .slog/{ext} training pairs in {store}")
    train_ds = build(train_files)
    if not holdout_files:
        return train_ds, train_ds
    holdout_ds = build(holdout_files)
    assert getattr(holdout_ds, hash_attr) == getattr(train_ds, hash_attr), (
        f"train/holdout {ext} pairs disagree on {hash_attr}"
    )
    return train_ds, holdout_ds


def _trajectory_dataset(files) -> TrajectoryDataset:
    """A trajectory dataset, the corpus's information-condition arm adopted
    first (it must precede the first dataset; re-adopting the same arm for
    the holdout is a no-op)."""
    adopt_information_condition(files)
    return TrajectoryDataset(files)


def load_datasets(store, params) -> tuple[TrajectoryDataset, TrajectoryDataset]:
    """The trajectory (.sobs) side of the split."""
    train_ds, holdout_ds = _load_split(store, params, ".sobs", _trajectory_dataset, "proposer_hash")
    if holdout_ds is train_ds:
        timed_print("no held-out pairs; metrics are on-train")
    return train_ds, holdout_ds


def _simmed_positions(mset_path) -> set[tuple[int, int]]:
    """MsetDataset's `select` for the distillation side: the stem's trajectory
    positions (evidence.dataset.trajectory_positions says why only those)."""
    return trajectory_positions(mset_path.with_suffix(".sobs"))


def _simmed_mset(files) -> MsetDataset:
    """An .mset dataset restricted to the stems' trajectory positions; a
    selection that matches nothing is a broken .sobs/.mset pairing and fails
    here rather than starving the joint step (cycle_batches would spin)."""
    ds = MsetDataset(mset_files=files, select=_simmed_positions)
    if ds.num_positions == 0:
        raise ValueError(f"no trajectory position found in the .mset labels of {files[0].parent}")
    return ds


def load_distill_datasets(store, params) -> tuple[MsetDataset, MsetDataset]:
    """The unfrozen mode's .mset side of the same split, restricted to the
    trajectory positions (the corpus's arm was adopted from the .sobs side;
    the .mset labels were made under it)."""
    return _load_split(store, params, ".mset", _simmed_mset, "model_hash")


def absorb_new_pairs(store, params, train_ds, holdout_ds, ext: str = ".sobs") -> int:
    """Ingest every `ext` pair delivered since the last pass into the side the
    split assigns it (a pair's side is fixed the first time it is seen)."""
    train_files, holdout_files = split_pairs(store, params.holdout_every, ext)
    seen = set(train_ds.files) | set(holdout_ds.files)
    added = train_ds.absorb(sorted(f for f in train_files if f not in seen))
    if holdout_ds is not train_ds:
        added += holdout_ds.absorb(sorted(f for f in holdout_files if f not in seen))
    return added


def epochs_left(params, state: EvidenceTrainState) -> bool:
    return params.train_epochs == 0 or state.settled_epochs < params.train_epochs


def build_optimizer(model, params) -> torch.optim.AdamW:
    """AdamW over the trainable params: the evidence path at `lr`, and --
    unfrozen -- the backbone as its own group at `lr * backbone_lr_mult`
    (train_loop.set_lr applies the schedule through the group's `lr_mult`)."""
    groups = [{"params": model.evidence_parameters()}]
    if params.unfreeze_backbone:
        groups.append({"params": model.backbone_parameters(), "lr_mult": params.backbone_lr_mult})
    return torch.optim.AdamW(groups, lr=params.lr, weight_decay=params.weight_decay)


def cycle_batches(dataset: MsetDataset, positions_per_batch: int, epoch: int):
    """An endless stream of distillation batches for one pass: the dataset's
    epoch under a pass-specific shuffle, restarted (reshuffled) whenever it
    runs dry, so the trajectory side alone paces the pass."""
    cycle = 0
    while True:
        yield from dataset.iter_batches(positions_per_batch, seed=cycle, epoch_index=epoch)
        cycle += 1


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
        self.max_e = max_evidence_width(params)
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


class StudentReference:
    """The frozen student's held-out sim soft-CE (`student_wld_ce`, `_ev`) --
    the flat reference line an unfrozen run's moving `plain_wld_ce` is read
    against, on the same eval prefix draw (evaluate's fixed seed). Computed
    from a pristine copy of the student, so it is what it claims after a
    resume too, and recomputed only when the holdout grows."""

    def __init__(self, student_checkpoint: str, device, batch_positions: int, max_e: int):
        self.model, _ = load_student(student_checkpoint, device)
        self.model.eval()
        self.device = device
        self.batch_positions = batch_positions
        self.max_e = max_e
        self._holdout_size = -1
        self._metrics: dict[str, float] = {}

    def metrics(self, holdout_ds: TrajectoryDataset) -> dict[str, float]:
        if holdout_ds.num_positions != self._holdout_size:
            m = evaluate(self.model, holdout_ds, self.device, self.batch_positions, self.max_e)
            self._metrics = {
                f"student_wld_ce{suffix}": m[f"plain_wld_ce{suffix}"]
                for suffix in ("", "_ev")
                if f"plain_wld_ce{suffix}" in m
            }
            self._holdout_size = holdout_ds.num_positions
        return dict(self._metrics)


def distill_metrics(model, holdout_ds: MsetDataset, device, batch_positions: int, cfg) -> dict:
    """The plain pass's distillation health on the .mset holdout, as
    `distill_*` series: the student trainer's ranking metrics against the
    teacher (recall@1 with the incumbent baseline, Spearman), the plane CE,
    and the distillation loss itself."""
    m = mset_eval.evaluate(
        model, holdout_ds, device, positions_per_batch=batch_positions, loss_cfg=cfg
    )
    out = {
        "distill_recall1": m["recall@1"],
        "distill_recall1_baseline": m["recall@1_baseline"],
        "distill_spearman": m["spearman"],
        "distill_loss": m["loss"],
        "distill_loss_wld": m["loss_wld"],
    }
    if "plane_ce" in m:
        out["distill_plane_ce"] = m["plane_ce"]
    return out


def save_epoch_checkpoint(paths, model, epoch: int, config: dict):
    """The per-pass checkpoint the trajectory pane loads: weights + config."""
    path = paths.checkpoint_path(epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, path)


def export_student(paths, model, epoch: int, student_cfg: dict):
    """The unfrozen mode's per-pass plain-student ONNX (models/
    model_epoch_NNNN.onnx), stamped with the student's arm and version as the
    mset trainer stamps its own. The export covers the plain path only (the
    evidence path's ONNX is roadmap item 3)."""
    export_onnx(
        model,
        paths.onnx_path(epoch),
        student_cfg["spatial_planes"],
        student_cfg["scalar_size"],
        opp_leave_input=student_cfg["open_leaves"],
        move_encoding_version=student_cfg["move_encoding_version"],
    )


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
    *_acc series (the Loss tab's Accuracy panel) with the rest in the table.
    An unfrozen pass adds its joint-step terms (loss_sim, loss_distill_*) and
    the student-reference / distill_* holdout series."""
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
    for k in DISTILL_LOSS_KEYS:
        if k in losses:
            record[f"loss_{k}"] = losses[k]
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
    record.update({k: v for k, v in m.items() if k.startswith(("posset_", "student_", "distill_"))})
    return record


def _distillation(params, ctx, epoch: int) -> Distillation | None:
    """The joint step's distillation side for pass `epoch`; None in frozen mode."""
    if ctx["distill_train_ds"] is None:
        return None
    batches = cycle_batches(ctx["distill_train_ds"], params.batch_positions, epoch)
    return Distillation(batches, ctx["distill_loss_cfg"], params.lambda_sim)


def _holdout_metrics(model, device, params, ctx) -> dict:
    """The pass's held-out readout: plain vs conditioned on the trajectory
    holdout and the position set; unfrozen, also the student reference and
    the plain pass's distillation health."""
    m = evaluate(model, ctx["holdout_ds"], device, params.batch_positions, ctx["max_e"])
    m.update(ctx["posset"].metrics(model, device))
    if ctx["distill_holdout_ds"] is not None:
        m.update(ctx["student_ref"].metrics(ctx["holdout_ds"]))
        m.update(
            distill_metrics(
                model,
                ctx["distill_holdout_ds"],
                device,
                params.batch_positions,
                ctx["distill_loss_cfg"],
            )
        )
    return m


def _pass_line(epoch, state, params, result, m, lr_now, train_s, settled, ctx) -> str:
    budget = (
        f"{state.settled_epochs}/{params.train_epochs}"
        if settled
        else f"corpus still growing, {ctx['train_ds'].num_positions} positions"
    )
    distill = ""
    if "distill_recall1" in m:
        distill = (
            f"student={m.get('student_wld_ce', float('nan')):.4f} "
            f"distill r@1={m['distill_recall1']:.3f} loss={m['distill_loss']:.4f} "
        )
    return (
        f"[pass {epoch}] rows={state.rows_trained} loss={result.losses['total']:.4f} "
        f"wld_ce cond={m.get('cond_wld_ce', float('nan')):.4f} "
        f"plain={m.get('plain_wld_ce', float('nan')):.4f} "
        f"(ev rows: cond={m.get('cond_wld_ce_ev', float('nan')):.4f} "
        f"plain={m.get('plain_wld_ce_ev', float('nan')):.4f}) {distill}"
        f"gain_mae={m.get('gain_mae', float('nan')):.4f} "
        f"gain_hit={m.get('gain_hit', float('nan')):.3f} "
        f"(base {m.get('gain_hit_baseline', float('nan')):.3f}) "
        f"p0_maxdiff={m['exact_p0_maxdiff']:.1e} lr={lr_now:.2e} "
        f"skipped={result.skipped} {train_s:.1f}s [{budget}]"
    )


def train_one_epoch(model, optimizer, recorder, paths, device, params, state, ctx, settled: bool):
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
        distill=_distillation(params, ctx, epoch),
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
    m = _holdout_metrics(model, device, params, ctx)
    eval_s = time.time() - t1
    lr_now = ctx["lr_controller"].current
    timed_print(_pass_line(epoch, state, params, result, m, lr_now, train_s, settled, ctx))
    record = _metrics_record(epoch, state, settled, result.losses, m, lr_now, result.skipped)
    checkpoint.save(paths, model, optimizer, state, ctx["config"])
    save_epoch_checkpoint(paths, model, epoch, ctx["config"])
    if params.unfreeze_backbone:
        # Frozen, the plain model is the student byte for byte; only an
        # unfrozen pass has a new plain student to export.
        export_student(paths, model, epoch, ctx["config"]["student"])
    # Delivered last: the record is what makes the pass visible, and every
    # artifact it stands for is now on disk.
    recorder.commit_generation(epoch, state.rows_trained, record)
    ctx["stats"].cycle_done(
        {"train_s": train_s, "eval_s": eval_s},
        units=state.rows_trained - rows_before,
        nbytes=0,
    )


def _loss_weights(params) -> dict:
    """Each recorded loss series' coefficient in the optimized total, for the
    dashboard's stacked loss panel: the sim terms (times lambda_sim in the
    joint total), plus the distillation terms when unfrozen."""
    sim = {"loss_wld": 1.0, "loss_score_diff": params.lambda_sd, "loss_gain": params.lambda_gain}
    if not params.unfreeze_backbone:
        return sim
    return {
        "loss_distill_wld": 1.0,
        "loss_distill_score_diff": params.lambda_sd,
        "loss_distill_planes": params.lambda_planes,
        **{k: params.lambda_sim * w for k, w in sim.items()},
    }


def _report_model(model, params) -> int:
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    what = "backbone unfrozen" if params.unfreeze_backbone else "fusion + proves-best head"
    print(f"Model: {n_total:,} parameters, {n_train:,} trainable ({what})")
    return n_train


def _distill_ctx(store, params, device, max_e) -> dict:
    """The unfrozen mode's run-context entries; None in frozen mode."""
    if not params.unfreeze_backbone:
        return {"distill_train_ds": None, "distill_holdout_ds": None}
    train_ds, holdout_ds = load_distill_datasets(store, params)
    print(
        f"distill: {train_ds.num_positions} positions / {train_ds.num_candidates} candidates; "
        f"eval: {holdout_ds.num_positions} positions / {holdout_ds.num_candidates} candidates"
    )
    return {
        "distill_train_ds": train_ds,
        "distill_holdout_ds": holdout_ds,
        "distill_loss_cfg": mset_train_loop.LossConfig.from_args(params),
        "student_ref": StudentReference(
            params.student_checkpoint, device, params.batch_positions, max_e
        ),
    }


def _absorb(store, params, ctx) -> int:
    """Take up the store's new pairs on both sides of the split; the count is
    the .sobs side's, which paces the corpus clock."""
    absorbed = absorb_new_pairs(store, params, ctx["train_ds"], ctx["holdout_ds"])
    if ctx["distill_train_ds"] is not None:
        absorb_new_pairs(store, params, ctx["distill_train_ds"], ctx["distill_holdout_ds"], ".mset")
    return absorbed


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

    model, student_cfg = load_student(
        params.student_checkpoint, device, freeze=not params.unfreeze_backbone
    )
    # The move rows this trainer encodes (candidates and evidence tokens) must
    # be the rows the student learned; a version bump in the engine would
    # otherwise train the fusion against embeddings of the wrong layout.
    if student_cfg["move_encoding_version"] != move_encoding_version():
        print(
            f"error: the student was trained under move encoding version "
            f"{student_cfg['move_encoding_version']} but the engine encodes version "
            f"{move_encoding_version()}"
        )
        return 1
    store = paths.data_dir / SLOGS_DIR
    wait_for_store(store, params)
    train_ds, holdout_ds = load_datasets(store, params)
    if train_ds.open_leaves != student_cfg["open_leaves"]:
        print(
            f"error: the corpus was simmed with open_leaves={train_ds.open_leaves} but the "
            f"student was trained with open_leaves={student_cfg['open_leaves']}"
        )
        return 1
    max_pool = max_pool_width(params)
    if train_ds.max_trajectory > max_pool:
        print(
            f"error: a trajectory holds {train_ds.max_trajectory} candidates, more than the "
            f"recipe's {max_pool} (1 + on_policy_max + off-policy floor): the corpus was not "
            "simmed with this tag's recipe"
        )
        return 1
    # The padded evidence width the model conditions on -- only the anchor and
    # on-policy picks ever enter an evidence set, never the off-policy floor.
    max_e = max_evidence_width(params)
    print(
        f"train: {train_ds.num_positions} positions / {train_ds.num_candidates} simmed candidates; "
        f"eval: {holdout_ds.num_positions} positions / {holdout_ds.num_candidates} candidates "
        f"(open_leaves={train_ds.open_leaves}, proposer {train_ds.proposer_hash[:12]})"
    )
    n_train = _report_model(model, params)
    optimizer = build_optimizer(model, params)

    recorder = TrainRecorder(ctx.sink)
    recorder.publish_run(
        ctx.tag, asdict(params), n_train, _loss_weights(params), default_controls()
    )
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
        **_distill_ctx(store, params, device, max_e),
    }

    state = checkpoint.resume(paths, model, optimizer, device, state_cls=EvidenceTrainState)
    run_ctx["lr_controller"] = WsdLrController(
        recorder, WsdSchedule.from_params(params), state.rows_trained
    )
    try:
        clock = pair_store.CorpusClock(store, params.target_pairs, ".sobs")
        while epochs_left(params, state):
            settled = clock.is_final(_absorb(store, params, run_ctx))
            train_one_epoch(
                model, optimizer, recorder, paths, device, params, state, run_ctx, settled
            )
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
