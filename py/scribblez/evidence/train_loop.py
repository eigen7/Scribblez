"""The evidence trainer's forward, loss, epoch loop, and held-out metrics.

The forward is the model's staged path: board trunk, move encodings, the
evidence-free first pass (which supplies the predicted half of every evidence
token, and is never a training path itself), then the fusion stage and the
conditioned re-score. Loss is taken on the held-out simmed candidates
(outside the prefix), against their own sim outcomes. With the backbone
frozen the trunk and move encodings run under no_grad and only EvidenceFusion
and the proves-best head learn; unfrozen, they carry gradients from the
conditioned loss, and each step is joint with a distillation batch over the
same games' .mset labels (Distillation) -- the ordinary student objective
anchoring the plain pass while the sim loss trains the conditioned one.

Metrics compare the conditioned pass with the plain one on the same held-out
rows, so "does conditioning learn anything from sim outcomes" is read directly:
soft-CE against the sim's W/D/L, value MAE, the proves-best gain error, and the
acquisition hit rate (does argmax gain over a position's held-out candidates
pick the one that actually simmed best; the plain value's argmax is the
baseline). Prefix-0 rows double as the exactness check -- conditioned and plain
must agree there to floating-point noise.
"""

from __future__ import annotations

import contextlib
import functools
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from scribblez.evidence_fusion import NUM_EVIDENCE_PLANES, NUM_EVIDENCE_SCALARS, EvidenceInputs
from scribblez.move_set_eval import train_loop as mset_train_loop
from scribblez.move_set_eval.evidence import observed_planes, observed_scalars
from scribblez.move_set_eval.model import footprint_cell_marginal, win_equity
from scribblez.sim_evidence.sobs import BOARD

# The sim-outcome loss terms every epoch reports; a joint (unfrozen) epoch adds
# DISTILL_LOSS_KEYS.
LOSS_KEYS = ("total", "wld", "score_diff", "gain")
DISTILL_LOSS_KEYS = (
    "sim",
    "distill",
    "distill_wld",
    "distill_score_diff",
    "distill_planes",
)

_INPUT_KEYS = ("input_spatial", "input_scalar")
_MOVE_KEYS = (
    "move_letters",
    "move_blanks",
    "move_squares",
    "move_tile_mask",
    "move_scalars",
    "move_pos_id",
)
_TARGET_KEYS = ("sim_wld", "sim_delta", "sim_value", "target_gain", "held_out")


@dataclass
class LossConfig:
    lambda_sd: float
    lambda_gain: float
    huber_delta_mean: float
    huber_delta_std: float
    huber_delta_gain: float
    grad_clip: float  # max gradient norm over the trainable params (0 = no clipping)

    @classmethod
    def from_args(cls, args) -> LossConfig:
        return cls(
            args.lambda_sd,
            args.lambda_gain,
            args.huber_delta_mean,
            args.huber_delta_std,
            args.huber_delta_gain,
            args.grad_clip,
        )


@dataclass
class Distillation:
    """The joint step's distillation side (backbone unfrozen): an endless
    stream of .mset batches -- one is consumed per trajectory batch -- and
    the student objective over each (move_set_eval's LossConfig). The step's
    total is distill + lambda_sim * sim."""

    batches: Iterator[dict]
    cfg: mset_train_loop.LossConfig
    lambda_sim: float


@dataclass
class EpochResult:
    losses: dict[str, float]  # held-out-row-weighted averages
    n_batches: int
    rows: int  # held-out rows this epoch
    rows_trained: int  # cumulative held-out rows across the run
    skipped: int = 0  # batches whose loss was non-finite (no step taken)


def _scatter_rows(rows: torch.Tensor, flat: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Selected rows scattered to their padded (P, max_e, ...) slots (zeros elsewhere)."""
    p, max_e = shape
    out = rows.new_zeros((p * max_e, *rows.shape[1:]))
    out[flat] = rows
    return out.view(p, max_e, *rows.shape[1:])


def batch_evidence_inputs(
    batch: dict, move_args: tuple, plain: dict[str, torch.Tensor], max_e: int, device
) -> EvidenceInputs:
    """The batch's evidence sets as (P, max_e, ...) inputs in one shot -- the
    batched sibling of move_set_eval.evidence.build_evidence_inputs (the
    single-position deployment builder), equal to collating it per position.

    The evidence rows are the batch's own candidate rows the subset marks
    (`in_evidence`): their move half is those rows' move inputs (same moves,
    same pre-move differential, so identical to a fresh encode), scattered to
    padded index pos_id * max_e + `ev_index` (the member's compact slot within
    its unit's set, so an arbitrary subset packs the way the deployment builder
    does); the predicted half is the plain pass over the same rows, on device;
    the observed half is the .sobs records the subset selects, gathered from the
    flattened records by the same membership mask, so both halves enumerate the
    unit blocks in ascending slot order and cannot drift apart.
    """
    letters, blanks, squares, tile_mask, scalars, pos_id = move_args
    p = len(batch["positions"])
    in_evidence = batch["in_evidence"]
    sel = in_evidence.to(device)
    flat = (pos_id * max_e + batch["ev_index"].to(device))[sel]
    dtype = scalars.dtype
    scatter = functools.partial(_scatter_rows, flat=flat, shape=(p, max_e))

    sel_np = in_evidence.numpy()
    moves_np = batch["all_moves"][sel_np]
    obs_np = batch["all_obs"][sel_np]
    observed_p = torch.from_numpy(observed_planes(moves_np, obs_np)).to(device=device, dtype=dtype)
    observed_s = torch.from_numpy(observed_scalars(obs_np)).to(device=device, dtype=dtype)

    planes = observed_p.new_zeros((int(sel.sum()), NUM_EVIDENCE_PLANES, BOARD, BOARD))
    planes[:, :4] = observed_p[:, :4]
    planes[:, 4:8] = footprint_cell_marginal(plain["planes"][sel]).to(dtype)
    planes[:, 8] = observed_p[:, 4]
    predicted_s = torch.cat(
        [torch.softmax(plain["wld"][sel], dim=1), plain["score_diff"][sel] / 100.0], dim=1
    ).to(dtype)
    obs_scalars = torch.cat([observed_s, predicted_s], dim=1)
    assert obs_scalars.shape[1] == NUM_EVIDENCE_SCALARS

    mask = torch.zeros(p * max_e, dtype=torch.bool, device=device)
    mask[flat] = True
    return EvidenceInputs(
        letters=scatter(letters[sel]),
        blanks=scatter(blanks[sel]),
        squares=scatter(squares[sel]),
        tile_mask=scatter(tile_mask[sel]),
        scalars=scatter(scalars[sel]),
        obs_planes=scatter(planes),
        obs_scalars=scatter(obs_scalars),
        mask=mask.view(p, max_e),
    )


def conditioned_forward(
    model, batch: dict, device, max_e: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """(plain, conditioned) score_moves outputs over the batch's flattened
    candidates; the conditioned pass reads each position's evidence prefix.
    The trunk and move encodings carry gradients only when the backbone is
    unfrozen; the plain pass never does -- it is the evidence tokens'
    predicted half, an input, so it is computed under no_grad in either
    mode."""
    spatial, scalar = (batch[k].to(device) for k in _INPUT_KEYS)
    move_args = tuple(batch[k].to(device) for k in _MOVE_KEYS)
    pos_id = move_args[-1]
    backbone_grad = torch.no_grad() if model.backbone_frozen else contextlib.nullcontext()
    with backbone_grad:
        board, g = model.encode_board(spatial, scalar)
        e = model.encode_moves(board, *move_args)
    with torch.no_grad():
        plain = model.score_moves(board, g, e, pos_id)
    evidence = batch_evidence_inputs(batch, move_args, plain, max_e, device)
    tokens, spatial_feats = model.encode_evidence(board, evidence)
    board_c, g_c = model.evidence_fusion(board, g, tokens, spatial_feats, evidence.mask)
    return plain, model.score_moves(board_c, g_c, e, pos_id)


def compute_loss(
    outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], cfg: LossConfig
) -> dict[str, torch.Tensor]:
    """Sim-outcome loss over the held-out rows: soft cross-entropy against the
    sim's W/D/L frequencies, Huber on the score-diff mean/std against the sim
    delta moments, Huber on the proves-best gain. Means over held-out rows."""
    held = targets["held_out"]
    log_pred = F.log_softmax(outputs["wld"][held], dim=1)
    loss_wld = -(targets["sim_wld"][held] * log_pred).sum(dim=1).mean()
    sd = outputs["score_diff"][held]
    t_sd = targets["sim_delta"][held]
    loss_sd = F.huber_loss(sd[:, 0], t_sd[:, 0], delta=cfg.huber_delta_mean) + F.huber_loss(
        sd[:, 1], t_sd[:, 1], delta=cfg.huber_delta_std
    )
    loss_gain = F.huber_loss(
        outputs["gain"][held], targets["target_gain"][held], delta=cfg.huber_delta_gain
    )
    total = loss_wld + cfg.lambda_sd * loss_sd + cfg.lambda_gain * loss_gain
    return {"total": total, "wld": loss_wld, "score_diff": loss_sd, "gain": loss_gain}


def _targets(batch: dict, device) -> dict[str, torch.Tensor]:
    return {k: batch[k].to(device) for k in _TARGET_KEYS}


def joint_loss(
    sim: dict[str, torch.Tensor], distill: dict[str, torch.Tensor], lambda_sim: float
) -> dict[str, torch.Tensor]:
    """The unfrozen step's losses: the sim-outcome terms as reported by
    compute_loss, the distillation terms under DISTILL_LOSS_KEYS, and the
    optimized total distill + lambda_sim * sim."""
    return {
        **sim,
        "total": distill["total"] + lambda_sim * sim["total"],
        "sim": sim["total"],
        "distill": distill["total"],
        "distill_wld": distill["wld"],
        "distill_score_diff": distill["score_diff"],
        "distill_planes": distill["planes"],
    }


def set_lr(optimizer, lr: float):
    """Apply the schedule's rate to every param group, times the group's own
    `lr_mult` when it carries one (the unfrozen backbone's group runs at a
    fraction of the evidence path's rate)."""
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_mult", 1.0)


def run_epoch(
    model,
    optimizer,
    batches: Iterable[dict],
    device,
    cfg: LossConfig,
    max_e: int,
    *,
    distill: Distillation | None = None,
    lr_fn: Callable[[int], float] | None = None,
    rows_trained: int = 0,
    on_batch: Callable[[int, int, float, int], None] | None = None,
) -> EpochResult:
    """One training pass. rows_trained counts held-out rows (the rows that
    carry loss) and keys the rows-clock learning rate. With `distill` each
    step is joint (joint_loss) over the trajectory batch and one distillation
    batch: one backward, one step. Gradients are clipped to cfg.grad_clip
    over the optimizer's params; a batch with a non-finite loss is skipped
    (see below)."""
    model.train()
    trainable = [p for group in optimizer.param_groups for p in group["params"]]
    keys = LOSS_KEYS if distill is None else LOSS_KEYS + DISTILL_LOSS_KEYS
    sums = {k: 0.0 for k in keys}
    n_batches = rows = skipped = 0
    t0 = last_progress = time.time()
    for batch in batches:
        targets = _targets(batch, device)
        m = int(targets["held_out"].sum())
        if m == 0:
            continue
        if lr_fn is not None:
            set_lr(optimizer, lr_fn(rows_trained))
        _, cond = conditioned_forward(model, batch, device, max_e)
        losses = compute_loss(cond, targets, cfg)
        if distill is not None:
            d = mset_train_loop.batch_loss(model, next(distill.batches), device, distill.cfg)
            losses = joint_loss(losses, d, distill.lambda_sim)
        # A non-finite loss must not reach the optimizer: one such step
        # poisons Adam's moments and every weight after it. Skip the batch
        # and count it; the pass reports the count and the trainer stops the
        # run when it is anything but rare.
        if not torch.isfinite(losses["total"]):
            skipped += 1
            continue
        optimizer.zero_grad()
        losses["total"].backward()
        # Clip, and skip a step whose gradient is non-finite (an overflow in
        # backward can leave inf/nan grads under a finite loss).
        norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip or float("inf"))
        if not torch.isfinite(norm):
            skipped += 1
            continue
        optimizer.step()
        n_batches += 1
        rows += m
        rows_trained += m
        for k in sums:
            sums[k] += losses[k].item() * m
        if on_batch is not None and time.time() - last_progress > 1.0:
            on_batch(n_batches, rows, time.time() - t0, rows_trained)
            last_progress = time.time()
    losses = {k: v / max(rows, 1) for k, v in sums.items()}
    return EpochResult(losses, n_batches, rows, rows_trained, skipped)


class _Accumulator:
    """Held-out-row sums for the plain-vs-conditioned metrics, overall and on
    the evidence-bearing (prefix > 0) rows."""

    def __init__(self):
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, key: str, value: float, n: int = 1):
        self.sums[key] = self.sums.get(key, 0.0) + value
        self.counts[key] = self.counts.get(key, 0) + n

    def means(self) -> dict[str, float]:
        return {k: self.sums[k] / max(self.counts[k], 1) for k in self.sums}


def _soft_ce(logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    return -(probs * F.log_softmax(logits, dim=1)).sum(dim=1)


def _hit_rate(acc: _Accumulator, key: str, score, value, pos_id, held):
    """Per position with >= 2 held-out candidates: whether argmax `score`
    picks the held-out candidate of greatest sim value."""
    for p in pos_id.unique().tolist():
        rows = (pos_id == p) & held
        if int(rows.sum()) < 2:
            continue
        acc.add(key, float(score[rows].argmax() == value[rows].argmax()))


@torch.no_grad()
def evaluate(model, dataset, device, positions_per_batch: int, max_e: int, seed: int = 0) -> dict:
    """Held-out metrics, plain vs conditioned, on a fixed prefix draw (seed)."""
    model.eval()
    acc = _Accumulator()
    exact = 0.0
    for batch in dataset.iter_batches(positions_per_batch, seed=seed, epoch_index=0):
        t = _targets(batch, device)
        held = t["held_out"]
        if not bool(held.any()):
            continue
        plain, cond = conditioned_forward(model, batch, device, max_e)
        pos_id = batch["move_pos_id"].to(device)
        size = batch["evidence_size"].to(device)[pos_id]
        with_ev = held & (size > 0)
        no_ev = held & (size == 0)
        if bool(no_ev.any()):
            exact = max(exact, float((cond["wld"][no_ev] - plain["wld"][no_ev]).abs().max()))
        v_plain = win_equity(torch.softmax(plain["wld"], dim=1))
        v_cond = win_equity(torch.softmax(cond["wld"], dim=1))
        for suffix, rows in (("", held), ("_ev", with_ev)):
            n = int(rows.sum())
            if n == 0:
                continue
            errors = {
                "plain_wld_ce": _soft_ce(plain["wld"][rows], t["sim_wld"][rows]),
                "cond_wld_ce": _soft_ce(cond["wld"][rows], t["sim_wld"][rows]),
                "plain_value_mae": (v_plain[rows] - t["sim_value"][rows]).abs(),
                "cond_value_mae": (v_cond[rows] - t["sim_value"][rows]).abs(),
                "gain_mae": (cond["gain"][rows] - t["target_gain"][rows]).abs(),
            }
            for key, err in errors.items():
                acc.add(f"{key}{suffix}", float(err.sum()), n)
        _hit_rate(acc, "gain_hit", cond["gain"], t["sim_value"], pos_id, held)
        _hit_rate(acc, "gain_hit_baseline", v_plain, t["sim_value"], pos_id, held)
        _hit_rate(acc, "gain_hit_ev", cond["gain"], t["sim_value"], pos_id, with_ev)
        _hit_rate(acc, "gain_hit_ev_baseline", v_plain, t["sim_value"], pos_id, with_ev)
    metrics = acc.means()
    metrics["exact_p0_maxdiff"] = exact
    metrics["rows"] = acc.counts.get("plain_wld_ce", 0)
    metrics["rows_ev"] = acc.counts.get("plain_wld_ce_ev", 0)
    return metrics
