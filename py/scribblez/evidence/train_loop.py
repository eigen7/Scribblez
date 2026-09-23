"""The evidence trainer's forward, loss, epoch loop and held-out metrics.

The forward follows the model's staged path: board trunk, move encodings, the
evidence-free first pass, then the fusion stage and the conditioned re-score.
The first pass only supplies the predicted half of each evidence token and
never receives gradient. The sole loss is on the held-out simmed candidates
(those outside the evidence subset), against their own sim outcomes.

With the backbone frozen, only EvidenceFusion and the proves-best head learn.
Unfrozen (the move proposal model, docs/roadmap.md item 5), the whole model
trains on the sim signal, the backbone at its own learning rate, and the
empty-subset rows keep the plain pass calibrated; there is no distillation
term.

The metrics compare the conditioned pass with the plain one on the same
held-out rows, which directly answers whether conditioning learns from sim
outcomes: soft-CE against the sim's W/D/L, value MAE, gain error, and the
acquisition hit rate (whether argmax gain over a position's held-out
candidates picks the one that simmed best, against the plain value's argmax as
baseline). Empty-subset rows double as an exactness check: conditioned and
plain must agree there up to floating-point noise.
"""

from __future__ import annotations

import contextlib
import functools
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from scribblez.evidence_fusion import (
    NUM_EVIDENCE_PLANES,
    NUM_EVIDENCE_SCALARS,
    NUM_OBSERVED_PLANES,
    NUM_PREDICTED_PLANES,
    EvidenceInputs,
    best_so_far,
)
from scribblez.move_set_eval.evidence import observed_scalars
from scribblez.move_set_eval.model import INPUT_KEYS, MOVE_KEYS, footprint_slot_planes, win_equity
from scribblez.sim_evidence.sobs import BOARD, candidate_slot_planes, observed_slot_planes

# The sim-outcome loss terms every epoch reports.
LOSS_KEYS = ("total", "wld", "score_diff", "gain")

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
    """The batch's evidence sets as (P, max_e, ...) inputs, equal to collating
    move_set_eval.evidence.build_evidence_inputs per position.

    Evidence rows are the batch's own candidate rows marked `in_evidence`,
    scattered to padded slot pos_id * max_e + ev_index. Their move inputs are
    reused as is (same moves and pre-move differential as a fresh encode), the
    predicted half comes from the plain pass over those rows, and the observed
    half from the .sobs records selected by the same mask. Both halves thus
    enumerate members in the same order.
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
    observed_p = torch.from_numpy(observed_slot_planes(obs_np)).to(device=device, dtype=dtype)
    candidate_p = torch.from_numpy(candidate_slot_planes(moves_np)).to(device=device, dtype=dtype)
    observed_s = torch.from_numpy(observed_scalars(obs_np)).to(device=device, dtype=dtype)

    pred_end = NUM_OBSERVED_PLANES + NUM_PREDICTED_PLANES
    planes = observed_p.new_zeros((int(sel.sum()), NUM_EVIDENCE_PLANES, BOARD, BOARD))
    planes[:, :NUM_OBSERVED_PLANES] = observed_p
    planes[:, NUM_OBSERVED_PLANES:pred_end] = footprint_slot_planes(plain["planes"][sel]).to(dtype)
    planes[:, pred_end:] = candidate_p
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
    candidates, the conditioned pass reading each unit's evidence subset.

    The trunk and move encodings carry gradient only when the backbone is
    unfrozen. The plain pass never does: it is an input (the tokens'
    predicted half), not a training path."""
    spatial, scalar = (batch[k].to(device) for k in INPUT_KEYS)
    move_args = tuple(batch[k].to(device) for k in MOVE_KEYS)
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
    best = best_so_far(evidence.obs_scalars, evidence.mask)
    return plain, model.score_moves(board_c, g_c, e, pos_id, best)


def compute_loss(
    outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], cfg: LossConfig
) -> dict[str, torch.Tensor]:
    """Sim-outcome loss, averaged over held-out rows: soft cross-entropy
    against the sim's W/D/L frequencies, and Huber losses for the score-diff
    mean/std (against the sim delta moments) and the proves-best gain."""
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


def set_lr(optimizer, lr: float):
    """Set every param group to `lr` times its `lr_mult` (default 1), which
    lets the unfrozen backbone run at a fraction of the evidence path's rate."""
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
    lr_fn: Callable[[int], float] | None = None,
    rows_trained: int = 0,
    on_batch: Callable[[int, int, float, int], None] | None = None,
) -> EpochResult:
    """One training pass. rows_trained counts held-out rows (the rows that
    carry loss) and drives lr_fn. Batches with a non-finite loss or gradient
    are skipped and counted in the result."""
    model.train()
    trainable = [p for group in optimizer.param_groups for p in group["params"]]
    sums = {k: 0.0 for k in LOSS_KEYS}
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
        # One non-finite step poisons Adam's moments and every later weight.
        # The trainer stops the run if skips are more than rare.
        if not torch.isfinite(losses["total"]):
            skipped += 1
            continue
        optimizer.zero_grad()
        losses["total"].backward()
        # Backward can overflow to inf/nan gradients under a finite loss.
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
    """Running sums and counts for per-key means."""

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
    """Held-out metrics, plain vs conditioned, over a subset draw fixed by
    `seed`. Keys ending in "_ev" cover only rows with non-empty evidence."""
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
