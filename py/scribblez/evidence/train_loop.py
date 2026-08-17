"""The evidence trainer's forward, loss, epoch loop, and held-out metrics.

The forward is the model's staged path with the backbone frozen: board trunk,
move encodings and the evidence-free first pass run under no_grad (the first
pass supplies the predicted half of every evidence token), then the fusion
stage and the conditioned re-score run with gradients -- only EvidenceFusion
and the proves-best head learn. Loss is taken on the held-out simmed
candidates (outside the prefix), against their own sim outcomes.

Metrics compare the conditioned pass with the plain one on the same held-out
rows, so "does conditioning learn anything from sim outcomes" is read directly:
soft-CE against the sim's W/D/L, value MAE, the proves-best gain error, and the
acquisition hit rate (does argmax gain over a position's held-out candidates
pick the one that actually simmed best; the plain value's argmax is the
baseline). Prefix-0 rows double as the exactness check -- conditioned and plain
must agree there to floating-point noise.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from scribblez.move_set_eval.evidence import build_evidence_inputs, collate_evidence
from scribblez.move_set_eval.model import win_equity

LOSS_KEYS = ("total", "wld", "score_diff", "gain")

_INPUT_KEYS = ("input_spatial", "input_scalar")
_MOVE_KEYS = (
    "move_letters",
    "move_blanks",
    "move_squares",
    "move_tile_mask",
    "move_scalars",
    "move_pos_id",
)
_TARGET_KEYS = ("sim_wld", "sim_delta", "sim_value", "target_gain", "held_out", "slot")


@dataclass
class LossConfig:
    lambda_sd: float
    lambda_gain: float
    huber_delta_mean: float
    huber_delta_std: float
    huber_delta_gain: float

    @classmethod
    def from_args(cls, args) -> LossConfig:
        return cls(
            args.lambda_sd,
            args.lambda_gain,
            args.huber_delta_mean,
            args.huber_delta_std,
            args.huber_delta_gain,
        )


@dataclass
class EpochResult:
    losses: dict[str, float]  # held-out-row-weighted averages
    n_batches: int
    rows: int  # held-out rows this epoch
    rows_trained: int  # cumulative held-out rows across the run


def _first_pass_rows(plain: dict[str, torch.Tensor], rows: torch.Tensor) -> dict:
    return {k: plain[k][rows] for k in ("wld", "score_diff", "planes")}


def conditioned_forward(
    model, batch: dict, device, max_e: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """(plain, conditioned) score_moves outputs over the batch's flattened
    candidates. The plain pass is computed without gradients (the backbone is
    frozen and its outputs are the evidence tokens' predicted half); the
    conditioned pass reads each position's evidence prefix."""
    spatial, scalar = (batch[k].to(device) for k in _INPUT_KEYS)
    move_args = tuple(batch[k].to(device) for k in _MOVE_KEYS)
    pos_id, slot = move_args[-1], batch["slot"].to(device)
    with torch.no_grad():
        board, g = model.encode_board(spatial, scalar)
        e = model.encode_moves(board, *move_args)
        plain = model.score_moves(board, g, e, pos_id)

    items = []
    for p, pos in enumerate(batch["positions"]):
        k = int(batch["prefix_sizes"][p])
        rows = (pos_id == p) & (slot < k)
        items.append(
            build_evidence_inputs(
                pos.moves[:k],
                pos.obs[:k],
                int(batch["pre_move_diff"][p]),
                _first_pass_rows(plain, rows),
                max_e=max_e,
                device=device,
            )
        )
    evidence = collate_evidence(items)
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
    carry loss) and keys the rows-clock learning rate."""
    model.train()
    sums = {k: 0.0 for k in LOSS_KEYS}
    n_batches = rows = 0
    t0 = last_progress = time.time()
    for batch in batches:
        targets = _targets(batch, device)
        m = int(targets["held_out"].sum())
        if m == 0:
            continue
        if lr_fn is not None:
            for group in optimizer.param_groups:
                group["lr"] = lr_fn(rows_trained)
        _, cond = conditioned_forward(model, batch, device, max_e)
        losses = compute_loss(cond, targets, cfg)
        optimizer.zero_grad()
        losses["total"].backward()
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
    return EpochResult(losses, n_batches, rows, rows_trained)


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
        prefix = batch["prefix_sizes"].to(device)[pos_id]
        with_ev = held & (prefix > 0)
        no_ev = held & (prefix == 0)
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
