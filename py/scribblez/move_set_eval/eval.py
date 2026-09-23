"""Evaluation for the move set evaluation model: how well it reproduces the
teacher's ranking of each position's candidate set.

The model's job is to filter candidates, so its top-K must keep the moves the
teacher would pick. Per position, candidates are ranked by the teacher's stored
win-equity and by the model's predicted win-equity (model.win_equity), and the
module reports, averaged over positions:

  * recall@K: the fraction of the teacher's top-K that is in the model's top-K.
  * regret@K: the teacher win-equity lost by keeping only the model's top-K.
    Recall counts dropping a near-tie the same as dropping the only winning
    move; regret prices the miss.
  * spearman: rank correlation of the two rankings over the whole set
    (positions with >= 2 candidates).
  * exch_retention@K: how often the teacher's best exchange makes the model's
    top-K over all candidates (positions with any exchange).
  * exch_rank_regret: teacher win-equity lost by taking the model's favourite
    exchange over the teacher's (positions with >= 2 exchanges). This measures
    whether the model ranks which tiles to keep.

Every metric has a "_baseline" twin: the same metric for the incumbent's
ranking, recovered from the stored candidate order (_baseline_ranking).

The numbers are meaningful only on a full-sweep dataset, where a position's
candidates are all of its legal moves. A stratified sample has ~15 candidates
per position, the same kind the model trained on, and never contains the tail
moves the filter exists to catch. This module treats both alike, so the caller
chooses the dataset (see eval_slice_line).
"""

from __future__ import annotations

import numpy as np
import torch

from .model import MOVE_KEYS, win_equity
from .train_loop import TARGET_KEYS

DEFAULT_KS = (1, 3, 5)

# Moves per forward pass. Swept positions have hundreds of candidates, so the
# position count alone no longer bounds a batch's memory. The model also pads
# queries to a (positions x largest candidate set) grid, a second cost of
# similar size that this budget does not bound: one near-cap position among
# small ones costs more than its candidate count suggests. Measured
# whole-forward peaks under no_grad, C=192, 64 positions per batch:
#   +429 MiB  worst admitted shape: one position at the 1500 sweep cap, M=16305
#   +366 MiB  same M, all positions the same size
#   +342 MiB  same grid, M=1563
# On stratified data the position bound always binds first.
MAX_CANDIDATES_PER_BATCH = 16384


def eval_slice_line(dataset) -> str:
    """A startup-log line saying what slice the metrics are read on: a
    stratified holdout (provisional numbers) or a full sweep with its
    legal-move coverage under the generator's cap."""
    if not dataset.full_sweep:
        return "eval slice: stratified (recall/regret are provisional -- no tail coverage)"
    coverage, truncated = dataset.sweep_coverage
    return (
        f"eval slice: full sweep, {coverage:.3f} mean coverage of legal moves, "
        f"{truncated}/{dataset.num_positions} positions truncated by the cap"
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation. Ties break arbitrarily, which is harmless
    for continuous equity values."""
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0:
        return 0.0
    return float((ra * rb).sum() / denom)


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k best-scored candidates (k capped at the count)."""
    return np.argsort(-scores)[: min(k, len(scores))]


def _topk_recall(teacher: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Fraction of the teacher's top-k that is in the model's top-k."""
    teacher_top = set(_topk_indices(teacher, k).tolist())
    pred_top = set(_topk_indices(pred, k).tolist())
    return len(teacher_top & pred_top) / len(teacher_top)


def _regret(teacher: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Teacher's best win-equity minus the best among `pred`'s top-k."""
    return float(teacher.max() - teacher[_topk_indices(pred, k)].max())


def _exchange_mask(scalars: np.ndarray) -> np.ndarray:
    """Exchange candidates, from the encoded move scalars: not a play and at
    least one tile surrendered (a pass has neither)."""
    return (scalars[:, 2] == 0.0) & (scalars[:, 1] > 0.0)


def _baseline_ranking(n: int) -> np.ndarray:
    """The incumbent's ranking as scores: earlier stored index ranks higher.

    On a full-sweep position the stored order is exactly the static-equity
    ranking (move_set_eval_candidates.h keeps the sweep a subsequence of it),
    so the baseline metrics are the true incumbent's.

    On a stratified position the generator stores the played move first, then
    any forced trajectory candidates, then the head of the equity ranking,
    then shuffled samples. The baseline is then exact only for top-1 (and for
    k <= 1 + StratumQuotas::top when nothing was forced); Spearman is a floor.
    If the non-finite-target filter dropped the played move, index 0 is
    instead the next surviving candidate."""
    return -np.arange(n, dtype=np.float64)


@torch.no_grad()
def evaluate(
    model,
    dataset,
    device,
    positions_per_batch: int = 64,
    ks=DEFAULT_KS,
    seed: int = 0,
    max_candidates_per_batch: int = MAX_CANDIDATES_PER_BATCH,
    loss_cfg=None,
) -> dict[str, float]:
    """Run the model over `dataset` and return the metrics named in the module
    docstring (with "_baseline" twins), the "positions" and
    "positions_with_exchanges" denominators, and "plane_ce" when the slice has
    plane targets. With a train_loop.LossConfig as `loss_cfg`, also returns
    the candidate-weighted distillation loss as "loss" and "loss_<term>".
    """
    model.eval()
    sums = {}
    loss_sums: dict[str, float] = {}
    exch_sums = {"exch_rank_regret": 0.0, "exch_rank_regret_baseline": 0.0}
    for k in ks:
        sums[f"recall@{k}"] = sums[f"recall@{k}_baseline"] = 0.0
        sums[f"regret@{k}"] = sums[f"regret@{k}_baseline"] = 0.0
        exch_sums[f"exch_retention@{k}"] = exch_sums[f"exch_retention@{k}_baseline"] = 0.0
    n_positions = 0
    plane_ce_sum = 0.0
    plane_candidates = 0
    spearman_sums = {"spearman": 0.0, "spearman_baseline": 0.0}
    n_ranked = 0
    n_exch = 0  # positions with any exchange candidate (retention denominator)
    n_exch_ranked = 0  # positions with >= 2 exchanges (rank-regret denominator)

    for batch in dataset.iter_batches(
        positions_per_batch, seed=seed, max_candidates=max_candidates_per_batch
    ):
        inputs = (batch["input_spatial"].to(device), batch["input_scalar"].to(device))
        move_args = tuple(batch[key].to(device) for key in MOVE_KEYS)
        out = model(*inputs, *move_args)
        if loss_cfg is not None:
            _accumulate_loss(loss_sums, out, batch, device, loss_cfg)
        if "target_planes" in batch:
            m = batch["move_pos_id"].shape[0]
            # Same as compute_loss's plane term.
            log_pred = torch.nn.functional.log_softmax(out["planes"], dim=-1)
            ce = -(batch["target_planes"].to(device) * log_pred).sum(dim=-1).mean()
            plane_ce_sum += ce.item() * m
            plane_candidates += m
        pred_eq = win_equity(out["wld"].softmax(dim=1)).cpu().numpy()
        teacher_eq = win_equity(batch["target_wld"]).numpy()
        pos_id = batch["move_pos_id"].numpy()
        exchange = _exchange_mask(batch["move_scalars"].numpy())

        for p in np.unique(pos_id):
            sel = pos_id == p
            t = teacher_eq[sel]
            rankings = {"": pred_eq[sel], "_baseline": _baseline_ranking(len(t))}
            n_positions += 1
            for suffix, q in rankings.items():
                for k in ks:
                    sums[f"recall@{k}{suffix}"] += _topk_recall(t, q, k)
                    sums[f"regret@{k}{suffix}"] += _regret(t, q, k)
            if len(t) >= 2:
                for suffix, q in rankings.items():
                    spearman_sums[f"spearman{suffix}"] += _spearman(t, q)
                n_ranked += 1

            exch_idx = np.flatnonzero(exchange[sel])
            if len(exch_idx) == 0:
                continue
            n_exch += 1
            best_exch = exch_idx[np.argmax(t[exch_idx])]
            for suffix, q in rankings.items():
                for k in ks:
                    retained = best_exch in _topk_indices(q, k)
                    exch_sums[f"exch_retention@{k}{suffix}"] += float(retained)
            if len(exch_idx) >= 2:
                n_exch_ranked += 1
                for suffix, q in rankings.items():
                    picked = exch_idx[np.argmax(q[exch_idx])]
                    exch_sums[f"exch_rank_regret{suffix}"] += float(t[best_exch] - t[picked])

    metrics = {name: total / max(n_positions, 1) for name, total in sums.items()}
    for name, total in spearman_sums.items():
        metrics[name] = total / max(n_ranked, 1)
    for name, total in exch_sums.items():
        denom = n_exch_ranked if name.startswith("exch_rank_regret") else n_exch
        metrics[name] = total / max(denom, 1)
    metrics["positions"] = n_positions
    metrics["positions_with_exchanges"] = n_exch
    if plane_candidates:
        metrics["plane_ce"] = plane_ce_sum / plane_candidates
    if loss_cfg is not None:
        n = max(loss_sums.pop("_candidates", 0), 1)
        metrics.update({name: total / n for name, total in loss_sums.items()})
    return metrics


def _accumulate_loss(sums: dict, out: dict, batch: dict, device, loss_cfg):
    """Add one batch's candidate-weighted loss terms to `sums`, with the
    "_candidates" denominator."""
    targets = {k: batch[k].to(device) for k in TARGET_KEYS if k in batch}
    losses = loss_cfg.loss(out, targets)
    m = targets["target_wld"].shape[0]
    sums["_candidates"] = sums.get("_candidates", 0) + m
    for term, value in losses.items():
        name = "loss" if term == "total" else f"loss_{term}"
        sums[name] = sums.get(name, 0.0) + value.item() * m
