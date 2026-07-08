"""Evaluation for the pre-move value model: how well M_pre reproduces the
teacher's ranking of a position's candidate set (docs/roadmap2.md, A3 gate).

The filter's one job is recall: M_pre's top-K must contain the moves M_post
would pick. For each labeled position we rank its candidates two ways -- by the
teacher's stored win-equity and by M_pre's predicted win-equity -- and report:

  * top-K recall: the fraction of the teacher's top-K candidates that fall in
    M_pre's top-K (averaged over positions), for each K;
  * rank correlation: the Spearman correlation between the two rankings over the
    whole candidate set (averaged over positions with >= 2 candidates).

Ranking is by win-equity P(win)+0.5*P(draw), the same scalar both models are
scored on, so the comparison is apples-to-apples.
"""

from __future__ import annotations

import numpy as np
import torch

from .model import win_equity

DEFAULT_KS = (1, 3, 5)

# Move-input tensors passed positionally to the model's forward.
_MOVE_KEYS = (
    "move_letters",
    "move_blanks",
    "move_squares",
    "move_tile_mask",
    "move_scalars",
    "move_pos_id",
)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two 1-D arrays (Pearson of their ordinal
    ranks). Ties are broken arbitrarily, which is immaterial for the continuous
    equity values ranked here."""
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0:
        return 0.0
    return float((ra * rb).sum() / denom)


def _topk_recall(teacher: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Fraction of the teacher's top-k candidates that appear in the model's
    top-k (k capped at the candidate count)."""
    k = min(k, len(teacher))
    teacher_top = set(np.argsort(-teacher)[:k].tolist())
    pred_top = set(np.argsort(-pred)[:k].tolist())
    return len(teacher_top & pred_top) / k


@torch.no_grad()
def evaluate(
    model,
    dataset,
    device,
    positions_per_batch: int = 64,
    ks=DEFAULT_KS,
    seed: int = 0,
) -> dict[str, float]:
    """Run M_pre over `dataset` and return the ranking-recall metrics.

    Returns {"recall@K": ..., "spearman": ..., "positions": n} where the
    recall keys are one per K in `ks`.
    """
    model.eval()
    recall_sums = {k: 0.0 for k in ks}
    n_positions = 0
    spearman_sum = 0.0
    n_ranked = 0

    for batch in dataset.iter_batches(positions_per_batch, seed=seed):
        inputs = (batch["input_spatial"].to(device), batch["input_scalar"].to(device))
        move_args = tuple(batch[key].to(device) for key in _MOVE_KEYS)
        out = model(*inputs, *move_args)
        pred_eq = win_equity(out["wld"].softmax(dim=1)).cpu().numpy()
        teacher_eq = win_equity(batch["target_wld"]).numpy()
        pos_id = batch["move_pos_id"].numpy()

        for p in np.unique(pos_id):
            sel = pos_id == p
            t = teacher_eq[sel]
            q = pred_eq[sel]
            n_positions += 1
            for k in ks:
                recall_sums[k] += _topk_recall(t, q, k)
            if len(t) >= 2:
                spearman_sum += _spearman(t, q)
                n_ranked += 1

    metrics = {f"recall@{k}": recall_sums[k] / max(n_positions, 1) for k in ks}
    metrics["spearman"] = spearman_sum / max(n_ranked, 1)
    metrics["positions"] = n_positions
    return metrics
