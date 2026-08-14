"""Evaluation for the move set evaluation model: how well it reproduces the
teacher's ranking of a position's candidate set (docs/roadmap.md, A3 gate).

The filter's job is measured two ways: recall (this model's top-K must
contain the moves the position evaluation model would pick) and teacher-value
regret (what a miss costs, the roadmap's other named A3 gate metric). For
each labeled position we rank its candidates two ways -- by the teacher's
stored win-equity and by this model's predicted win-equity -- and report:

  * top-K recall: the fraction of the teacher's top-K candidates that fall in
    this model's top-K (averaged over positions), for each K;
  * teacher-value regret@K: the teacher win-equity forfeited by keeping only
    the top-K -- recall scores dropping a near-tie like dropping a uniquely
    winning move; regret prices the miss;
  * rank correlation: the Spearman correlation between the two rankings over the
    whole candidate set (averaged over positions with >= 2 candidates);
  * exchange-slice metrics, the readout that decides whether exchanges need a
    dedicated head (docs/roadmap.md A4): exch_retention@K -- how often the
    teacher's best exchange survives into the model's global top-K, over
    positions with any exchange -- and exch_rank_regret, the teacher win-equity
    lost by taking the model's favourite exchange instead of the teacher's,
    over positions with >= 2 exchanges. The latter is exactly the
    which-tiles-to-keep ranking the pre-v1 encoding could not express.

Every metric is paired with an incumbent-baseline value computed on the same
positions from the stored candidate order (_baseline_ranking). Ranking is by
win-equity P(win)+0.5*P(draw), the same scalar both models are scored on, so
the comparison is apples-to-apples.

The metrics mean what the roadmap's A3 gate asks only on a full-sweep dataset,
where a position's candidates are all of its legal moves: over a stratified
sample they score the model on ~15 candidates it was trained on and cannot see
the tail moves the filter exists to catch. Nothing here treats the two
differently -- a swept position is just a position with far more candidates --
so the caller chooses the dataset the numbers are read on.
"""

from __future__ import annotations

import numpy as np
import torch

from .model import win_equity

DEFAULT_KS = (1, 3, 5)

# Moves per forward pass. A swept position carries hundreds of candidates
# against a stratified one's ~15, so over full-sweep positions the position
# count alone stops describing what a batch costs and the candidate count takes
# over -- but not alone: the model also builds a padded (positions x largest
# candidate set) query grid, so a batch mixing one near-cap position with small
# ones costs more than its candidate count suggests. The two terms are
# comparable, and this budget bounds only the first. Whole-forward peaks under
# no_grad at C=192, positions_per_batch at its 64 default: +429 MiB at the worst
# shape admitted here (one position at the generator's 1500 sweep cap, M=16305),
# +366 MiB for that M with every position the same size, +342 MiB for that grid
# at M=1563. Over stratified positions this never binds before the position
# bound does.
MAX_CANDIDATES_PER_BATCH = 16384

# Move-input tensors passed positionally to the model's forward.
_MOVE_KEYS = (
    "move_letters",
    "move_blanks",
    "move_squares",
    "move_tile_mask",
    "move_scalars",
    "move_pos_id",
)


def eval_slice_line(dataset) -> str:
    """One line describing what the gate metrics are being read on, for a
    trainer's startup log. Recall and regret are A3 gate numbers only over full
    sweeps; over a stratified holdout they are a provisional reading, and that
    belongs next to them. For a sweep, how much of each position the generator's
    candidate cap reached is part of the reading."""
    if not dataset.full_sweep:
        return "eval slice: stratified (recall/regret are provisional -- no tail coverage)"
    coverage, truncated = dataset.sweep_coverage
    return (
        f"eval slice: full sweep, {coverage:.3f} mean coverage of legal moves, "
        f"{truncated}/{dataset.num_positions} positions truncated by the cap"
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


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k best-scored candidates (k capped at the count)."""
    return np.argsort(-scores)[: min(k, len(scores))]


def _topk_recall(teacher: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Fraction of the teacher's top-k candidates that appear in the model's
    top-k."""
    teacher_top = set(_topk_indices(teacher, k).tolist())
    pred_top = set(_topk_indices(pred, k).tolist())
    return len(teacher_top & pred_top) / len(teacher_top)


def _regret(teacher: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Teacher win-equity forfeited by keeping only the ranking's top-k: the
    gap between the teacher's best candidate and the best it retains (the
    module docstring carries the rationale)."""
    return float(teacher.max() - teacher[_topk_indices(pred, k)].max())


def _exchange_mask(scalars: np.ndarray) -> np.ndarray:
    """Which candidates are EXCHANGE moves, off the encoded scalars: not a
    play, at least one tile surrendered (a pass has neither)."""
    return (scalars[:, 2] == 0.0) & (scalars[:, 1] > 0.0)


def _baseline_ranking(n: int) -> np.ndarray:
    """The incumbent's ranking, encoded descending-by-stored-index, which the
    generator's storage order is chosen to make exact.

    On a full-sweep position the stored order IS the static-equity ranking
    (move_set_eval_candidates.h keeps the sweep a subsequence of it), so every
    baseline metric here is the true incumbent's -- which is what the A3 gate
    compares the learned filter against.

    On a stratified position the generator stores the played move first
    (HastyBot's own choice: the equity argmax outside the endgame, the solver's
    move inside it), then the equity ranking's head; beyond the top stratum the
    order is a shuffled sample. Baseline rank metrics over the full set
    (Spearman) are then a floor, and top-k metrics are exact only for
    k <= 1 + quota_top. One known smudge: on the rare positions where the
    dataset's non-finite-target filter dropped the stored-first candidate,
    index 0 is the next surviving candidate rather than the move the incumbent
    played."""
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
) -> dict[str, float]:
    """Run the model over `dataset` and return the ranking metrics, each
    paired with the incumbent baseline computed on the same positions (see
    _baseline_ranking).

    Returns, per K in `ks`: "recall@K" / "recall@K_baseline" (top-k set
    overlap with the teacher's) and "regret@K" / "regret@K_baseline" (mean
    teacher win-equity forfeited by the top-k, lower is better); plus
    "spearman" / "spearman_baseline", the exchange-slice metrics
    ("exch_retention@K", "exch_rank_regret", each with its baseline; see the
    module docstring), and the "positions" / "positions_with_exchanges"
    denominators.
    """
    model.eval()
    sums = {}
    exch_sums = {"exch_rank_regret": 0.0, "exch_rank_regret_baseline": 0.0}
    for k in ks:
        sums[f"recall@{k}"] = sums[f"recall@{k}_baseline"] = 0.0
        sums[f"regret@{k}"] = sums[f"regret@{k}_baseline"] = 0.0
        exch_sums[f"exch_retention@{k}"] = exch_sums[f"exch_retention@{k}_baseline"] = 0.0
    n_positions = 0
    plane_bce_sum = 0.0
    plane_candidates = 0
    spearman_sums = {"spearman": 0.0, "spearman_baseline": 0.0}
    n_ranked = 0
    n_exch = 0  # positions with any exchange candidate (retention denominator)
    n_exch_ranked = 0  # positions with >= 2 exchanges (rank-regret denominator)

    for batch in dataset.iter_batches(
        positions_per_batch, seed=seed, max_candidates=max_candidates_per_batch
    ):
        inputs = (batch["input_spatial"].to(device), batch["input_scalar"].to(device))
        move_args = tuple(batch[key].to(device) for key in _MOVE_KEYS)
        out = model(*inputs, *move_args)
        if "target_planes" in batch:
            m = batch["move_pos_id"].shape[0]
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                out["planes"], batch["target_planes"].to(device)
            )
            plane_bce_sum += bce.item() * m
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
    # Plane-readout quality, only when the slice carries plane targets (the
    # full-sweep holdout does not; the stratified fallback holdout does).
    if plane_candidates:
        metrics["plane_bce"] = plane_bce_sum / plane_candidates
    return metrics
