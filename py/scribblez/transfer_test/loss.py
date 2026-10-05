"""The reader's training objective and its held-out readout
(docs/plans/supreme_bot_m1a.md, PR 4).

Every pick query is scored against its candidate's label, the label rollouts'
empirical outcome:

    wld        cross-entropy against the win/draw/loss frequencies
    score      Gaussian negative log-likelihood of the label's score
               differences, log sd + (var + (mean - mu)^2) / (2 sd^2): the
               expected NLL of one rollout under the head's Gaussian
    expected   Gaussian negative log-likelihood of the label's expected score
               under the head's estimate and spread, the label's own sampling
               variance (bounded by E (1 - E) / n) added as above, so a noisy
               label does not teach false confidence

Both likelihoods are beta-NLL (_beta_nll), so their means learn as by squared
error.
    footprint  cross-entropy against the label's opponent-reply and own-next
               footprint frequencies, the two heads averaged
    rank       among the candidates queried at the same prefix of a row, a
               logistic loss on every pair's expected-score order, weighted by
               the gap between their labels, so near-ties cost little

The game-result anchor belongs to the label loop and is left out (plan, PR 4).

The readout is M1a's headline, read during training: at the full context,
the error of each candidate's expected score after centering predictions and
labels within the row, on the held-out candidates (and, for contrast, the
probed ones), against the same error of the teacher's prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from scribblez.transfer_test.reader import FOOTPRINT_HEADS
from scribblez.transfer_test.rows import SCORE_SCALE
from scribblez.transfer_test.tokens import TokenBatch

LOSS_TERMS = ("wld", "score", "expected", "footprint", "rank")
LOG_SD_MIN, LOG_SD_MAX = -7.0, 3.0
# Typical variances, the beta-NLL terms' scales: a game's score difference
# (about 50 points, in SCORE_SCALE units) and the expected score's within-row
# spread between candidates (step 0 measured 0.037).
SCORE_VAR_SCALE = 0.5**2
EXPECTED_VAR_SCALE = 0.03**2


@dataclass(frozen=True)
class LossWeights:
    wld: float = 1.0
    score: float = 1.0
    expected: float = 1.0
    footprint: float = 0.5
    rank: float = 1.0
    rank_temperature: float = 0.01  # expected-score difference per logit

    def of(self, term: str) -> float:
        return getattr(self, term)


@dataclass
class Queries:
    """The batch's real queries, flat: each output, its candidate's label, and
    the row and prefix it was asked at."""

    out: dict[str, torch.Tensor]
    target: dict[str, torch.Tensor]
    row: torch.Tensor  # (N,)
    prefix: torch.Tensor  # (N,)
    candidate: torch.Tensor  # (N,) flat candidate index


def flat_queries(out: dict[str, torch.Tensor], b: TokenBatch) -> Queries:
    valid = ~b.query_pad
    candidate = b.query_candidate[valid]
    rows = torch.arange(valid.shape[0], device=valid.device)[:, None].expand_as(valid)
    return Queries(
        out={k: v[valid].float() for k, v in out.items()},
        target={k: v[candidate] for k, v in b.target.items()},
        row=rows[valid],
        prefix=b.query_prefix[valid],
        candidate=candidate,
    )


def expected_label(target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """The label's expected score and its sampling variance bound."""
    e = target["wld"][:, 0] + 0.5 * target["wld"][:, 1]
    return e, (e * (1 - e)).clamp(min=1e-4) / target["n"]


def _beta_nll(mean, log_sd, target_mean, target_var, scale: float) -> torch.Tensor:
    """The expected Gaussian negative log-likelihood of a target with mean
    `target_mean` and variance `target_var`, each query's term weighted by its
    predicted variance, detached, over `scale` (beta-NLL with beta = 1,
    Seitzer et al. 2022). Plain NLL lets a head lower its loss by widening its
    spread instead of moving its mean, and starves the mean of gradient where
    the spread is wide; the weighting gives the mean a squared-error gradient
    while the spread still learns the misfit. `scale`, a typical variance,
    keeps the term near unit size."""
    log_sd = log_sd.clamp(LOG_SD_MIN, LOG_SD_MAX)
    var = torch.exp(2 * log_sd)
    nll = log_sd + (target_var + (target_mean - mean) ** 2) / (2 * var)
    return (var.detach() / scale * nll).mean()


def _soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against unnormalized target frequencies, averaged over
    the rows with any mass (zero when none has: a batch of decided endgames)."""
    mass = target.sum(dim=-1, keepdim=True)
    ce = -(target / mass.clamp(min=1e-9) * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    return ce.sum() / (mass > 0).sum().clamp(min=1)


def _same_prefix_groups(q: Queries) -> torch.Tensor:
    """(N, N) bool: the two queries share a row and a prefix."""
    return (q.row[:, None] == q.row[None, :]) & (q.prefix[:, None] == q.prefix[None, :])


def _rank_loss(q: Queries, temperature: float) -> torch.Tensor:
    e, _ = expected_label(q.target)
    mu = q.out["expected"][:, 0]
    gap = e[:, None] - e[None, :]
    ordered = _same_prefix_groups(q) & (gap > 0)
    margin = (mu[:, None] - mu[None, :]) / temperature
    weighted = gap * F.softplus(-margin)
    return weighted[ordered].sum() / gap[ordered].sum().clamp(min=1e-6)


def losses(out: dict[str, torch.Tensor], b: TokenBatch, w: LossWeights) -> dict[str, torch.Tensor]:
    """Each term of the module docstring, and their weighted "total"."""
    q = flat_queries(out, b)
    t = q.target
    e, e_var = expected_label(t)
    terms = {
        "wld": -(t["wld"] * torch.log_softmax(q.out["wld"], dim=-1)).sum(dim=-1).mean(),
        "score": _beta_nll(
            q.out["score"][:, 0],
            q.out["score"][:, 1],
            t["score_mean"] / SCORE_SCALE,
            t["score_var"] / SCORE_SCALE**2,
            SCORE_VAR_SCALE,
        ),
        "expected": _beta_nll(
            q.out["expected"][:, 0], q.out["expected"][:, 1], e, e_var, EXPECTED_VAR_SCALE
        ),
        "footprint": sum(_soft_ce(q.out[h], t[h]) for h in FOOTPRINT_HEADS) / len(FOOTPRINT_HEADS),
        "rank": _rank_loss(q, w.rank_temperature),
    }
    terms["total"] = sum(w.of(k) * terms[k] for k in LOSS_TERMS)
    return terms


def _row_centered(x: torch.Tensor, row: torch.Tensor, rows: int) -> torch.Tensor:
    total = torch.zeros(rows, device=x.device).index_add_(0, row, x)
    count = torch.zeros(rows, device=x.device).index_add_(0, row, torch.ones_like(x))
    return x - (total / count.clamp(min=1))[row]


@torch.no_grad()
def readout(out: dict[str, torch.Tensor], b: TokenBatch) -> dict[str, torch.Tensor]:
    """Sums for the module docstring's readout over one batch: squared
    within-row errors of the reader and of the prior, on held-out and on
    probed candidates, and their counts. `finish_readout` turns accumulated
    sums into RMSEs."""
    q = flat_queries(out, b)
    rows = b.query_pad.shape[0]
    last = torch.zeros(rows, dtype=q.prefix.dtype, device=q.prefix.device)
    last = last.scatter_reduce(0, q.row, q.prefix, reduce="amax")
    full = q.prefix == last[q.row]
    row, cand = q.row[full], q.candidate[full]
    e, _ = expected_label({k: v[full] for k, v in q.target.items()})
    prior_value = b.candidate["prior_value"][cand]
    estimates = {
        "reader": q.out["expected"][full, 0],
        "prior": prior_value[:, 0] + 0.5 * prior_value[:, 1],
    }
    label = _row_centered(e, row, rows)
    held = b.held_out[cand]
    sums = {}
    for name, x in estimates.items():
        sq = (_row_centered(x.float(), row, rows) - label) ** 2
        sums[f"{name}_held_sq"] = sq[held].sum()
        sums[f"{name}_probed_sq"] = sq[~held].sum()
    sums["held"] = held.sum().float()
    sums["probed"] = (~held).sum().float()
    return sums


def finish_readout(sums: dict[str, float]) -> dict[str, float]:
    """RMSEs from summed readouts: heldout_rmse_<arm>, probed_rmse_<arm>."""
    out = {}
    for arm in ("reader", "prior"):
        out[f"heldout_rmse_{arm}"] = (sums[f"{arm}_held_sq"] / max(sums["held"], 1)) ** 0.5
        out[f"probed_rmse_{arm}"] = (sums[f"{arm}_probed_sq"] / max(sums["probed"], 1)) ** 0.5
    return out
