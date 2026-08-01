"""Sequential match statistics (roadmap E2).

The shared discipline for head-to-head match play: games arrive in mirrored
pairs (the engine's --paired mode), each pair collapses to one score for the
player under test, and a generalized sequential probability ratio test decides
when the accumulated pairs settle the match. Owning this here keeps every
match-play experiment (A1's training-time eval first) from reinventing its
stopping rule.

Pairing is what makes the arithmetic honest: the two games of a pair share a
game seed, so per-seed tile luck lands on both sides of the comparison and
cancels out of the pair score. The unit of observation is therefore the pair,
and the pentanomial pair-score distribution -- not the per-game trinomial --
is what the variance estimate must come from, because the two games of a pair
are correlated.
"""

import math
from dataclasses import dataclass

# Pair scores for the player under test: each game contributes win=1, draw=0.5,
# loss=0; the pair score is the mean of its two games.
PAIR_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)


def pair_score_counts(pair_scores: list[float]) -> list[int]:
    """Pentanomial counts from per-pair scores (each the mean of a pair's two
    game scores, so one of PAIR_SCORES; anything else raises)."""
    counts = [0] * 5
    for score in pair_scores:
        counts[PAIR_SCORES.index(score)] += 1
    return counts


def mean_and_variance(counts: list[int]) -> tuple[float, float]:
    """Sample mean and (biased, 1/n) variance of the pair-score distribution."""
    n = sum(counts)
    if n == 0:
        return 0.5, 0.0
    mean = sum(c * s for c, s in zip(counts, PAIR_SCORES)) / n
    var = sum(c * (s - mean) ** 2 for c, s in zip(counts, PAIR_SCORES)) / n
    return mean, var


@dataclass(frozen=True)
class SprtResult:
    llr: float
    lower: float  # accept H0 (score <= p0) at or below this
    upper: float  # accept H1 (score >= p1) at or above this
    decision: str  # "H0" | "H1" | "continue"


def sprt(
    counts: list[int], p0: float, p1: float, alpha: float = 0.05, beta: float = 0.05
) -> SprtResult:
    """Generalized SPRT over pentanomial pair counts, testing H0: expected pair
    score = p0 against H1: it is p1. Returns an SprtResult.

    The log-likelihood ratio uses the normal approximation with the observed
    pair-score variance, llr = n (p1 - p0)(2 mean - p0 - p1) / (2 var) -- the
    chess-testing GSPRT. A degenerate sample (every pair identical) has no
    variance estimate; it decides only when the mean sits clear of both
    hypotheses, and otherwise waits for more pairs.
    """
    if not p0 < p1:
        raise ValueError(f"need p0 < p1, got {p0} >= {p1}")
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    n = sum(counts)
    mean, var = mean_and_variance(counts)
    if var > 0:
        llr = n * (p1 - p0) * (2.0 * mean - p0 - p1) / (2.0 * var)
    elif n > 0 and mean > max(p0, p1):
        llr = math.inf
    elif n > 0 and mean < min(p0, p1):
        llr = -math.inf
    else:
        llr = 0.0
    decision = "continue"
    if llr >= upper:
        decision = "H1"
    elif llr <= lower:
        decision = "H0"
    return SprtResult(llr=llr, lower=lower, upper=upper, decision=decision)


def score_confidence_interval(counts: list[int], z: float = 1.96) -> tuple[float, float]:
    """Normal-approximation confidence interval (mean, half_width) for the
    expected pair score. Pointwise -- for plotting a win-rate curve, not for
    stopping decisions, which belong to sprt()."""
    n = sum(counts)
    mean, var = mean_and_variance(counts)
    if n < 2:
        return mean, 0.5
    return mean, z * math.sqrt(var / n)
