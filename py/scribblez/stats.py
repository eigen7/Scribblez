"""Match statistics over paired game play (roadmap E2).

The shared discipline for head-to-head match play: games arrive in mirrored
pairs (the engine's --paired mode), each pair collapses to one score for the
player under test, and the accumulated pairs give that player's win rate with
a confidence interval. Owning this here keeps every match-play experiment
(A1's training-time eval first) from reinventing it.

Pairing is what makes the arithmetic honest: the two games of a pair share a
game seed, so per-seed tile luck lands on both sides of the comparison and
cancels out of the pair score. The unit of observation is therefore the pair,
and the pentanomial pair-score distribution -- not the per-game trinomial --
is what the variance estimate must come from, because the two games of a pair
are correlated.
"""

import math

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
    mean = sum(c * s for c, s in zip(counts, PAIR_SCORES, strict=True)) / n
    var = sum(c * (s - mean) ** 2 for c, s in zip(counts, PAIR_SCORES, strict=True)) / n
    return mean, var


def score_confidence_interval(counts: list[int], z: float = 1.96) -> tuple[float, float]:
    """Normal-approximation confidence interval (mean, half_width) for the
    expected pair score."""
    n = sum(counts)
    mean, var = mean_and_variance(counts)
    if n < 2:
        return mean, 0.5
    return mean, z * math.sqrt(var / n)
