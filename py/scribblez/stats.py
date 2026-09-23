"""Win-rate statistics for head-to-head matches played in mirrored pairs.

play_game's --paired mode plays each game seed twice with the seats swapped,
so tile luck lands on both sides and largely cancels within a pair. The two
games of a pair are therefore correlated, and the unit of observation must be
the pair: each pair reduces to one score for the player under test, and the
variance comes from the five-valued (pentanomial) pair-score distribution
rather than from per-game win/draw/loss counts.
"""

import math

# Pair scores for the player under test: each game contributes win=1, draw=0.5,
# loss=0; the pair score is the mean of its two games.
PAIR_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)


def pair_score_counts(pair_scores: list[float]) -> list[int]:
    """Counts of each PAIR_SCORES value; any other score raises."""
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
    """(mean, half_width) of a normal-approximation confidence interval for
    the expected pair score."""
    n = sum(counts)
    mean, var = mean_and_variance(counts)
    if n < 2:
        return mean, 0.5
    return mean, z * math.sqrt(var / n)
