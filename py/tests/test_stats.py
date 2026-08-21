"""Tests for the paired match statistics (scribblez.stats)."""

import pytest
from scribblez import stats


def test_pair_score_counts_bins_pairs():
    counts = stats.pair_score_counts([1.0, 0.5, 0.25, 0.0, 1.0])
    assert counts == [1, 1, 1, 0, 2]


def test_pair_score_counts_rejects_non_pair_scores():
    with pytest.raises(ValueError):
        stats.pair_score_counts([0.3])


def test_mean_and_variance_balanced_sample():
    mean, var = stats.mean_and_variance([1, 0, 0, 0, 1])
    assert mean == 0.5
    assert var == 0.25


def test_confidence_interval_narrows_with_n():
    small_mean, small_hw = stats.score_confidence_interval([2, 2, 2, 2, 2])
    big_mean, big_hw = stats.score_confidence_interval([20, 20, 20, 20, 20])
    assert small_mean == big_mean == 0.5
    assert big_hw < small_hw
    assert stats.score_confidence_interval([1, 0, 0, 0, 0]) == (0.0, 0.5)
