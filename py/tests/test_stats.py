"""Tests for the sequential match statistics (scribblez.stats)."""

import math

import pytest
from scribblez import stats


def test_pair_score_counts_bins_pairs():
    # (1, 1) -> 1.0, (1, 0) -> 0.5, (0.5, 0) -> 0.25, (0, 0) -> 0.0
    counts = stats.pair_score_counts([1, 1, 1, 0, 0.5, 0, 0, 0])
    assert counts == [1, 1, 1, 0, 1]


def test_pair_score_counts_rejects_odd_game_count():
    with pytest.raises(ValueError):
        stats.pair_score_counts([1, 0, 1])


def test_mean_and_variance_balanced_sample():
    mean, var = stats.mean_and_variance([1, 0, 0, 0, 1])
    assert mean == 0.5
    assert var == 0.25


def test_sprt_needs_ordered_hypotheses():
    with pytest.raises(ValueError):
        stats.sprt([0, 0, 4, 0, 0], p0=0.55, p1=0.5)


def test_sprt_continues_on_a_balanced_sample():
    r = stats.sprt([5, 5, 5, 5, 5], p0=0.5, p1=0.55)
    assert r.decision == "continue"
    assert r.lower < r.llr < r.upper


def test_sprt_accepts_h1_on_a_dominant_sample():
    # 200 pairs at 60% mean score with realistic spread.
    r = stats.sprt([10, 20, 60, 70, 40], p0=0.5, p1=0.55)
    assert r.llr >= r.upper
    assert r.decision == "H1"


def test_sprt_accepts_h0_on_a_losing_sample():
    r = stats.sprt([40, 70, 60, 20, 10], p0=0.5, p1=0.55)
    assert r.llr <= r.lower
    assert r.decision == "H0"


def test_sprt_llr_sign_tracks_the_mean():
    winning = stats.sprt([0, 5, 10, 15, 10], p0=0.5, p1=0.55)
    losing = stats.sprt([10, 15, 10, 5, 0], p0=0.5, p1=0.55)
    assert winning.llr > 0 > losing.llr


def test_sprt_degenerate_sweep_decides_only_when_clear():
    all_wins = stats.sprt([0, 0, 0, 0, 8], p0=0.5, p1=0.55)
    assert all_wins.llr == math.inf
    assert all_wins.decision == "H1"
    all_losses = stats.sprt([8, 0, 0, 0, 0], p0=0.5, p1=0.55)
    assert all_losses.decision == "H0"
    # Every pair split 0.5/0.5: no variance, mean between the hypotheses.
    stuck = stats.sprt([0, 0, 8, 0, 0], p0=0.45, p1=0.55)
    assert stuck.decision == "continue"
    assert stats.sprt([0, 0, 0, 0, 0], p0=0.5, p1=0.55).decision == "continue"


def test_confidence_interval_narrows_with_n():
    small_mean, small_hw = stats.score_confidence_interval([2, 2, 2, 2, 2])
    big_mean, big_hw = stats.score_confidence_interval([20, 20, 20, 20, 20])
    assert small_mean == big_mean == 0.5
    assert big_hw < small_hw
    assert stats.score_confidence_interval([1, 0, 0, 0, 0]) == (0.0, 0.5)
