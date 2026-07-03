"""Unit tests for reuse-factor <-> epochs conversion."""

import pytest
from scribblez.generational.reuse import effective_reuse, epochs_for_reuse


def test_turns_per_game_zero_ignores_game_length():
    # reuse = window * epochs; avg_eligible must not matter.
    assert epochs_for_reuse(4, window=4, turns_per_game=0, avg_eligible=20) == 1
    assert epochs_for_reuse(4, window=4, turns_per_game=0, avg_eligible=999) == 1
    assert epochs_for_reuse(6, window=2, turns_per_game=0, avg_eligible=20) == 3


def test_turns_per_game_positive_uses_game_length():
    # reuse = window * epochs * K / avg_eligible  ->  epochs = reuse * avg / (w*K)
    assert epochs_for_reuse(1.0, window=4, turns_per_game=1, avg_eligible=20) == 5
    assert epochs_for_reuse(2.0, window=2, turns_per_game=2, avg_eligible=20) == 10
    assert epochs_for_reuse(1.0, window=1, turns_per_game=1, avg_eligible=20) == 20


def test_epochs_at_least_one():
    assert epochs_for_reuse(0.1, window=8, turns_per_game=0, avg_eligible=20) == 1
    assert epochs_for_reuse(0.0, window=1, turns_per_game=1, avg_eligible=20) == 1


def test_empty_generation_falls_back_to_reuse_epochs():
    assert epochs_for_reuse(3.0, window=1, turns_per_game=1, avg_eligible=0) == 3


def test_effective_reuse_inverts_epochs_for_reuse():
    # reuse=8 with avg_eligible=18 is representable in whole epochs for every
    # (window, turns_per_game) here, so the round-trip is exact.
    for k in (0, 1, 3):
        for w in (1, 4):
            epochs = epochs_for_reuse(8.0, window=w, turns_per_game=k, avg_eligible=18.0)
            r = effective_reuse(epochs, window=w, turns_per_game=k, avg_eligible=18.0)
            assert r == pytest.approx(8.0, abs=0.01)


def test_turns_per_game_zero_reuse_granularity_is_window():
    # With all-turns sampling, achievable reuse is a multiple of window, so a
    # target below the window floors to one epoch = `window` passes.
    assert epochs_for_reuse(2.0, window=4, turns_per_game=0, avg_eligible=18) == 1
    assert effective_reuse(1, window=4, turns_per_game=0, avg_eligible=18) == 4.0


def test_effective_reuse_zero_turns_is_window_times_epochs():
    assert effective_reuse(3, window=4, turns_per_game=0, avg_eligible=20) == 12.0
