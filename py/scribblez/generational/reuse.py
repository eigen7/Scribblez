"""Convert a target reuse factor into a per-generation epoch count.

The natural training knob is the *reuse factor*: how many gradient passes each
unique eligible position gets over a generation's lifetime in the window. The
mechanical knob the loop consumes is epochs-per-generation. This module maps
between them given the window size, the turns-per-game sampling, and the
generation's average eligible turns per game -- which is measured, not assumed:
it is `SlogDataset.num_samples / num_games`, i.e. the sum of the .slog headers'
eligible-turn counts over the game count, so nothing extra is stored on disk.

A generation is trained over `window * epochs` epochs total (it sits in the
window for `window` generation-steps, trained `epochs` epochs at each). Per
epoch, each game presents either all ~`avg_eligible` of its eligible turns
(`turns_per_game <= 0`) or `turns_per_game` of them. So the lifetime passes per
unique eligible position are:

    turns_per_game <= 0:  window * epochs                      (game length cancels)
    turns_per_game  > 0:  window * epochs * turns_per_game / avg_eligible
"""

from __future__ import annotations


def epochs_for_reuse(reuse: float, window: int, turns_per_game: int, avg_eligible: float) -> int:
    """Epochs-per-generation that yield ~`reuse` passes per unique eligible
    position, rounded to at least 1. `avg_eligible` is ignored when
    `turns_per_game <= 0` (all eligible turns are used each epoch, so the game
    length cancels)."""
    w = max(1, window)
    if turns_per_game <= 0:
        epochs = reuse / w
    elif avg_eligible <= 0:
        epochs = reuse  # degenerate (empty generation); fall back to reuse epochs
    else:
        epochs = reuse * avg_eligible / (w * turns_per_game)
    return max(1, round(epochs))


def effective_reuse(epochs: int, window: int, turns_per_game: int, avg_eligible: float) -> float:
    """The actual passes per unique eligible position for a chosen epoch count --
    the inverse of epochs_for_reuse, for reporting what a config really does."""
    w = max(1, window)
    if turns_per_game <= 0:
        return float(w * epochs)
    if avg_eligible <= 0:
        return float(w * epochs * turns_per_game)
    return w * epochs * turns_per_game / avg_eligible
