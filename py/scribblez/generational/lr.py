"""Rows-clock learning-rate control.

The generational trainer does not compute the learning rate from a fixed
schedule -- an open-ended, moving-target self-play run has no known annealing
horizon. Instead the *base* rate is a live operator control (the dashboard
Controls tab), stepped down by hand off the loss plots. The only automatic shape
is a startup warmup on the rows-clock, which attenuates the base rate over the
first `warmup_rows` rows of the whole run to prevent early instability. On a
mid-run restart the warmup is a no-op, since the cumulative row count is already
past `warmup_rows`.

See docs/generational_training.md, "Learning rate: a persisted manual control".
"""

from __future__ import annotations

from collections.abc import Callable


def warmup_factor(rows_trained: int, warmup_rows: int) -> float:
    """Linear warmup multiplier in [0, 1]: ramps 0 -> 1 over the first
    `warmup_rows` rows, then stays 1. `warmup_rows <= 0` disables warmup."""
    if warmup_rows <= 0 or rows_trained >= warmup_rows:
        return 1.0
    return rows_trained / warmup_rows


def effective_lr(base_lr: float, rows_trained: int, warmup_rows: int) -> float:
    """The learning rate to apply this step: the live base rate scaled by the
    rows-clock warmup ramp. `base_lr` is the manual control value; warmup only
    attenuates it during the first `warmup_rows` of the whole run."""
    return base_lr * warmup_factor(rows_trained, warmup_rows)


def make_lr_fn(base_lr: float, warmup_rows: int) -> Callable[[int], float]:
    """Build the per-step learning-rate callable run_epoch expects: it maps the
    running rows count to `effective_lr(base_lr, rows, warmup_rows)`.

    `base_lr` is captured at build time. A later increment that makes the base
    rate a live dashboard control rebuilds this (or reads the control inside)
    when the operator changes it; for now it is the fixed configured rate."""

    def lr_fn(rows_trained: int) -> float:
        return effective_lr(base_lr, rows_trained, warmup_rows)

    return lr_fn
