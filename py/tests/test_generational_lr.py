"""Unit tests for the rows-clock learning-rate control."""

import pytest
from scribblez.generational.lr import effective_lr, warmup_factor


def test_warmup_ramps_linearly_then_holds():
    assert warmup_factor(0, 1000) == 0.0
    assert warmup_factor(250, 1000) == pytest.approx(0.25)
    assert warmup_factor(1000, 1000) == 1.0  # at the boundary, warmup is done
    assert warmup_factor(5000, 1000) == 1.0  # and stays done afterward


def test_warmup_disabled_when_nonpositive():
    assert warmup_factor(0, 0) == 1.0
    assert warmup_factor(0, -1) == 1.0
    assert warmup_factor(10, 0) == 1.0


def test_effective_lr_scales_base_during_warmup():
    # Half-way through warmup the effective rate is half the base.
    assert effective_lr(1e-3, 500, 1000) == pytest.approx(5e-4)
    # Past warmup the base is applied verbatim (the manual control value).
    assert effective_lr(1e-3, 2000, 1000) == pytest.approx(1e-3)
    # A later manual step-down of the base is reflected immediately.
    assert effective_lr(2e-4, 2000, 1000) == pytest.approx(2e-4)


def test_effective_lr_no_warmup():
    assert effective_lr(3e-4, 0, 0) == pytest.approx(3e-4)
