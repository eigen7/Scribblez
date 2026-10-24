"""generational.finite_guard: a non-finite training step or weight stops the
trainer with WorkerFatal, naming the first bad step and value, with no
per-step host sync."""

import math

import pytest
import torch
from scribblez.generational.finite_guard import FiniteGuard, check_parameters
from scribblez.workloads.worker import WorkerFatal
from torch import nn


def test_finite_steps_pass_and_the_first_bad_step_is_named():
    guard = FiniteGuard()
    guard.observe("batch 0", {"loss": torch.tensor(1.0), "norm": torch.tensor(2.0)})
    guard.check()
    guard.observe("batch 1", {"loss": torch.tensor(1.0), "norm": torch.tensor(3.0)})
    guard.observe("batch 2", {"loss": torch.tensor(math.nan), "norm": torch.tensor(math.inf)})
    guard.observe("batch 3", {"loss": torch.tensor(math.nan), "norm": torch.tensor(1.0)})
    with pytest.raises(WorkerFatal, match=r"non-finite loss, norm at batch 2"):
        guard.check()


def test_a_check_forgets_the_steps_it_read():
    guard = FiniteGuard()
    guard.observe("batch 0", {"loss": torch.tensor(math.inf)})
    with pytest.raises(WorkerFatal):
        guard.check()
    guard.check()  # nothing left to read


def test_non_finite_weights_are_refused_before_a_checkpoint():
    good, bad = nn.Linear(2, 2), nn.Linear(2, 2)
    check_parameters([("good", good)], "after generation 3")
    with torch.no_grad():
        bad.weight[0, 1] = math.nan
    with pytest.raises(WorkerFatal, match="bad's weights are non-finite after generation 3"):
        check_parameters([("good", good), ("bad", bad)], "after generation 3")
