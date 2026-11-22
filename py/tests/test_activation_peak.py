"""ActivationPeak: the largest input or output of any module in a model's
forwards, while open, and no hooks left behind once closed."""

import torch
import torch.nn as nn
from scribblez.activation_peak import ActivationPeak


def _model() -> nn.Sequential:
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.ReLU(), nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[100.0, 0.0], [0.0, -1.0]]))
        model[2].weight.copy_(torch.tensor([[1.0, 1.0]]))
    return model


def test_the_peak_is_the_largest_activation_anywhere():
    model = _model()
    with torch.no_grad(), ActivationPeak(model) as peak:
        assert peak.value() == 0.0
        model(torch.tensor([[0.5, 3.0]]))  # the first linear's output peaks: 50
        model(torch.tensor([[-0.2, -7.0]]))  # smaller everywhere: the first linear's 20
    assert peak.value() == 50.0


def test_closing_removes_the_hooks():
    model = _model()
    with ActivationPeak(model):
        pass
    assert not any(m._forward_hooks for m in model.modules())
