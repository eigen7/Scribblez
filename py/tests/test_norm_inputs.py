"""NormInputSquares: the mean square of every normalization layer's input,
collected per training forward, eager and compiled, and penalizing a scale
the model's output cannot see."""

import torch
import torch.nn as nn
from scribblez.norm_inputs import NormInputSquares
from scribblez.transformer_tower import RMSNorm


class Toy(nn.Module):
    """Two linears feeding three kinds of norm, bracketing its forward with
    the collector as the leaf model does."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.ln = nn.LayerNorm(8)
        self.b = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)
        self.rms = RMSNorm(8)
        self.collector = NormInputSquares(self)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.collector.start()
        h = self.a(x)
        y = self.rms(self.bn(self.b(self.ln(h))))
        return {"y": y, "norm_input": self.collector.take()}

    def expected(self, x: torch.Tensor) -> torch.Tensor:
        """The collector's value, computed by hand."""
        h = self.a(x)
        ln_in, bn_in = h, self.b(self.ln(h))
        rms_in = self.bn(bn_in)
        return torch.stack([t.square().mean() for t in (ln_in, bn_in, rms_in)]).mean()


def _inputs() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(16, 4) * 30


def test_the_collector_reports_each_norms_mean_squared_input():
    torch.manual_seed(0)
    model = Toy()
    x = _inputs()
    model(x)  # an earlier forward's norms are not counted again
    got = model(x)["norm_input"]
    torch.testing.assert_close(got, model.expected(x))


def test_the_collector_needs_its_forward_bracketed():
    torch.manual_seed(0)
    model = Toy()
    model.ln(torch.ones(2, 8))  # outside start()/take(): not recorded
    model.collector.start()
    model.ln(torch.full((2, 8), 3.0))
    assert model.collector.take().item() == 9.0


def test_compiled_matches_eager_in_value_and_gradient():
    """torch.compile traces the hooks and the collector's list into the graph:
    the trainer's compiled forward reports the same penalty and gradient. The
    eager backend is Dynamo's capture without Inductor's code generation, which
    is the part the hooks depend on, at a fraction of the compile time."""
    torch.manual_seed(0)
    eager = Toy()
    compiled_src = Toy()
    compiled_src.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled_src, backend="eager")
    x = _inputs()
    for model in (eager, compiled):
        model(x)["norm_input"].backward()
    torch.testing.assert_close(compiled(x)["norm_input"], eager(x)["norm_input"])
    torch.testing.assert_close(compiled_src.a.weight.grad, eager.a.weight.grad)


def test_the_penalized_scale_is_invisible_to_the_output():
    """Shrinking a tensor that feeds a norm leaves the output as it was and
    quarters its mean square: the penalty reads a scale the output cannot see."""
    torch.manual_seed(0)
    model = Toy().eval()  # BatchNorm on its running stats, as served
    x = _inputs()
    model.collector.start()
    before = model.ln(model.a(x))
    big = model.collector.take()
    with torch.no_grad():
        model.a.weight.mul_(0.5)
        model.a.bias.mul_(0.5)
    model.collector.start()
    after = model.ln(model.a(x))
    small = model.collector.take()
    torch.testing.assert_close(after, before, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(small, big / 4)
