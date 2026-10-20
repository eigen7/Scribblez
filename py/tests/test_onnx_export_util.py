"""export_graph (scribblez.onnx_export_util): it leaves every submodule's
train/eval mode as it found it, and it refuses an export that disagrees with
the module, writing nothing."""

import pytest
import torch
from scribblez.onnx_export_util import eval_mode, export_graph
from torch import nn


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.body(x))


class _Wrapper(nn.Module):
    """An export wrapper, built fresh and so in train mode, as each exporter's
    is."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)


class _BranchOnData(nn.Module):
    """A Python branch on the input's values, which the tracer bakes in for
    the dummy input: an export that disagrees with the module elsewhere."""

    def forward(self, x):
        return x * 2 if bool((x > 0).any()) else x


def _export(graph: nn.Module, path):
    export_graph(
        graph,
        (torch.zeros(1, 4),),
        path,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        metadata={"graph": "test"},
        opset=17,
    )


def _modes(module: nn.Module) -> list[bool]:
    return [m.training for m in module.modules()]


def test_export_leaves_every_mode_as_it_found_it(tmp_path):
    net = _Net().train()
    net.body.eval()  # a deliberately frozen submodule
    before = _modes(net)
    _export(_Wrapper(net), tmp_path / "net.onnx")
    assert _modes(net) == before
    assert (tmp_path / "net.onnx").exists()


def test_eval_mode_restores_on_error():
    net = _Net().train()
    with pytest.raises(RuntimeError), eval_mode(net):
        assert not any(_modes(net))
        raise RuntimeError
    assert all(_modes(net))


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
def test_a_disagreeing_export_raises_and_writes_nothing(tmp_path):
    path = tmp_path / "branch.onnx"
    with pytest.raises(ValueError, match="disagrees"):
        _export(_BranchOnData(), path)
    assert list(tmp_path.iterdir()) == []
