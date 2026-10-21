"""Export a tilted-head LeaveInferenceModel to ONNX for the engine's TensorRT
loader (engine/include/nn/model_specs.h, LeaveInferenceSpec).

The graph takes a row (`input_spatial`, `input_scalar`) and emits
`leave_params`, (batch, 29): theta, then a and b converted from the trainer's
units (loss.VALUE_SCALE points) to points, the layout of the engine's
belief::LeaveParams. The metadata carries the format version and the leave
values the tilt was trained against, which the engine gates on, and the
generation's provenance.
"""

from pathlib import Path

import torch
from torch import nn

from scribblez.ffi import DEFAULT_LEXICON, format_layout
from scribblez.leave_inference.heads import HEAD_TILTED
from scribblez.leave_inference.loss import VALUE_SCALE
from scribblez.onnx_export_util import export_graph

# Must match the engine's nn::kGraphLeaveInference and LeaveParamsOutput.
GRAPH = "leave_inference"
OUTPUT = "leave_params"
# The entries the engine gates on, which checkpoints also record.
FORMAT_KEY = "leave_inference_format_version"
TABLE_KEY = "leave_values"


def params_in_points(theta: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N, 29): theta, then a per point and b per squared point of leave
    value, from the model's outputs."""
    return torch.cat([theta, (a / VALUE_SCALE)[:, None], (b / VALUE_SCALE**2)[:, None]], dim=1)


class _ExportGraph(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_spatial: torch.Tensor, input_scalar: torch.Tensor) -> torch.Tensor:
        out = self.model(input_spatial, input_scalar)
        return params_in_points(out["theta"], out["a"], out["b"])


def export_onnx(
    model: nn.Module,
    path: str | Path,
    spatial_planes: int,
    scalar_size: int,
    *,
    leave_values: str,
    provenance: dict[str, object],
    board_size: int = 15,
    opset: int = 17,
):
    """Export `model` in eval mode to `path`, atomically. `leave_values` is
    the content hash of the leave values it was trained against;
    `provenance` (tag, generation, policy) is recorded verbatim."""
    if model.head_kind != HEAD_TILTED:
        raise ValueError(f"only the {HEAD_TILTED} head is served, not {model.head_kind!r}")
    device = next(model.parameters()).device
    dummy_spatial = torch.zeros(1, spatial_planes, board_size, board_size, device=device)
    dummy_scalar = torch.zeros(1, scalar_size, device=device)
    names = ("input_spatial", "input_scalar", OUTPUT)
    export_graph(
        _ExportGraph(model),
        (dummy_spatial, dummy_scalar),
        path,
        input_names=names[:2],
        output_names=names[2:],
        dynamic_axes={name: {0: "batch"} for name in names},
        metadata={
            "graph": GRAPH,
            FORMAT_KEY: str(format_layout()["constants"]["leave_inference"]["format_version"]),
            TABLE_KEY: leave_values,
            "lexicon": DEFAULT_LEXICON,
            "head": model.head_kind,
            **{k: str(v) for k, v in provenance.items()},
        },
        opset=opset,
    )
