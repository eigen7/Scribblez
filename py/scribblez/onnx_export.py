"""Export a trained ScribblezModel to ONNX.

The exported graph takes the same two inputs as the PyTorch model
(`input_spatial`, `input_scalar`) and produces all three head outputs
(`wld`, `score_diff`, `opp_next_placement`). The batch dimension is dynamic
so the same file serves single-position and batched inference.
"""

from __future__ import annotations

from pathlib import Path

import torch


def export_onnx(
    model: torch.nn.Module,
    path: str | Path,
    spatial_planes: int,
    scalar_size: int,
    board_size: int = 15,
    opset: int = 17,
) -> None:
    """Trace `model` and write an ONNX graph to `path` (eval mode, dynamic batch)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    dummy_spatial = torch.zeros(1, spatial_planes, board_size, board_size, device=device)
    dummy_scalar = torch.zeros(1, scalar_size, device=device)

    torch.onnx.export(
        model,
        (dummy_spatial, dummy_scalar),
        str(path),
        input_names=["input_spatial", "input_scalar"],
        output_names=["wld", "score_diff", "opp_next_placement"],
        dynamic_axes={
            "input_spatial": {0: "batch"},
            "input_scalar": {0: "batch"},
            "wld": {0: "batch"},
            "score_diff": {0: "batch"},
            "opp_next_placement": {0: "batch"},
        },
        opset_version=opset,
        dynamo=False,
    )
    if was_training:
        model.train()
