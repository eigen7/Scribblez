"""Export a LeafEvalModel to ONNX for the engine's TensorRT loader
(engine/include/nn/model_specs.h, LeafEvalSpec).

The graph takes a row (`input_spatial`, (batch, 95, 15, 15), and
`input_scalar`, (batch, 85)) and emits `wld` logits (batch, 3) and
`score_diff` (batch, 2: mean, std). A model trained without the join planes
still takes the full row; its graph slices the planes it reads. The metadata
carries the format version, the leave values the row's scalars were priced
with (which the engine checks before loading: nn::check_leave_values), and
the generation's provenance.

The trainer exports every generation into its tag's models/, where a
policy_set task pins one. As a script: export a tag's rolling checkpoint as
its latest generation. The checkpoint holds the served weights: the trainer
checkpoints in eval mode, the averaged iterate with recalibrated batch norm.

    python -m scribblez.deep_bot_leaf_eval.onnx_export --tag leaf-conv-nojoin
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from scribblez.deep_bot_leaf_eval.model import LAYOUT, LeafEvalModel, LeafEvalShape
from scribblez.ffi import DEFAULT_LEXICON
from scribblez.onnx_export_util import export_graph
from scribblez.paths import DEFAULT_MOUNT_ROOT, TagPaths

# Must match the engine's nn::kGraphDeepBotLeafEval and LeafEvalSpec.
GRAPH = "deep_bot_leaf_eval"
OUTPUTS = ("wld", "score_diff")
FORMAT_KEY = "leaf_eval_format_version"
TABLE_KEY = "leave_values"
WORKLOAD = "deep_bot_leaf_eval"


class _ExportGraph(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_spatial: torch.Tensor, input_scalar: torch.Tensor):
        out = self.model(input_spatial, input_scalar)
        return tuple(out[name] for name in OUTPUTS)


def export_onnx(
    model: LeafEvalModel,
    path: str | Path,
    *,
    leave_values: str,
    provenance: dict[str, object],
    board_size: int = 15,
    opset: int = 17,
):
    """Export `model` in eval mode to `path`, atomically. `leave_values` is
    the content hash of the leave values its rows were priced with;
    `provenance` (tag, generation, rows) is recorded verbatim."""
    device = next(model.parameters()).device
    spatial_planes = LAYOUT["join_plane0"] + LAYOUT["join_planes"]
    dummy_spatial = torch.zeros(1, spatial_planes, board_size, board_size, device=device)
    dummy_scalar = torch.zeros(1, LAYOUT["pool_size"] + 1, device=device)
    names = ("input_spatial", "input_scalar", *OUTPUTS)
    export_graph(
        _ExportGraph(model),
        (dummy_spatial, dummy_scalar),
        path,
        input_names=names[:2],
        output_names=names[2:],
        dynamic_axes={name: {0: "batch"} for name in names},
        metadata={
            "graph": GRAPH,
            FORMAT_KEY: str(LAYOUT["format_version"]),
            TABLE_KEY: leave_values,
            "lexicon": DEFAULT_LEXICON,
            "trunk": model.shape.trunk,
            "join_planes": "1" if model.shape.join_planes else "0",
            **{k: str(v) for k, v in provenance.items()},
        },
        opset=opset,
    )


def export_checkpoint(ckpt: dict, out: Path, tag: str) -> dict[str, object]:
    """Export the loaded rolling checkpoint `ckpt` to `out`; returns the
    provenance recorded."""
    params = ckpt["config"]
    if int(params[FORMAT_KEY]) != int(LAYOUT["format_version"]):
        raise ValueError(
            f"checkpoint row format {params[FORMAT_KEY]} != engine's {LAYOUT['format_version']}"
        )
    shape = LeafEvalShape(
        trunk=params["trunk"],
        join_planes=bool(params["join_planes"]),
        width=int(params["width"]),
        depth=int(params["depth"]),
        num_heads=int(params["num_heads"]),
    )
    model = LeafEvalModel(shape)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # generation_index is the next generation to train; the weights are the
    # previous one's.
    provenance = {
        "tag": tag,
        "generation": int(ckpt["generation_index"]) - 1,
        "rows_trained": int(ckpt["rows_trained"]),
    }
    export_onnx(model, out, leave_values=params[TABLE_KEY], provenance=provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", required=True, help=f"a {WORKLOAD} tag name")
    parser.add_argument(
        "--out", help="output .onnx (default: the tag's export of that generation, in models/)"
    )
    args = parser.parse_args()
    paths = TagPaths(args.tag, WORKLOAD, DEFAULT_MOUNT_ROOT)
    ckpt = torch.load(paths.rolling_checkpoint, map_location="cpu", weights_only=False)
    generation = int(ckpt["generation_index"]) - 1
    out = Path(args.out) if args.out else paths.onnx_path(generation)
    provenance = export_checkpoint(ckpt, out, args.tag)
    print(f"exported {out} ({provenance})")


if __name__ == "__main__":
    main()
