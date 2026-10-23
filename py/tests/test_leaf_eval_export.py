"""The DeepBot leaf eval model's ONNX export: the graph takes the full row
(join planes included, whether or not the model reads them), emits the two
heads, and carries the metadata the engine's LeafEvalSpec gates on."""

from pathlib import Path

import onnx
import pytest
import torch

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)

from scribblez.deep_bot_leaf_eval.model import LAYOUT, LeafEvalModel, LeafEvalShape  # noqa: E402
from scribblez.deep_bot_leaf_eval.onnx_export import (  # noqa: E402
    FORMAT_KEY,
    GRAPH,
    OUTPUTS,
    TABLE_KEY,
    export_onnx,
)

SPATIAL_PLANES = LAYOUT["join_plane0"] + LAYOUT["join_planes"]
SCALARS = LAYOUT["pool_size"] + 1


def _dims(value_info) -> list:
    return [d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim]


@pytest.mark.parametrize("join_planes", [False, True])
def test_export_carries_the_row_and_the_metadata(tmp_path, join_planes):
    torch.manual_seed(0)
    model = LeafEvalModel(
        LeafEvalShape(trunk="conv", join_planes=join_planes, width=8, depth=1, num_heads=1)
    )
    path = tmp_path / "leaf.onnx"
    export_onnx(model, path, leave_values="abc123", provenance={"tag": "t", "generation": 3})
    graph = onnx.load(str(path))
    meta = {p.key: p.value for p in graph.metadata_props}
    assert meta["graph"] == GRAPH
    assert meta[FORMAT_KEY] == str(LAYOUT["format_version"])
    assert meta[TABLE_KEY] == "abc123"
    assert meta["join_planes"] == ("1" if join_planes else "0")
    assert meta["tag"] == "t" and meta["generation"] == "3"
    inputs = {i.name: _dims(i) for i in graph.graph.input}
    assert inputs["input_spatial"] == ["batch", SPATIAL_PLANES, 15, 15]
    assert inputs["input_scalar"] == ["batch", SCALARS]
    outputs = {o.name: _dims(o) for o in graph.graph.output}
    assert list(outputs) == list(OUTPUTS)
    assert outputs["wld"] == ["batch", 3]
    assert outputs["score_diff"] == ["batch", 2]
