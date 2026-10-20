"""Export a PositionEvalModel to ONNX for the engine's TensorRT loader.

The graph has the model's inputs (`input_spatial`, `input_scalar`) and its head
outputs by name (`wld`, `score_diff`, then one raw footprint-logit output per
placement head; consumers apply masking and softmax), with a dynamic batch
dimension.
"""

import functools
import os
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, numpy_helper

from scribblez.onnx_export_util import common_metadata, export_graph

from .model import PLACEMENT_HEAD_NAMES

# The frozen compiled-lexicon buffers (~24 MB) are identical in every checkpoint,
# so instead of being baked into each export they live in one shared blob beside
# the models, referenced as ONNX external data.
_LEXICON_BLOB = "lexicon_frozen.bin"


def _frozen_lexicon_names(model: torch.nn.Module) -> set[str]:
    """ONNX initializer names of the trunk's lexicon-module buffers, if any."""
    lex = getattr(getattr(model, "trunk", None), "lexicon_module", None)
    if lex is None:
        return set()
    return {f"trunk.lexicon_module.{name}" for name, _ in lex.named_buffers()}


def _externalize_frozen_lexicon(path: Path, frozen_names: set[str]):
    """Move the frozen lexicon initializers into the shared blob beside `path`
    and point the graph at it."""
    if not frozen_names:
        return
    model = onnx.load(str(path))
    inits = {i.name: i for i in model.graph.initializer}
    frozen = [inits[n] for n in sorted(frozen_names) if n in inits]
    if not frozen:
        return

    # Sorted-name layout, identical across generations because the compiled
    # lexicon never changes, so the blob only needs writing once.
    blob, chunks, layout, offset = path.parent / _LEXICON_BLOB, [], [], 0
    for init in frozen:
        raw = np.ascontiguousarray(numpy_helper.to_array(init)).tobytes()
        layout.append((init, offset, len(raw)))
        chunks.append(raw)
        offset += len(raw)
    if not blob.exists() or blob.stat().st_size != offset:
        # Beside and renamed over, like the export: a collection must never
        # take a half-written blob (cloud/ssh_transfer.py skips `.tmp` names).
        tmp = blob.with_name(blob.name + ".tmp")
        tmp.write_bytes(b"".join(chunks))
        os.replace(tmp, blob)

    for init, off, length in layout:
        init.ClearField("raw_data")
        init.data_location = TensorProto.EXTERNAL
        del init.external_data[:]
        refs = (("location", _LEXICON_BLOB), ("offset", str(off)), ("length", str(length)))
        for key, val in refs:
            entry = init.external_data.add()
            entry.key, entry.value = key, val
    onnx.save(model, str(path))


def export_onnx(
    model: torch.nn.Module,
    path: str | Path,
    spatial_planes: int,
    scalar_size: int,
    *,
    opp_leave_input: bool,
    board_size: int = 15,
    opset: int = 17,
):
    """Export `model` in eval mode to `path`, atomically."""
    device = next(model.parameters()).device
    dummy_spatial = torch.zeros(1, spatial_planes, board_size, board_size, device=device)
    dummy_scalar = torch.zeros(1, scalar_size, device=device)
    names = ("input_spatial", "input_scalar", "wld", "score_diff", *PLACEMENT_HEAD_NAMES)
    frozen = _frozen_lexicon_names(model)
    export_graph(
        model,
        (dummy_spatial, dummy_scalar),
        path,
        input_names=names[:2],
        output_names=names[2:],
        dynamic_axes={name: {0: "batch"} for name in names},
        # The external-data location is a bare filename, resolved relative to
        # the model's directory, so it stays valid after the rename to `path`.
        postprocess=functools.partial(_externalize_frozen_lexicon, frozen_names=frozen),
        metadata={**common_metadata(opp_leave_input), "graph": "position_eval"},
        opset=opset,
    )
