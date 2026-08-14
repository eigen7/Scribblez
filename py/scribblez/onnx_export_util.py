"""Shared machinery for the ONNX exporters (position_eval, move_set_eval).

Every exporter in this repo targets the same C++ TensorRT consumer, so they
share the constraints these helpers exist for: every weight must survive as a
plain named initializer (the parser-refitter maps refit weights by initializer
name; aliased or folded weights break the architecture-shared plan cache), the
metadata_props are the explicit contract engine loaders recover model
properties from, and an export must land atomically so a reader never sees a
partial file.
"""

import contextlib
import hashlib
import os
from pathlib import Path

import onnx
import torch

from scribblez.ffi import DEFAULT_LEXICON, format_layout


def undo_initializer_dedup(path: Path):
    """torch.onnx.export emits a single initializer for byte-identical parameter
    tensors (common in a freshly initialized model, where every BatchNorm layer
    starts from the same gamma/beta/running stats) and aliases the other
    parameter names to it through Identity nodes. TensorRT's ONNX parser folds
    weights reached through such aliases into anonymous derived tensors that
    its refitter cannot map back to any initializer, which breaks refitting a
    model's weights onto an architecture-shared cached engine plan. Materialize
    each alias as its own initializer and drop the Identity nodes so every
    parameter is a plain named initializer."""
    model = onnx.load(str(path))
    inits = {i.name: i for i in model.graph.initializer}
    aliases = [
        node
        for node in model.graph.node
        if node.op_type == "Identity" and node.input[0] in inits and node.output[0] not in inits
    ]
    if not aliases:
        return
    for node in aliases:
        dup = onnx.TensorProto()
        dup.CopyFrom(inits[node.input[0]])
        dup.name = node.output[0]
        model.graph.initializer.append(dup)
        model.graph.node.remove(node)
    onnx.save(model, str(path))


def architecture_signature(model: torch.nn.Module, opset: int) -> str:
    """md5 fingerprint of the model's architecture: the module tree's repr (layer
    structure and shapes, but not weights) plus the export opset and the
    torch/onnx versions that shape the emitted graph. Two checkpoints of the
    same architecture produce the same signature.

    The position runtime's C++ loader keys its engine-plan cache on it, so such
    checkpoints share one cached plan and load by refitting it with their own
    weights. MoveSetNet stamps the same signature but keys its cache on the
    model's content hash instead, building a plan per checkpoint and never
    refitting (engine/include/nn/trt_util.h)."""
    components = [str(model), f"opset={opset}", torch.__version__, onnx.__version__]
    return hashlib.md5("\n".join(components).encode()).hexdigest()


def write_metadata(path: Path, entries: dict[str, str]):
    """Record `entries` in the ONNX metadata_props -- the explicit contract
    consumers recover model properties from (the dashboard's what-if runner
    reads the encoding arm; C++ loaders cross-check the arm against the
    declared input dims and key the TensorRT engine-plan cache on the
    architecture signature; the graph/version keys let a loader reject a
    mismatched graph kind or move encoding once it checks them)."""
    m = onnx.load(str(path), load_external_data=False)
    for key, value in entries.items():
        entry = m.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(m, str(path))


def common_metadata(contingent_features: bool, opp_leave_input: bool) -> dict[str, str]:
    """The metadata entries every exporter stamps -- the input-encoding arm,
    the board-row encoding version (the engine's loader rejects a checkpoint
    trained under a different one), and the lexicon -- for the exporter to
    extend with its graph-specific keys."""
    return {
        "contingent_features": "true" if contingent_features else "false",
        "opp_leave_input": "true" if opp_leave_input else "false",
        "input_encoding_version": str(format_layout()["constants"]["input_encoding_version"]),
        "lexicon": DEFAULT_LEXICON,
    }


@contextlib.contextmanager
def atomic_output(path: Path):
    """Yield the temp path an export (and its in-place transforms) should land
    on, renaming it onto `path` with a single os.replace on success. A reader
    that sees `path` exist therefore always sees a complete file -- never a
    partially written or partially transformed one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    yield tmp_path
    os.replace(tmp_path, path)
