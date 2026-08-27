"""Export a trained PositionEvalModel to ONNX.

The exported graph takes the same two inputs as the PyTorch model
(`input_spatial`, `input_scalar`) and produces all three head outputs
(`wld`, `score_diff`, then one mask output per
`scribblez.position_eval.model.MASK_HEAD_NAMES` entry). The batch dimension
is dynamic
so the same file serves single-position and batched inference.
"""

import warnings
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, numpy_helper

from scribblez.onnx_export_util import (
    architecture_signature,
    atomic_output,
    common_metadata,
    undo_initializer_dedup,
    write_metadata,
)

from .model import MASK_HEAD_NAMES

# Frozen compiled-lexicon buffers are identical across every checkpoint, so rather
# than bake ~24 MB into each per-generation ONNX they are shared: moved into one blob
# beside the models that all generations reference via ONNX external data.
_LEXICON_BLOB = "lexicon_frozen.bin"


def _frozen_lexicon_names(model: torch.nn.Module) -> set[str]:
    """The ONNX initializer names of the trunk's frozen lexicon-tool buffers (empty
    if the model has no lexicon module)."""
    lex = getattr(getattr(model, "trunk", None), "lexicon_module", None)
    if lex is None:
        return set()
    return {f"trunk.lexicon_module.{name}" for name, _ in lex.named_buffers()}


def _externalize_frozen_lexicon(path: Path, frozen_names: set[str]):
    """Move the frozen lexicon initializers out of the just-written ONNX into a shared
    `lexicon_frozen.bin` beside it (ONNX external data). Every generation lays the same
    (frozen) tensors out identically, so the blob is written once and later generations
    only re-point at it; onnxruntime resolves it transparently at load."""
    if not frozen_names:
        return
    model = onnx.load(str(path))
    inits = {i.name: i for i in model.graph.initializer}
    frozen = [inits[n] for n in sorted(frozen_names) if n in inits]
    if not frozen:
        return

    # Deterministic sorted-name layout with cumulative offsets -- identical across
    # generations because the compiled DAWG never changes, so the blob is write-once.
    blob, chunks, layout, offset = path.parent / _LEXICON_BLOB, [], [], 0
    for init in frozen:
        raw = np.ascontiguousarray(numpy_helper.to_array(init)).tobytes()
        layout.append((init, offset, len(raw)))
        chunks.append(raw)
        offset += len(raw)
    if not blob.exists() or blob.stat().st_size != offset:
        blob.write_bytes(b"".join(chunks))

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
    """Trace `model` and write an ONNX graph to `path` atomically (eval mode,
    dynamic batch), stamping the input-encoding arm into its metadata_props."""
    path = Path(path)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    dummy_spatial = torch.zeros(1, spatial_planes, board_size, board_size, device=device)
    dummy_scalar = torch.zeros(1, scalar_size, device=device)

    # The legacy TorchScript exporter (dynamo=False) is pinned deliberately: it
    # produces the exact output names/order the C++ TensorRT decode binds to and
    # that the parity tests assert. PyTorch 2.9 deprecated that path in favor of
    # the torch.export-based exporter, so silence its expected DeprecationWarnings
    # at this one call site rather than letting them flood every export.
    with atomic_output(path) as tmp_path, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            model,
            (dummy_spatial, dummy_scalar),
            str(tmp_path),
            input_names=["input_spatial", "input_scalar"],
            output_names=["wld", "score_diff", *MASK_HEAD_NAMES],
            dynamic_axes={
                name: {0: "batch"}
                for name in ("input_spatial", "input_scalar", "wld", "score_diff", *MASK_HEAD_NAMES)
            },
            opset_version=opset,
            dynamo=False,
            # Constant folding bakes some weights into derived constants that
            # TensorRT's parser-refitter cannot map back to ONNX initializers,
            # which breaks weight refit onto an architecture-shared cached
            # engine plan. Every weight must survive as a plain initializer.
            do_constant_folding=False,
        )
        undo_initializer_dedup(tmp_path)
        # Share the frozen compiled-lexicon buffers across generations instead
        # of baking them into every export (no-op when the model has no lexicon
        # module). The external-data location recorded inside the graph is the
        # bare blob filename (resolved relative to the directory the model file
        # is loaded from, not to the model file's own name), so it stays
        # correct once tmp_path is renamed to path.
        _externalize_frozen_lexicon(tmp_path, _frozen_lexicon_names(model))
        write_metadata(
            tmp_path,
            {
                **common_metadata(opp_leave_input),
                "model-architecture-signature": architecture_signature(model, opset),
                "graph": "position_eval",
            },
        )
    if was_training:
        model.train()
