"""Shared machinery for the ONNX exporters (position_eval, move_set_eval).

Every exporter targets the same C++ TensorRT loader, which imposes three
constraints:

- Every weight must appear as a plain named initializer. The loader caches one
  engine plan per architecture and refits each checkpoint's weights onto it by
  initializer name; aliased or folded weights cannot be refit.
- metadata_props is the contract from which the loader recovers model
  properties.
- An export must land atomically, so a reader never sees a partial file.
"""

import contextlib
import hashlib
import os
from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import numpy_helper

from scribblez.ffi import DEFAULT_LEXICON, format_layout


def undo_initializer_dedup(path: Path):
    """Give every parameter its own named initializer again.

    torch.onnx.export emits one initializer for byte-identical parameters
    (common in a freshly initialized model, where every BatchNorm starts from
    the same statistics) and aliases the other names to it through Identity
    nodes. TensorRT folds such aliases into anonymous tensors its refitter
    cannot map back to a name. This materializes each alias as its own
    initializer and drops the Identity nodes."""
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


def load_onnx_initializers(model: nn.Module, path: Path):
    """Load an export's initializers back into `model`'s parameters and buffers.

    Exports keep every parameter and buffer as an initializer under its
    state-dict name, so a generation whose torch checkpoint was not kept can be
    rebuilt from its ONNX export for offline evaluation. BatchNorm's
    `num_batches_tracked` is the only state absent from a graph; any other
    mismatch is an architecture mismatch and raises."""
    graph = onnx.load(str(path)).graph
    weights = {
        init.name: torch.from_numpy(numpy_helper.to_array(init).copy())
        for init in graph.initializer
    }
    expected = {k for k in model.state_dict() if not k.endswith("num_batches_tracked")}
    if set(weights) != expected:
        missing = sorted(expected - set(weights))
        unexpected = sorted(set(weights) - expected)
        raise ValueError(
            f"{path}: initializers do not match the model -- "
            f"missing {missing[:5]}, unexpected {unexpected[:5]}"
        )
    model.load_state_dict(weights, strict=False)


def architecture_signature(model: torch.nn.Module, opset: int) -> str:
    """A fingerprint of the model's architecture, independent of its weights:
    the module tree's repr plus the opset and torch/onnx versions, which all
    shape the emitted graph.

    The C++ loaders key their engine-plan cache on it, so checkpoints of one
    architecture share a cached plan and load by refitting their own weights
    onto it (engine/include/nn/trt_util.h)."""
    components = [str(model), f"opset={opset}", torch.__version__, onnx.__version__]
    return hashlib.md5("\n".join(components).encode()).hexdigest()


def weight_fingerprint(model: torch.nn.Module) -> str:
    """A fingerprint of the model's weights. Graphs that must come from the same
    model (the move-proposal cache/step pair) carry it so a loader can reject a
    mismatched pair."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --- export-safe re-expressions ------------------------------------------
#
# Two torch constructs export badly and are re-expressed here for the
# exporters' wrapper modules:
# - nn.MultiheadAttention's packed in_proj traces as a bare Constant with no
#   initializer behind it, so it cannot be refit.
# - A Linear over cat([per-row, shared]) has to Expand the shared part across
#   the dynamic row axis, and the legacy TorchScript tracer bakes the traced
#   row count into that Expand.


def split_mha_qkv(mha: nn.MultiheadAttention) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    """Copy a packed nn.MultiheadAttention's in_proj into three plain q/k/v
    nn.Linears. out_proj is already a plain Linear; use it directly."""
    c = mha.embed_dim
    q, k, v = nn.Linear(c, c), nn.Linear(c, c), nn.Linear(c, c)
    with torch.no_grad():
        q.weight.copy_(mha.in_proj_weight[:c])
        q.bias.copy_(mha.in_proj_bias[:c])
        k.weight.copy_(mha.in_proj_weight[c : 2 * c])
        k.bias.copy_(mha.in_proj_bias[c : 2 * c])
        v.weight.copy_(mha.in_proj_weight[2 * c :])
        v.bias.copy_(mha.in_proj_bias[2 * c :])
    return q, k, v


def split_concat_linear(linear: nn.Linear, split: int) -> tuple[nn.Linear, nn.Linear]:
    """Split a Linear over cat([a, b]) (a has `split` features) into a_part,
    which carries the bias, and a bias-free b_part, so that
    a_part(a) + b_part(b) equals the original. A shared `b` of shape (1, ...)
    then broadcasts in the add instead of being Expanded across rows."""
    out_features = linear.out_features
    a_part = nn.Linear(split, out_features)
    b_part = nn.Linear(linear.in_features - split, out_features, bias=False)
    with torch.no_grad():
        a_part.weight.copy_(linear.weight[:, :split])
        a_part.bias.copy_(linear.bias)
        b_part.weight.copy_(linear.weight[:, split:])
    return a_part, b_part


def cross_attention_2d(
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    out_proj: nn.Linear,
    num_heads: int,
    queries: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """Unmasked, unbatched multi-head attention over the split_mha_qkv
    projections, matching an eval-mode nn.MultiheadAttention:
    queries (Nq, C), keys (Nk, C) -> (Nq, C)."""
    c = q_proj.out_features
    d = c // num_heads
    q = q_proj(queries).view(-1, num_heads, d).transpose(0, 1)  # (H, Nq, d)
    k = k_proj(keys).view(-1, num_heads, d).transpose(0, 1)  # (H, Nk, d)
    v = v_proj(keys).view(-1, num_heads, d).transpose(0, 1)
    attn = F.softmax(q @ k.transpose(1, 2) * d**-0.5, dim=-1)  # (H, Nq, Nk)
    ctx = (attn @ v).transpose(0, 1).reshape(-1, c)  # (Nq, C)
    return out_proj(ctx)


def write_metadata(path: Path, entries: dict[str, str]):
    """Append `entries` to the file's ONNX metadata_props.

    Consumers read model properties from these entries rather than inferring
    them. The C++ loaders check the encoding arm against the declared input
    dims, reject a mismatched graph kind or encoding version, and key the
    engine-plan cache on the architecture signature. The dashboard's what-if
    runner reads the encoding arm."""
    m = onnx.load(str(path), load_external_data=False)
    for key, value in entries.items():
        entry = m.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(m, str(path))


def common_metadata(opp_leave_input: bool) -> dict[str, str]:
    """The metadata entries every export carries: the input-encoding arm, the
    board-row encoding version, and the lexicon. Exporters add their
    graph-specific keys."""
    return {
        "opp_leave_input": "true" if opp_leave_input else "false",
        "input_encoding_version": str(format_layout()["constants"]["input_encoding_version"]),
        "lexicon": DEFAULT_LEXICON,
    }


@contextlib.contextmanager
def atomic_output(path: Path):
    """Yield a temp path for the export and its in-place post-processing, then
    os.replace it onto `path` on success, so a reader never sees a partial
    file. On failure the temp file is removed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        yield tmp_path
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, path)
