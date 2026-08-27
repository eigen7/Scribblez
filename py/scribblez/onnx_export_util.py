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
import torch.nn as nn
import torch.nn.functional as F

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

    The C++ loaders key the engine-plan cache on it, so such checkpoints share
    one cached plan and load by refitting it with their own weights
    (engine/include/nn/trt_util.h)."""
    components = [str(model), f"opset={opset}", torch.__version__, onnx.__version__]
    return hashlib.md5("\n".join(components).encode()).hexdigest()


def weight_fingerprint(model: torch.nn.Module) -> str:
    """sha256 over the model's parameters and buffers in a deterministic order
    -- unlike architecture_signature, this changes with the WEIGHTS, so it
    distinguishes two checkpoints of the same architecture. Used to tie graphs
    that must come from one in-memory model (the move-proposal cache/step pair)
    so a loader can reject a stale graph paired with a newer one."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --- refit-discipline re-expressions -------------------------------------
#
# The TensorRT parser-refitter maps refit weights by initializer name, so every
# weight must trace as a plain named initializer. Two torch constructs defeat
# that and are re-expressed here, shared by every exporter: an
# nn.MultiheadAttention's packed in_proj (traces as a bare Constant with no
# initializer behind it), and a Linear over a concatenation (whose per-move
# broadcast Expands across the dynamic row axis, the legacy tracer's
# shape-baking hazard).


def split_mha_qkv(mha: nn.MultiheadAttention) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    """Re-express a packed nn.MultiheadAttention's in_proj as three plain q/k/v
    nn.Linears holding copied weight/bias slices as their own parameters, so
    each exports as a plain named initializer. out_proj is already a plain
    Linear -- read it off `mha` directly."""
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
    """Re-associate a Linear over cat([a (split), b (rest)]) into an `a`
    sub-Linear (carrying the bias) and a bias-free `b` sub-Linear, each holding a
    copied weight slice. `a_part(a) + b_part(b)` equals the original applied to
    the concatenation, but `b` stays (1, ...) and broadcasts in the add rather
    than Expanding across the dynamic row axis."""
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
    """Batchless multi-head attention (the plain-Linear form of an eval-mode
    nn.MultiheadAttention, no mask): q from `queries` (Nq, C), k/v from `keys`
    (Nk, C) -> (Nq, C)."""
    c = q_proj.out_features
    d = c // num_heads
    q = q_proj(queries).view(-1, num_heads, d).transpose(0, 1)  # (H, Nq, d)
    k = k_proj(keys).view(-1, num_heads, d).transpose(0, 1)  # (H, Nk, d)
    v = v_proj(keys).view(-1, num_heads, d).transpose(0, 1)
    attn = F.softmax(q @ k.transpose(1, 2) * d**-0.5, dim=-1)  # (H, Nq, Nk)
    ctx = (attn @ v).transpose(0, 1).reshape(-1, c)  # (Nq, C)
    return out_proj(ctx)


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


def common_metadata(opp_leave_input: bool) -> dict[str, str]:
    """The metadata entries every exporter stamps -- the input-encoding arm,
    the board-row encoding version (the engine's loader rejects a checkpoint
    trained under a different one), and the lexicon -- for the exporter to
    extend with its graph-specific keys."""
    return {
        "opp_leave_input": "true" if opp_leave_input else "false",
        "input_encoding_version": str(format_layout()["constants"]["input_encoding_version"]),
        "lexicon": DEFAULT_LEXICON,
    }


@contextlib.contextmanager
def atomic_output(path: Path):
    """Yield the temp path an export (and its in-place transforms) should land
    on, renaming it onto `path` with a single os.replace on success. A reader
    that sees `path` exist therefore always sees a complete file -- never a
    partially written or partially transformed one. On failure the temp file
    is removed too: a rejected export (the FP16 gate's designed outcome)
    leaves nothing beside the served models."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        yield tmp_path
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, path)
