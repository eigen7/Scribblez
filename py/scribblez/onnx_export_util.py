"""Shared machinery for the ONNX exporters (position_eval, move_set_eval).

Every exporter traces through export_graph, which owns the steps every export
needs: the module in eval mode and each submodule's mode restored afterwards,
the tracer settings, the metadata, and a check of the written file against
the module itself.

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
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
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


# --- export_graph -----------------------------------------------------------

# How far the written graph, run by onnxruntime on the CPU, may stray from the
# module run eagerly at full FP32. Kernel and reduction order alone stay near
# 1e-5; an export defect (a shape baked in by the tracer, a folded or
# mis-bound weight, a module traced in the wrong mode) moves outputs by
# order-one amounts.
CHECK_RTOL = 1e-3
CHECK_ATOL = 1e-3

# Rows the check feeds along each dynamic axis: unlike any exporter's dummy
# count, so a tracer that baked the dummy's size into the graph fails.
CHECK_DYNAMIC_ROWS = 3


@contextlib.contextmanager
def eval_mode(module: nn.Module):
    """Every submodule of `module` in eval mode inside the block, each back in
    its own mode afterwards, even on error.

    Restoring submodule by submodule, rather than calling train() on the top,
    keeps a submodule that was deliberately in eval mode (a frozen backbone)
    in eval mode. torch.onnx.export restores only the module it is given,
    recursively, so a wrapper left in train mode would put everything it
    wraps in train mode."""
    modes = [(m, m.training) for m in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for m, training in modes:
            m.training = training


@contextlib.contextmanager
def _full_fp32():
    """Eager matmuls and convolutions in full FP32, not TF32, so the check
    compares the export against the module's exact arithmetic."""
    matmul = torch.get_float32_matmul_precision()
    cudnn = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(matmul)
        torch.backends.cudnn.allow_tf32 = cudnn


def _check_inputs(args, input_names, dynamic_axes, seed: int = 0) -> list[torch.Tensor]:
    """Fresh inputs shaped like `args`: CHECK_DYNAMIC_ROWS rows along each
    dynamic axis 0, floats random normal, integer inputs (indices, masks)
    zero, which every integer input accepts."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for name, arg in zip(input_names, args, strict=True):
        shape = list(arg.shape)
        if 0 in dynamic_axes.get(name, {}):
            shape[0] = CHECK_DYNAMIC_ROWS
        if arg.dtype.is_floating_point:
            t = torch.randn(shape, generator=gen, dtype=arg.dtype)
        else:
            t = torch.zeros(shape, dtype=arg.dtype)
        out.append(t.to(arg.device))
    return out


def _flat_outputs(out) -> list[torch.Tensor]:
    """A forward's outputs in the order the tracer flattens them into graph
    outputs: a tensor alone, a tuple in order, a dict by its values."""
    if isinstance(out, torch.Tensor):
        return [out]
    return list(out.values()) if isinstance(out, dict) else list(out)


def check_against_module(path: Path, graph: nn.Module, inputs, input_names, output_names):
    """Raise ValueError unless the ONNX at `path`, run by onnxruntime, gives
    `graph`'s eager outputs on `inputs`."""
    with eval_mode(graph), torch.no_grad(), _full_fp32():
        want = graph(*inputs)
    want = _flat_outputs(want)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {name: t.cpu().numpy() for name, t in zip(input_names, inputs, strict=True)}
    got = session.run(list(output_names), feed)
    for name, g, w in zip(output_names, got, want, strict=True):
        w = w.cpu().numpy()
        if g.shape != w.shape or not np.allclose(g, w, rtol=CHECK_RTOL, atol=CHECK_ATOL):
            err = np.abs(g - w).max() if g.shape == w.shape else "shape"
            raise ValueError(
                f"{path}: exported output '{name}' disagrees with the module "
                f"(shape {g.shape} vs {w.shape}, max error {err})"
            )


def export_graph(
    graph: nn.Module,
    args: tuple[torch.Tensor, ...],
    path: str | Path,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: dict[str, dict[int, str]],
    metadata: dict[str, str],
    opset: int,
    postprocess: Callable[[Path], None] | None = None,
):
    """Trace `graph` on the dummy inputs `args` to `path`, atomically, and
    stamp `metadata` plus its architecture signature.

    The trace runs in eval mode (eval_mode), with the legacy TorchScript
    exporter, which produces the output names and order the engine binds to,
    and without constant folding, which would turn weights into derived
    constants the TensorRT refitter cannot map back to initializers.
    `postprocess(tmp_path)` runs after the initializer de-duplication, before
    the metadata (e.g. to move weights into external data). Before the file
    lands, check_against_module runs it on fresh inputs; on a mismatch it
    raises and no file is written."""
    path = Path(path)
    with atomic_output(path) as tmp_path:
        with eval_mode(graph), warnings.catch_warnings():
            # Deprecated since PyTorch 2.9; still the exporter we target.
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(
                graph,
                args,
                str(tmp_path),
                input_names=list(input_names),
                output_names=list(output_names),
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                training=torch.onnx.TrainingMode.EVAL,
                dynamo=False,
                do_constant_folding=False,
            )
        undo_initializer_dedup(tmp_path)
        if postprocess:
            postprocess(tmp_path)
        write_metadata(
            tmp_path,
            {"model-architecture-signature": architecture_signature(graph, opset), **metadata},
        )
        inputs = _check_inputs(args, input_names, dynamic_axes)
        check_against_module(tmp_path, graph, inputs, input_names, output_names)
