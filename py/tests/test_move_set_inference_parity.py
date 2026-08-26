"""Inference-parity tests for the move-set-eval ONNX export (hop A).

Three claims, each pinned before any TensorRT runtime exists (PR 4 covers hop
B separately):

  (a) The P=1 export wrapper IS the training forward: over a ragged
      multi-position batch, decomposing into per-position P=1 calls reproduces
      `MoveSetEvalModel.forward` (allclose, not bitwise -- the re-associated
      head and the collapsed grid reorder float sums).
  (b) The exported graph is truly dynamic in M: traced at one M, it reproduces
      the wrapper under ONNXRuntime at different Ms -- a single-M check would
      pass with a shape silently baked by the legacy tracer.
  (c) The file contract holds: input/output names, dtypes and the "moves"
      axis; the metadata keys engine loaders read (graph kind, move-encoding
      version, arm, signature); the re-associated head weights present as
      plain initializers with no Identity aliasing left.

The model is randomly initialized: this tests the plumbing's fidelity, not any
trained weights, so the tests stay hermetic (no checkpoint, no GPU).
"""

import json
import subprocess
import sys

import numpy as np
import onnx
import pytest
import torch
from scribblez.ffi import get_input_shapes
from scribblez.move_set_eval.model import MoveSetEvalModel
from scribblez.move_set_eval.moves import move_encoding_dims
from scribblez.move_set_eval.onnx_export import (
    MOVE_INPUT_NAMES,
    OUTPUT_NAMES,
    MoveSetEvalExportModel,
    export_onnx,
)

_input_shapes = {s.name: s.dims for s in get_input_shapes()}
SPATIAL_PLANES, BOARD_SIZE, _ = _input_shapes["input_spatial"]
SCALAR_SIZE = _input_shapes["input_scalar"][0]
MAX_PLACED, NUM_SCALARS, LETTER_VOCAB, CELLS = move_encoding_dims()


def _random_model(seed: int = 0) -> MoveSetEvalModel:
    torch.manual_seed(seed)
    model = MoveSetEvalModel(
        spatial_planes=SPATIAL_PLANES,
        scalar_size=SCALAR_SIZE,
        trunk_channels=16,  # tiny: numerics, not capacity
        num_blocks=3,  # includes one global-pooling block (covers its ops)
        num_heads=2,
    )
    model.eval()
    return model


def _random_moves(m: int, seed: int):
    """A candidate block mixing realistic plays, an exchange (letters but no
    squares, is_play 0), and a pass row -- so the is_play gate and the pad
    slots are exercised, not just dense plays."""
    rng = np.random.default_rng(seed)
    letters = rng.integers(1, LETTER_VOCAB, (m, MAX_PLACED), dtype=np.int32)
    blanks = rng.integers(0, 2, (m, MAX_PLACED)).astype(np.uint8)
    squares = rng.integers(0, CELLS, (m, MAX_PLACED), dtype=np.int32)
    tile_mask = (rng.random((m, MAX_PLACED)) < 0.6).astype(np.uint8)
    scalars = rng.standard_normal((m, NUM_SCALARS)).astype(np.float32)
    scalars[:, 2] = 1.0  # plays...
    if m >= 2:  # ...except an exchange row (tiles, no squares) and a pass row
        squares[-2] = 0
        scalars[-2, 2] = 0.0
        letters[-1] = 0
        tile_mask[-1] = 0
        squares[-1] = 0
        scalars[-1, 2] = 0.0
    return letters, blanks, squares, tile_mask, scalars


def _board_inputs(p: int, seed: int):
    rng = np.random.default_rng(seed)
    spatial = rng.standard_normal((p, SPATIAL_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    scalar = rng.standard_normal((p, SCALAR_SIZE), dtype=np.float32)
    return spatial, scalar


def _training_batch(counts, seed: int = 1):
    """The flattened multi-position batch the training forward consumes."""
    spatial, scalar = _board_inputs(len(counts), seed)
    per_pos = [_random_moves(m, seed + 10 + i) for i, m in enumerate(counts)]
    moves = [np.concatenate(arrays) for arrays in zip(*per_pos, strict=True)]
    pos_id = np.concatenate([np.full(m, i, dtype=np.int64) for i, m in enumerate(counts)])
    return spatial, scalar, per_pos, moves, pos_id


def _torch_move_args(letters, blanks, squares, tile_mask, scalars):
    """The training forward's dtypes (dataset.py upcasts the FFI arrays)."""
    return (
        torch.from_numpy(letters).long(),
        torch.from_numpy(blanks).bool(),
        torch.from_numpy(squares).long(),
        torch.from_numpy(tile_mask).bool(),
        torch.from_numpy(scalars),
    )


def test_export_wrapper_matches_training_forward_on_ragged_batches():
    counts = [1, 5, 12, 3]
    spatial, scalar, per_pos, moves, pos_id = _training_batch(counts)
    model = _random_model()
    wrapper = MoveSetEvalExportModel(model)
    wrapper.eval()

    with torch.no_grad():
        ref = model(
            torch.from_numpy(spatial),
            torch.from_numpy(scalar),
            *_torch_move_args(*moves),
            torch.from_numpy(pos_id),
        )
        start = 0
        for i, m in enumerate(counts):
            letters, blanks, squares, tile_mask, scalars = per_pos[i]
            wld, score_diff = wrapper(
                torch.from_numpy(spatial[i : i + 1]),
                torch.from_numpy(scalar[i : i + 1]),
                torch.from_numpy(letters),
                torch.from_numpy(blanks),
                torch.from_numpy(squares),
                torch.from_numpy(tile_mask),
                torch.from_numpy(scalars),
            )
            np.testing.assert_allclose(
                wld.numpy(), ref["wld"][start : start + m].numpy(), atol=1e-5, rtol=1e-4
            )
            np.testing.assert_allclose(
                score_diff.numpy(),
                ref["score_diff"][start : start + m].numpy(),
                atol=1e-5,
                rtol=1e-4,
            )
            start += m


def _export(tmp_path, model, move_encoding_version=1):
    path = tmp_path / "mset.onnx"
    export_onnx(
        model,
        path,
        SPATIAL_PLANES,
        SCALAR_SIZE,
        opp_leave_input=False,
        move_encoding_version=move_encoding_version,
    )
    return path


def test_onnx_runtime_matches_torch_at_other_ms(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model = _random_model()
    wrapper = MoveSetEvalExportModel(model)
    wrapper.eval()
    path = _export(tmp_path, model)  # traced at M=5 inside export_onnx
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    spatial, scalar = _board_inputs(1, seed=3)
    for m in (1, 37):  # neither is the traced M
        letters, blanks, squares, tile_mask, scalars = _random_moves(m, seed=40 + m)
        feeds = {
            "input_spatial": spatial,
            "input_scalar": scalar,
            "move_letters": letters,
            "move_blanks": blanks,
            "move_squares": squares,
            "move_tile_mask": tile_mask,
            "move_scalars": scalars,
        }
        got = sess.run(list(OUTPUT_NAMES), feeds)
        with torch.no_grad():
            want = wrapper(
                torch.from_numpy(spatial),
                torch.from_numpy(scalar),
                torch.from_numpy(letters),
                torch.from_numpy(blanks),
                torch.from_numpy(squares),
                torch.from_numpy(tile_mask),
                torch.from_numpy(scalars),
            )
        for got_arr, want_t in zip(got, want, strict=True):
            assert got_arr.shape == tuple(want_t.shape)
            np.testing.assert_allclose(got_arr, want_t.numpy(), atol=1e-5, rtol=1e-4)


def test_exported_file_contract(tmp_path):
    path = _export(tmp_path, _random_model(), move_encoding_version=1)
    m = onnx.load(str(path))

    meta = {e.key: e.value for e in m.metadata_props}
    assert meta["graph"] == "move_set_eval"
    assert meta["move_encoding_version"] == "1"
    assert meta["opp_leave_input"] == "false"
    assert "model-architecture-signature" in meta

    assert [o.name for o in m.graph.output] == list(OUTPUT_NAMES)
    inputs = {i.name: i for i in m.graph.input}
    assert set(inputs) == {"input_spatial", "input_scalar", *MOVE_INPUT_NAMES}
    itypes = {name: i.type.tensor_type.elem_type for name, i in inputs.items()}
    assert itypes["input_spatial"] == onnx.TensorProto.FLOAT
    assert itypes["move_letters"] == onnx.TensorProto.INT32
    assert itypes["move_squares"] == onnx.TensorProto.INT32
    assert itypes["move_blanks"] == onnx.TensorProto.UINT8
    assert itypes["move_tile_mask"] == onnx.TensorProto.UINT8
    assert itypes["move_scalars"] == onnx.TensorProto.FLOAT
    for name in MOVE_INPUT_NAMES:
        dim0 = inputs[name].type.tensor_type.shape.dim[0]
        assert dim0.dim_param == "moves", name
    # The board inputs are static (1, ...): the P=1 specialization.
    assert inputs["input_spatial"].type.tensor_type.shape.dim[0].dim_value == 1

    # Refit discipline: the re-associated head is materialized as plain named
    # initializers, and no Identity-aliased weights remain.
    init_names = {i.name for i in m.graph.initializer}
    assert any("head_attended" in n for n in init_names)
    assert any("head_g" in n for n in init_names)
    for proj in ("q_proj", "k_proj", "v_proj"):  # the hand-rolled attention
        assert any(proj in n for n in init_names), proj
    aliased = [n for n in m.graph.node if n.op_type == "Identity" and n.input[0] in init_names]
    assert not aliased


# The legacy-checkpoint path runs in a subprocess: adopting the information
# condition sets the process-wide FFI session arm, which this test process may
# already have created under a different arm (the same isolation the trainer
# tests use).
_LEGACY_CONDITION_DRIVER = """
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scribblez.move_set_eval import targets as T
from scribblez.move_set_eval.onnx_export import legacy_checkpoint_condition

root, flags = Path(sys.argv[1]), int(sys.argv[2])
store = root / "slogs"
store.mkdir(parents=True)
hdr = np.zeros(1, dtype=T._FILE_HEADER)
hdr["magic"], hdr["version"] = T.MSET_MAGIC, T.MSET_VERSION
hdr["record_floats"], hdr["flags"] = 5, flags
hdr["model_hash"] = b"cafe"
(store / "x.mset").write_bytes(hdr.tobytes())

got = legacy_checkpoint_condition(SimpleNamespace(data_dir=root))
assert got["move_encoding_version"] == 0, got  # pre-fix rows, never the live constant
assert got["open_leaves"] == bool(flags & T.MSET_FLAG_OPEN_LEAVES), got
import json
print("RESULT " + json.dumps(got))
"""


def _run_legacy_condition(tmp_path, name: str, flags: int) -> dict:
    script = tmp_path / "driver.py"
    script.write_text(_LEGACY_CONDITION_DRIVER)
    out = subprocess.run(
        [sys.executable, str(script), str(tmp_path / name), str(flags)],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("RESULT "))
    return json.loads(line.removeprefix("RESULT "))


def test_legacy_checkpoint_condition_recovers_the_arm(tmp_path):
    """The standalone exporter's fallback for checkpoints predating the
    self-describing config: the arm comes off the corpus header and the
    version is pinned to 0, the stamp an engine enforcing the move-encoding
    version will rightly refuse against a newer encoder. Both arms run (each
    in its own subprocess -- the session arm is process-wide), and the
    open-leaves corpus must yield a strictly wider scalar block: the
    inequality is what proves the adopted arm actually reached the layout,
    not just the returned flag."""
    from scribblez.move_set_eval.targets import MSET_FLAG_OPEN_LEAVES

    hidden = _run_legacy_condition(tmp_path, "hidden", 0)
    open_ = _run_legacy_condition(tmp_path, "open", MSET_FLAG_OPEN_LEAVES)
    assert hidden["spatial_planes"] == open_["spatial_planes"] > 0
    assert open_["scalar_size"] > hidden["scalar_size"] > 0  # the opp-leave block
