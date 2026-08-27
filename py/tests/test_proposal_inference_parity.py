"""Inference-parity tests for the move-proposal split ONNX export (roadmap item 3).

The move proposal model runs incrementally in the engine as two graphs -- a
`move_proposal_cache` graph once per turn, then a `move_proposal_step` graph per
evidence-loop iteration reusing the cache. These tests pin the Python side of
that split before any TensorRT runtime exists (PR3 covers the engine):

  (a) The two export wrappers, composed, ARE `MoveSetEvalModel.forward`: the
      cache wrapper's plain heads and the step wrapper's evidence-conditioned
      heads reproduce the monolithic forward over empty, partial, and full
      evidence sets (allclose, not bitwise -- the re-associated heads and the
      re-expressed attentions reorder float sums). Evidence tokens gather the
      cache's own move encodings for candidates that are in the scored set, so
      the reference's evidence moves are the first k candidates.
  (b) Both graphs are truly dynamic in M: traced at one M, they reproduce the
      wrappers under ONNXRuntime at other Ms -- and the step graph's fixed-width
      E evidence block stays inert past its mask.
  (c) The file contract: names, dtypes, the "moves" axis, the fixed-width
      leading-1 evidence inputs, the metadata engine loaders read (graph kind,
      move-encoding version, the shared proposal_export_id tying the pair), and
      the refit discipline (split heads + hand-rolled attentions as plain
      initializers, no Identity aliasing).

The model is randomly initialized, with its evidence path perturbed off its
zero-init so a populated set actually differs from the plain pass -- this tests
plumbing fidelity, not trained weights, and stays hermetic (no checkpoint, no
GPU).
"""

import numpy as np
import onnx
import pytest
import torch
from scribblez.evidence_fusion import (
    NUM_EVIDENCE_PLANES,
    NUM_EVIDENCE_SCALARS,
    EvidenceInputs,
)
from scribblez.ffi import get_input_shapes
from scribblez.move_set_eval.model import MoveSetEvalModel
from scribblez.move_set_eval.moves import move_encoding_dims
from scribblez.move_set_eval.proposal_export import (
    CACHE_INPUT_NAMES,
    CACHE_OUTPUT_NAMES,
    GRAPH_CACHE,
    GRAPH_STEP,
    STEP_INPUT_NAMES,
    STEP_OUTPUT_NAMES,
    ProposalCacheExportModel,
    ProposalStepExportModel,
    export_proposal_cache,
    export_proposal_step,
    proposal_export_id,
)

_input_shapes = {s.name: s.dims for s in get_input_shapes()}
SPATIAL_PLANES, BOARD_SIZE, _ = _input_shapes["input_spatial"]
SCALAR_SIZE = _input_shapes["input_scalar"][0]
MAX_PLACED, NUM_SCALARS, LETTER_VOCAB, CELLS = move_encoding_dims()
CHANNELS = 16
MAX_E = 6


def _random_model(seed: int = 0) -> MoveSetEvalModel:
    torch.manual_seed(seed)
    model = MoveSetEvalModel(
        spatial_planes=SPATIAL_PLANES,
        scalar_size=SCALAR_SIZE,
        trunk_channels=CHANNELS,  # tiny: numerics, not capacity
        num_blocks=3,  # includes one global-pooling block (covers its ops)
        num_heads=2,
    )
    # The fusion projections and proves-best head are zero-init, which would make
    # every conditioned pass equal the plain one -- perturb them so a populated
    # evidence set genuinely exercises the fusion + gain path.
    with torch.no_grad():
        for module in (model.evidence_fusion, model.proves_best):
            for p in module.parameters():
                p.add_(0.1 * torch.randn_like(p))
    model.eval()
    return model


def _random_moves(m: int, seed: int):
    """A candidate block mixing plays, an exchange (letters, no squares), and a
    pass row -- exercising the is_play gate and the pad slots."""
    rng = np.random.default_rng(seed)
    letters = rng.integers(1, LETTER_VOCAB, (m, MAX_PLACED), dtype=np.int32)
    blanks = rng.integers(0, 2, (m, MAX_PLACED)).astype(np.uint8)
    squares = rng.integers(0, CELLS, (m, MAX_PLACED), dtype=np.int32)
    tile_mask = (rng.random((m, MAX_PLACED)) < 0.6).astype(np.uint8)
    scalars = rng.standard_normal((m, NUM_SCALARS)).astype(np.float32)
    scalars[:, 2] = 1.0  # plays...
    if m >= 2:  # ...except an exchange row and a pass row
        squares[-2] = 0
        scalars[-2, 2] = 0.0
        letters[-1] = 0
        tile_mask[-1] = 0
        squares[-1] = 0
        scalars[-1, 2] = 0.0
    return letters, blanks, squares, tile_mask, scalars


def _board_inputs(seed: int):
    rng = np.random.default_rng(seed)
    spatial = rng.standard_normal((1, SPATIAL_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    scalar = rng.standard_normal((1, SCALAR_SIZE), dtype=np.float32)
    return spatial, scalar


def _evidence_inputs(moves, k: int, seed: int):
    """A width-MAX_E evidence set whose k real tokens describe the first k
    candidate moves (so the cache-move-encoding gather equals a re-encode), with
    random observation planes/scalars. Returns (EvidenceInputs, ev_obs_planes,
    ev_obs_scalars, ev_mask_uint8) -- the same observation arrays the step graph
    is fed, so the reference and the graph see identical evidence."""
    letters, blanks, squares, tile_mask, scalars = moves
    el = np.zeros((MAX_E, MAX_PLACED), np.int64)
    eb = np.zeros((MAX_E, MAX_PLACED), np.int64)
    es = np.zeros((MAX_E, MAX_PLACED), np.int64)
    et = np.zeros((MAX_E, MAX_PLACED), np.float32)
    esc = np.zeros((MAX_E, NUM_SCALARS), np.float32)
    el[:k] = letters[:k]
    eb[:k] = blanks[:k]
    es[:k] = squares[:k]
    et[:k] = tile_mask[:k]
    esc[:k] = scalars[:k]

    rng = np.random.default_rng(seed)
    obs_planes = np.abs(
        rng.standard_normal((1, MAX_E, NUM_EVIDENCE_PLANES, BOARD_SIZE, BOARD_SIZE))
    ).astype(np.float32)
    obs_scalars = rng.standard_normal((1, MAX_E, NUM_EVIDENCE_SCALARS)).astype(np.float32)
    mask = np.zeros((1, MAX_E), dtype=bool)
    mask[0, :k] = True

    ev = EvidenceInputs(
        letters=torch.from_numpy(el)[None],
        blanks=torch.from_numpy(eb)[None],
        squares=torch.from_numpy(es)[None],
        tile_mask=torch.from_numpy(et)[None],
        scalars=torch.from_numpy(esc)[None],
        obs_planes=torch.from_numpy(obs_planes),
        obs_scalars=torch.from_numpy(obs_scalars),
        mask=torch.from_numpy(mask),
    )
    return ev, obs_planes, obs_scalars, mask.astype(np.uint8)


def _reference(model, spatial, scalar, moves, ev):
    """MoveSetEvalModel.forward over one position's candidate set, optionally
    evidence-conditioned."""
    letters, blanks, squares, tile_mask, scalars = moves
    m = len(scalars)
    with torch.no_grad():
        return model(
            torch.from_numpy(spatial),
            torch.from_numpy(scalar),
            torch.from_numpy(letters).long(),
            torch.from_numpy(blanks).bool(),
            torch.from_numpy(squares).long(),
            torch.from_numpy(tile_mask).float(),
            torch.from_numpy(scalars),
            torch.zeros(m, dtype=torch.long),
            evidence=ev,
        )


@pytest.mark.parametrize("k", [0, 3, MAX_E])
def test_wrappers_match_training_forward(k):
    model = _random_model()
    cache = ProposalCacheExportModel(model).eval()
    step = ProposalStepExportModel(model).eval()

    spatial, scalar = _board_inputs(seed=3)
    m = 11
    moves = _random_moves(m, seed=40)
    ev, obs_planes, obs_scalars, mask = _evidence_inputs(moves, k, seed=7)
    ref = _reference(model, spatial, scalar, moves, ev if k else None)

    letters, blanks, squares, tile_mask, scalars = moves
    with torch.no_grad():
        board, g, move_enc, c_wld, c_sd, c_planes = cache(
            torch.from_numpy(spatial),
            torch.from_numpy(scalar),
            torch.from_numpy(letters),
            torch.from_numpy(blanks),
            torch.from_numpy(squares),
            torch.from_numpy(tile_mask),
            torch.from_numpy(scalars),
        )
        if k == 0:  # empty evidence: the cache's plain heads ARE the forward
            _assert_close({"wld": c_wld, "score_diff": c_sd, "planes": c_planes}, ref)
            return
        ev_move_enc = torch.cat([move_enc[:k], move_enc.new_zeros(MAX_E - k, CHANNELS)], 0)[None]
        s_wld, s_sd, s_planes, s_gain = step(
            board,
            g,
            move_enc,
            ev_move_enc,
            torch.from_numpy(obs_planes),
            torch.from_numpy(obs_scalars),
            torch.from_numpy(mask),
        )
    _assert_close({"wld": s_wld, "score_diff": s_sd, "planes": s_planes, "gain": s_gain}, ref)


def _assert_close(got: dict, ref: dict):
    for name, value in got.items():
        np.testing.assert_allclose(
            value.numpy(), ref[name].numpy(), atol=1e-5, rtol=1e-4, err_msg=name
        )


def _export_pair(tmp_path, model):
    xid = proposal_export_id(model)
    cache_path = tmp_path / "cache.onnx"
    step_path = tmp_path / "step.onnx"
    export_proposal_cache(
        model,
        cache_path,
        SPATIAL_PLANES,
        SCALAR_SIZE,
        opp_leave_input=False,
        move_encoding_version=1,
        proposal_export_id=xid,
    )
    export_proposal_step(
        model,
        step_path,
        opp_leave_input=False,
        move_encoding_version=1,
        proposal_export_id=xid,
        max_evidence=MAX_E,
    )
    return cache_path, step_path


@pytest.mark.parametrize("k", [0, 4, MAX_E])
def test_onnx_runtime_matches_torch_at_other_ms(tmp_path, k):
    ort = pytest.importorskip("onnxruntime")
    model = _random_model()
    cache_path, step_path = _export_pair(tmp_path, model)  # both traced at M=5
    cache_sess = ort.InferenceSession(str(cache_path), providers=["CPUExecutionProvider"])
    step_sess = ort.InferenceSession(str(step_path), providers=["CPUExecutionProvider"])

    spatial, scalar = _board_inputs(seed=3)
    for m in (1, 23):  # neither is the traced M
        moves = _random_moves(m, seed=40 + m)
        ev, obs_planes, obs_scalars, mask = _evidence_inputs(moves, min(k, m), seed=7 + m)
        ref = _reference(model, spatial, scalar, moves, ev if k else None)
        letters, blanks, squares, tile_mask, scalars = moves

        cache_out = cache_sess.run(
            list(CACHE_OUTPUT_NAMES),
            {
                "input_spatial": spatial,
                "input_scalar": scalar,
                "move_letters": letters,
                "move_blanks": blanks,
                "move_squares": squares,
                "move_tile_mask": tile_mask,
                "move_scalars": scalars,
            },
        )
        cache = dict(zip(CACHE_OUTPUT_NAMES, cache_out, strict=True))
        if k == 0:
            for name in ("wld", "score_diff", "planes"):
                np.testing.assert_allclose(cache[name], ref[name].numpy(), atol=1e-5, rtol=1e-4)
            continue

        kk = min(k, m)
        move_enc = cache["move_enc"]
        ev_move_enc = np.zeros((1, MAX_E, CHANNELS), np.float32)
        ev_move_enc[0, :kk] = move_enc[:kk]
        step_out = step_sess.run(
            list(STEP_OUTPUT_NAMES),
            {
                "board": cache["board"],
                "g": cache["g"],
                "move_enc": move_enc,
                "ev_move_enc": ev_move_enc,
                "ev_obs_planes": obs_planes,
                "ev_obs_scalars": obs_scalars,
                "ev_mask": mask,
            },
        )
        step = dict(zip(STEP_OUTPUT_NAMES, step_out, strict=True))
        for name in STEP_OUTPUT_NAMES:
            assert step[name].shape[0] == m, name
            np.testing.assert_allclose(
                step[name], ref[name].numpy(), atol=1e-5, rtol=1e-4, err_msg=name
            )


def test_exported_file_contract(tmp_path):
    model = _random_model()
    cache_path, step_path = _export_pair(tmp_path, model)
    cache = onnx.load(str(cache_path))
    step = onnx.load(str(step_path))

    cmeta = {e.key: e.value for e in cache.metadata_props}
    smeta = {e.key: e.value for e in step.metadata_props}
    assert cmeta["graph"] == GRAPH_CACHE
    assert smeta["graph"] == GRAPH_STEP
    assert cmeta["move_encoding_version"] == "1"
    assert cmeta["opp_leave_input"] == "false"
    assert "model-architecture-signature" in cmeta
    # The shared fingerprint ties a cache graph to its step graph.
    assert cmeta["proposal_export_id"] == smeta["proposal_export_id"]

    assert [o.name for o in cache.graph.output] == list(CACHE_OUTPUT_NAMES)
    assert [o.name for o in step.graph.output] == list(STEP_OUTPUT_NAMES)
    assert {i.name for i in cache.graph.input} == set(CACHE_INPUT_NAMES)
    assert {i.name for i in step.graph.input} == set(STEP_INPUT_NAMES)

    cin = {i.name: i for i in cache.graph.input}
    assert cin["move_letters"].type.tensor_type.elem_type == onnx.TensorProto.INT32
    assert cin["move_blanks"].type.tensor_type.elem_type == onnx.TensorProto.UINT8
    assert cin["input_spatial"].type.tensor_type.shape.dim[0].dim_value == 1
    for name in ("move_letters", "move_scalars"):
        assert cin[name].type.tensor_type.shape.dim[0].dim_param == "moves", name
    # The cache emits the handoff tensors as static (1, ...) outputs.
    cout = {o.name: o for o in cache.graph.output}
    assert cout["board"].type.tensor_type.shape.dim[0].dim_value == 1
    assert cout["move_enc"].type.tensor_type.shape.dim[0].dim_param == "moves"

    sin = {i.name: i for i in step.graph.input}
    assert sin["move_enc"].type.tensor_type.shape.dim[0].dim_param == "moves"
    assert sin["ev_mask"].type.tensor_type.elem_type == onnx.TensorProto.UINT8
    assert sin["ev_obs_planes"].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    # The evidence inputs are fixed-width leading-1 batches (E folded into the
    # row, so M stays the only dynamic axis).
    for name in ("ev_move_enc", "ev_obs_planes", "ev_obs_scalars", "ev_mask"):
        shape = sin[name].type.tensor_type.shape.dim
        assert shape[0].dim_value == 1, name
        assert shape[1].dim_value == MAX_E, name

    # Refit discipline: split heads and hand-rolled attentions as plain named
    # initializers, no Identity aliasing left.
    for graph in (cache, step):
        init_names = {i.name for i in graph.graph.initializer}
        for stem in ("head_attended", "head_g", "plane_attended", "plane_g", "q_proj", "k_proj"):
            assert any(stem in n for n in init_names), stem
        aliased = [
            n for n in graph.graph.node if n.op_type == "Identity" and n.input[0] in init_names
        ]
        assert not aliased
    step_inits = {i.name for i in step.graph.initializer}
    for stem in ("sa_q", "sa_k", "sa_v", "pb_attended", "pb_g"):  # the fusion self-attn + gain
        assert any(stem in n for n in step_inits), stem
