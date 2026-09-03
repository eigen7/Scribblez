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
      cache's own move encodings by scored index (scattered, not the first k, so
      a move_enc/ev_move_enc routing swap can't alias away), and the reference's
      evidence moves are those same scored candidates.
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
from scribblez.move_set_eval.model import MoveSetEvalModel, footprint_slot_planes
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


def _evidence_indices(m: int, kind: str, seed: int) -> list[int]:
    """Scored-candidate indices for an evidence set. Deliberately scattered
    (not the first k) and, for the partial case, carrying a duplicate: an
    evidence candidate is one of the scored candidates, so the runtime GATHERS
    its move encoding from the cache's per-candidate output by this index --
    scattered/duplicate indices make ev_move_enc differ from any contiguous
    move_enc slice, so a move_enc/ev_move_enc routing swap shows up as a
    mismatch rather than aliasing away."""
    if kind == "empty":
        return []
    rng = np.random.default_rng(seed)
    k = min(3 if kind == "partial" else MAX_E, m)
    idx = rng.permutation(m)[:k].tolist()
    if kind == "partial" and k >= 2:
        idx[1] = idx[0]  # the same simmed candidate appearing twice
    return idx


def _evidence_inputs(moves, indices: list[int], seed: int):
    """A width-MAX_E evidence set describing the candidate moves at `indices`,
    with random observation planes/scalars. Returns (EvidenceInputs,
    ev_obs_planes, ev_obs_scalars, ev_mask_uint8) -- the same observation arrays
    the step graph is fed, so the reference and the graph see identical
    evidence. The evidence move data is the scored candidate's own (moves[idx]),
    so the reference's re-encode equals the runtime's gather of move_enc[idx]."""
    letters, blanks, squares, tile_mask, scalars = moves
    k = len(indices)
    el = np.zeros((MAX_E, MAX_PLACED), np.int64)
    eb = np.zeros((MAX_E, MAX_PLACED), np.int64)
    es = np.zeros((MAX_E, MAX_PLACED), np.int64)
    et = np.zeros((MAX_E, MAX_PLACED), np.float32)
    esc = np.zeros((MAX_E, NUM_SCALARS), np.float32)
    for j, idx in enumerate(indices):
        el[j], eb[j], es[j], et[j], esc[j] = (
            letters[idx],
            blanks[idx],
            squares[idx],
            tile_mask[idx],
            scalars[idx],
        )

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


def _gather_ev_move_enc(move_enc: np.ndarray, indices: list[int]) -> np.ndarray:
    """(1, MAX_E, C) evidence move encodings gathered from the cache's per-
    candidate move_enc (num_scored, C) by scored index, zero-padded past k."""
    out = np.zeros((1, MAX_E, move_enc.shape[-1]), np.float32)
    for j, idx in enumerate(indices):
        out[0, j] = move_enc[idx]
    return out


def _reference(model, spatial, scalar, moves, ev):
    """MoveSetEvalModel.forward over one position's candidate set, optionally
    evidence-conditioned."""
    letters, blanks, squares, tile_mask, scalars = moves
    m = len(scalars)
    with torch.no_grad():
        out = dict(
            model(
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
        )
    # The proposal graphs serve the footprint heads' slot-channel probabilities
    # (the evidence-plane layout), so the reference decodes its footprint
    # logits the same way.
    out["planes"] = footprint_slot_planes(out["planes"]).flatten(2)
    return out


@pytest.mark.parametrize("kind", ["empty", "partial", "full"])
def test_wrappers_match_training_forward(kind):
    model = _random_model()
    cache = ProposalCacheExportModel(model).eval()
    step = ProposalStepExportModel(model).eval()

    spatial, scalar = _board_inputs(seed=3)
    m = 11
    moves = _random_moves(m, seed=40)
    indices = _evidence_indices(m, kind, seed=5)
    ev, obs_planes, obs_scalars, mask = _evidence_inputs(moves, indices, seed=7)
    # An empty set still runs the step wrapper (its all-empty-mask gate must
    # reproduce the plain forward -- the deployment loop's first iteration).
    ref = _reference(model, spatial, scalar, moves, ev)
    plain_ref = _reference(model, spatial, scalar, moves, None)

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
        ev_move_enc = _gather_ev_move_enc(move_enc.numpy(), indices)
        s_wld, s_sd, s_gain = step(
            board,
            g,
            move_enc,
            torch.from_numpy(ev_move_enc),
            torch.from_numpy(obs_planes),
            torch.from_numpy(obs_scalars),
            torch.from_numpy(mask),
        )
    # The cache graph's heads are the plain forward's, its planes included (the
    # evidence-free planes every evidence token's predicted half is gathered
    # from); the step graph's are the conditioned forward's, planes-free.
    _assert_close({"wld": c_wld, "score_diff": c_sd, "planes": c_planes}, plain_ref)
    outputs = {"wld": s_wld, "score_diff": s_sd, "gain": s_gain}
    _assert_close(outputs, ref)
    if kind == "empty":  # and the empty step equals the cache's plain heads, finite
        _assert_close({"wld": s_wld, "score_diff": s_sd}, {"wld": c_wld, "score_diff": c_sd})
        assert all(torch.isfinite(v).all() for v in outputs.values())


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


@pytest.mark.parametrize("kind", ["empty", "partial", "full"])
def test_onnx_runtime_matches_torch_at_other_ms(tmp_path, kind):
    ort = pytest.importorskip("onnxruntime")
    model = _random_model()
    cache_path, step_path = _export_pair(tmp_path, model)  # both traced at M=5
    cache_sess = ort.InferenceSession(str(cache_path), providers=["CPUExecutionProvider"])
    step_sess = ort.InferenceSession(str(step_path), providers=["CPUExecutionProvider"])

    spatial, scalar = _board_inputs(seed=3)
    for m in (1, 23):  # neither is the traced M
        moves = _random_moves(m, seed=40 + m)
        indices = _evidence_indices(m, kind, seed=5 + m)
        ev, obs_planes, obs_scalars, mask = _evidence_inputs(moves, indices, seed=7 + m)
        ref = _reference(model, spatial, scalar, moves, ev)
        plain_ref = _reference(model, spatial, scalar, moves, None)
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
        move_enc = cache["move_enc"]
        for name in ("wld", "score_diff", "planes"):
            assert cache[name].shape[0] == m, name
            np.testing.assert_allclose(
                cache[name], plain_ref[name].numpy(), atol=1e-5, rtol=1e-4, err_msg=name
            )
        # Always run the step graph -- including the empty-mask case, which must
        # reproduce the plain forward through the graph itself, not a shortcut.
        step_out = step_sess.run(
            list(STEP_OUTPUT_NAMES),
            {
                "board": cache["board"],
                "g": cache["g"],
                "move_enc": move_enc,
                "ev_move_enc": _gather_ev_move_enc(move_enc, indices),
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
    # Both graphs of one exported pair share the fingerprint (its discriminating
    # power -- rejecting a mismatched pair -- is tested separately below).
    assert cmeta["proposal_export_id"] == smeta["proposal_export_id"]

    assert [o.name for o in cache.graph.output] == list(CACHE_OUTPUT_NAMES)
    assert [o.name for o in step.graph.output] == list(STEP_OUTPUT_NAMES)
    assert {i.name for i in cache.graph.input} == set(CACHE_INPUT_NAMES)
    assert {i.name for i in step.graph.input} == set(STEP_INPUT_NAMES)

    # The move inputs' dtypes match move_set_encoder.h so the engine feeds them
    # zero-copy, and every move input (plus the M-riding outputs) rides "moves".
    move_dtypes = {
        "move_letters": onnx.TensorProto.INT32,
        "move_blanks": onnx.TensorProto.UINT8,
        "move_squares": onnx.TensorProto.INT32,
        "move_tile_mask": onnx.TensorProto.UINT8,
        "move_scalars": onnx.TensorProto.FLOAT,
    }
    cin = {i.name: i for i in cache.graph.input}
    for name, dtype in move_dtypes.items():
        assert cin[name].type.tensor_type.elem_type == dtype, name
        assert cin[name].type.tensor_type.shape.dim[0].dim_param == "moves", name
    assert cin["input_spatial"].type.tensor_type.shape.dim[0].dim_value == 1
    # The cache emits the handoff tensors as static (1, ...) outputs; wld /
    # score_diff / planes / move_enc ride "moves".
    cout = {o.name: o for o in cache.graph.output}
    assert cout["board"].type.tensor_type.shape.dim[0].dim_value == 1
    assert cout["g"].type.tensor_type.shape.dim[0].dim_value == 1
    for name in ("move_enc", "wld", "score_diff", "planes"):
        assert cout[name].type.tensor_type.shape.dim[0].dim_param == "moves", name

    sin = {i.name: i for i in step.graph.input}
    assert sin["move_enc"].type.tensor_type.shape.dim[0].dim_param == "moves"
    assert sin["ev_mask"].type.tensor_type.elem_type == onnx.TensorProto.UINT8
    assert sin["ev_obs_planes"].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    sout = {o.name: o for o in step.graph.output}
    for name in STEP_OUTPUT_NAMES:
        assert sout[name].type.tensor_type.shape.dim[0].dim_param == "moves", name
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
        for stem in ("head_attended", "head_g", "q_proj", "k_proj"):
            assert any(stem in n for n in init_names), stem
        aliased = [
            n for n in graph.graph.node if n.op_type == "Identity" and n.input[0] in init_names
        ]
        assert not aliased
    cache_inits = {i.name for i in cache.graph.initializer}
    step_inits = {i.name for i in step.graph.initializer}
    for stem in ("plane_attended", "plane_g"):  # the plane head: cache graph only
        assert any(stem in n for n in cache_inits), stem
        assert not any(stem in n for n in step_inits), stem
    for stem in ("sa_q", "sa_k", "sa_v", "pb_attended", "pb_g"):  # the fusion self-attn + gain
        assert any(stem in n for n in step_inits), stem


def test_proposal_export_id_discriminates_models():
    """The fingerprint tying a cache/step pair is deterministic for one model
    and differs for another -- both a different checkpoint of the same
    architecture (weight-sensitive) and a different architecture -- so a loader
    can actually reject a mismatched pair, not just pass a tautology."""
    m0 = _random_model(seed=0)
    assert proposal_export_id(m0) == proposal_export_id(_random_model(seed=0))  # deterministic
    assert proposal_export_id(m0) != proposal_export_id(_random_model(seed=1))  # weight-sensitive
    other_arch = MoveSetEvalModel(
        spatial_planes=SPATIAL_PLANES,
        scalar_size=SCALAR_SIZE,
        trunk_channels=CHANNELS,
        num_blocks=4,  # a different architecture
        num_heads=2,
    )
    assert proposal_export_id(m0) != proposal_export_id(other_arch)
