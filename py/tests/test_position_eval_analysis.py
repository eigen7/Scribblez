"""Tests for the position-evaluation analysis encoder's arm contract: a
position is encoded under the arm the caller names (a served model's own, not
the process-wide session's), and a width the arm does not encode is refused
rather than filled.

The dashboard serves models of every arm from one session, so encoding under
the session's arm and slicing is how a model of one arm comes to read another
arm's scalars."""

from pathlib import Path

import numpy as np
import pytest
from scribblez.ffi import (
    BOARD_CELLS,
    InputArm,
    analyze_position_eval_gcg,
    analyze_position_eval_gcg_leaves,
    collapse_position_eval_placement,
    format_layout,
    legal_position_eval_placement,
    position_eval_board_json,
    session_input_arm,
)
from scribblez.position_eval import analysis

# A frozen .gcg fixture. Live position sets under positions/ are regenerated
# and renamed; a test must never read one.
_GCG = Path(__file__).resolve().parents[2] / "engine" / "tests" / "data" / "masked-racks.gcg"


def _text() -> str:
    """The position's GCG, or a skip when the lexicon (which the session needs)
    is unavailable."""
    try:
        session_input_arm()
    except OSError:
        pytest.skip("lexicon unavailable")
    return _GCG.read_text()


def _arm(opp_leave: bool) -> InputArm:
    """An arm's widths, derived from the session's by the registry's block size
    (the opp-leave block is 27 scalars)."""
    base = session_input_arm()
    assert not base.opp_leave_input
    return InputArm(
        opp_leave,
        base.spatial_planes,
        base.scalar_size + (27 if opp_leave else 0),
    )


def test_the_session_arm_round_trips():
    text = _text()
    arm = session_input_arm()
    row = analyze_position_eval_gcg(text, arm)
    assert row.shape == (arm.input_floats,)
    spatial, scalar = arm.split(row)
    assert spatial.shape == (arm.spatial_planes, 15, 15)
    assert scalar.shape == (arm.scalar_size,)


def test_another_arm_is_encoded_as_itself():
    """A face-up-leaves row is encoded under the arm the caller named, not the
    session's: it is 27 scalars longer than the session's row, agrees with it
    on every shared block, and carries the opponent-leave block as its tail --
    all zeros here, the opponent having bingoed."""
    text = _text()
    arm = _arm(opp_leave=True)
    base_arm = session_input_arm()
    base = analyze_position_eval_gcg(text, base_arm)
    face_up = analyze_position_eval_gcg(text, arm)
    assert face_up.shape == (base.size + 27,)
    planes = arm.spatial_planes
    np.testing.assert_array_equal(face_up[: planes * BOARD_CELLS], base[: planes * BOARD_CELLS])
    scalar_base = base_arm.split(base)[1]
    scalar_face_up = arm.split(face_up)[1]
    np.testing.assert_array_equal(scalar_face_up[:-27], scalar_base)
    assert not scalar_face_up[-27:].any()  # the opponent kept nothing


def test_a_width_the_arm_does_not_encode_is_refused():
    text = _text()
    arm = session_input_arm()
    stale = InputArm(arm.opp_leave_input, arm.spatial_planes, 1)
    with pytest.raises(ValueError, match="the arm encodes"):
        analyze_position_eval_gcg(text, stale)
    with pytest.raises(ValueError, match="the arm encodes"):
        analyze_position_eval_gcg_leaves(text, "CEMR", None, stale)


def test_a_bad_position_is_an_os_error_not_a_width_error():
    _text()
    with pytest.raises(OSError):
        analyze_position_eval_gcg("#character-encoding UTF-8\n", session_input_arm())


def _placement_shape() -> tuple[int, int]:
    consts = format_layout()["constants"]
    return len(consts["placement_head_names"]), consts["footprint"]["num_classes"]


def test_collapse_placement_produces_board_marginals():
    """The engine collapse turns the four heads' raw footprint logits into
    board-frame per-cell occupancy marginals: a (heads, 15, 15) stack of
    probabilities in [0, 1], nonzero because legal footprints exist here."""
    text = _text()
    heads, classes = _placement_shape()
    raw = np.random.default_rng(0).standard_normal((heads, classes)).astype(np.float32)
    planes = collapse_position_eval_placement(text, raw)
    assert planes.shape == (heads, 15, 15)
    assert np.isfinite(planes).all()
    assert (planes >= 0.0).all()
    assert (planes <= 1.0 + 1e-5).all()  # a per-cell marginal, not a summed logit
    assert planes.sum() > 0.0  # some footprint is legal on this board


def test_collapse_placement_bad_gcg_is_os_error():
    _text()
    heads, classes = _placement_shape()
    raw = np.zeros((heads, classes), dtype=np.float32)
    with pytest.raises(OSError):
        collapse_position_eval_placement("#character-encoding UTF-8\n", raw)


def test_legal_placement_covers_all_probability_mass():
    """legal_position_eval_placement's plane is model-independent (unlike
    collapse_position_eval_placement, it takes no `raw`), but must still be a
    superset of wherever a masked-softmax collapse can ever land mass: a
    masked softmax puts zero probability on an illegal class, so a cell with
    nonzero collapsed probability under ANY logits must be legal. An occupied
    board square is never legal (no footprint covers a square with a tile on
    it already)."""
    text = _text()
    heads, classes = _placement_shape()
    legal = legal_position_eval_placement(text)
    assert legal.shape == (heads, 15, 15)
    assert legal.dtype == bool

    raw = np.random.default_rng(0).standard_normal((heads, classes)).astype(np.float32)
    planes = collapse_position_eval_placement(text, raw)
    assert np.all(legal[planes > 1e-6])

    board = position_eval_board_json(text)["board"]
    occupied = [(r, c) for r in range(15) for c in range(15) if board[r][c] is not None]
    assert occupied  # the fixture has moves recorded
    for r, c in occupied:
        assert not legal[:, r, c].any(), (r, c)


def test_legal_placement_bad_gcg_is_os_error():
    _text()
    with pytest.raises(OSError):
        legal_position_eval_placement("#character-encoding UTF-8\n")


def test_ground_truth_is_per_condition():
    d = analysis.DEFAULT_DATASET
    assert analysis.ground_truth_path(d, True) == d / "monte-carlo-sim-results.face-up-leaves.json"
    assert analysis.ground_truth_path(d, False) == d / "monte-carlo-sim-results.hidden-leaves.json"


def test_alternate_leaves_reproduce_the_recorded_encoding():
    """Re-submitting the recorded leaves as alternates encodes the same row; an
    alternate opponent leave must match the recorded one's size (empty here:
    the opponent bingoed), and the POV leave may not spend tiles twice."""
    text = _text()
    arm = _arm(opp_leave=True)
    bundle = position_eval_board_json(text)
    leave = "".join(t["letter"] for t in bundle["rack"])  # a blank renders as "?"
    assert bundle["opp_leave"] == ""
    recorded = analyze_position_eval_gcg(text, arm)
    np.testing.assert_array_equal(analyze_position_eval_gcg_leaves(text, leave, "", arm), recorded)
    np.testing.assert_array_equal(
        analyze_position_eval_gcg_leaves(text, leave, None, arm), recorded
    )
    with pytest.raises(ValueError, match="opponent leave must have 0 tile"):
        analyze_position_eval_gcg_leaves(text, leave, "A", arm)
    with pytest.raises(ValueError, match="POV leave must have"):
        analyze_position_eval_gcg_leaves(text, leave + "A", None, arm)
