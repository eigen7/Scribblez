"""Tests for the position-evaluation analysis encoder's arm contract: a
position is encoded under the arm the caller names (a served model's own, not
the process-wide session's), and a width the arm does not encode is refused
rather than filled.

The dashboard serves models of every arm from one session, so encoding under
the session's arm and slicing was how a face-up-leaves model came to read
contingent scalars in its opponent-leave block."""

import numpy as np
import pytest
from scribblez.ffi import (
    BOARD_CELLS,
    InputArm,
    analyze_position_eval_gcg,
    analyze_position_eval_gcg_leave,
    session_input_arm,
)
from scribblez.position_eval import analysis

_GCG = analysis.DEFAULT_DATASET / "pos-8.gcg"


def _text() -> str:
    """The position's GCG, or a skip when the dataset or the lexicon (which the
    session needs) is unavailable."""
    if not _GCG.exists():
        pytest.skip("position-evaluation dataset unavailable")
    try:
        session_input_arm()
    except OSError:
        pytest.skip("lexicon unavailable")
    return _GCG.read_text()


def _arm(contingent: bool, opp_leave: bool) -> InputArm:
    """An arm's widths, derived from the session's by the registry's block sizes
    (the opp-leave block is 27 scalars, the contingent tail 3 planes + 56)."""
    base = session_input_arm()
    assert base.contingent_features and not base.opp_leave_input
    return InputArm(
        contingent,
        opp_leave,
        base.spatial_planes - (0 if contingent else 3),
        base.scalar_size - (0 if contingent else 56) + (27 if opp_leave else 0),
    )


def test_the_session_arm_round_trips():
    text = _text()
    arm = session_input_arm()
    row = analyze_position_eval_gcg(text, arm)
    assert row.shape == (arm.input_floats,)
    spatial, scalar = arm.split(row)
    assert spatial.shape == (arm.spatial_planes, 15, 15)
    assert scalar.shape == (arm.scalar_size,)


def test_another_arm_is_encoded_as_itself_not_sliced():
    """A face-up-leaves (opp-leave, non-contingent) row is not a prefix of the
    session's full row: its tail is the opponent-leave block -- all zeros here,
    the opponent having bingoed -- where a prefix slice would carry contingent
    scalars. The shared prefix (board planes, rack, pool, ...) does agree."""
    text = _text()
    full = analyze_position_eval_gcg(text, session_input_arm())
    face_up = analyze_position_eval_gcg(text, _arm(contingent=False, opp_leave=True))
    planes = _arm(False, True).spatial_planes
    np.testing.assert_array_equal(face_up[: planes * BOARD_CELLS], full[: planes * BOARD_CELLS])
    scalar_full = session_input_arm().split(full)[1]
    scalar_face_up = _arm(False, True).split(face_up)[1]
    np.testing.assert_array_equal(scalar_face_up[:-27], scalar_full[: len(scalar_face_up) - 27])
    assert not scalar_face_up[-27:].any()  # the opponent kept nothing
    assert scalar_full[len(scalar_face_up) - 27 : len(scalar_face_up)].any()  # what a slice fed


def test_a_width_the_arm_does_not_encode_is_refused():
    text = _text()
    arm = session_input_arm()
    stale = InputArm(arm.contingent_features, arm.opp_leave_input, arm.spatial_planes, 1)
    with pytest.raises(ValueError, match="the arm encodes"):
        analyze_position_eval_gcg(text, stale)
    with pytest.raises(ValueError, match="the arm encodes"):
        analyze_position_eval_gcg_leave(text, "CEMR", stale)


def test_a_bad_position_is_an_os_error_not_a_width_error():
    _text()
    with pytest.raises(OSError):
        analyze_position_eval_gcg("#character-encoding UTF-8\n", session_input_arm())
