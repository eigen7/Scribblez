"""Golden pins for the FFI-served format layout (scribblez.ffi.format_layout /
struct_dtype). The served offsets and sizes are the C++ compiler's own, so
these tests exist to catch transport and dtype-construction bugs -- a handful
of independently-known values, not a re-mirror of the structs."""

from pathlib import Path

import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)


def test_struct_dtypes_pin_known_layouts():
    from scribblez.ffi import struct_dtype

    move = struct_dtype("Move")
    assert move.itemsize == 16
    assert move.fields["glyphs"][0].shape == (7,)
    assert move.fields["square_mask"][1] == 12  # after the zeroed padding byte

    turn = struct_dtype("SlogTurnBlob")
    assert turn.itemsize == 24
    assert turn.fields["move"][0] == move  # nested reference resolves


def test_magics_spell_their_extensions():
    from scribblez.sim_evidence.slog_meta import SLOG_MAGIC

    assert SLOG_MAGIC == int.from_bytes(b"SLOG", "little")


def test_placement_head_names_are_pinned():
    from scribblez.position_eval.model import PLACEMENT_HEAD_NAMES

    golden = (
        "opp_next_placement",
        "self_next_placement",
        "opp_win_placement",
        "self_win_placement",
    )
    assert PLACEMENT_HEAD_NAMES == golden
