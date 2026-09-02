"""Golden pins for the FFI-served format layout (scribblez.ffi.format_layout /
struct_dtype). The served offsets and sizes are the C++ compiler's own, so
these tests exist to catch transport and dtype-construction bugs -- a handful
of independently-known values, not a re-mirror of the structs."""

from pathlib import Path

import numpy as np
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

    record = struct_dtype("SobsRecord")
    assert record.fields["move"][0] == move  # nested reference resolves
    # The sim observation is a dense footprint-class histogram (.sobs v5).
    assert record.fields["obs"][0].fields["opp_next_count"][0].shape == (2927,)

    assert struct_dtype("MsetFileHeader").fields["model_hash"][0] == np.dtype("S64")
    assert struct_dtype("SlogTurnBlob").itemsize == 24


def test_magics_spell_their_extensions():
    from scribblez.move_set_eval.targets import MSET_MAGIC
    from scribblez.sim_evidence.slog_meta import SLOG_MAGIC
    from scribblez.sim_evidence.sobs import SOBS_MAGIC

    assert MSET_MAGIC == int.from_bytes(b"MSET", "little")
    assert SOBS_MAGIC == int.from_bytes(b"SOBS", "little")
    assert SLOG_MAGIC == int.from_bytes(b"SLOG", "little")


def test_glyph_char_follows_the_engine_code_table():
    """glyph_char's code table is a deliberate replica of glyph.h's (the
    encoding is effectively frozen); this pin and its C++ twin
    (Glyph.CodeTablePinnedForCrossLanguageReaders) keep the two in lockstep."""
    from scribblez.sim_evidence.sobs import glyph_char

    assert glyph_char(0) == "?"  # empty
    assert glyph_char(1) == "A" and glyph_char(26) == "Z"
    # Designated blanks start right after the letters (glyph.h: 27..52) and
    # render lowercased; the unassigned blank (53) has no letter to show.
    assert glyph_char(27) == "a" and glyph_char(52) == "z"
    assert glyph_char(53) == "?"


def test_placement_head_names_are_the_target_heads():
    from scribblez.move_set_eval.targets import PLANE_NAMES
    from scribblez.position_eval.model import PLACEMENT_HEAD_NAMES

    golden = (
        "opp_next_placement",
        "self_next_placement",
        "opp_win_placement",
        "self_win_placement",
    )
    assert PLACEMENT_HEAD_NAMES == golden
    assert PLANE_NAMES == golden
