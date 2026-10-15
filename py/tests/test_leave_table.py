"""The leave table through the FFI (scribblez.leave_inference.leave_table):
row counts, counts that sum to each length, the KLV identity, and values that
match the engine's leave values."""

from pathlib import Path

import numpy as np
import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)
if not LEAVES.exists():
    pytest.skip("HastyBot leave values not installed", allow_module_level=True)

from scribblez.ffi import LeaveTableHandle  # noqa: E402
from scribblez.leave_inference.leave_table import MAX_LEAVE_LENGTH, LeaveTable  # noqa: E402

# From docs/plans/deep_bot.md; they depend only on the tile distribution.
ROW_COUNTS = {1: 27, 2: 373, 3: 3509, 4: 25254, 5: 148150, 6: 737311}


@pytest.fixture(scope="module")
def table() -> LeaveTable:
    return LeaveTable()


def test_row_counts(table):
    assert MAX_LEAVE_LENGTH == 6
    assert {k: len(r.value) for k, r in table.rows.items()} == ROW_COUNTS


def test_rows_are_distinct_leaves_of_their_length(table):
    for k, r in table.rows.items():
        assert r.counts.shape == (ROW_COUNTS[k], 27)
        np.testing.assert_array_equal(r.counts.sum(axis=1), k)
        assert len(np.unique(r.counts, axis=0)) == ROW_COUNTS[k]


def test_values_are_the_leave_values(table):
    # The best single tile is the blank, a known positive leave; a lone Q is
    # a known negative one.
    singles = table.rows[1]
    by_tile = {int(np.argmax(c)): v for c, v in zip(singles.counts, singles.value, strict=True)}
    assert by_tile[26] > 15 > 0 > by_tile[16]


def test_klv_identity_is_recorded(table):
    assert len(table.klv_hash) == 16
    assert int(table.klv_hash, 16) >= 0


def test_handle_rejects_lengths_outside_the_table():
    handle = LeaveTableHandle()
    with pytest.raises(ValueError):
        handle.rows(0)
    with pytest.raises(ValueError):
        handle.rows(7)
    handle.close()
