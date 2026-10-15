"""The engine's LeaveValueArrays through the FFI (scribblez.ffi): row counts,
leaves of each length sorted by value, and the leave values' identity."""

from pathlib import Path

import numpy as np
import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)
if not LEAVES.exists():
    pytest.skip("HastyBot leave values not installed", allow_module_level=True)

from scribblez.ffi import LeaveValueArrays  # noqa: E402

# From docs/plans/deep_bot.md; they depend only on the tile distribution.
ROW_COUNTS = {1: 27, 2: 373, 3: 3509, 4: 25254, 5: 148150, 6: 737311}


@pytest.fixture(scope="module")
def arrays() -> LeaveValueArrays:
    return LeaveValueArrays()


def test_rows_are_distinct_leaves_sorted_by_value(arrays):
    for k, count in ROW_COUNTS.items():
        tiles, values = arrays.of_length(k)
        assert tiles.shape == (count, 6) and values.shape == (count,)
        real = tiles[:, :k]
        assert (real < 27).all() and (tiles[:, k:] == LeaveValueArrays.NO_TILE).all()
        assert (np.diff(real.astype(int), axis=1) >= 0).all()
        assert len(np.unique(real, axis=0)) == count
        assert (np.diff(values) >= 0).all()


def test_known_values(arrays):
    # A lone blank is the best single tile; a lone Q is among the worst.
    tiles, values = arrays.of_length(1)
    by_tile = dict(zip(tiles[:, 0].tolist(), values.tolist(), strict=True))
    assert by_tile[26] == max(by_tile.values()) > 15 > 0 > by_tile[16]


def test_identity_is_the_leave_values_file(arrays):
    assert len(arrays.content_hash) == 16
    int(arrays.content_hash, 16)


def test_rejects_lengths_outside_the_arrays(arrays):
    for k in (0, 7):
        with pytest.raises(ValueError):
            arrays.of_length(k)
