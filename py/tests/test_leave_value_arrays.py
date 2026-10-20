"""The engine's LeaveValueArrays through the FFI (scribblez.ffi): row counts,
per-slot codes that decode to the distinct leaves of each length, and the
leave values' identity."""

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


def test_codes_decode_to_distinct_leaves_of_their_length(arrays):
    copies = arrays.copies_per_tile
    assert arrays.num_codes == 27 * copies
    for k, count in ROW_COUNTS.items():
        codes, values = arrays.of_length(k)
        assert codes.shape == (k, count) and values.shape == (count,)
        assert (codes < arrays.num_codes).all()
        assert (np.diff(codes.astype(int), axis=0) >= 0).all()
        letters, copy = codes // copies, codes % copies
        # A slot's copy number counts the earlier slots with its letter.
        earlier = (letters[None, :, :] == letters[:, None, :]) & (
            np.arange(k)[None, :, None] < np.arange(k)[:, None, None]
        )
        np.testing.assert_array_equal(copy, earlier.sum(axis=1))
        # Distinct rows: one byte per slot packs a row into an int64 key, and
        # sorting 1-D keys is far cheaper than np.unique over rows.
        assert arrays.num_codes <= 256
        keys = (codes.astype(np.int64) << (8 * np.arange(k, dtype=np.int64))[:, None]).sum(axis=0)
        assert (np.diff(np.sort(keys)) != 0).all()


def test_known_values(arrays):
    # A lone blank is the best single tile; a lone Q is among the worst.
    codes, values = arrays.of_length(1)
    by_letter = dict(
        zip((codes[0] // arrays.copies_per_tile).tolist(), values.tolist(), strict=True)
    )
    assert by_letter[26] == max(by_letter.values()) > 15 > 0 > by_letter[16]


def test_identity_is_the_leave_values_file(arrays):
    assert len(arrays.content_hash) == 16
    int(arrays.content_hash, 16)


def test_rejects_lengths_outside_the_arrays(arrays):
    for k in (0, 7):
        with pytest.raises(ValueError):
            arrays.of_length(k)
