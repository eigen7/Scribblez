"""The .ltab reader (scribblez.leave_inference.leave_table) over a table
leave_table_tool writes: row counts, mask/count consistency, the header's
identity fields, and refusal of another format version."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

# This checkout's own binaries, so a worktree's tests exercise the code built
# beside them.
_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
TOOL = _ENGINE_DIR / "leave_table_tool"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)

from scribblez.leave_inference.leave_table import (  # noqa: E402
    HEADER,
    LTAB_FORMAT_VERSION,
    LTAB_MAGIC,
    MAX_LEAVE_LENGTH,
    read_leave_table,
)

# From docs/plans/deep_bot.md; they depend only on the tile distribution.
ROW_COUNTS = {1: 27, 2: 373, 3: 3509, 4: 25254, 5: 148150, 6: 737311}


@pytest.fixture(scope="module")
def ltab_path(tmp_path_factory) -> Path:
    if not TOOL.exists():
        pytest.skip("leave_table_tool not built")
    if not LEAVES.exists():
        pytest.skip("HastyBot leave values not installed")
    path = tmp_path_factory.mktemp("ltab") / "NWL23.ltab"
    subprocess.run([str(TOOL), "--out", str(path)], check=True, capture_output=True)
    return path


def test_magic_spells_ltab():
    assert int.from_bytes(b"LTAB", "little") == LTAB_MAGIC


def test_rows_and_header(ltab_path):
    table = read_leave_table(ltab_path)
    assert table.lexicon == "NWL23"
    assert len(table.klv_hash) == 16 and len(table.content_id) == 16
    assert set(table.klv_hash + table.content_id) <= set("0123456789abcdef")
    assert sorted(table.rows) == list(range(1, MAX_LEAVE_LENGTH + 1))
    for k, rows in table.rows.items():
        assert rows.counts.shape == (ROW_COUNTS[k], 27) and rows.counts.dtype == np.uint8
        assert rows.mask.dtype == np.uint32 and rows.value.dtype == np.float32
        assert (rows.counts.sum(axis=1) == k).all()
        bits = (rows.mask[:, None] >> np.arange(27, dtype=np.uint32)) & 1
        assert (bits.astype(bool) == (rows.counts > 0)).all()
        assert (rows.mask >> 27 == 0).all()
        assert np.isfinite(rows.value).all()


def test_other_format_version_is_refused(ltab_path, tmp_path):
    data = bytearray(ltab_path.read_bytes())
    offset = HEADER.fields["version"][1]
    data[offset : offset + 2] = (LTAB_FORMAT_VERSION + 1).to_bytes(2, "little")
    tampered = tmp_path / "tampered.ltab"
    tampered.write_bytes(bytes(data))
    both = f"version {LTAB_FORMAT_VERSION + 1}.*version {LTAB_FORMAT_VERSION}"
    with pytest.raises(ValueError, match=both):
        read_leave_table(tampered)
