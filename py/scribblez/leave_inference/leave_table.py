"""Pure-numpy reader for the leave table (.ltab): every leave drawable from a
full bag, for each length 1..6, with its static leave value.

engine/include/belief/leave_table.h owns the layout and leave_table_tool
writes the file; the dtypes and the format version come from the engine's
format-layout document, so this reader cannot drift from the writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scribblez.ffi import format_layout, struct_dtype

_CONST = format_layout()["constants"]["ltab"]

LTAB_MAGIC = _CONST["magic"]
LTAB_FORMAT_VERSION = _CONST["version"]
MAX_LEAVE_LENGTH = _CONST["max_length"]

HEADER = struct_dtype("LeaveTableHeader")
ROW = struct_dtype("LeaveTableRow")


@dataclass
class LeaveRows:
    """One length's leaves, in the engine's enumeration order."""

    counts: np.ndarray  # (R, 27) uint8, indexed by tile code; the blank is 26
    mask: np.ndarray  # (R,) uint32, bit l set iff counts[:, l] > 0
    value: np.ndarray  # (R,) float32, the static leave value


@dataclass
class LeaveTable:
    lexicon: str
    klv_hash: str  # content hash of the .klv2 the values came from
    content_id: str  # the table's identity, which models trained against it stamp
    rows: dict[int, LeaveRows]  # keyed by leave length, 1..MAX_LEAVE_LENGTH


def read_leave_table(path: str | Path) -> LeaveTable:
    """Parse a .ltab file. Raises ValueError on bad magic, a format version
    other than the engine's, or a size that disagrees with the row counts."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size < HEADER.itemsize:
        raise ValueError(f"{path}: too short for a leave table header")
    hdr = np.frombuffer(buf, dtype=HEADER, count=1)[0]
    if hdr["magic"] != LTAB_MAGIC:
        raise ValueError(f"{path}: not a leave table (bad magic)")
    if hdr["version"] != LTAB_FORMAT_VERSION:
        raise ValueError(
            f"{path}: leave table format version {hdr['version']}, "
            f"but this build reads version {LTAB_FORMAT_VERSION}"
        )
    row_counts = [int(n) for n in hdr["row_counts"]]
    expected = HEADER.itemsize + sum(row_counts) * ROW.itemsize
    if buf.size != expected:
        raise ValueError(f"{path}: {buf.size} bytes, but its row counts imply {expected}")
    all_rows = np.frombuffer(buf, dtype=ROW, offset=HEADER.itemsize)
    starts = np.concatenate([[0], np.cumsum(row_counts)])
    rows = {}
    for k in range(1, MAX_LEAVE_LENGTH + 1):
        part = all_rows[starts[k - 1] : starts[k]]
        rows[k] = LeaveRows(
            counts=np.ascontiguousarray(part["counts"]),
            mask=np.ascontiguousarray(part["mask"]),
            value=np.ascontiguousarray(part["value"]),
        )
    return LeaveTable(
        lexicon=hdr["lexicon"].decode(),
        klv_hash=hdr["klv_hash"].decode(),
        content_id=hdr["content_id"].decode(),
        rows=rows,
    )
