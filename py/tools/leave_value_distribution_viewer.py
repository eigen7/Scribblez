#!/usr/bin/env python3
"""Write an interactive, self-contained HTML page of the leave value
distribution.

The page histograms the value of every leave of length 1..6 in the engine's
leave values (the .klv2 HastyBot plays with) and lets you constrain the leaves
-- tiles they must contain or must not contain, their lengths -- and compare the
constrained distribution with the unconstrained one. Each leave counts once,
or weighted by its probability of being drawn from a full bag.

The page embeds every leave and value (gzipped), so it opens from disk, with no
server, in any modern browser.

Examples:
  leave_distribution.py
  leave_distribution.py --out /tmp/leaves.html
"""

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np
from scribblez.ffi import DEFAULT_LEXICON, LeaveValueArrays
from scribblez.supply_registers import TILE_COUNTS
from util.argparse_ext import ArgumentDefaultsHelpFormatter

TEMPLATE = Path(__file__).with_suffix(".html")
DATA_MARKER = "/*__LEAVE_DATA__*/null"


def gzip_b64(buf: bytes) -> str:
    return base64.b64encode(gzip.compress(buf, compresslevel=9)).decode("ascii")


def pack_leaves(arrays: LeaveValueArrays) -> dict:
    """Every leave as tile indices (0..25 = A..Z, 26 = blank) and values, the
    lengths concatenated in order 1..max_length: the tiles row-major, k bytes
    per length-k leave, and one float32 value per leave."""
    tiles, values, rows = [], [], []
    for k in range(1, arrays.max_length + 1):
        codes, vals = arrays.of_length(k)
        tiles.append((codes.T // arrays.copies_per_tile).astype(np.uint8).ravel())
        values.append(vals.astype("<f4"))
        rows.append(len(vals))
    return {
        "lexicon": DEFAULT_LEXICON,
        "source": "leaves.klv2",
        "contentHash": arrays.content_hash,
        "bagCounts": list(TILE_COUNTS),
        "rows": rows,
        "tiles": gzip_b64(np.concatenate(tiles).tobytes()),
        "values": gzip_b64(np.concatenate(values).tobytes()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("leave_distribution.html"))
    args = p.parse_args()

    data = pack_leaves(LeaveValueArrays())
    html = TEMPLATE.read_text()
    if DATA_MARKER not in html:
        raise RuntimeError(f"{TEMPLATE} lacks the data marker {DATA_MARKER!r}")
    # The template is a fragment (it also publishes as a hosted page, which
    # supplies its own document skeleton); a file on disk needs the doctype
    # to render in standards mode.
    args.out.write_text("<!doctype html>\n" + html.replace(DATA_MARKER, json.dumps(data)))
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size_mb:.1f} MB, {sum(data['rows']):,} leaves)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
