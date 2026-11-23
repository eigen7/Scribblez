#!/usr/bin/env python3
"""Serve an interactive page of the leave value distribution.

The page histograms the value of every leave of length 1..6 in the engine's
leave values (the .klv2 HastyBot plays with) and lets you constrain the leaves
-- tiles they must contain or must not contain, their lengths -- and compare the
constrained distribution with the unconstrained one. Each leave counts once,
or weighted by its probability of being drawn from a full bag.

By default the script serves the page until Ctrl-C and prints its URL. The page
embeds every leave and value (gzipped), so --out instead writes it to a file
that opens from disk, with no server, in any modern browser.

Examples:
  leave_value_distribution_viewer.py
  leave_value_distribution_viewer.py --out /tmp/leaves.html
"""

import argparse
import base64
import gzip
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scribblez.dashboard.react_server import reclaim_port
from scribblez.ffi import DEFAULT_LEXICON, LeaveValueArrays
from scribblez.service_urls import service_url
from scribblez.supply_registers import TILE_COUNTS
from util.argparse_ext import ArgumentDefaultsHelpFormatter

TEMPLATE = Path(__file__).with_suffix(".html")
DATA_MARKER = "/*__LEAVE_DATA__*/null"
# devenv.toml's [services] entry, which the gateway routes to the host browser.
SERVICE = "leaves"
DEFAULT_PORT = 5182


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


def build_page() -> bytes:
    """The template with the leaves filled in. The template is a fragment (it
    also publishes as a hosted page, which supplies its own document
    skeleton); a standalone page needs the doctype to render in standards
    mode."""
    html = TEMPLATE.read_text()
    if DATA_MARKER not in html:
        raise RuntimeError(f"{TEMPLATE} lacks the data marker {DATA_MARKER!r}")
    data = json.dumps(pack_leaves(LeaveValueArrays()))
    return ("<!doctype html>\n" + html.replace(DATA_MARKER, data)).encode()


def page_handler(page: bytes) -> type[BaseHTTPRequestHandler]:
    """A request handler answering GET / with `page`."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, format, *args):
            pass  # one line per request is noise under the URL banner

    return Handler


def serve(page: bytes, port: int):
    """Serve `page` until Ctrl-C. Binds every interface, as the Vite dev
    servers do, so the gateway can reach it; a previous run on the port is
    replaced."""
    reclaim_port(port)
    server = ThreadingHTTPServer(("0.0.0.0", port), page_handler(page))
    url = service_url(SERVICE, port, DEFAULT_PORT)
    print(f"\nLeave value distribution: {url}  (Ctrl-C to stop)\n", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--out", type=Path, help="Write the page to this file instead of serving it.")
    args = p.parse_args()

    page = build_page()
    if args.out:
        args.out.write_bytes(page)
        print(f"wrote {args.out} ({len(page) / 1e6:.1f} MB)")
    else:
        serve(page, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
