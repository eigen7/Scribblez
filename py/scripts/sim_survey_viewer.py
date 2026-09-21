#!/usr/bin/env python3
"""Browse the positions a sim candidate survey found, in the browser.

Serves a finished survey directory (the --slog-dir of sim_candidate_survey.py,
with its .simsurvey.json files and gcg/ exports) to the web viewer and launches
it: one position at a time, the plays from outside the HastyBot top moves that
out-simmed them listed above those top moves, the selected move previewed on
the board, and its confirming-sim statistics beside it. Left/right arrows step
through the positions, strongest first.

Usage:
    ./py/scripts/sim_survey_viewer.py --survey-dir /workspace/mount/sim-surveys/all-plays-seed1
    ./py/scripts/sim_survey_viewer.py --tag <a blind_spots dashboard tag>
"""

import argparse
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from scribblez.dashboard.react_server import WEB_DIR, reclaim_port
from scribblez.service_urls import service_url
from scribblez.sim_candidate_survey import MIN_SIGMA
from scribblez.sim_survey_viewer import viewer_data
from scribblez.workloads import blind_spots
from util.argparse_ext import ArgumentDefaultsHelpFormatter

DEFAULT_API_PORT = 8091
DEFAULT_DEV_PORT = 5181


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--survey-dir", type=Path, help="a finished survey's --slog-dir")
    p.add_argument("--gcg-dir", type=Path, help="the exported games (default: <survey-dir>/gcg)")
    p.add_argument(
        "--tag",
        help="browse this blind_spots dashboard tag instead of a --survey-dir",
    )
    p.add_argument(
        "--min-sigmas",
        type=float,
        default=MIN_SIGMA,
        help="show positions where an outside play sits this many standard errors above the "
        "best top move",
    )
    p.add_argument("--api-port", type=int, default=DEFAULT_API_PORT)
    p.add_argument("--dev-port", type=int, default=DEFAULT_DEV_PORT)
    return p.parse_args()


def survey_handler(payload: bytes) -> type[BaseHTTPRequestHandler]:
    """A request handler answering GET /api/survey with `payload`."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/api/survey":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass  # one line per request is noise under the viewer's banner

    return Handler


def main() -> int:
    args = parse_args()
    if args.tag:
        args.survey_dir, args.gcg_dir = blind_spots.survey_dirs(args.tag)
    elif not args.survey_dir:
        raise SystemExit("pass --survey-dir or --tag")
    data = viewer_data(args.survey_dir, args.min_sigmas, args.gcg_dir)
    if not data["positions"]:
        print(f"no position in {args.survey_dir} clears {args.min_sigmas} sigma", file=sys.stderr)
        return 1
    reclaim_port(args.api_port)
    reclaim_port(args.dev_port)
    # Loopback only: the browser reaches it through Vite's /api proxy.
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.api_port), survey_handler(json.dumps(data).encode())
    )
    Thread(target=server.serve_forever, daemon=True).start()
    env = {
        **os.environ,
        "VITE_TOOL": "survey",
        "VITE_DEV_PORT": str(args.dev_port),
        "VITE_API_PORT": str(args.api_port),
    }
    vite = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=WEB_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = service_url("survey", args.dev_port, DEFAULT_DEV_PORT)
    print(f"\n{len(data['positions'])} positions. Survey viewer: {url}\n", file=sys.stderr)
    try:
        vite.wait()
    except KeyboardInterrupt:
        pass
    finally:
        vite.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
