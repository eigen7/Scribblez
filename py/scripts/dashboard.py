#!/usr/bin/env python3
"""Launch a per-task training-metrics dashboard (Bokeh server).

Usage:
    python -m scripts.dashboard                          # post-move (default)
    python -m scripts.dashboard --task max_move_per_lane
    python -m scripts.dashboard --port 5006

Serves the per-tag dashboard.db stores under <mount-root>/tags/*/. Run this when
a trainer isn't already serving the dashboard.
"""

import argparse
import os
import sys
from pathlib import Path

from scribblez.dashboard import server

_APPS = {
    "post_move_value": server.POST_MOVE_VALUE_APP,
    "max_move_per_lane": server.MAX_MOVE_PER_LANE_APP,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task",
        choices=sorted(_APPS),
        default="post_move_value",
        help="Which model's dashboard to serve.",
    )
    parser.add_argument("--port", type=int, default=server.DEFAULT_PORT, help="Server port.")
    args = parser.parse_args()

    app = _APPS[args.task]
    cmd = server.serve_command(args.port, app=app)
    print(f"Serving dashboard: http://localhost:{args.port}/{Path(app).stem}")
    os.execvp(cmd[0], cmd)  # replace this process with `bokeh serve`
    return 0  # unreachable


if __name__ == "__main__":
    sys.exit(main())
