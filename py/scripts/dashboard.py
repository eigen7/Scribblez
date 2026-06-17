#!/usr/bin/env python3
"""Launch the training-metrics dashboard (Bokeh server).

Usage:
    python -m scripts.dashboard                 # serve on the default port
    python -m scripts.dashboard --port 5006

Serves the per-tag dashboard.db stores under <mount-root>/tags/*/ at
http://localhost:<port>/app. Run this when train.py isn't already serving it.
"""

import argparse
import os
import sys

from scribblez.dashboard import server


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=server.DEFAULT_PORT, help="Server port.")
    args = parser.parse_args()

    cmd = server.serve_command(args.port)
    print(f"Serving dashboard: http://localhost:{args.port}/app")
    os.execvp(cmd[0], cmd)  # replace this process with `bokeh serve`
    return 0  # unreachable


if __name__ == "__main__":
    sys.exit(main())
