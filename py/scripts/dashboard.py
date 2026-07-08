#!/usr/bin/env python3
"""Launch the web dashboard (React app + Python data API).

With no --task this serves the master dashboard: the entrypoint for all work,
where you pick a workload, create or open a tag, attach local/cloud workers,
and watch progress (docs/master_dashboard.md). With --task it opens directly
on one training task's view, which is also what each trainer auto-spawns.
Reclaims the API/Vite ports from any stale dashboard first.

Usage:
    ./py/scripts/dashboard.py                       # master dashboard
    ./py/scripts/dashboard.py --task position_eval
"""

import argparse
import sys

from scribblez.dashboard import react_server
from scribblez.paths import MAX_MOVE_PER_LANE, POSITION_EVAL
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--task",
        choices=[POSITION_EVAL, MAX_MOVE_PER_LANE],
        default=None,
        help="open directly on one training task's view (default: the master dashboard)",
    )
    p.add_argument(
        "--tag", default=None, help="Open the dashboard on this tag (default: the first available)."
    )
    p.add_argument("--mount-root", default="/workspace/mount")
    p.add_argument("--api-port", type=int, default=react_server.DEFAULT_API_PORT)
    p.add_argument("--dev-port", type=int, default=react_server.DEFAULT_DEV_PORT)
    args = p.parse_args()
    react_server.launch(args.task, args.mount_root, args.api_port, args.dev_port, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
