#!/usr/bin/env python3
"""Launch the training dashboard (React app + Python data API) for a task.

Run this to view a tag's dashboard when a trainer isn't already serving one (each
trainer launches it automatically). It serves the per-tag dashboard.db stores under
<mount-root>/tags/<task>/, reclaiming its ports if a stale dashboard holds them.

Usage:
    ./py/scripts/dashboard.py --task post_move_value
    ./py/scripts/dashboard.py --task max_move_per_lane --dev-port 5180
"""

import argparse
import sys

from scribblez.dashboard import react_server
from scribblez.paths import MAX_MOVE_PER_LANE, POST_MOVE_VALUE
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--task",
        choices=[POST_MOVE_VALUE, MAX_MOVE_PER_LANE],
        default=POST_MOVE_VALUE,
        help="Which model's runs to serve.",
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
