#!/usr/bin/env python3
"""Launch the React training dashboard (Python data API + Vite dev server).

This is the React/embedded-Bokeh dashboard (docs/react_dashboard.md), served
alongside the legacy Bokeh dashboard during the migration.

Usage:
    python -m scripts.react_dashboard --task max_move_per_lane
    python -m scripts.react_dashboard --task post_move_value --dev-port 5180
"""

import argparse
import sys

from scribblez.dashboard import react_server
from scribblez.paths import MAX_MOVE_PER_LANE, POST_MOVE_VALUE


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--task", choices=[POST_MOVE_VALUE, MAX_MOVE_PER_LANE], default=POST_MOVE_VALUE,
                   help="Which model's runs to serve.")
    p.add_argument("--mount-root", default="/workspace/mount")
    p.add_argument("--api-port", type=int, default=react_server.DEFAULT_API_PORT)
    p.add_argument("--dev-port", type=int, default=react_server.DEFAULT_DEV_PORT)
    args = p.parse_args()
    react_server.launch(args.task, args.mount_root, args.api_port, args.dev_port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
