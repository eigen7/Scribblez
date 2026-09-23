#!/usr/bin/env python3
"""Launch the master web dashboard: the React app and its Python API server.

The dashboard is where workloads run. Pick a workload, create or open a tag
(one task of that workload), attach workers to it -- local, ssh or rented,
plus the singleton trainer of a training workload -- and watch progress and
analysis. See docs/master_dashboard.md. Any process still listening on the API
or Vite port, typically a stale dashboard, is killed first.

Usage:
    ./py/scripts/dashboard.py
    ./py/scripts/dashboard.py --workload position_eval --tag mytag
"""

import argparse
import sys

from scribblez import workloads
from scribblez.dashboard import react_server
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--workload",
        choices=sorted(workloads.WORKLOADS),
        default=None,
        help="open the dashboard on this workload's tag list",
    )
    p.add_argument("--tag", default=None, help="open the dashboard on this tag's task view")
    p.add_argument(
        "--mount-root", default="/workspace/mount", help="persistent data mount (holds tags/)"
    )
    p.add_argument(
        "--api-port", type=int, default=react_server.DEFAULT_API_PORT, help="Python API server port"
    )
    p.add_argument(
        "--dev-port", type=int, default=react_server.DEFAULT_DEV_PORT, help="Vite dev server port"
    )
    args = p.parse_args()
    react_server.launch(args.mount_root, args.api_port, args.dev_port, args.workload, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
