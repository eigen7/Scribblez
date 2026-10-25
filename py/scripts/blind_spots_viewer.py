#!/usr/bin/env python3
"""Browse the hidden-leaves blind spots a hidden_blind_spots tag kept, in the
browser.

The sim-survey viewer, fed from the tag's .blindspots.json reports and .gcg
exports (scribblez/blind_spots_viewer.py): one position at a time, the plays
from outside HastyBot's top moves that out-simmed them listed above those top
moves, the selected move previewed on the board beside its confirming and
screening win rates. The left and right arrow keys step through the positions
in name order. It serves on the sim-survey viewer's ports, so it replaces a
running sim_survey_viewer.py.

Usage:
    ./py/scripts/blind_spots_viewer.py --tag v1
    ./py/scripts/blind_spots_viewer.py --tag v1 --min-bag 8
    ./py/scripts/blind_spots_viewer.py --found-dir <dir of reports> --gcg-dir <dir of .gcg>
"""

import argparse
import sys
from pathlib import Path

from scribblez.blind_spots_viewer import viewer_data
from scribblez.paths import add_mount_root_argument
from scribblez.survey_viewer_server import DEFAULT_API_PORT, DEFAULT_DEV_PORT, serve
from scribblez.workloads import hidden_blind_spots
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--tag", help="a hidden_blind_spots dashboard tag")
    p.add_argument("--found-dir", type=Path, help="the reports, instead of a --tag")
    p.add_argument("--gcg-dir", type=Path, help="the kept positions' games, with --found-dir")
    p.add_argument(
        "--min-bag", type=int, default=0, help="show positions with at least this many in the bag"
    )
    p.add_argument("--api-port", type=int, default=DEFAULT_API_PORT, help="survey API server port")
    p.add_argument("--dev-port", type=int, default=DEFAULT_DEV_PORT, help="Vite dev server port")
    add_mount_root_argument(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.tag:
        args.found_dir, args.gcg_dir = hidden_blind_spots.report_dirs(args.tag, args.mount_root)
    elif not (args.found_dir and args.gcg_dir):
        raise SystemExit("pass --tag, or --found-dir and --gcg-dir")
    data = viewer_data(args.found_dir, args.gcg_dir, args.min_bag)
    if not data["positions"]:
        print(f"no kept position in {args.found_dir} with bag >= {args.min_bag}", file=sys.stderr)
        return 1
    return serve(data, args.api_port, args.dev_port, f"{len(data['positions'])} blind spots.")


if __name__ == "__main__":
    sys.exit(main())
