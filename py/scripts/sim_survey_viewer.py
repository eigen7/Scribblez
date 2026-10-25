#!/usr/bin/env python3
"""Browse the positions a sim candidate survey found, in the browser.

Serves a finished survey to the web viewer and launches it. The survey is
either a directory written by sim_candidate_survey.py (its --slog-dir, holding
the .simsurvey.json files and gcg/ exports) or a blind_spots dashboard tag.
The viewer shows one position at a time: the plays from outside HastyBot's top
moves that out-simmed them, listed above those top moves, with the selected
move previewed on the board beside its confirming-sim statistics. The left and
right arrow keys step through the positions in name order.

Usage:
    ./py/scripts/sim_survey_viewer.py --survey-dir /workspace/mount/sim-surveys/all-plays-seed1
    ./py/scripts/sim_survey_viewer.py --tag <a blind_spots dashboard tag>
"""

import argparse
import sys
from pathlib import Path

from scribblez.paths import add_mount_root_argument
from scribblez.sim_candidate_survey import MIN_SIGMA
from scribblez.sim_survey_viewer import viewer_data
from scribblez.survey_viewer_server import DEFAULT_API_PORT, DEFAULT_DEV_PORT, serve
from scribblez.workloads import blind_spots
from util.argparse_ext import ArgumentDefaultsHelpFormatter


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
    p.add_argument("--api-port", type=int, default=DEFAULT_API_PORT, help="survey API server port")
    p.add_argument("--dev-port", type=int, default=DEFAULT_DEV_PORT, help="Vite dev server port")
    add_mount_root_argument(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.tag:
        args.survey_dir, args.gcg_dir = blind_spots.survey_dirs(args.tag, args.mount_root)
    elif not args.survey_dir:
        raise SystemExit("pass --survey-dir or --tag")
    data = viewer_data(args.survey_dir, args.min_sigmas, args.gcg_dir)
    if not data["positions"]:
        print(f"no position in {args.survey_dir} clears {args.min_sigmas} sigma", file=sys.stderr)
        return 1
    banner = f"{len(data['positions'])} positions."
    return serve(data, args.api_port, args.dev_port, banner)


if __name__ == "__main__":
    sys.exit(main())
