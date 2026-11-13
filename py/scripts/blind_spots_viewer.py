#!/usr/bin/env python3
"""Browse the hidden-leaves blind spots a hidden_blind_spots tag kept, in the
browser.

The sim-survey viewer, fed from the tag's .blindspots.json reports and .gcg
exports (scribblez/blind_spots_viewer.py): one position at a time, the plays
from outside HastyBot's top moves that out-simmed them listed above those top
moves, the selected move previewed on the board beside its confirming and
screening win rates, and what the mover believes about the opponent's rack
under the leave inference model the pools were labeled with: a per-tile chart
against no information, and sample racks. The left and right arrow keys step
through the positions in name order. --leave-tag shows another leave inference
model's beliefs in the panel instead, e.g. when the pools' own model is gone.
It serves on the sim-survey viewer's ports, so it replaces a running
sim_survey_viewer.py.

Usage:
    ./py/scripts/blind_spots_viewer.py --tag v1
    ./py/scripts/blind_spots_viewer.py --tag v1 --min-bag 8
    ./py/scripts/blind_spots_viewer.py --tag v1 --no-beliefs
    ./py/scripts/blind_spots_viewer.py --tag v1 --leave-tag v1 --leave-generation 292
    ./py/scripts/blind_spots_viewer.py --found-dir <dir of reports> --gcg-dir <dir of .gcg>
"""

import argparse
import sys
from pathlib import Path

from scribblez.blind_spots_viewer import BeliefPanel, leave_model_path, report_header, viewer_data
from scribblez.leave_inference.beliefs import LeaveBeliefs
from scribblez.paths import TagPaths, add_mount_root_argument
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
    beliefs = p.add_mutually_exclusive_group()
    beliefs.add_argument(
        "--no-beliefs",
        action="store_true",
        help="skip the opponent-rack panel, e.g. when the pools' leave inference model is gone",
    )
    beliefs.add_argument(
        "--leave-tag",
        help="a leave_inference tag whose beliefs the panel shows, instead of the pools' own model",
    )
    p.add_argument(
        "--leave-generation",
        type=int,
        help="the --leave-tag generation; default: its latest export",
    )
    p.add_argument("--api-port", type=int, default=DEFAULT_API_PORT, help="survey API server port")
    p.add_argument("--dev-port", type=int, default=DEFAULT_DEV_PORT, help="Vite dev server port")
    add_mount_root_argument(p)
    args = p.parse_args()
    if args.leave_generation is not None and not args.leave_tag:
        p.error("--leave-generation needs --leave-tag")
    return args


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
    routes = {}
    banner = f"{len(data['positions'])} blind spots."
    model = belief_model(args)
    if model is not None:  # the opponent-rack panel
        panel = BeliefPanel(data, args.found_dir, LeaveBeliefs(model))
        routes["/api/leave_samples"] = panel.samples
        if args.leave_tag:
            banner += f" Beliefs from {model}, not the pools' own leave inference model."
    return serve(data, args.api_port, args.dev_port, banner, routes)


def belief_model(args: argparse.Namespace) -> Path | None:
    """The leave inference export the panel shows: --leave-tag's, else the one
    the pools were labeled under; None for no panel."""
    if args.no_beliefs:
        return None
    if not args.leave_tag:
        return leave_model_path(report_header(args.found_dir), args.mount_root)
    paths = TagPaths(args.leave_tag, "leave_inference", args.mount_root)
    exported = paths.exported_generations()
    if not exported:
        raise SystemExit(f"leave_inference/{args.leave_tag} has no exported model")
    generation = exported[-1] if args.leave_generation is None else args.leave_generation
    if generation not in exported:
        raise SystemExit(f"leave_inference/{args.leave_tag} has no export of gen {generation}")
    return paths.onnx_path(generation)


if __name__ == "__main__":
    sys.exit(main())
