#!/usr/bin/env python3
"""Write the .beliefs.json sidecars (scribblez/blind_spots_beliefs.py) of a
hidden_blind_spots tag whose reports predate them, so the blind-spots viewer
shows their opponent-rack panel.

By default under the leave inference model the reports name, which the pools
were labeled with. --leave-tag stands another model in for it, e.g. when that
one has been deleted; the sidecars name the stand-in and the viewer flags it.
Reports that already have a sidecar are skipped, so a stopped run resumes.

    py/tools/one_off/backfill_blind_spots_beliefs.py --tag v1 --leave-tag v1 --leave-generation 292
"""

import argparse
import sys
from pathlib import Path

from scribblez.blind_spots_beliefs import BeliefWriter
from scribblez.blind_spots_viewer import BELIEFS_SUFFIX, REPORT_SUFFIX, report_header
from scribblez.leave_inference.beliefs import LeaveBeliefs
from scribblez.paths import TagPaths, add_mount_root_argument
from scribblez.workloads import hidden_blind_spots
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--tag", required=True, help="a hidden_blind_spots dashboard tag")
    p.add_argument("--leave-tag", help="a stand-in leave_inference tag; default: the reports'")
    p.add_argument(
        "--leave-generation", type=int, help="the --leave-tag generation; default: its latest"
    )
    add_mount_root_argument(p)
    args = p.parse_args()
    if args.leave_generation is not None and not args.leave_tag:
        p.error("--leave-generation needs --leave-tag")
    return args


def leave_model(args: argparse.Namespace, found_dir: Path) -> tuple[str, int]:
    """The (tag, generation) of the model to write the sidecars under."""
    if not args.leave_tag:
        model = report_header(found_dir).get("leave_model", {})
        if not model.get("tag"):
            raise SystemExit("the reports name no leave inference model (uniform beliefs)")
        return model["tag"], model["generation"]
    exported = TagPaths(args.leave_tag, "leave_inference", args.mount_root).exported_generations()
    if not exported:
        raise SystemExit(f"leave_inference/{args.leave_tag} has no exported model")
    return args.leave_tag, exported[-1] if args.leave_generation is None else args.leave_generation


def main() -> int:
    args = parse_args()
    found_dir, gcg_dir = hidden_blind_spots.report_dirs(args.tag, args.mount_root)
    tag, generation = leave_model(args, found_dir)
    onnx = TagPaths(tag, "leave_inference", args.mount_root).onnx_path(generation)
    if not onnx.exists():
        raise SystemExit(f"no export at {onnx}")
    writer = BeliefWriter(LeaveBeliefs(onnx), tag, generation)
    reports = sorted(found_dir.glob(f"*{REPORT_SUFFIX}"))
    todo = [
        r
        for r in reports
        if not r.with_name(r.name.removesuffix(REPORT_SUFFIX) + BELIEFS_SUFFIX).exists()
    ]
    print(f"{len(todo)} of {len(reports)} reports to write, under {tag} gen {generation}")
    for i, report in enumerate(todo, 1):
        writer.write(report, gcg_dir)
        print(f"\r{i}/{len(todo)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
