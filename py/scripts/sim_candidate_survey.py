#!/usr/bin/env python3
"""Measure what a static-equity candidate cut costs the self-play corpus.

Runs sim_candidate_survey_tool over a directory of .slog files (files it has
already surveyed are skipped) and reports how often, and by how much, a
Monte-Carlo sim prefers a candidate outside the head of the HastyBot equity
ranking -- PR 0 of docs/plans/sim_labeled_candidates.md. The default candidate
sample is 64 per position: the played move and the rest of the top 32 in
full, 28 from the tail, 4 exchanges.

Usage:
    ./py/scripts/sim_candidate_survey.py --slog-dir <dir> --limit-games 300 [--open-leaves]
    ./py/scripts/sim_candidate_survey.py --slog-dir <dir> --report-only
"""

import argparse
import subprocess
import sys
from pathlib import Path

from scribblez.paths import ENGINE_DIR
from scribblez.sim_candidate_survey import SURVEY_SUFFIX, load_survey, report
from util.argparse_ext import ArgumentDefaultsHelpFormatter

SURVEY_TOOL = str(ENGINE_DIR / "sim_candidate_survey_tool")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--slog-dir", type=Path, required=True, help="directory of .slog files")
    p.add_argument("--open-leaves", action="store_true", help="a face-up-leaves corpus")
    p.add_argument("--rollouts", type=int, default=300, help="rollouts per candidate per replica")
    p.add_argument("--limit-games", type=int, default=0, help="first N games per file (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cut", type=int, default=10, help="the equity top-K cut being priced")
    p.add_argument("--report-only", action="store_true", help="skip the sims; read existing rows")
    return p.parse_args()


def run_survey(args: argparse.Namespace):
    cmd = [SURVEY_TOOL, "--slog-dir", str(args.slog_dir), "--rollouts", str(args.rollouts)]
    cmd += ["--limit-games", str(args.limit_games), "--seed", str(args.seed)]
    if args.open_leaves:
        cmd.append("--open-leaves")
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    if not args.report_only:
        run_survey(args)
    paths = sorted(args.slog_dir.glob(f"*{SURVEY_SUFFIX}"))
    if not paths:
        print(f"no *{SURVEY_SUFFIX} files in {args.slog_dir}", file=sys.stderr)
        return 1
    print(report(load_survey(paths), args.cut))
    return 0


if __name__ == "__main__":
    sys.exit(main())
