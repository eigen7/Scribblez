#!/usr/bin/env python3
"""Measure what a static-equity candidate cut costs the self-play corpus.

Runs sim_candidate_survey_tool over a directory of .slog files (files it has
already surveyed are skipped) and reports how often, and by how much, a
Monte-Carlo sim prefers a candidate outside the head of the HastyBot equity
ranking -- PR 0 of docs/plans/sim_labeled_candidates.md. The default candidate
sample is 64 per position: the played move and the rest of the top 32 in
full, 28 from the tail, 4 exchanges.

--recipe setup asks the sharper question: only positions with a high-value
setup play outside the cut (a J/Q/X/Z kept, a tile laid beside a premium
square where it hooks), simming the top of the ranking plus every such play.
--review-dir then collects the games where the setup gained most.

--generate-games makes the whole run reproducible from nothing: it first plays
that many HastyBot-vs-HastyBot games (greedy, random opening of mean 2 plies,
face-up leaves with --open-leaves -- the position_eval corpus's recipe) into
--slog-dir on one thread, which with a fixed --game-seed yields the same
games every time, and names the files by that seed. The review README records
the command line, so anyone can regenerate its files.

Usage:
    ./py/scripts/sim_candidate_survey.py --slog-dir <dir> --limit-games 300 [--open-leaves]
    ./py/scripts/sim_candidate_survey.py --slog-dir <dir> --generate-games 3000 --open-leaves \\
        --recipe setup --rollouts 1000 --review-dir positions/NWL23/setup-survey-examples
    ./py/scripts/sim_candidate_survey.py --slog-dir <dir> --report-only
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from scribblez.paths import ENGINE_DIR
from scribblez.selfplay import hasty_player_spec, run_games
from scribblez.sim_candidate_survey import (
    SURVEY_SUFFIX,
    load_survey,
    report,
    setup_report,
    write_review_dir,
)
from util.argparse_ext import ArgumentDefaultsHelpFormatter

SURVEY_TOOL = str(ENGINE_DIR / "sim_candidate_survey_tool")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--slog-dir", type=Path, required=True, help="directory of .slog files")
    p.add_argument("--generate-games", type=int, default=0, help="play this many games first")
    p.add_argument("--game-seed", type=int, default=1, help="play_game seed for --generate-games")
    p.add_argument("--recipe", choices=("stratified", "setup"), default="stratified")
    p.add_argument("--max-positions", type=int, default=100, help="setup: positions per file")
    p.add_argument("--review-dir", type=Path, help="setup: collect the strongest examples here")
    p.add_argument("--review-count", type=int, default=15, help="examples collected")
    p.add_argument("--open-leaves", action="store_true", help="a face-up-leaves corpus")
    p.add_argument("--rollouts", type=int, default=300, help="rollouts per candidate per replica")
    p.add_argument("--limit-games", type=int, default=0, help="first N games per file (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cut", type=int, default=10, help="the equity top-K cut being priced")
    p.add_argument("--report-only", action="store_true", help="skip the sims; read existing rows")
    return p.parse_args()


def generate_games(args: argparse.Namespace):
    """Fill an empty --slog-dir with --generate-games games. One game thread and a
    fixed seed make the games reproducible; play_game names its files by
    timestamp, so they are renamed after the seed to make everything downstream
    (survey rows, GCG names) reproducible too."""
    if any(args.slog_dir.glob("*.slog")):
        print(f"{args.slog_dir} already has games; not generating")
        return
    code = run_games(
        args.slog_dir,
        args.generate_games,
        threads=1,
        player_spec=hasty_player_spec(),
        seed=args.game_seed,
        random_opening_mean=2.0,
        face_up_leaves=args.open_leaves,
    )
    if code != 0:
        raise SystemExit(code)
    for i, path in enumerate(sorted(args.slog_dir.glob("*.slog"))):
        path.rename(args.slog_dir / f"hasty-seed{args.game_seed}-{i}.slog")


def gcg_dir(args: argparse.Namespace) -> Path:
    return args.slog_dir / "gcg"


def run_survey(args: argparse.Namespace):
    cmd = [SURVEY_TOOL, "--slog-dir", str(args.slog_dir), "--rollouts", str(args.rollouts)]
    cmd += ["--limit-games", str(args.limit_games), "--seed", str(args.seed)]
    cmd += ["--recipe", args.recipe, "--cut", str(args.cut)]
    cmd += ["--max-positions", str(args.max_positions)]
    if args.recipe == "setup":
        cmd += ["--gcg-dir", str(gcg_dir(args))]
    if args.open_leaves:
        cmd.append("--open-leaves")
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    if args.generate_games and not args.report_only:
        generate_games(args)
    if not args.report_only:
        run_survey(args)
    paths = sorted(args.slog_dir.glob(f"*{SURVEY_SUFFIX}"))
    if not paths:
        print(f"no *{SURVEY_SUFFIX} files in {args.slog_dir}", file=sys.stderr)
        return 1
    survey = load_survey(paths)
    print(report(survey, args.cut))
    if args.recipe == "setup":
        print()
        print(setup_report(survey, args.cut))
        if args.review_dir:
            command = shlex.join(["./py/scripts/sim_candidate_survey.py", *sys.argv[1:]])
            write_review_dir(
                survey, args.cut, gcg_dir(args), args.review_dir, args.review_count, command
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
