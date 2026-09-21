#!/usr/bin/env python3
"""Find the plays a static-equity candidate cut hides from HastyBot self-play.

Runs sim_candidate_survey_tool over a directory of .slog files (files it has
already surveyed are skipped) and reports how often, and by how much, a
Monte-Carlo sim prefers a candidate outside the head of the HastyBot equity
ranking -- PR 0 of docs/plans/sim_labeled_candidates.md. Each position is
simmed twice: a screen of every candidate (racing: a candidate clearly below
the leader stops early) singles out its few best moves outside the cut, and a
longer confirming sim on fresh rollouts re-sims those beside the cut's, free
of the screen's best-of-hundreds bias. A move counts only when the confirming
sim puts it at least two standard errors above the cut's best. A stopped run
resumes where it left off when rerun with the same options.

Recipes (sim_candidate_survey_tool.cpp describes them fully):
  all         every legal play that places no blank, at a sample of every
              eligible turn -- exhaustive, and expensive (about a minute a
              position at 1000 rollouts on 28 threads)
  setup       only positions with a high-value setup play outside the cut (a
              J/Q/X/Z kept, a tile laid beside a premium square where it
              hooks), simming just those plays plus the cut
  stratified  a per-game turn sample, 64 stratified candidates each

Each .slog gets a .simsurvey.json sidecar carrying, per candidate (from the
screen, and again from the confirming sim), the outcome counts, the
final-margin histogram, both sides' next-move score statistics and the
end-of-game rack settlement -- enough for a later tool to classify why an
outside play wins.

The confirming sim's rollouts solve their endgames, rather than play them
greedily, at positions with at most --solve-max-unseen unseen tiles: greedy
endgames misjudge late-game candidates by tens of win%.

--generate-games makes the whole run reproducible from nothing: it first plays
that many HastyBot-vs-HastyBot games (greedy, random opening of mean 2 plies,
face-up leaves with --open-leaves -- the position_eval corpus's recipe) into
--slog-dir on one thread, which with a fixed --game-seed yields the same
games every time, and names the files by that seed. --target-positions keeps
playing and surveying further batches until that many positions are found;
games cost nothing beside the sims, so the way to collect positions is one
game a batch with every eligible turn surveyed (--generate-games 1
--max-positions 0), which wastes no game and stops within a game of the
target.
--review-dir collects the
games of the confirmed positions, and its README records the command line, so
anyone can regenerate its files.

Usage:
    ./py/scripts/sim_candidate_survey.py --slog-dir /workspace/mount/sim-surveys/blind-spots \\
        --generate-games 1 --max-positions 0 --target-positions 100 --open-leaves \\
        --review-dir positions/NWL23/best-bot-blind-spots
    ./py/scripts/sim_candidate_survey.py --slog-dir <dir> --recipe setup --report-only
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
    write_review_dir,
)
from util.argparse_ext import ArgumentDefaultsHelpFormatter

SURVEY_TOOL = str(ENGINE_DIR / "sim_candidate_survey_tool")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--slog-dir", type=Path, required=True, help="directory of .slog files")
    p.add_argument("--generate-games", type=int, default=0, help="play this many games first")
    p.add_argument("--game-seed", type=int, default=1, help="play_game seed for --generate-games")
    p.add_argument(
        "--target-positions",
        type=int,
        default=0,
        help="with --generate-games: keep playing batches of that many games (seeds game-seed, "
        "game-seed + 1, ...) and surveying them until this many positions have an outside play "
        "that beats the cut",
    )
    p.add_argument(
        "--solve-max-unseen",
        type=int,
        default=14,
        help="confirming rollouts solve their endgames at positions with at most this many "
        "unseen tiles (-1 = never)",
    )
    p.add_argument("--recipe", choices=("all", "setup", "stratified"), default="all")
    p.add_argument("--max-positions", type=int, default=100, help="positions per file (0 = all)")
    p.add_argument("--review-dir", type=Path, help="collect the confirmed positions' games here")
    p.add_argument("--open-leaves", action="store_true", help="a face-up-leaves corpus")
    p.add_argument("--rollouts", type=int, default=1000, help="screening rollouts per candidate")
    p.add_argument("--confirm-rollouts", type=int, default=5000, help="confirming rollouts")
    p.add_argument("--confirm-picks", type=int, default=5, help="outside moves confirmed")
    p.add_argument("--limit-games", type=int, default=0, help="first N games per file (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cut", type=int, default=10, help="the equity top-K cut being priced")
    p.add_argument("--report-only", action="store_true", help="skip the sims; read existing rows")
    return p.parse_args()


def generate_games(args: argparse.Namespace, seed: int):
    """Play --generate-games games under `seed` into --slog-dir, unless that seed's
    files are already there. One game thread and a fixed seed make the games
    reproducible; play_game names its files by timestamp, so they are renamed
    after the seed to make everything downstream (survey files, GCG names)
    reproducible too."""
    if any(args.slog_dir.glob(f"hasty-seed{seed}-*.slog")):
        return
    code = run_games(
        args.slog_dir,
        args.generate_games,
        threads=1,
        player_spec=hasty_player_spec(),
        seed=seed,
        random_opening_mean=2.0,
        face_up_leaves=args.open_leaves,
    )
    if code != 0:
        raise SystemExit(code)
    fresh = sorted(p for p in args.slog_dir.glob("*.slog") if not p.name.startswith("hasty-seed"))
    for i, path in enumerate(fresh):
        path.rename(args.slog_dir / f"hasty-seed{seed}-{i}.slog")


def survey_files(args: argparse.Namespace) -> list[Path]:
    return sorted(args.slog_dir.glob(f"*{SURVEY_SUFFIX}"))


def generate_and_survey(args: argparse.Namespace):
    """One batch of games and its survey -- or, with --target-positions, batch after
    batch until enough positions have an outside play that beats the cut. Batches
    already played and files already surveyed are skipped, so a stopped run
    picks up where it left off."""
    batch = 0
    while True:
        generate_games(args, args.game_seed + batch)
        run_survey(args)
        found = len(load_survey(survey_files(args)).winning_positions)
        print(f"batch {batch}: {found} positions so far", file=sys.stderr)
        if found >= args.target_positions:
            return
        batch += 1


def gcg_dir(args: argparse.Namespace) -> Path:
    return args.slog_dir / "gcg"


def run_survey(args: argparse.Namespace):
    cmd = [SURVEY_TOOL, "--slog-dir", str(args.slog_dir), "--rollouts", str(args.rollouts)]
    cmd += ["--confirm-rollouts", str(args.confirm_rollouts)]
    cmd += ["--confirm-picks", str(args.confirm_picks)]
    cmd += ["--solve-max-unseen", str(args.solve_max_unseen)]
    cmd += ["--limit-games", str(args.limit_games), "--seed", str(args.seed)]
    cmd += ["--recipe", args.recipe, "--cut", str(args.cut)]
    cmd += ["--max-positions", str(args.max_positions)]
    if args.recipe != "stratified":
        cmd += ["--gcg-dir", str(gcg_dir(args))]
    if args.open_leaves:
        cmd.append("--open-leaves")
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    if args.generate_games and not args.report_only:
        generate_and_survey(args)
    elif not args.report_only:
        run_survey(args)
    paths = survey_files(args)
    if not paths:
        print(f"no *{SURVEY_SUFFIX} files in {args.slog_dir}", file=sys.stderr)
        return 1
    survey = load_survey(paths)
    print(report(survey))
    if args.review_dir:
        # --report-only is how a finished run is re-read; the recorded command is
        # the one that regenerates everything.
        argv = [a for a in sys.argv[1:] if a != "--report-only"]
        command = shlex.join(["./py/scripts/sim_candidate_survey.py", *argv])
        write_review_dir(survey, args.cut, gcg_dir(args), args.review_dir, command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
