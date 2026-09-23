#!/usr/bin/env python3
"""Turn a blind_spots dashboard tag into an examples directory ready to commit.

The tag's workers deliver the positions they find to
/workspace/mount/tags/blind_spots/<tag>/data/. This copies those positions'
games into a directory under positions/, with a README table of what the sims
said about each, ready to `git add`. The directory is deleted and rebuilt from
the tag on every run, so rerun it whenever the tag has grown. See
docs/blind_spots.md.

Usage:
    ./py/scripts/blind_spots_collect.py --tag seed-corpus
    ./py/scripts/blind_spots_collect.py --tag seed-corpus --min-gain 2
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from scribblez.paths import REPO_ROOT
from scribblez.sim_candidate_survey import SURVEY_SUFFIX, load_survey, report, write_review_dir
from scribblez.workloads import blind_spots
from util.argparse_ext import ArgumentDefaultsHelpFormatter

DEFAULT_REVIEW_DIR = REPO_ROOT / "positions" / "NWL23" / "best-bot-blind-spots"

# The README's instructions for regenerating or extending the corpus.
HOW_TO = """\
# In the dashboard: create a tag of the "Collect HastyBot blind spots" workload and add Surveyor
# workers (local, ssh or rented); then: ./py/scripts/blind_spots_collect.py --tag <tag>
# Or on one machine, without the dashboard:
./py/scripts/sim_candidate_survey.py --slog-dir <scratch dir> --generate-games 1 \\
    --max-positions 0 --target-positions 100 --open-leaves --review-dir <this directory>"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--tag", required=True, help="the blind_spots tag to collect")
    p.add_argument(
        "--review-dir",
        type=Path,
        default=DEFAULT_REVIEW_DIR,
        help="output directory; deleted and rebuilt on each run",
    )
    p.add_argument(
        "--min-gain",
        type=float,
        default=0.0,
        help="keep only outside plays that gain at least this many win%% over the best top move "
        "(on top of the survey's 2-sigma bar, which a large sim passes on tiny edges)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    survey_dir, gcg_dir = blind_spots.survey_dirs(args.tag)
    paths = sorted(survey_dir.glob(f"*{SURVEY_SUFFIX}"))
    if not paths:
        print(f"no survey files in {survey_dir}", file=sys.stderr)
        return 1
    survey = load_survey(paths)
    kept = [f for f in survey.findings if 100 * f.gain >= args.min_gain]
    survey = dataclasses.replace(survey, findings=kept)
    print(report(survey))
    cut = json.loads(paths[0].read_text())["cut"]  # one tag, one set of params
    write_review_dir(survey, cut, gcg_dir, args.review_dir, HOW_TO)
    print(f"\n{len(survey.winning_positions)} positions -> {args.review_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
