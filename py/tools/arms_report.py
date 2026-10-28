#!/usr/bin/env python3
"""Tabulate match_arms tags: each arm's record, score and spread, and, against
a baseline arm, the pair-by-pair difference (match_eval/arms_report.py).

    py/tools/arms_report.py                          # every match_arms tag
    py/tools/arms_report.py --baseline base          # ... with deltas vs arm "base"
    py/tools/arms_report.py ablate-base ablate-uniform --baseline base

An arm still playing shows its games so far and no time.
"""

import argparse
import math
import sys
from pathlib import Path

from scribblez.match_eval import arms_report
from scribblez.match_eval.arms_report import Arm, Interval
from scribblez.paths import add_mount_root_argument

FOOTNOTE = """
± is a 95% interval over deal pairs. Δ is the arm less the baseline over the pairs both
played; 'same' counts games that ended with the baseline's exact spread."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tags", nargs="*", help="match_arms tags (default: all)")
    parser.add_argument("--baseline", help="the arm (or its tag) every other arm is compared with")
    add_mount_root_argument(parser)
    return parser.parse_args()


def fmt(iv: Interval, digits: int, signed: bool = False) -> str:
    sign = "+" if signed else ""
    width = "" if math.isnan(iv.half_width) else f" ±{iv.half_width:.{digits}f}"
    return f"{iv.mean:{sign}.{digits}f}{width}"


def fmt_time(arm: Arm) -> str:
    if arm.elapsed_s is None:
        return "playing"
    return f"{arm.elapsed_s / 60:.0f} min"


def find_baseline(arms: list[Arm], name: str) -> Arm:
    matches = [a for a in arms if name in (a.name, a.tag)]
    if len(matches) != 1:
        sys.exit(f"--baseline {name!r} matches {len(matches)} arms; name one arm or tag")
    return matches[0]


def row(arm: Arm, baseline: Arm | None) -> list[str]:
    s = arms_report.summarize(arm.games)
    cells = [
        arm.name,
        str(s.games),
        f"{s.wins}-{s.draws}-{s.losses}",
        fmt(s.score, 3),
        fmt(s.spread, 1),
        fmt_time(arm),
    ]
    if baseline is not None:
        if arm is baseline:
            return cells + ["baseline", "", "", ""]
        c = arms_report.compare(arm.games, baseline.games)
        cells += [fmt(c.score, 3, signed=True), fmt(c.spread, 1, signed=True)]
        cells += [str(c.pairs), f"{c.identical}/{2 * c.pairs}"]
    return cells


def table_line(cells: list[str], widths: list[int]) -> str:
    """The arm name left-aligned, the numbers right-aligned."""
    padded = [c.rjust(w) for c, w in zip(cells, widths, strict=True)]
    padded[0] = cells[0].ljust(widths[0])
    return "  ".join(padded)


def print_table(header: list[str], rows: list[list[str]]):
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    for r in [header, ["-" * w for w in widths], *rows]:
        print(table_line(r, widths))


def main():
    args = parse_args()
    mount_root = Path(args.mount_root)
    tags = args.tags or arms_report.all_tags(mount_root)
    arms = [a for tag in tags for a in arms_report.read_tag(tag, mount_root) if a.games]
    if not arms:
        sys.exit("no arm has saved games")
    baseline = find_baseline(arms, args.baseline) if args.baseline else None
    if baseline is not None:
        arms.sort(key=lambda a: a is not baseline)
    header = ["arm", "games", "W-D-L", "score", "spread", "time"]
    if baseline is not None:
        header += ["Δscore", "Δspread", "pairs", "same"]
    print_table(header, [row(a, baseline) for a in arms])
    print(FOOTNOTE)


if __name__ == "__main__":
    main()
