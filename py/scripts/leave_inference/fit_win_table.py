#!/usr/bin/env python3
"""Fit a win table from a corpus of hidden-leaves .slog files, outside a tag.

The leave_inference workload refits its own table every generation; this
script fits one from any corpus, for example an existing HastyBot self-play
tag's slogs (docs/plans/deep_bot_v0.md, "The win table"). Every
`--holdout-every`-th file (by a hash of its name) is held out, and the
report compares the fitted table's log loss on those files with Macondo's
bundled table.

Usage:
    ./py/scripts/leave_inference/fit_win_table.py \\
        --slogs /workspace/mount/tags/position_eval/hidden-v1/data/gen_* \\
        --policy "--type=hasty" --out /workspace/mount/win_tables/hasty-v0.csv
"""

import argparse
import sys
import zlib
from dataclasses import fields
from pathlib import Path

from scribblez.leave_inference import win_table
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def _slog_files(roots: list[Path]) -> list[Path]:
    files = []
    for root in roots:
        files.extend(sorted(root.rglob("*.slog")) if root.is_dir() else [root])
    return files


def _held_out(path: Path, every: int) -> bool:
    return every > 0 and zlib.crc32(path.stem.encode()) % every == 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--slogs", type=Path, nargs="+", required=True, help=".slog files or dirs.")
    p.add_argument("--out", type=Path, required=True, help="The table to write.")
    p.add_argument("--policy", required=True, help="The player spec that played the corpus.")
    p.add_argument("--holdout-every", type=int, default=10, help="Hold out 1 file in N; 0: none.")
    defaults = win_table.FitParams()
    for f in fields(win_table.FitParams):
        p.add_argument(
            f"--{f.name.replace('_', '-')}", type=f.type, default=getattr(defaults, f.name)
        )
    args = p.parse_args()

    files = _slog_files(args.slogs)
    train = [f for f in files if not _held_out(f, args.holdout_every)]
    held = [f for f in files if _held_out(f, args.holdout_every)]
    params = win_table.FitParams(**{f.name: getattr(args, f.name) for f in fields(defaults)})

    samples = win_table.read_samples(train)
    grid = win_table.fit(samples, params)
    provenance = {
        "source": " ".join(str(r) for r in args.slogs),
        "policy": args.policy,
        "games": samples.games,
        "positions": len(samples),
        **{f.name: getattr(params, f.name) for f in fields(params)},
    }
    win_table.write(args.out, grid, provenance)
    print(f"fitted {len(samples)} positions from {samples.games} games -> {args.out}")

    if held:
        test = win_table.read_samples(held)
        macondo = win_table.log_loss(win_table.MACONDO_WIN_TABLE, test, macondo_format=True)
        print(f"held out: {len(test)} positions from {test.games} games")
        print(f"  log loss, fitted:  {win_table.log_loss(args.out, test):.5f}")
        print(f"  log loss, Macondo: {macondo:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
