#!/usr/bin/env python3
"""Sample a frozen evaluation subset from a tag's held-out test split.

Usage:
    ./py/scripts/post_move_value/sample_test_subset.py -t mytag -n 12

Writes tags/<tag>/test-subset/positions.slog (a standalone .slog holding the
sampled positions) plus pos-NN.txt ASCII dumps. The structural probes read this
.slog directly. train.py builds it automatically if missing; run this to
(re)create or resize it on demand.
"""

import argparse
import sys

from scribblez.paths import POST_MOVE_VALUE, TagPaths
from scribblez.post_move_value.eval.sampling import build_test_subset
from scribblez.post_move_value.eval.web_render import render_position_images
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample an evaluation subset .slog from the test split.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--tag", required=True, help="Tag (per-tag artifact root).")
    parser.add_argument("-n", "--num-positions", type=int, default=12, help="Positions to sample.")
    args = parser.parse_args()

    paths = TagPaths(args.tag, POST_MOVE_VALUE)
    if not paths.test_dir.exists() or not any(paths.test_dir.glob("*.slog")):
        print(f"No test split at {paths.test_dir}.", file=sys.stderr)
        return 1

    n = build_test_subset(paths.test_dir, paths.test_subset_slog, num_positions=args.num_positions)
    render_position_images(paths.test_subset_slog, paths.test_subset_dir)
    print(f"Wrote {n} positions to {paths.test_subset_slog}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
