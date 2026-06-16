#!/usr/bin/env python3
"""Sample a frozen evaluation subset from a tag's held-out test split.

Usage:
    python -m scripts.sample_test_subset -t mytag -n 12

Writes tags/<tag>/test-subset/positions.slog (a standalone .slog holding the
sampled positions) plus pos-NN.txt ASCII dumps. The structural probes read this
.slog directly. train.py builds it automatically if missing; run this to
(re)create or resize it on demand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Importable both as a module (python -m scripts.sample_test_subset) and directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scribblez.eval.probe_eval import write_position_dumps
from scribblez.eval.sampling import build_test_subset
from scribblez.paths import TagPaths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample an evaluation subset .slog from the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--tag", required=True, help="Tag (per-tag artifact root).")
    parser.add_argument("-n", "--num-positions", type=int, default=12, help="Positions to sample.")
    parser.add_argument("--mount-root", type=str, default="/workspace/mount", help="tags/ root.")
    args = parser.parse_args()

    paths = TagPaths(args.tag, args.mount_root)
    if not paths.test_dir.exists() or not any(paths.test_dir.glob("*.slog")):
        print(f"No test split at {paths.test_dir}.", file=sys.stderr)
        return 1

    n = build_test_subset(paths.test_dir, paths.test_subset_slog, num_positions=args.num_positions)
    write_position_dumps(paths.test_subset_slog, paths.test_subset_dir)
    print(f"Wrote {n} positions to {paths.test_subset_slog} (+ pos-NN.txt dumps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
