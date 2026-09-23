#!/usr/bin/env python3
"""Generate data for the sim-evidence kill-test on this machine, without the dashboard.

Runs kill_test generation cycles until Ctrl-C, accumulating .slog/.sobs pairs
under <mount>/tags/kill_test/<tag>/data/slogs. The cycle itself and its
parameters live in scribblez/workloads/kill_test.py, shared with the dashboard's
workers; the parameter flags are generated from KillTestParams.

A rerun resumes where the last run stopped. Train the experiment's arms on the
accumulated pairs with scripts/kill_test.py -t <tag>; that may run while this
keeps generating, since it uses whatever complete pairs exist. See the
kill-test section of docs/plans/sim_residual_feedback.md.

Usage:
    ./py/scripts/generate_kill_test_data.py -t apple
"""

import argparse
import sys

from scribblez import params as params_mod
from scribblez.hardware import default_thread_count
from scribblez.workloads.kill_test import KillTestParams, run_one_cycle, slog_dir
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "-t",
        "--tag",
        required=True,
        help="tag to accumulate data under (its data/slogs dir)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=default_thread_count(),
        help="engine threads for self-play and sims (default: all logical processors)",
    )
    params_mod.add_arguments(p, KillTestParams)
    args = p.parse_args()
    params = params_mod.from_args(KillTestParams, args)

    out_dir = slog_dir(args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Accumulating kill-test data under {out_dir} (Ctrl-C to stop)")

    batch = 0
    try:
        while True:
            batch += 1
            existing = len(list(out_dir.glob("*.sobs")))
            print(f"\n=== cycle {batch} ({existing} .sobs files so far) ===")
            result = run_one_cycle(out_dir, params, args.threads)
            if result.returncode != 0:
                return result.returncode
    except KeyboardInterrupt:
        # An interrupt loses at most the in-flight cycle's unfinished work:
        # .slog batches and .sobs sidecars both land atomically, and the next
        # run re-derives any missing sidecar from its .slog.
        print("\nstopped; data is consistent and generation resumes on rerun")
    return 0


if __name__ == "__main__":
    sys.exit(main())
