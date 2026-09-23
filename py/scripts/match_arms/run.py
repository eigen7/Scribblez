#!/usr/bin/env python3
"""Run a match_arms experiment on a tag without the dashboard.

Normally an arms experiment runs from the master dashboard: create a
match_arms task, whose params define the arms, the shared opponent and the
number of game pairs, and attach its singleton arms worker. This CLI calls the
same role runner directly, for headless runs and debugging. Results land in
the tag's dashboard.db either way, so the dashboard's Arms tab shows them.

The workload flags are generated from its params dataclass (MatchArmsParams),
so they always match the dashboard's task form.

Usage:
    ./py/scripts/match_arms/run.py -t sweep1 --threads 8 \
        --arms "k5=--type=neural-sim --model=m.onnx --sim-top-k=5; ..." \
        --opponent "--type=sim"
"""

import argparse
import socket
import sys

from cloud.sinks import LocalSink
from scribblez import workloads
from scribblez.workloads.base import WorkerContext
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def main() -> int:
    spec = workloads.get("match_arms")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "-t", "--tag", required=True, help="Tag to run under; its directory holds all outputs."
    )
    p.add_argument("--threads", type=int, default=8, help="Game threads per match round.")
    spec.add_cli_arguments(p)
    args = p.parse_args()
    params = spec.params_from_args(args)

    role = spec.role("arms")
    ctx = WorkerContext(
        spec=spec,
        role=role,
        tag=args.tag,
        params=params,
        worker_id=f"cli-{socket.gethostname()}",
        threads=args.threads,
        max_cycles=0,
        sink=LocalSink(spec.data_dir(args.tag)),
    )
    return workloads.resolve(role.runner)(ctx)


if __name__ == "__main__":
    sys.exit(main())
