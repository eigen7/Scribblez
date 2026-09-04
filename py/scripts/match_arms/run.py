#!/usr/bin/env python3
"""Headless CLI over the match_arms arms role.

The master dashboard is the normal way to run an arms experiment: create a
match_arms task (its frozen params define the arms, the opponent, and the pair
budget) and attach the singleton arms worker. This CLI invokes the same role
runner directly on a tag for headless runs and debugging; results land in the
tag's dashboard.db either way, so the Arms tab renders them identically.

The flags are generated from the workload's params dataclass, so this CLI
cannot drift from the dashboard's task form.

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
    p.add_argument("-t", "--tag", required=True, help="Tag (per-tag artifact root).")
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
