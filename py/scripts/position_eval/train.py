#!/usr/bin/env python3
"""Headless CLI over the position-evaluation train role.

The master dashboard is the normal way to run training: create a position_eval
task, attach generator workers and the singleton trainer, and the dashboard's
generation scheduler fills generations from their staged chunks. This CLI
invokes the same train-role runner (scribblez/position_eval/trainer.py)
directly on a tag for debugging -- it consumes complete generations under the
tag exactly like the dashboard-launched trainer, so something must be filling
them (the dashboard server with generator workers attached to the same tag).

The flags are generated from the workload's params dataclass, so this CLI
cannot drift from the dashboard's task form.

Usage:
    ./py/scripts/position_eval/train.py -t mytag
"""

import argparse
import os
import socket
import sys

from cloud.sinks import LocalSink
from scribblez import params as params_mod
from scribblez import workloads
from scribblez.workloads.base import WorkerContext
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def main() -> int:
    spec = workloads.get("position_eval")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("-t", "--tag", required=True, help="Tag (per-tag artifact root).")
    p.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda).")
    params_mod.add_arguments(p, spec.params_cls)
    args = p.parse_args()
    params = params_mod.from_args(spec.params_cls, args)

    os.environ["SCZ_DEVICE"] = args.device
    role = spec.role("train")
    ctx = WorkerContext(
        spec=spec,
        role=role,
        tag=args.tag,
        params=params,
        worker_id=f"cli-{socket.gethostname()}",
        threads=0,
        max_cycles=0,
        sink=LocalSink(spec.data_dir(args.tag)),
    )
    return workloads.resolve(role.runner)(ctx)


if __name__ == "__main__":
    sys.exit(main())
