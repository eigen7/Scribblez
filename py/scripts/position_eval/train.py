#!/usr/bin/env python3
"""Run the position_eval trainer on a tag without the dashboard, for debugging.

Normally the dashboard runs training: create a position_eval task, attach
generator workers and the singleton trainer, and the dashboard's generation
scheduler assembles generations from the workers' staged chunks. This CLI calls
the same train-role runner (scribblez/position_eval/trainer.py) directly. It
trains on the tag's complete generations exactly as the dashboard's trainer
does, so something must still be producing them: a dashboard server with
generator workers attached to the same tag.

The workload flags are generated from its params dataclass, so they always
match the dashboard's task form.

The trainer writes its metrics as records under the tag's records/ dir. A
running dashboard server ingests them into dashboard.db; with none up, run
scripts/ingest_train_records.py afterwards.

Usage:
    ./py/scripts/position_eval/train.py -t mytag
"""

import argparse
import os
import socket
import sys

from cloud.sinks import LocalSink
from scribblez import workloads
from scribblez.workloads.base import WorkerContext
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def main() -> int:
    spec = workloads.get("position_eval")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "-t", "--tag", required=True, help="Tag to train; its directory holds all outputs."
    )
    p.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda).")
    spec.add_cli_arguments(p)
    args = p.parse_args()
    params = spec.params_from_args(args)

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
