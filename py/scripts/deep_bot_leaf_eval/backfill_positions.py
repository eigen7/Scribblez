#!/usr/bin/env python3
"""Add the known positions a tag's analyses lack, without the dashboard: the
backfill role's runner (workloads/positions_backfill.py) on chosen
generations, for a quick look or a tag with few analyses. A deep_bot_leaf_eval
generation is an export's, a leaf_bot_generation one a release's. Not while
the tag's backfill slot runs, which would rewrite the same files.

    py/scripts/deep_bot_leaf_eval/backfill_positions.py \\
        --workload deep_bot_leaf_eval --tag v8 --generations 1270

Without --generations, every analyzed generation, newest first. Restart the
dashboard afterwards only if it predates the backfill: it reads a rewritten
analysis again from then on.
"""

import argparse
import shutil
import socket
import sys

from cloud.sinks import LocalSink
from scribblez import params as params_mod
from scribblez.dashboard import tasks
from scribblez.paths import add_mount_root_argument
from scribblez.workloads import deep_bot_leaf_eval, leaf_bot_generation, positions_backfill
from scribblez.workloads.base import WorkerContext
from util.argparse_ext import ArgumentDefaultsHelpFormatter

EXPORTS = {
    deep_bot_leaf_eval.SPEC.name: (deep_bot_leaf_eval.SPEC, deep_bot_leaf_eval.analysis_exports),
    leaf_bot_generation.SPEC.name: (leaf_bot_generation.SPEC, leaf_bot_generation.analysis_exports),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--workload", required=True, choices=sorted(EXPORTS))
    p.add_argument("--tag", required=True)
    p.add_argument("--generations", type=int, nargs="+", help="default: every analyzed one")
    p.add_argument("--threads", type=int, default=28, help="the engine's CPU threads")
    add_mount_root_argument(p)
    args = p.parse_args()
    spec, exports = EXPORTS[args.workload]
    raw = tasks.read_params(spec, args.tag, args.mount_root)
    if raw is None:
        print(f"no {args.workload} tag '{args.tag}' under {args.mount_root}", file=sys.stderr)
        return 1
    ctx = WorkerContext(
        spec=spec,
        role=spec.role(positions_backfill.ROLE),
        tag=args.tag,
        params=params_mod.validate(spec.params_cls, raw),
        worker_id=f"cli-{socket.gethostname()}",
        threads=args.threads,
        max_cycles=0,
        sink=LocalSink(spec.data_dir(args.tag, args.mount_root)),
        mount_root=args.mount_root,
    )
    # No stats record: the tag's Stats tab lists its slots, which this is not.
    for _ in positions_backfill.backfill_all(ctx, exports, args.generations):
        pass
    shutil.rmtree(ctx.tag_paths().work_dir(ctx.worker_id), ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
