#!/usr/bin/env python3
"""Pull a tag's cloud-delivered data from the bucket into the local tag dir.

Copies the prefixes remote workers write to: the workload's sync_data_dirs
(e.g. kill_test's slogs/, the training workloads' staging/) plus the workers'
stats/ and params/ records. Files merge with anything generated locally under
the same tag; output filenames carry per-worker suffixes, so they never
collide. With --trainer-outputs it also pulls what a remotely running trainer
delivers (scribblez/paths.py TRAINER_OUTPUT_*): records, exports, the rolling
checkpoint and the train_state.json cursor. It never uploads, and it deletes
local files only in the mirrored export dir (MIRRORED_OUTPUT_DIRS); the
bucket stays the durable archive.

Prefixes this host itself writes to the bucket, such as the generation dirs the
scheduler publishes, are not pulled: the local mount already holds them.

Usage:
    ./py/scripts/cloud_sync.py -t hello            one sync
    ./py/scripts/cloud_sync.py -t hello --watch    resync every --interval sec
"""

import argparse
import sys
import time

from cloud.credentials import load_credentials
from cloud.r2 import bucket_path, rclone
from scribblez import workloads
from scribblez.paths import TRAINER_OUTPUT_DIRS, TRAINER_OUTPUT_FILES, TRAINER_OUTPUT_IMMUTABLE
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def _targets(spec: workloads.WorkloadSpec, tag: str, trainer_outputs: bool):
    """(bucket sub-prefix, local dir, extra rclone flags) for every directory
    a sync pulls. Prefixes whose objects never change are compared by size
    alone: on an S3-style remote the listing carries no modtime, so the
    default comparison would ask for every unchanged export one by one."""
    paths = spec.paths(tag)
    targets = [(sub, paths.data_dir / sub, ()) for sub in spec.sync_data_dirs]
    targets += [("stats", paths.stats_dir, ()), ("params", paths.root / "params", ())]
    if trainer_outputs:
        targets += [
            (sub, paths.root / sub, ("--size-only",) if sub in TRAINER_OUTPUT_IMMUTABLE else ())
            for sub in TRAINER_OUTPUT_DIRS
        ]
    return targets


# Trainer-output directories the local copy MIRRORS rather than accumulates:
# a trainer that prunes its exports (move_set_eval's retention) deletes them
# from the bucket, and a copy that only ever added would keep every one it
# had pulled before the prune. rclone sync deletes locally what the bucket
# no longer has; the directory is the trainer's alone, so nothing else's is
# at risk.
MIRRORED_OUTPUT_DIRS = ("models",)


def _pull_file(r2, spec: workloads.WorkloadSpec, tag: str, name: str) -> int:
    """Pull one root-level file the trainer rewrites in place, if the bucket
    has it yet (a trainer that has not checkpointed has published nothing)."""
    src = bucket_path(r2, spec.name, tag, name)
    if not rclone(r2, "lsf", src, capture=True).stdout.strip():
        return 0
    return rclone(r2, "copyto", src, str(spec.paths(tag).root / name)).returncode


def sync_once(r2, spec: workloads.WorkloadSpec, tag: str, trainer_outputs: bool = False) -> int:
    paths = spec.paths(tag)
    targets = _targets(spec, tag, trainer_outputs)
    for sub, dest, flags in targets:
        dest.mkdir(parents=True, exist_ok=True)
        verb = "sync" if trainer_outputs and sub in MIRRORED_OUTPUT_DIRS else "copy"
        res = rclone(r2, verb, *flags, bucket_path(r2, spec.name, tag, sub), str(dest))
        if res.returncode != 0:
            print(f"sync of {sub}/ failed", file=sys.stderr)
            return res.returncode
    if trainer_outputs:
        for name in TRAINER_OUTPUT_FILES:
            rc = _pull_file(r2, spec, tag, name)
            if rc != 0:
                print(f"sync of {name} failed", file=sys.stderr)
                return rc
    counts = ", ".join(
        f"{sub}: {sum(1 for _ in dest.iterdir()) if dest.is_dir() else 0}"
        for sub, dest, _ in targets
    )
    print(f"{paths.root}: {counts}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("-t", "--tag", required=True, help="tag to sync")
    p.add_argument(
        "--workload",
        choices=sorted(workloads.WORKLOADS),
        default="kill_test",
        help="tag's workload",
    )
    p.add_argument("--watch", action="store_true", help="keep syncing until Ctrl-C")
    p.add_argument("--interval", type=int, default=60, help="seconds between --watch syncs")
    p.add_argument(
        "--trainer-outputs",
        action="store_true",
        help="also pull what a trainer running elsewhere delivers "
        "(records, exports, checkpoint, cursor)",
    )
    args = p.parse_args()

    spec = workloads.get(args.workload)
    r2 = load_credentials().r2
    while True:
        rc = sync_once(r2, spec, args.tag, args.trainer_outputs)
        if rc != 0 or not args.watch:
            return rc
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
