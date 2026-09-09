#!/usr/bin/env python3
"""Pull a tag's cloud-delivered data down to the local mount dir.

Copies the workload's inbound bucket prefixes -- its declared data dirs (e.g.
kill_test's slogs/, the training workloads' staging/) plus the workers'
stats/ and params/ records -- into the tag's local dir, merging with anything
generated locally under the same tag (output filenames carry per-worker
suffixes, so local and cloud files coexist). With --trainer-outputs, also
what a trainer running elsewhere delivers (scribblez/paths.py
TRAINER_OUTPUT_*): its records, exports and rolling checkpoint, and the
cursor it publishes. Never uploads or deletes anything; the bucket remains
the durable archive.

Deliberately NOT synced: prefixes the controller host itself maintains in the
bucket (the generation dirs the scheduler's ingest mirroring and publishing
populate) -- pulling those would re-download data the local mount already
holds.

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
        res = rclone(r2, "copy", *flags, bucket_path(r2, spec.name, tag, sub), str(dest))
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
    p.add_argument("-t", "--tag", required=True)
    p.add_argument("--workload", choices=sorted(workloads.WORKLOADS), default="kill_test")
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
