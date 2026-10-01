#!/usr/bin/env python3
"""Pull a tag's cloud-delivered data from the bucket into the local tag dir.

Copies the prefixes rented generators deliver to: the workload's
sync_data_dirs (e.g. kill_test's slogs/, the training workloads' staging/).
Files merge with anything generated locally under the same tag; output
filenames carry per-worker suffixes, so they never collide. Workers' records
and trainers' outputs do not come this way: the controller collects them over
ssh (WorkerManager._transfer_target). It never uploads or deletes; the bucket
copy stays until its generator's data reaches the controller another way.

A tag whose data home ingests bucket staging itself (generational/data_home.py)
gets no watcher at all (WorkerManager._ensure_sync): a copy here would bring
back chunks the data home has already moved into generations.

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
from scribblez.paths import TagPaths, add_mount_root_argument
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def sync_once(r2, spec: workloads.WorkloadSpec, paths: TagPaths) -> int:
    for sub in spec.sync_data_dirs:
        dest = paths.data_dir / sub
        dest.mkdir(parents=True, exist_ok=True)
        res = rclone(r2, "copy", bucket_path(r2, spec.name, paths.tag, sub), str(dest))
        if res.returncode != 0:
            print(f"sync of {sub}/ failed", file=sys.stderr)
            return res.returncode
    counts = ", ".join(
        f"{sub}: {sum(1 for _ in (paths.data_dir / sub).iterdir())}" for sub in spec.sync_data_dirs
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
    add_mount_root_argument(p)
    p.add_argument("--watch", action="store_true", help="keep syncing until Ctrl-C")
    p.add_argument("--interval", type=int, default=60, help="seconds between --watch syncs")
    args = p.parse_args()

    spec = workloads.get(args.workload)
    r2 = load_credentials().r2
    while True:
        rc = sync_once(r2, spec, spec.paths(args.tag, args.mount_root))
        if rc != 0 or not args.watch:
            return rc
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
