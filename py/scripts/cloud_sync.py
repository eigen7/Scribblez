#!/usr/bin/env python3
"""Pull a tag's cloud-generated data down to the local mount dir.

Copies the workload's inbound bucket prefixes -- its declared data dirs (e.g.
kill_test's slogs/, the training workloads' staging/) plus the workers'
stats/ and params/ records -- into the tag's local dir, merging with anything
generated locally under the same tag (output filenames carry per-worker
suffixes, so local and cloud files coexist). Never uploads or deletes
anything; the bucket remains the durable archive.

Deliberately NOT synced: prefixes the controller host itself maintains in the
bucket (the generation dirs the scheduler's ingest mirroring populates) --
pulling those would re-download data the local mount already holds.

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
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def sync_once(r2, spec: workloads.WorkloadSpec, tag: str) -> int:
    paths = spec.paths(tag)
    targets = [(sub, paths.data_dir / sub) for sub in spec.sync_data_dirs]
    targets += [("stats", paths.stats_dir), ("params", paths.root / "params")]
    for sub, dest in targets:
        dest.mkdir(parents=True, exist_ok=True)
        res = rclone(r2, "copy", bucket_path(r2, spec.name, tag, sub), str(dest))
        if res.returncode != 0:
            print(f"sync of {sub}/ failed", file=sys.stderr)
            return res.returncode
    counts = ", ".join(
        f"{sub}: {sum(1 for _ in dest.iterdir()) if dest.is_dir() else 0}" for sub, dest in targets
    )
    print(f"{paths.root}: {counts}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("-t", "--tag", required=True)
    p.add_argument("--workload", choices=sorted(workloads.WORKLOADS), default="kill_test")
    p.add_argument("--watch", action="store_true", help="keep syncing until Ctrl-C")
    p.add_argument("--interval", type=int, default=60, help="seconds between --watch syncs")
    args = p.parse_args()

    spec = workloads.get(args.workload)
    r2 = load_credentials().r2
    while True:
        rc = sync_once(r2, spec, args.tag)
        if rc != 0 or not args.watch:
            return rc
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
