#!/usr/bin/env python3
"""Pull a tag's cloud-generated kill-test data down to the local mount dir.

Copies the tag's bucket prefix (slogs/ plus the workers' params/ manifests)
into <mount>/kill_test/<tag>/, merging with anything generated locally under
the same tag -- output filenames are nanosecond timestamps, so local and cloud
batches coexist in one directory. Never uploads or deletes anything; the
bucket remains the durable archive. Analysis (kill_test.py) then runs on the
merged local directory exactly as for purely local data.

Usage:
    ./py/scripts/cloud_sync.py -t hello            one sync
    ./py/scripts/cloud_sync.py -t hello --watch    resync every --interval sec
"""

import argparse
import sys
import time
from pathlib import Path

from cloud.credentials import load_credentials
from cloud.r2 import bucket_path, rclone
from scripts.generate_kill_test_data import MOUNT_ROOT
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def sync_once(r2, tag: str, dest: Path) -> int:
    res = rclone(r2, "copy", bucket_path(r2, "kill_test", tag), str(dest))
    if res.returncode != 0:
        print("sync failed", file=sys.stderr)
        return res.returncode
    pairs = sum(1 for s in (dest / "slogs").glob("*.slog") if s.with_suffix(".sobs").exists())
    print(f"{dest}: {pairs} complete pair(s)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("-t", "--tag", required=True)
    p.add_argument("--watch", action="store_true", help="keep syncing until Ctrl-C")
    p.add_argument("--interval", type=int, default=60, help="seconds between --watch syncs")
    args = p.parse_args()

    r2 = load_credentials().r2
    dest = MOUNT_ROOT / "kill_test" / args.tag
    dest.mkdir(parents=True, exist_ok=True)
    while True:
        rc = sync_once(r2, args.tag, dest)
        if rc != 0 or not args.watch:
            return rc
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
