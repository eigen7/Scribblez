#!/usr/bin/env python3
"""Roll the dashboard's control store back into JSON files
(docs/plans/dashboard_state_model.md §10).

Writes every tag's whole record to its task.json and the pool and queue to
pool.json and queue.json, as the dashboard kept them before the control store,
then empties the store. Code from before the control store then runs on the
mount as it was left; a later start of the current dashboard imports the files
afresh.

Run it with the dashboard stopped; it refuses otherwise.

Usage:
    ./py/scripts/export_control_db.py
    ./py/scripts/export_control_db.py --mount-root /path/to/mount
"""

import argparse

from scribblez.dashboard.api import acquire_control_lock
from scribblez.dashboard.workers import WorkerManager
from scribblez.paths import add_mount_root_argument
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    add_mount_root_argument(p)
    args = p.parse_args()
    acquire_control_lock(args.mount_root)
    for written in WorkerManager(args.mount_root).export_json_stores():
        print(f"wrote {written}")


if __name__ == "__main__":
    main()
