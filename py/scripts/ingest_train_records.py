#!/usr/bin/env python3
"""Ingest a tag's trainer records into its dashboard.db, once.

The dashboard server does this on every reconcile pass for every training
workload (generational/train_ingest.py), so it is never needed while the
server runs. It is for a trainer driven headlessly (the scripts/<workload>/
train.py CLIs) with no server up: the records sit under the tag's records/
until something writes them, and this is that something.

Usage:
    ./py/scripts/ingest_train_records.py -w position_eval -t mytag
"""

import argparse
import sys

from scribblez import workloads
from scribblez.dashboard import db
from scribblez.generational import train_ingest
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("-w", "--workload", required=True, choices=sorted(workloads.WORKLOADS))
    p.add_argument("-t", "--tag", required=True, help="Tag whose records to ingest.")
    args = p.parse_args()
    paths = workloads.get(args.workload).paths(args.tag)
    if not paths.records_dir.is_dir():
        print(f"no records under {paths.records_dir}")
        return 1
    conn = db.connect(paths.dashboard_db)
    try:
        written = train_ingest.ingest(paths, conn)
    finally:
        conn.close()
    print(f"ingested {len(written)} record(s) into {paths.dashboard_db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
