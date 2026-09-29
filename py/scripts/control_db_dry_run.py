#!/usr/bin/env python3
"""Dry-run the control database's migration (docs/plans/dashboard_state_model.md
§10) against a mount root, reading it and writing nothing there.

Imports the root's task.json, pool.json and queue.json into a scratch control
database, then prints every tag's projected state and every finding: rows of
the decision table the migration would refuse, constraint violations, and
disagreements between the projection and what the JSON stores say today. Run
it against the live mount before the database becomes authoritative (PR 3);
every finding must be settled first.

Usage:
    ./py/scripts/control_db_dry_run.py
    ./py/scripts/control_db_dry_run.py --mount-root /path/to/copy --out /tmp/control.db
"""

import argparse
import sys
import tempfile
from pathlib import Path

from scribblez.dashboard import control_db
from scribblez.dashboard.workers import WorkerManager
from scribblez.paths import add_mount_root_argument
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    add_mount_root_argument(p)
    p.add_argument(
        "--out", type=Path, default=None, help="where to write the database (default: a temp file)"
    )
    args = p.parse_args()

    manager = WorkerManager(args.mount_root)
    with tempfile.TemporaryDirectory() as scratch:
        conn = control_db.connect(args.out or Path(scratch) / "control.db")
        findings = control_db.import_stores(conn, manager)
        projected = control_db.project(conn)
        conn.close()
    legacy = control_db.legacy_states(manager)
    for (workload, tag), state in sorted(projected.items()):
        agrees = (
            ""
            if legacy.get((workload, tag)) == state
            else f"   (stores: {legacy.get((workload, tag))})"
        )
        print(f"{workload}/{tag}: {state}{agrees}")
    print()
    for f in findings:
        print(f"finding: {f}")
    disagreements = [k for k in projected if legacy.get(k) != projected[k]]
    print(f"\n{len(projected)} tags, {len(findings)} findings, {len(disagreements)} disagreements")
    return 1 if findings or disagreements else 0


if __name__ == "__main__":
    sys.exit(main())
