#!/usr/bin/env python3
"""Move a renamed workload's tags under its new name.

A workload's name is its tags' directory (<mount>/tags/<workload>/<tag>), and
it is stored in each tag's task.json, in its workers' params/<worker>.json,
and in the key of the dashboard's control record of the tag (control.db,
"<workload>/<tag>"). A role's name is stored in the same params files, in the
workers of the control record, and in stats/<worker>.json. Renaming a
workload or a role in code leaves its tags under the old names, which the
dashboard no longer knows. This moves every tag of OLD under NEW and rewrites
each stored name, renaming any of its roles given with --role, and any data/
subdirectory given with --data-dir that the workload delivers into.

The control database's other tables are rebuilt from the records on every
reconcile pass (dashboard/control_db.py), so only the records are rewritten.
A pool or queue record that names OLD (a machine leased to one of its tags, a
tag waiting in the queue) is refused rather than rewritten: release the tag
first.

Run it with the dashboard stopped (it takes the dashboard's lock, and waits
for it, then refuses, if a dashboard holds it) and no worker of the tags
alive; start the dashboard afterwards. --dry-run says what would change, and
runs beside a dashboard.

Usage:
    ./py/scripts/rename_workload_tags.py leaf_eval_truth leaf_eval_hasty_rollouts \\
        --role truth=rollouts --data-dir truth=hasty_rollouts
"""

import argparse
import json
import os
import sys
from pathlib import Path

from scribblez.dashboard.api import acquire_control_lock
from scribblez.dashboard.control_store import ControlStore
from scribblez.paths import add_mount_root_argument
from util.argparse_ext import ArgumentDefaultsHelpFormatter

TASK = "task"
# The records that name tags only inside their bodies.
SHARED_RECORDS = ("pool", "queue")


def renamed(names: dict[str, str], name: str) -> str:
    return names.get(name, name)


def rewrite(stored: dict, workload: tuple[str, str], roles: dict[str, str]) -> dict:
    """`stored` (a task.json, a params or stats file, a control record) with
    the workload renamed where it names one, and its roles renamed: its own
    and its workers'."""
    old, new = workload
    out = dict(stored)
    if out.get("workload") == old:
        out["workload"] = new
    if "role" in out:
        out["role"] = renamed(roles, out["role"])
    if "workers" in out:
        out["workers"] = [w | {"role": renamed(roles, w["role"])} for w in out["workers"]]
    return out


def _write_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def stored_files(root: Path) -> list[Path]:
    """The tag's files that store its workload's or its roles' names."""
    files = [root / "task.json"] if (root / "task.json").is_file() else []
    for folder in ("params", "stats"):
        files.extend(sorted((root / folder).glob("*.json")))
    return files


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def task_record(store: ControlStore, root: Path, key: str) -> dict:
    """The tag's control record, or before the dashboard moved it into the
    store, its task.json."""
    body = store.get(TASK, key)
    if body is not None:
        return json.loads(body)
    task_file = root / "task.json"
    return json.loads(task_file.read_text()) if task_file.is_file() else {}


def refusal(store, root: Path, dest: Path, key: str, data_dirs: dict[str, str]) -> str | None:
    """Why the tag at `root` cannot move to `dest`, or None."""
    if dest.exists():
        return f"{dest} already exists"
    alive = [
        w["worker_id"]
        for w in task_record(store, root, key).get("workers", [])
        if _pid_alive(w.get("pid"))
    ]
    if alive:
        return f"workers still alive: {', '.join(alive)}; pause them first"
    for old, new in data_dirs.items():
        if (root / "data" / old).exists() and (root / "data" / new).exists():
            return f"both data/{old} and data/{new} exist"
    return None


def check_tags(store, mount_root: Path, tags: list[str], workload, data_dirs):
    """Exit, before any tag changes, if one of `tags` cannot move."""
    old, new = workload
    for tag in tags:
        root, dest = mount_root / "tags" / old / tag, mount_root / "tags" / new / tag
        why = refusal(store, root, dest, f"{old}/{tag}", data_dirs)
        if why:
            sys.exit(f"error: {old}/{tag}: {why}")


def move_tag(store, mount_root: Path, tag: str, workload, roles, data_dirs, dry_run: bool):
    """Move one tag of the old workload under the new one (check_tags first)."""
    old, new = workload
    root = mount_root / "tags" / old / tag
    dest = mount_root / "tags" / new / tag
    for path in stored_files(root):
        print(f"  {path.relative_to(root)}: rewrite")
        if not dry_run:
            _write_json(path, rewrite(json.loads(path.read_text()), workload, roles))
    for a, b in data_dirs.items():
        if (root / "data" / a).is_dir():
            print(f"  data/{a} -> data/{b}")
            if not dry_run:
                (root / "data" / a).rename(root / "data" / b)
    body = store.get(TASK, f"{old}/{tag}")
    print(f"  {root} -> {dest}" + ("; control record rekeyed" if body is not None else ""))
    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    root.rename(dest)
    if body is not None:
        with store.transaction():
            store.put(TASK, f"{new}/{tag}", json.dumps(rewrite(json.loads(body), workload, roles)))
            store.delete(TASK, f"{old}/{tag}")


def check_shared_records(store: ControlStore, old: str):
    """Exit if the pool or the queue names the old workload."""
    for kind in SHARED_RECORDS:
        body = store.get(kind, "")
        if body is not None and f'"{old}"' in body:
            sys.exit(f"error: the {kind} names {old}; release its tags from the {kind} first")


def pairs(values: list[str], flag: str) -> dict[str, str]:
    out = {}
    for value in values:
        a, sep, b = value.partition("=")
        if not (sep and a and b):
            sys.exit(f"error: {flag} takes OLD=NEW, got {value!r}")
        out[a] = b
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("old", help="the workload's old name")
    p.add_argument("new", help="its new name")
    p.add_argument("--role", action="append", default=[], help="a renamed role, OLD=NEW")
    p.add_argument(
        "--data-dir", action="append", default=[], help="a renamed data/ subdirectory, OLD=NEW"
    )
    p.add_argument("--dry-run", action="store_true", help="say what would change, change nothing")
    add_mount_root_argument(p)
    args = p.parse_args()
    mount_root = Path(args.mount_root)
    if not args.dry_run:
        acquire_control_lock(args.mount_root)
    store = ControlStore(mount_root)
    check_shared_records(store, args.old)
    roles, data_dirs = pairs(args.role, "--role"), pairs(args.data_dir, "--data-dir")
    tags = sorted(d.name for d in (mount_root / "tags" / args.old).glob("*") if d.is_dir())
    if not tags:
        print(f"no tags under {mount_root / 'tags' / args.old}")
        return 0
    workload = (args.old, args.new)
    check_tags(store, mount_root, tags, workload, data_dirs)
    for tag in tags:
        print(f"{args.old}/{tag} -> {args.new}/{tag}")
        move_tag(store, mount_root, tag, workload, roles, data_dirs, args.dry_run)
    if not args.dry_run:
        (mount_root / "tags" / args.old).rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
