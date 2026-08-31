#!/usr/bin/env python3
"""Bring a live tag's stored params forward after a workload's param schema changes.

When a workload param is renamed, dropped, or added-without-a-default, every stored
copy of a live tag's params stops validating and the scheduler wedges with
"unknown parameter 'X'" (or a missing-field error) on each reconcile / dispatch.
The params are snapshotted in more than one place -- the tag's task.json and one
per-worker provenance manifest under params/<worker_id>.json -- so hand-editing
task.json alone leaves the tag broken.

This applies --rename / --drop / --set to the params in ALL of a tag's stored
files, validates the result against the workload's CURRENT schema, and writes each
back atomically (tmp + os.replace). Operations apply in that order per file, so a
rename can feed a later set. A file already matching the target schema is left
untouched and reported.

Refuses to run while any of the tag's workers is alive -- their processes hold the
old params and would rewrite a snapshot out from under you. Pause the workers and
stop the dashboard first; restart the dashboard afterwards so it reloads the
migrated tags.

Usage:
    ./py/scripts/migrate_tag_params.py position_eval --tag footprints-official \
        --drop mask_placement

    ./py/scripts/migrate_tag_params.py evidence_trajectories --all \
        --rename traj_rollouts=rollouts --set temperature=1.0 --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

from scribblez import params as params_mod
from scribblez import workloads
from scribblez.paths import DEFAULT_MOUNT_ROOT, TagPaths
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _write_json(path: Path, obj: dict):
    """Atomic write: a crash never leaves a half-written params file the scheduler
    would then choke on."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def apply_ops(params: dict, renames: dict, drops: list, sets: dict) -> tuple[dict, bool]:
    """A new params dict with renames, then drops, then sets applied, plus whether
    anything changed. Pure -- the file I/O and validation live in the caller."""
    out = dict(params)
    changed = False
    for old, new in renames.items():
        if old in out:
            out[new] = out.pop(old)
            changed = True
    for key in drops:
        if key in out:
            del out[key]
            changed = True
    for key, value in sets.items():
        if out.get(key) != value:
            out[key] = value
            changed = True
    return out, changed


def param_files(root: Path) -> list[Path]:
    """Every stored file under a tag whose ["params"] holds the workload params:
    task.json, then each per-worker provenance manifest, in a stable order."""
    files = [root / "task.json"] if (root / "task.json").is_file() else []
    params_dir = root / "params"
    if params_dir.is_dir():
        files.extend(sorted(params_dir.glob("*.json")))
    return files


def alive_workers(root: Path) -> list[str]:
    task_file = root / "task.json"
    if not task_file.is_file():
        return []
    task = json.loads(task_file.read_text())
    return [w["worker_id"] for w in task.get("workers", []) if _pid_alive(w.get("pid"))]


def migrate_tag(spec, tag, mount_root, renames, drops, sets, dry_run) -> int:
    """Migrate one tag's files; return the count of files changed (0 if clean)."""
    root = TagPaths(tag, spec.name, mount_root).root
    if not root.is_dir():
        sys.exit(f"error: no such tag dir {root}")
    alive = alive_workers(root)
    if alive and not dry_run:
        # A live worker holds the old params and would rewrite its snapshot on top
        # of the migration. A dry run touches nothing, so it may still preview.
        sys.exit(
            f"error: {tag}: worker(s) still running {alive}; "
            "pause them and stop the dashboard first"
        )

    files = param_files(root)
    if not files:
        print(f"{tag}: no param files")
        return 0

    print(f"{tag}:")
    if alive:
        print(f"  (warning: worker(s) {alive} still running; writes would be refused)")
    changed_count = 0
    for f in files:
        doc = json.loads(f.read_text())
        params = doc.get("params")
        rel = f.relative_to(root)
        if not isinstance(params, dict):
            print(f"  {rel}: no params block, skipped")
            continue
        new_params, changed = apply_ops(params, renames, drops, sets)
        # Validate against the CURRENT schema -- the whole point is that the tag
        # loads again, so refuse to write anything that still would not.
        try:
            params_mod.validate(spec.params_cls, new_params)
        except params_mod.ParamsError as e:
            sys.exit(f"error: {tag}/{rel}: result still invalid against {spec.name} schema: {e}")
        if not changed:
            print(f"  {rel}: already clean")
            continue
        changed_count += 1
        if dry_run:
            print(f"  {rel}: would migrate")
            continue
        doc["params"] = new_params
        _write_json(f, doc)
        print(f"  {rel}: migrated")
    return changed_count


def _split_pair(text: str, flag: str) -> tuple[str, str]:
    if "=" not in text:
        sys.exit(f"error: {flag} expects KEY=VALUE, got {text!r}")
    key, value = text.split("=", 1)
    return key, value


def _json_value(text: str):
    """Parse a --set value as JSON (so true/1/1.5 get their real types), falling
    back to the raw string for a bare word. validate() coerces from there."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def resolve_tags(spec, tags: list, want_all: bool, mount_root) -> list[str]:
    tags_root = TagPaths("_", spec.name, mount_root).root.parent
    if want_all:
        if tags:
            sys.exit("error: pass either --all or --tag, not both")
        if not tags_root.is_dir():
            return []
        return sorted(d.name for d in tags_root.iterdir() if d.is_dir())
    if not tags:
        sys.exit("error: no tags; pass --tag NAME (repeatable) or --all")
    return tags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("workload", help="workload name, e.g. position_eval")
    ap.add_argument("--tag", action="append", default=[], help="tag to migrate (repeatable)")
    ap.add_argument("--all", action="store_true", help="every tag under the workload")
    ap.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                    help="rename a param (repeatable)")
    ap.add_argument("--drop", action="append", default=[], metavar="PARAM",
                    help="drop a param (repeatable)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="sets",
                    help="set a param, value parsed as JSON then coerced (repeatable)")
    ap.add_argument("--mount-root", default=str(DEFAULT_MOUNT_ROOT))
    ap.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = ap.parse_args()

    spec = workloads.get(args.workload)  # asserts a known workload
    renames = dict(_split_pair(p, "--rename") for p in args.rename)
    sets = {k: _json_value(v) for k, v in (_split_pair(p, "--set") for p in args.sets)}
    if not (renames or args.drop or sets):
        sys.exit("error: nothing to do; pass --rename / --drop / --set")

    tags = resolve_tags(spec, args.tag, args.all, args.mount_root)
    if not tags:
        print("no tags found")
        return

    total = sum(migrate_tag(spec, t, args.mount_root, renames, args.drop, sets, args.dry_run)
                for t in tags)
    if args.dry_run:
        print(f"\ndry run: {total} file(s) would change; nothing written")
    elif total:
        print(f"\ndone: {total} file(s) migrated. Restart the dashboard to reload the tags.")
    else:
        print("\nnothing to change; all files already match the schema")


if __name__ == "__main__":
    main()
