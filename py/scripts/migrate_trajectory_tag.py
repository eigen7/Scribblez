#!/usr/bin/env python3
"""Convert a trajectory-generating move_set_eval tag into an evidence_trajectories tag.

A one-off migration for tags created when evidence trajectories were a
move_set_eval option (`traj_every` > 0). Such a tag's data is exactly what
evidence_trajectories produces, .slog/.sobs/.mset triples in data/slogs with no
workload identity in any file, so migrating it only moves the tag directory to
the new workload and rewrites task.json:

  * the workload name;
  * the params: those listed in RENAMES carry over (the traj_* ones renamed),
    the rest are dropped;
  * the generate worker slots, reset to paused and unlaunched, since their
    processes ran with the old workload's environment and must be respawned.
    Other roles' slots are dropped.

Refuses to run while any of the tag's workers is alive; pause them first.

Usage:
    ./py/scripts/migrate_trajectory_tag.py trajectories
"""

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict

from scribblez import params as params_mod
from scribblez.paths import DEFAULT_MOUNT_ROOT, EVIDENCE_TRAJECTORIES, MOVE_SET_EVAL, TagPaths
from scribblez.workloads import evidence_trajectories
from util.argparse_ext import ArgumentDefaultsHelpFormatter

# move_set_eval param -> evidence_trajectories param, for those that carry over.
RENAMES = {
    "teacher_model": "teacher_model",
    "proposer_model": "proposer_model",
    "games_per_batch": "games_per_batch",
    "traj_positions_per_game": "positions_per_game",
    "traj_rollouts": "rollouts",
    "traj_proposals_min": "on_policy_min",
    "traj_proposals_max": "on_policy_max",
    "traj_temperature": "temperature",
    "quota_top": "quota_top",
    "quota_mid": "quota_mid",
    "quota_tail": "quota_tail",
    "quota_exchange": "quota_exchange",
    "mid_rank_limit": "mid_rank_limit",
    "hasty_temperature": "hasty_temperature",
    "hasty_top_k": "hasty_top_k",
    "random_opening_mean": "random_opening_mean",
    "face_up_leaves": "face_up_leaves",
    "target_pairs": "target_pairs",
}


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def migrate_params(old: dict) -> dict:
    """The new workload's params from the old tag's, validated against its schema."""
    if old.get("traj_every", 0) <= 0:
        sys.exit("error: this tag never generated trajectories (traj_every = 0)")
    new = {RENAMES[k]: v for k, v in old.items() if k in RENAMES}
    return asdict(params_mod.validate(evidence_trajectories.SPEC.params_cls, new))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("tag", help="the move_set_eval tag to migrate")
    ap.add_argument("--mount-root", default=str(DEFAULT_MOUNT_ROOT))
    args = ap.parse_args()

    src = TagPaths(args.tag, MOVE_SET_EVAL, args.mount_root).root
    dst = TagPaths(args.tag, EVIDENCE_TRAJECTORIES, args.mount_root).root
    task_file = src / "task.json"
    if not task_file.is_file():
        sys.exit(f"error: {task_file} not found")
    if dst.exists():
        sys.exit(f"error: {dst} already exists")
    task = json.loads(task_file.read_text())
    if task["workload"] != MOVE_SET_EVAL:
        sys.exit(f"error: tag's workload is {task['workload']!r}, not {MOVE_SET_EVAL!r}")
    alive = [w["worker_id"] for w in task["workers"] if _pid_alive(w.get("pid"))]
    if alive:
        sys.exit(f"error: worker(s) still running: {alive}; pause them first")

    task["workload"] = EVIDENCE_TRAJECTORIES
    task["params"] = migrate_params(task["params"])
    task["workers"] = [
        {**w, "desired_state": "paused", "launched": False, "pid": None, "observed_running": False}
        for w in task["workers"]
        if w["role"] == "generate"
    ]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    (dst / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    print(f"moved {src} -> {dst}")
    print("params:", json.dumps(task["params"], indent=2))


if __name__ == "__main__":
    main()
