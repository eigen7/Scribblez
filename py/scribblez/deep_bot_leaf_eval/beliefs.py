"""The .belief sidecars the leaf's belief-mode rows read
(engine/include/data/belief_log.h; docs/plans/deep_bot_belief_leaf.md, "The
belief sidecar"): one beside each .slog, under the tag's pinned leave
inference model.

The trainer writes them, since they need a GPU and the generators have none:
for each generation of its window before training on it, after the .forgone
sidecars the tool reads (leave_inference/forgone.py), and for the held-out
games, which also store 16 of the opponent's leaves per turn as DeepBot draws
them. A directory whose every .slog has its .belief is done: the tool writes
each file atomically, after the whole directory's .forgone sidecars.

A generation can arrive without them: the controller's durable copy is pulled
when the generation completes, so a home seeded from it lacks them, and they
are written again there.
"""

import subprocess
from pathlib import Path

from scribblez.leave_inference import forgone
from scribblez.paths import ENGINE_DIR

SUFFIX = ".belief"
TOOL = ENGINE_DIR / "belief_sidecar_tool"
# The opponent's leaves the held-out sidecars store per turn: DeepBot's
# default match_opp_leaves.
HOLDOUT_DRAWS = 16


def write_sidecars(slog_dir: Path, leave_model: Path, threads: int, draws: int = 0):
    """Write a fresh sidecar beside every .slog in `slog_dir`, each storing
    `draws` leaves per turn. Raises subprocess.CalledProcessError if the tool
    fails."""
    cmd = [str(TOOL), "--slog-dir", str(slog_dir), "--leave-inference", str(leave_model)]
    cmd += ["--threads", str(threads), "--draws", str(draws)]
    subprocess.run(cmd, check=True)


def complete(slog_dir: Path) -> bool:
    """Whether every .slog in `slog_dir` has its .belief."""
    return all(p.with_suffix(SUFFIX).is_file() for p in slog_dir.glob("*.slog"))


def ensure_generation(gen_dir: Path, leave_model: Path, threads: int) -> bool:
    """Write the generation's .forgone and .belief sidecars unless it has them.
    Returns whether it wrote them."""
    if complete(gen_dir):
        return False
    forgone.write_sidecars(gen_dir, threads)
    write_sidecars(gen_dir, leave_model, threads)
    return True


def ensure_holdout(holdout_dir: Path, leave_model: Path, threads: int) -> bool:
    """Write the held-out set's .belief sidecars, with HOLDOUT_DRAWS leaves per
    turn, unless it has them; its .forgone sidecars are holdout.ensure's.
    Returns whether it wrote them."""
    if complete(holdout_dir):
        return False
    write_sidecars(holdout_dir, leave_model, threads, HOLDOUT_DRAWS)
    return True
