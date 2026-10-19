"""The .forgone sidecars the leave inference rows read (engine
data/forgone_play_log.h; docs/plans/deep_bot_v0.md, "Forgone plays"): one
beside each .slog, written by forgone_play_tool where the games are played."""

import subprocess
from pathlib import Path

from scribblez.paths import ENGINE_DIR

SUFFIX = ".forgone"
TOOL = ENGINE_DIR / "forgone_play_tool"


def write_sidecars(slog_dir: Path, threads: int):
    """Write a fresh sidecar beside every .slog in `slog_dir`. Raises
    subprocess.CalledProcessError if the tool fails."""
    subprocess.run([str(TOOL), "--slog-dir", str(slog_dir), "--threads", str(threads)], check=True)
