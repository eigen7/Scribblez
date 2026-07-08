"""Task-independent training scaffolding shared by the generational trainers
(position evaluation, max-move-per-lane): `reset_tag` (run-artifact reset) and
`timed_print` (progress lines)."""

import shutil
import sys
from datetime import datetime
from pathlib import Path

from .paths import TagPaths


def timed_print(msg: str):
    """Print `msg` with a millisecond-resolution local-time prefix, e.g.
    '2026-06-29 11:45:09.815 <msg>'. Used for the trainers' progress lines."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} {msg}")


def reset_tag(paths: TagPaths):
    """Wipe prior run artifacts (checkpoints, onnx, dashboard DB). Keeps any val set."""
    print(f"--restart: clearing prior run artifacts under {paths.root}", file=sys.stderr)
    shutil.rmtree(paths.checkpoints_dir, ignore_errors=True)
    shutil.rmtree(paths.onnx_dir, ignore_errors=True)
    for suffix in ("", "-wal", "-shm"):
        Path(str(paths.dashboard_db) + suffix).unlink(missing_ok=True)
