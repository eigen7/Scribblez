"""Small helpers shared by the task-specific trainers."""

import shutil
import sys
from datetime import datetime
from pathlib import Path

from .paths import TagPaths


def timed_print(msg: str):
    """Print `msg` prefixed with a millisecond-resolution local timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} {msg}")


def reset_tag(paths: TagPaths):
    """Delete a tag's checkpoints, ONNX exports and dashboard DB, keeping any val set."""
    print(f"--restart: clearing prior run artifacts under {paths.root}", file=sys.stderr)
    shutil.rmtree(paths.checkpoints_dir, ignore_errors=True)
    shutil.rmtree(paths.onnx_dir, ignore_errors=True)
    for suffix in ("", "-wal", "-shm"):
        Path(str(paths.dashboard_db) + suffix).unlink(missing_ok=True)
