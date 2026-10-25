"""Renames into place that survive a power loss.

A rename makes a file appear at its destination whole or not at all, but only
to a running system. After a power loss the filesystem may keep the rename and
lose data still in the page cache: a delivered .pool came back as its first
8 MiB, and every trainer start then failed reading it. Flushing the data
before the rename, and the directory after it, keeps the order the rename was
made in.
"""

import os
from pathlib import Path


def flush(path: Path):
    """fsync `path`, a file or a directory."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def flush_tree(root: Path):
    """flush every file and directory under `root`, `root` included."""
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            flush(Path(dirpath) / name)
        flush(Path(dirpath))


def replace(src: Path, dest: Path):
    """os.replace(src, dest) such that, after a power loss, `dest` holds all
    of `src`'s data or the rename did not happen. `src` may be a directory."""
    if src.is_dir():
        flush_tree(src)
    else:
        flush(src)
    os.replace(src, dest)
    flush(dest.parent)
