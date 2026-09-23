"""Generation lifecycle: on-disk bookkeeping for generational training.

A generation is one batch of self-play games in its own directory under the
tag's data/generations/. The trainer trains over a sliding window of the most
recent complete generations and evicts older ones. This module owns the
per-directory manifests, window selection, eviction, and the small
train_state.json cursor the trainer publishes. The scheduler, which fills
generations from staged chunks, and the trainer, which consumes them,
coordinate entirely through these files.

The manifest is the authority on a directory's status. Completeness (status
plus committed game count) and publication to the results bucket are recorded
facts, never inferred from a file listing. Everything here reads manifests
only, never .slog headers, so it stays cheap and independent of the C++
loader.

See docs/position_eval_workload.md for the surrounding protocol.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from ..paths import TagPaths

MANIFEST_NAME = "manifest.json"

# Manifest status values.
GENERATING = "generating"
COMPLETE = "complete"

# Manifest key set once the generation is in the results bucket.
PUBLISHED = "published"

# The gen_<NNNNNN> directory-name prefix produced by TagPaths.generation_dir.
_DIR_PREFIX = "gen_"


# ---------------------------------------------------------------------------
# Manifest read/write
# ---------------------------------------------------------------------------


def manifest_path(gen_dir: Path) -> Path:
    return gen_dir / MANIFEST_NAME


def read_manifest(gen_dir: Path) -> dict | None:
    """The directory's manifest, or None if absent/unreadable."""
    p = manifest_path(gen_dir)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_manifest(gen_dir: Path, manifest: dict):
    """Write the manifest atomically, so a crash never leaves a half-written
    one that would misclassify the directory."""
    gen_dir.mkdir(parents=True, exist_ok=True)
    tmp = gen_dir / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, manifest_path(gen_dir))


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


def open_generation(paths: TagPaths, index: int, *, target_games: int) -> Path:
    """Create generation `index`'s directory and write its `generating`
    manifest; return the directory."""
    gen_dir = paths.generation_dir(index)
    write_manifest(
        gen_dir,
        {
            "index": index,
            "target_games": target_games,
            "status": GENERATING,
            "committed_games": 0,
        },
    )
    return gen_dir


def update_committed(gen_dir: Path, committed_games: int):
    """Record the directory's committed game count, for display. The scheduler
    recomputes it from .slog headers every tick, so a stale value heals
    itself."""
    manifest = read_manifest(gen_dir)
    if manifest is None:
        raise FileNotFoundError(f"no manifest to update in {gen_dir}")
    if manifest.get("committed_games") != committed_games:
        manifest["committed_games"] = committed_games
        write_manifest(gen_dir, manifest)


def mark_complete(gen_dir: Path, committed_games: int):
    """Mark the directory complete with its final committed game count. Raises
    if there is no manifest to update."""
    manifest = read_manifest(gen_dir)
    if manifest is None:
        raise FileNotFoundError(f"no manifest to complete in {gen_dir}")
    manifest["status"] = COMPLETE
    manifest["committed_games"] = committed_games
    write_manifest(gen_dir, manifest)


def is_complete(gen_dir: Path) -> bool:
    manifest = read_manifest(gen_dir)
    return manifest is not None and manifest.get("status") == COMPLETE


def mark_published(gen_dir: Path):
    """Record that the complete generation is in the results bucket, whole."""
    manifest = read_manifest(gen_dir)
    if manifest is None:
        raise FileNotFoundError(f"no manifest to mark published in {gen_dir}")
    manifest[PUBLISHED] = True
    write_manifest(gen_dir, manifest)


def is_published(gen_dir: Path) -> bool:
    manifest = read_manifest(gen_dir)
    return manifest is not None and bool(manifest.get(PUBLISHED))


# ---------------------------------------------------------------------------
# Discovery and windowing
# ---------------------------------------------------------------------------


def list_generation_indices(paths: TagPaths) -> list[int]:
    """Sorted indices of every generation directory present, complete or not."""
    root = paths.generations_dir
    if not root.is_dir():
        return []
    indices = []
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith(_DIR_PREFIX):
            try:
                indices.append(int(d.name[len(_DIR_PREFIX) :]))
            except ValueError:
                continue
    return sorted(indices)


def complete_indices_upto(paths: TagPaths, latest_index: int) -> list[int]:
    """Sorted indices of complete generations at or before `latest_index`."""
    return [
        i
        for i in list_generation_indices(paths)
        if i <= latest_index and is_complete(paths.generation_dir(i))
    ]


def window_dirs(paths: TagPaths, latest_index: int, window: int) -> list[Path]:
    """Directories of the up-to-`window` most recent complete generations at or
    before `latest_index`, oldest first. `window <= 0` means all complete
    generations (unbounded corpus)."""
    complete = complete_indices_upto(paths, latest_index)
    if window > 0:
        complete = complete[-window:]
    return [paths.generation_dir(i) for i in complete]


def evict_beyond_window(paths: TagPaths, latest_index: int, window: int) -> list[int]:
    """Delete complete generations older than the window ending at
    `latest_index`, returning their indices. Never touches incomplete
    generations or any past `latest_index`. `window <= 0` evicts nothing."""
    if window <= 0:
        return []
    complete = complete_indices_upto(paths, latest_index)
    kept = set(complete[-window:])
    evicted = []
    for idx in complete:
        if idx not in kept:
            shutil.rmtree(paths.generation_dir(idx), ignore_errors=True)
            evicted.append(idx)
    return evicted


# ---------------------------------------------------------------------------
# The trainer's published cursor
# ---------------------------------------------------------------------------


def read_train_state(paths: TagPaths) -> dict:
    """The trainer's cursor ({rows_trained, generation_index}), or {} before a
    trainer has ever checkpointed."""
    try:
        return json.loads(paths.train_state_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def write_train_state(paths: TagPaths, state: dict):
    """Atomically publish the trainer's cursor, which the scheduler and the
    dashboard read instead of loading the torch checkpoint."""
    path = paths.train_state_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
