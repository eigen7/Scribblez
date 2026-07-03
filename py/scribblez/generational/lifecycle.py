"""Generation lifecycle: on-disk bookkeeping for the generational trainer.

A *generation* is one batch of self-play games written to its own directory under
a tag's data/generations/. The trainer trains over a sliding window of the most
recent complete generations; older ones are evicted. This module owns creating a
generation directory and its manifest, marking it complete, selecting the
training window, evicting stale generations, and reconciling disk state on
restart -- so the orchestrator is indifferent to how a generation got filled
(in-process producer, local worker, or remote worker).

The manifest is the authority for a generation's status: completeness is a
recorded fact (status + committed game count), never inferred from a file glob.
A crash mid-generation therefore leaves an incomplete manifest -- detected as a
partial to finish or regenerate -- rather than a directory that looks finished.
The core here reads manifests only (no .slog header I/O), so it stays cheap and
free of the C++ loader; measuring how far a partial got is a separate,
dependency-injected helper (`count_games_on_disk`).

See docs/generational_training.md, "Generations and the sliding window" and
"Restart and state".
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..paths import TagPaths

MANIFEST_NAME = "manifest.json"

# Manifest status values.
GENERATING = "generating"
COMPLETE = "complete"

# The gen_<NNNNNN> directory-name prefix produced by TagPaths.generation_dir.
_DIR_PREFIX = "gen_"


# ---------------------------------------------------------------------------
# Manifest read/write
# ---------------------------------------------------------------------------


def manifest_path(gen_dir: Path) -> Path:
    return gen_dir / MANIFEST_NAME


def read_manifest(gen_dir: Path) -> dict | None:
    """The generation's manifest, or None if absent/unreadable."""
    p = manifest_path(gen_dir)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_manifest(gen_dir: Path, manifest: dict) -> None:
    """Write the manifest atomically (tmp file + os.replace) so a crash never
    leaves a half-written manifest that would misclassify the generation."""
    gen_dir.mkdir(parents=True, exist_ok=True)
    tmp = gen_dir / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, manifest_path(gen_dir))


# ---------------------------------------------------------------------------
# Generation state transitions
# ---------------------------------------------------------------------------


def open_generation(
    paths: TagPaths, index: int, *, target_games: int, seed: int, player_spec: str
) -> Path:
    """Create generation `index`'s directory and write its `generating` manifest;
    return the directory. Overwrites any prior manifest for the index, so this
    also restarts a partial generation from scratch."""
    gen_dir = paths.generation_dir(index)
    write_manifest(
        gen_dir,
        {
            "index": index,
            "target_games": target_games,
            "seed": seed,
            "player_spec": player_spec,
            "status": GENERATING,
            "committed_games": 0,
        },
    )
    return gen_dir


def mark_complete(gen_dir: Path, committed_games: int) -> None:
    """Flip the generation's manifest to `complete`, recording the authoritative
    committed game count. Raises if there is no manifest to update."""
    manifest = read_manifest(gen_dir)
    if manifest is None:
        raise FileNotFoundError(f"no manifest to complete in {gen_dir}")
    manifest["status"] = COMPLETE
    manifest["committed_games"] = committed_games
    write_manifest(gen_dir, manifest)


def is_complete(gen_dir: Path) -> bool:
    manifest = read_manifest(gen_dir)
    return manifest is not None and manifest.get("status") == COMPLETE


# ---------------------------------------------------------------------------
# Discovery and windowing
# ---------------------------------------------------------------------------


def list_generation_indices(paths: TagPaths) -> list[int]:
    """Sorted indices of every generation directory present (complete or not).
    Matches the gen_<NNNNNN> naming produced by TagPaths.generation_dir."""
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


def latest_complete_index(paths: TagPaths) -> int | None:
    """Highest generation index whose manifest is marked complete, or None."""
    for idx in reversed(list_generation_indices(paths)):
        if is_complete(paths.generation_dir(idx)):
            return idx
    return None


def window_dirs(paths: TagPaths, latest_index: int, window: int) -> list[Path]:
    """Directories of the up-to-`window` most recent complete generations at or
    before `latest_index`, oldest first. `window <= 0` means all complete
    generations (unbounded corpus)."""
    complete = complete_indices_upto(paths, latest_index)
    if window > 0:
        complete = complete[-window:]
    return [paths.generation_dir(i) for i in complete]


def evict_beyond_window(paths: TagPaths, latest_index: int, window: int) -> list[int]:
    """Delete complete generations older than the window ending at `latest_index`,
    returning the evicted indices. Never touches partial generations or any
    generation past `latest_index` (e.g. a background refill in progress).
    `window <= 0` evicts nothing."""
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


def count_games_on_disk(gen_dir: Path, count_file_games: Callable[[Path], int]) -> int:
    """Sum game counts across the generation's .slog files, using the injected
    `count_file_games(path) -> int` (the ffi header reader in production, a fake
    in tests). Used to measure how far a partial generation got before a crash,
    without pulling the C++ loader into this module's import graph."""
    return sum(count_file_games(f) for f in sorted(gen_dir.glob("*.slog")))


# ---------------------------------------------------------------------------
# Restart reconciliation
# ---------------------------------------------------------------------------


@dataclass
class ReconcileResult:
    """A classification of on-disk generations for restart.

    `complete` and `partials` are disjoint, sorted index lists over every
    generation directory present. `next_index` is the index to open for a
    brand-new generation once any partials are resolved.
    """

    complete: list[int]
    partials: list[int]
    next_index: int

    @property
    def latest_complete(self) -> int | None:
        return self.complete[-1] if self.complete else None


def reconcile(paths: TagPaths) -> ReconcileResult:
    """Classify on-disk generations for restart. Reads manifests only, so it is
    cheap and free of .slog header I/O.

    A generation is complete iff its manifest says so; anything else (missing or
    `generating` manifest) is a partial that the orchestrator must finish or
    regenerate before training over the window."""
    indices = list_generation_indices(paths)
    complete = [i for i in indices if is_complete(paths.generation_dir(i))]
    complete_set = set(complete)
    partials = [i for i in indices if i not in complete_set]
    next_index = (indices[-1] + 1) if indices else 0
    return ReconcileResult(complete=complete, partials=partials, next_index=next_index)
