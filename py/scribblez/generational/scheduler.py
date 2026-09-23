"""The generation scheduler: turns staged self-play chunks into generations and
paces the generators against the trainer.

Generators know nothing about generations. They deliver whole .slog chunks
into the tag's staging area, directly for local workers and via cloud sync
for bucket-delivering ones. The scheduler, ticked per task by the dashboard
server's reconcile loop, is the single writer of generation structure:

  1. It keeps one generation open at a time, moving staged chunks into it by
     atomic rename and marking it complete in its manifest once it holds the
     target game count.
  2. It opens a generation only while its index is within `open_ahead` of the
     trainer's published cursor (train_state.json). When none may be opened,
     it gates (parks) the generate role until the trainer advances.
  3. It recomputes committed game counts from .slog headers every tick rather
     than tracking them, so a crash between a rename and a manifest write heals
     itself.

An ingest ledger (one chunk name per line) keeps a chunk from being assigned
twice: a chunk that reappears in staging, because cloud sync downloaded it
again before the bucket-side move landed, is deleted. The ledger line is
written before the rename, so a crash between the two loses that chunk rather
than duplicating it.

Since a single process assigns each whole file with one rename, every .slog
arrives whole and belongs to exactly one generation. The C++ loader relies on
that, and so does the training design's bound on how often a game is reused
(window x turns_per_game passes; docs/generational_training.md).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from scribblez import params as params_mod
from scribblez.paths import TagPaths

from . import lifecycle

GENERATE_ROLE = "generate"
GATE_REASON_AHEAD = "ahead of trainer"

# One ingested-chunk name per line, under the tag's data/ dir.
LEDGER_NAME = "ingest_log.txt"

# (chunk path) -> games in the chunk, from its .slog header.
ChunkGamesFn = Callable[[Path], int]


def _header_games(chunk: Path) -> int:
    from scribblez.ffi import read_file_header

    return read_file_header(chunk)[0]


@dataclass(frozen=True)
class SchedulerConfig:
    games_per_generation: int
    open_ahead: int


def tick_for_task(spec, task, hooks):
    """The WorkloadSpec.scheduler entry: one tick for one task."""
    params = params_mod.validate(spec.params_cls, task.params)
    cfg = SchedulerConfig(
        games_per_generation=params.games_per_generation,
        open_ahead=params.open_ahead,
    )
    tick(spec.paths(task.tag), cfg, hooks)


def tick(paths: TagPaths, cfg: SchedulerConfig, hooks, chunk_games: ChunkGamesFn = _header_games):
    """One scheduling pass: move staged chunks into generations as far as
    pacing allows, updating manifests and the generate-role gate; then publish
    any complete generation not yet in the bucket."""
    paths.staging_dir.mkdir(parents=True, exist_ok=True)
    _drain(paths, cfg, hooks, chunk_games)
    if hooks.publish:
        _publish_complete(paths, hooks)


def _publish_complete(paths: TagPaths, hooks):
    """Publish every complete generation whose manifest does not yet record
    it, marking each only after the hook returns: a failed upload is retried
    next tick, and a controller restart picks up where it left off. Window
    eviction keeps the scan short."""
    for index in lifecycle.list_generation_indices(paths):
        gen_dir = paths.generation_dir(index)
        if lifecycle.is_complete(gen_dir) and not lifecycle.is_published(gen_dir):
            hooks.publish(f"generations/{gen_dir.name}")
            lifecycle.mark_published(gen_dir)


def _drain(paths: TagPaths, cfg: SchedulerConfig, hooks, chunk_games: ChunkGamesFn):
    staged = _staged_chunks(paths)

    cursor = lifecycle.read_train_state(paths).get("generation_index", 0)
    while True:
        open_index = _open_index(paths)
        if open_index is None:
            next_index = _next_index(paths, cursor)
            if next_index > cursor + cfg.open_ahead:
                hooks.gate(GENERATE_ROLE, GATE_REASON_AHEAD)
                return
            lifecycle.open_generation(paths, next_index, target_games=cfg.games_per_generation)
            open_index = next_index
        hooks.gate(GENERATE_ROLE, None)
        gen_dir = paths.generation_dir(open_index)
        dest_rel = f"generations/{gen_dir.name}"
        staged = _fill(paths, gen_dir, dest_rel, cfg.games_per_generation, staged, hooks,
                       chunk_games)  # fmt: skip
        if not lifecycle.is_complete(gen_dir):
            return  # needs more chunks; staging is drained


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def _ledger_path(paths: TagPaths) -> Path:
    return paths.data_dir / LEDGER_NAME


def _read_ledger(paths: TagPaths) -> set[str]:
    try:
        return set(_ledger_path(paths).read_text().split())
    except FileNotFoundError:
        return set()


def _append_ledger(paths: TagPaths, name: str):
    with open(_ledger_path(paths), "a") as f:
        f.write(name + "\n")


def _staged_chunks(paths: TagPaths) -> list[Path]:
    """Staged chunks awaiting assignment, sorted by name. A chunk already in
    the ledger is a duplicate (see the module docstring) and is deleted."""
    ledger = _read_ledger(paths)
    staged = []
    for f in sorted(paths.staging_dir.glob("*.slog")):
        if f.name in ledger:
            f.unlink()
        else:
            staged.append(f)
    return staged


def _committed_games(dest_dir: Path, chunk_games: ChunkGamesFn) -> int:
    return sum(chunk_games(f) for f in sorted(dest_dir.glob("*.slog")))


def _fill(
    paths: TagPaths,
    dest_dir: Path,
    dest_rel: str,
    target: int,
    staged: list[Path],
    hooks,
    chunk_games: ChunkGamesFn,
) -> list[Path]:
    """Assign staged chunks into `dest_dir` until its target game count is
    reached, then mark it complete. Returns the chunks still unassigned."""
    committed = _committed_games(dest_dir, chunk_games)
    while staged and committed < target:
        chunk = staged.pop(0)
        try:
            games = chunk_games(chunk)
        except OSError:
            # Chunks arrive whole, so an unreadable header means damage, not a
            # file still in flight. Set it aside rather than poison the
            # generation or retry it every tick.
            print(f"scheduler: quarantining unreadable chunk {chunk.name}")
            os.replace(chunk, chunk.with_suffix(".bad"))
            continue
        _append_ledger(paths, chunk.name)
        os.replace(chunk, dest_dir / chunk.name)
        if hooks.mirror:
            hooks.mirror(chunk.name, dest_rel)
        committed += games
    if committed >= target:
        lifecycle.mark_complete(dest_dir, committed)
    else:
        lifecycle.update_committed(dest_dir, committed)
    return staged


# ---------------------------------------------------------------------------
# Generation indexing
# ---------------------------------------------------------------------------


def _open_index(paths: TagPaths) -> int | None:
    """The index of the open (still generating) generation, or None. Only one
    is ever opened at a time; if a crash left several, the lowest wins."""
    for i in lifecycle.list_generation_indices(paths):
        manifest = lifecycle.read_manifest(paths.generation_dir(i))
        if manifest is not None and manifest.get("status") == lifecycle.GENERATING:
            return i
    return None


def _next_index(paths: TagPaths, cursor: int) -> int:
    """The index the next generation would get: past every existing directory
    (evicted ones never come back) and never behind the trainer's cursor."""
    existing = lifecycle.list_generation_indices(paths)
    return max([cursor] + [i + 1 for i in existing])


# ---------------------------------------------------------------------------
# Progress (the tag listing / Overview counters)
# ---------------------------------------------------------------------------


def progress(spec, tag: str) -> list[tuple[str, object]]:
    paths = spec.paths(tag)
    out: list[tuple[str, object]] = []
    open_index = _open_index(paths)
    if open_index is not None:
        m = lifecycle.read_manifest(paths.generation_dir(open_index))
        out.append(
            ("filling", f"gen {open_index}: {m['committed_games']}/{m['target_games']} games")
        )
    complete = [
        i
        for i in lifecycle.list_generation_indices(paths)
        if lifecycle.is_complete(paths.generation_dir(i))
    ]
    if complete:
        out.append(("generations", f"{len(complete)} complete (latest gen {max(complete)})"))
    state = lifecycle.read_train_state(paths)
    if state:
        out.append(("trainer", f"gen {state.get('generation_index', 0)}"))
        out.append(("rows", state.get("rows_trained", 0)))
    return out
