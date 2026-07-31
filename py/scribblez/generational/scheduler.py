"""The generation scheduler: controller-side ingest, lifecycle, and pacing.

Generators are generation-agnostic -- they deliver whole .slog chunks into the
tag's staging area (directly for local workers, via cloud sync for pods). The
scheduler, ticked per task by the dashboard server's reconcile loop, is the
single writer of generation structure:

  1. It fills the frozen test split first (when the task wants one), then keeps
     exactly one generation open at a time, assigning staged chunks by atomic
     rename and recording completion in the directory manifest when the target
     game count is reached.
  2. It paces the fleet: a generation is opened only while its index is within
     `open_ahead` of the trainer's published cursor (train_state.json); when
     nothing may be opened, the generate role is gated (parked) until the
     trainer advances.
  3. Committed counts are recomputed from .slog headers each tick rather than
     tracked incrementally, so a crash between a rename and a manifest write
     self-heals.

An ingest ledger (one chunk name per line) makes assignment idempotent: a
chunk re-appearing in staging (a cloud sync racing an ingest) is deleted, not
assigned twice. The ledger line is written before the rename, so a crash
between the two loses that one chunk rather than duplicating it.

Because assignment is one rename per whole file by a single process, every
.slog belongs to exactly one generation and arrives whole -- the invariants
the C++ loader and the reuse accounting rely on.
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
    test_games: int  # 0 = no frozen test split
    open_ahead: int


def tick_for_task(spec, task, hooks):
    """The WorkloadSpec.scheduler entry: one tick for one task."""
    params = params_mod.validate(spec.params_cls, task.params)
    cfg = SchedulerConfig(
        games_per_generation=params.games_per_generation,
        test_games=params.test_games,
        open_ahead=params.open_ahead,
    )
    tick(spec.paths(task.tag), cfg, hooks)


def tick(paths: TagPaths, cfg: SchedulerConfig, hooks, chunk_games: ChunkGamesFn = _header_games):
    """One scheduling pass: drain staging into fill targets as far as pacing
    allows, updating manifests and the generate-role gate."""
    paths.staging_dir.mkdir(parents=True, exist_ok=True)
    staged = _staged_chunks(paths)

    if cfg.test_games > 0:
        if lifecycle.read_manifest(paths.test_dir) is None:
            lifecycle.open_split(paths.test_dir, target_games=cfg.test_games)
        if not lifecycle.is_complete(paths.test_dir):
            _fill(paths, paths.test_dir, "test", cfg.test_games, staged, hooks, chunk_games)
            if not lifecycle.is_complete(paths.test_dir):
                hooks.gate(GENERATE_ROLE, None)
                return

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
    """Staged chunks awaiting assignment, oldest name first. A chunk already in
    the ingest ledger is a duplicate (a cloud sync re-downloaded it before the
    bucket-side mirror move landed) and is deleted."""
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
            # An unreadable header means a corrupt chunk (it can only have
            # arrived whole, so this is damage, not an in-flight file); set it
            # aside rather than poisoning the generation or rescanning forever.
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
    """The index of the open (status: generating) generation, or None. At most
    one is open by construction; the lowest wins if a crash left several."""
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
    test_manifest = lifecycle.read_manifest(paths.test_dir)
    if test_manifest is not None and test_manifest.get("status") == lifecycle.GENERATING:
        out.append(
            ("test fill", f"{test_manifest['committed_games']}/{test_manifest['target_games']}")
        )
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
