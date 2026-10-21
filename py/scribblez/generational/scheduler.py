"""The generation scheduler: turns staged self-play chunks into generations and
paces the generators against the trainer.

Generators know nothing about generations. They deliver whole .slog chunks
into the tag's staging area: directly for local workers, through the
controller's collection over ssh for remote ones. The scheduler, ticked by
the tag's data home beside its trainer (generational/data_home.py), is the
single writer of generation structure:

  1. It keeps one generation open at a time, moving staged chunks into it by
     atomic rename and marking it complete in its manifest once it holds the
     target game count.
  2. It opens a generation only while its index is within `open_ahead` of the
     trainer's published cursor (train_state.json). When none may be opened,
     it gates (parks) the generate role until the trainer advances.
  3. It recomputes committed game counts from .slog headers every tick rather
     than tracking them, so a crash between a rename and a manifest write heals
     itself.
  4. The controller gates the generators from the data home's published state
     and, once the trainer has reached the task's `max_rows`, finishes the
     generate role for good (tick_for_task).

An ingest ledger (one chunk name per line) keeps a chunk from being assigned
twice: a chunk that reappears in staging, because a collection or relay
delivered it again after its delete failed, is deleted. The ledger line is
written before the rename, so a crash between the two loses that chunk rather
than duplicating it.

A workload whose rows read a per-.slog sidecar (SchedulerConfig.chunk_sidecar)
moves each chunk as a set: the sidecar first, the .slog last, and duplicates
and quarantines take the sidecar along. Generators deliver the sidecar before
its .slog, so a staged .slog without one is damage, quarantined like an
unreadable header. A crash mid-move strands at most a sidecar, which nothing
reads without its .slog.

A workload that evaluates on held-out chunks (SchedulerConfig.holdout_every)
has the scheduler route about one staged chunk in `holdout_every`, chosen by
a hash of its name as pair_store.split_pair_stems chooses, into the held-out
set (TagPaths.held_out_chunks_dir) instead of a generation. The set takes the
first `holdout_chunks` such chunks and is then marked complete, like a
generation; later ones are ledgered and deleted. The set is never evicted.

Since a single process assigns each whole file with one rename, every .slog
arrives whole and belongs to exactly one generation. The C++ loader relies on
that, and so does the training design's bound on how often a game is reused
(window x turns_per_game passes; docs/generational_training.md).
"""

from __future__ import annotations

import json
import os
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from scribblez import params as params_mod
from scribblez.paths import SCHEDULER_STATE_REL, TagPaths

from . import lifecycle

# The WorkloadSpec.scheduler of the workloads this module schedules.
TICK_FOR_TASK = "scribblez.generational.scheduler:tick_for_task"

GENERATE_ROLE = "generate"
TRAIN_ROLE = "train"
GATE_REASON_AHEAD = "ahead of trainer"
GATE_REASON_NO_TRAINER = "trainer not running"
GATE_REASON_NO_HEARTBEAT = "no heartbeat from the trainer's data home"

# How old a data home's heartbeat may be before its generators are parked. It
# publishes every data_home.POLL_SECONDS (5 s), and the controller reads a
# remote home's record once per reconcile pass; the margin covers passes slowed
# by other tags' ssh calls.
HEARTBEAT_STALE_SECONDS = 120

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
    # The suffix of the sidecar each chunk .slog carries, or "" for none.
    chunk_sidecar: str = ""
    # Held-out chunks (see the module docstring): about one in
    # `holdout_every` (0: none), into a set of `holdout_chunks`.
    holdout_every: int = 0
    holdout_chunks: int = 0


def tick_for_task(spec, task, hooks):
    """The WorkloadSpec.scheduler entry, run by the controller: gate the
    generators from the data home's record (home_gate), or, once the trainer
    has trained the task's `max_rows`, finish them. The scheduling itself
    (`tick`) runs beside the trainer (generational/data_home.py).

    The trainer exits on its own at max_rows, but the generators would only be
    gated once they ran `open_ahead` generations ahead, and a gated slot still
    wants to run, so the dashboard's idle rule would never stop a rented
    machine. Finishing the role stops its containers and lets the machines go.
    """
    params = params_mod.validate(spec.params_cls, task.params)
    paths = hooks.paths
    if complete(spec, paths, params):
        hooks.finish(GENERATE_ROLE)
        return
    hooks.gate(GENERATE_ROLE, home_gate(paths, hooks.role_running(TRAIN_ROLE), time.time()))


# Per data-home record: (the last heartbeat value seen, when this controller
# first saw it). Freshness is judged on this controller's clock alone, so a
# home machine's clock never matters. A controller restart forgets it, which
# counts every home fresh for one threshold rather than parking them all.
_heartbeats: dict[Path, tuple[float, float]] = {}


def home_gate(paths: TagPaths, trainer_running: bool, now: float) -> str | None:
    """The gate on a tag's generators: parked while the trainer is
    not running or its data home's heartbeat has not changed for
    HEARTBEAT_STALE_SECONDS (or there is no record), since nothing would take
    their chunks; otherwise whatever its scheduler decided."""
    if not trainer_running:
        return GATE_REASON_NO_TRAINER
    path = paths.root / SCHEDULER_STATE_REL
    try:
        record = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return GATE_REASON_NO_HEARTBEAT
    seen = _heartbeats.get(path)
    if seen is None or seen[0] != record["heartbeat"]:
        seen = _heartbeats[path] = (record["heartbeat"], now)
    if now - seen[1] > HEARTBEAT_STALE_SECONDS:
        return GATE_REASON_NO_HEARTBEAT
    return record["gate"]


def complete(spec, paths: TagPaths, params) -> bool:
    """The WorkloadSpec.complete entry: whether the trainer's published cursor
    has reached the task's `max_rows`."""
    rows = lifecycle.read_train_state(paths).get("rows_trained", 0)
    return params_mod.reached(rows, params.max_rows)


def tick(paths: TagPaths, cfg: SchedulerConfig, hooks, chunk_games: ChunkGamesFn = _header_games):
    """One scheduling pass: move staged chunks into generations as far as
    pacing allows, updating manifests and the generate-role gate."""
    paths.staging_dir.mkdir(parents=True, exist_ok=True)
    _drain(paths, cfg, hooks, chunk_games)


def _drain(paths: TagPaths, cfg: SchedulerConfig, hooks, chunk_games: ChunkGamesFn):
    staged = _route_held_out(paths, cfg, _staged_chunks(paths, cfg.chunk_sidecar), chunk_games)

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
        staged = _fill(
            paths, gen_dir, cfg.games_per_generation, staged, chunk_games, cfg.chunk_sidecar
        )
        if not lifecycle.is_complete(gen_dir):
            return  # needs more chunks; staging is drained


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def ledger_path(paths: TagPaths) -> Path:
    return paths.data_dir / LEDGER_NAME


def _read_ledger(paths: TagPaths) -> set[str]:
    try:
        return set(ledger_path(paths).read_text().split())
    except FileNotFoundError:
        return set()


def _append_ledger(paths: TagPaths, name: str):
    with open(ledger_path(paths), "a") as f:
        f.write(name + "\n")


def _chunk_files(chunk: Path, sidecar: str) -> list[Path]:
    """A chunk's files in move order: its sidecar, if any, then its .slog."""
    return [chunk.with_suffix(sidecar), chunk] if sidecar else [chunk]


def _quarantine(chunk: Path, sidecar: str, why: str):
    """Set a damaged chunk aside, rather than poison a generation or retry it
    every tick: the .slog as <stem>.bad, its sidecar with .bad appended."""
    print(f"scheduler: quarantining {why} chunk {chunk.name}")
    if sidecar and chunk.with_suffix(sidecar).exists():
        side = chunk.with_suffix(sidecar)
        os.replace(side, side.with_name(side.name + ".bad"))
    os.replace(chunk, chunk.with_suffix(".bad"))


def _staged_chunks(paths: TagPaths, sidecar: str) -> list[Path]:
    """Staged chunks awaiting assignment, sorted by name. A chunk already in
    the ledger is a duplicate (see the module docstring) and is deleted; one
    missing its sidecar is quarantined."""
    ledger = _read_ledger(paths)
    staged = []
    for f in sorted(paths.staging_dir.glob("*.slog")):
        if f.name in ledger:
            for part in _chunk_files(f, sidecar):
                part.unlink(missing_ok=True)
        elif sidecar and not f.with_suffix(sidecar).exists():
            _quarantine(f, sidecar, f"{sidecar}-less")
        else:
            staged.append(f)
    return staged


def _committed_games(dest_dir: Path, chunk_games: ChunkGamesFn) -> int:
    return sum(chunk_games(f) for f in sorted(dest_dir.glob("*.slog")))


def _fill(
    paths: TagPaths,
    dest_dir: Path,
    target: int,
    staged: list[Path],
    chunk_games: ChunkGamesFn,
    sidecar: str,
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
            # file still in flight.
            _quarantine(chunk, sidecar, "unreadable")
            continue
        _append_ledger(paths, chunk.name)
        for part in _chunk_files(chunk, sidecar):
            os.replace(part, dest_dir / part.name)
        committed += games
    if committed >= target:
        lifecycle.mark_complete(dest_dir, committed)
    else:
        lifecycle.update_committed(dest_dir, committed)
    return staged


def is_held_out(stem: str, holdout_every: int) -> bool:
    """Whether the chunk with this stem belongs to the held-out set: a hash of
    the name, so the choice survives a restart and a redelivery."""
    return holdout_every > 0 and zlib.crc32(stem.encode()) % holdout_every == 0


def _route_held_out(
    paths: TagPaths, cfg: SchedulerConfig, staged: list[Path], chunk_games: ChunkGamesFn
) -> list[Path]:
    """Move the held-out chunks among `staged` into the held-out set, or delete
    them once it is complete. Returns the chunks left for the generations."""
    rest = [c for c in staged if not is_held_out(c.stem, cfg.holdout_every)]
    held = [c for c in staged if is_held_out(c.stem, cfg.holdout_every)]
    dest = paths.held_out_chunks_dir
    for chunk in held:
        _append_ledger(paths, chunk.name)
        if lifecycle.is_complete(dest):
            for part in _chunk_files(chunk, cfg.chunk_sidecar):
                part.unlink()
            continue
        if lifecycle.read_manifest(dest) is None:
            lifecycle.write_manifest(
                dest, {"target_chunks": cfg.holdout_chunks, "status": lifecycle.GENERATING}
            )
        for part in _chunk_files(chunk, cfg.chunk_sidecar):
            os.replace(part, dest / part.name)
        chunks = sorted(dest.glob("*.slog"))
        if len(chunks) >= cfg.holdout_chunks:
            lifecycle.mark_complete(dest, sum(chunk_games(c) for c in chunks))
    return rest


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


def progress(spec, paths: TagPaths, params) -> list[tuple[str, object]]:
    """The latest complete generation and the rows trained."""
    out: list[tuple[str, object]] = []
    complete_gens = [
        i
        for i in lifecycle.list_generation_indices(paths)
        if lifecycle.is_complete(paths.generation_dir(i))
    ]
    if complete_gens:
        out.append(("generation", max(complete_gens)))
    state = lifecycle.read_train_state(paths)
    if state:
        out.append(("rows", state.get("rows_trained", 0)))
    return out
