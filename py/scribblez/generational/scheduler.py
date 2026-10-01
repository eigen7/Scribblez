"""The generation scheduler: turns staged self-play chunks into generations and
paces the generators against the trainer.

Generators know nothing about generations. They deliver whole .slog chunks
into the tag's staging area: directly for local workers, through the
controller's collection over ssh for remote ones. The scheduler, ticked per task by the dashboard
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
  4. Once the trainer has reached the task's `max_rows`, it finishes the
     generate role for good (tick_for_task).

An ingest ledger (one chunk name per line) keeps a chunk from being assigned
twice: a chunk that reappears in staging, because a collection or relay
delivered it again after its delete failed, is deleted. The ledger line is
written before the rename, so a crash between the two loses that chunk rather
than duplicating it.

Since a single process assigns each whole file with one rename, every .slog
arrives whole and belongs to exactly one generation. The C++ loader relies on
that, and so does the training design's bound on how often a game is reused
(window x turns_per_game passes; docs/generational_training.md).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from scribblez import params as params_mod
from scribblez.paths import SCHEDULER_STATE_REL, TagPaths

from . import lifecycle

# TaskRecord.data_plane's values: the controller ticks this scheduler on its
# own tag tree ("legacy"), or a data home beside the trainer does
# (generational/data_home.py) and the controller only gates generators.
DATA_PLANE_LEGACY, DATA_PLANE_HOME = "legacy", "home"

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


def tick_for_task(spec, task, hooks):
    """The WorkloadSpec.scheduler entry: one tick for one task, or, once the
    trainer has trained the task's `max_rows`, the end of its generators.

    The trainer exits on its own at max_rows, but the generators would only be
    gated once they ran `open_ahead` generations ahead, and a gated slot still
    wants to run, so the dashboard's idle rule would never stop a rented
    machine. Finishing the role stops its containers and lets the machines go.
    """
    params = params_mod.validate(spec.params_cls, task.params)
    paths = hooks.paths
    if _trainer_done(paths, params.max_rows):
        hooks.finish(GENERATE_ROLE)
        return
    if task.data_plane == DATA_PLANE_HOME:
        hooks.gate(GENERATE_ROLE, home_gate(paths, hooks.role_running(TRAIN_ROLE), time.time()))
        return
    cfg = SchedulerConfig(
        games_per_generation=params.games_per_generation,
        open_ahead=params.open_ahead,
    )
    tick(paths, cfg, hooks)


# Per data-home record: (the last heartbeat value seen, when this controller
# first saw it). Freshness is judged on this controller's clock alone, so a
# home machine's clock never matters. A controller restart forgets it, which
# counts every home fresh for one threshold rather than parking them all.
_heartbeats: dict[Path, tuple[float, float]] = {}


def home_gate(paths: TagPaths, trainer_running: bool, now: float) -> str | None:
    """The gate on a data-home tag's generators: parked while the trainer is
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


def _trainer_done(paths: TagPaths, max_rows: int) -> bool:
    """Whether the trainer's published cursor has reached `max_rows`."""
    return params_mod.reached(lifecycle.read_train_state(paths).get("rows_trained", 0), max_rows)


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
    it, marking each only once the hook says it is in the bucket: an upload
    still running is asked about again next tick, a failed one is retried,
    and a controller restart picks up where it left off. Window eviction
    keeps the scan short."""
    for index in lifecycle.list_generation_indices(paths):
        gen_dir = paths.generation_dir(index)
        if lifecycle.is_complete(gen_dir) and not lifecycle.is_published(gen_dir):
            if hooks.publish(f"generations/{gen_dir.name}"):
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
        staged = _fill(paths, gen_dir, cfg.games_per_generation, staged, chunk_games)
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
    target: int,
    staged: list[Path],
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


def progress(spec, paths: TagPaths, params) -> list[tuple[str, object]]:
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
