"""Unit tests for the generation scheduler (staging ingest, lifecycle, pacing).

Chunk game-counting is faked (each fake chunk's text is its game count), so the
assignment/completion/gating logic is exercised without the C++ loader.
"""

from pathlib import Path

import pytest
from scribblez.generational import lifecycle, scheduler
from scribblez.paths import POSITION_EVAL, TagPaths
from scribblez.workloads.base import SchedulerHooks


@pytest.fixture
def paths(tmp_path: Path) -> TagPaths:
    return TagPaths("t", POSITION_EVAL, mount_root=tmp_path)


def _chunk_games(chunk: Path) -> int:
    try:
        return int(chunk.read_text())
    except ValueError as e:
        raise OSError(str(e)) from e


def _stage(paths: TagPaths, name: str, games: int):
    paths.staging_dir.mkdir(parents=True, exist_ok=True)
    (paths.staging_dir / f"{name}.slog").write_text(str(games))


class Hooks(SchedulerHooks):
    def __init__(self):
        self.gates: dict[str, str | None] = {}
        self.mirrored: list[tuple[str, str]] = []
        super().__init__(
            gate=lambda role, reason: self.gates.__setitem__(role, reason),
            mirror=lambda name, dest: self.mirrored.append((name, dest)),
        )


def _tick(paths, hooks, *, games=100, test=0, ahead=1):
    cfg = scheduler.SchedulerConfig(games_per_generation=games, test_games=test, open_ahead=ahead)
    scheduler.tick(paths, cfg, hooks, chunk_games=_chunk_games)


def test_fills_and_completes_generation(paths):
    hooks = Hooks()
    _tick(paths, hooks)  # nothing staged: opens gen 0, stays open
    assert lifecycle.read_manifest(paths.generation_dir(0))["status"] == lifecycle.GENERATING
    assert hooks.gates["generate"] is None

    _stage(paths, "a", 60)
    _stage(paths, "b", 60)
    _tick(paths, hooks)
    gen0 = paths.generation_dir(0)
    assert lifecycle.is_complete(gen0)
    assert lifecycle.read_manifest(gen0)["committed_games"] == 120
    assert sorted(f.name for f in gen0.glob("*.slog")) == ["a.slog", "b.slog"]
    # With gen 0 complete and cursor still 0, gen 1 opens (open_ahead=1).
    assert lifecycle.read_manifest(paths.generation_dir(1))["status"] == lifecycle.GENERATING


def test_gates_when_ahead_of_trainer(paths):
    hooks = Hooks()
    _stage(paths, "a", 100)
    _stage(paths, "b", 100)
    _stage(paths, "c", 40)
    _tick(paths, hooks)
    # Gen 0 and gen 1 (cursor 0 + ahead 1) complete; gen 2 may not open.
    assert lifecycle.is_complete(paths.generation_dir(0))
    assert lifecycle.is_complete(paths.generation_dir(1))
    assert 2 not in lifecycle.list_generation_indices(paths)
    assert hooks.gates["generate"] == scheduler.GATE_REASON_AHEAD
    assert len(list(paths.staging_dir.glob("*.slog"))) == 1  # c waits in staging

    # The trainer advances; the next tick opens gen 2 and releases the gate.
    lifecycle.write_train_state(paths, {"generation_index": 1})
    _tick(paths, hooks)
    assert lifecycle.read_manifest(paths.generation_dir(2))["status"] == lifecycle.GENERATING
    assert lifecycle.read_manifest(paths.generation_dir(2))["committed_games"] == 40
    assert hooks.gates["generate"] is None


def test_test_split_fills_first(paths):
    hooks = Hooks()
    _stage(paths, "a", 30)
    _tick(paths, hooks, test=50)
    test_manifest = lifecycle.read_manifest(paths.test_dir)
    assert test_manifest["status"] == lifecycle.GENERATING
    assert test_manifest["committed_games"] == 30
    assert lifecycle.list_generation_indices(paths) == []  # no generation until test done
    assert hooks.gates["generate"] is None

    _stage(paths, "b", 30)
    _tick(paths, hooks, test=50)
    assert lifecycle.is_complete(paths.test_dir)
    # Generations start only after the test split froze.
    assert lifecycle.read_manifest(paths.generation_dir(0))["status"] == lifecycle.GENERATING


def test_ledger_deletes_resynced_duplicates(paths):
    hooks = Hooks()
    _stage(paths, "a", 100)
    _tick(paths, hooks)
    assert lifecycle.is_complete(paths.generation_dir(0))

    # The same chunk re-appears in staging (a cloud sync raced the ingest); it
    # is deleted, never assigned twice.
    _stage(paths, "a", 100)
    _tick(paths, hooks)
    assert list(paths.staging_dir.glob("*.slog")) == []
    assert lifecycle.read_manifest(paths.generation_dir(1))["committed_games"] == 0


def test_unreadable_chunk_is_quarantined(paths):
    hooks = Hooks()
    paths.staging_dir.mkdir(parents=True, exist_ok=True)
    (paths.staging_dir / "bad.slog").write_text("garbage")
    _stage(paths, "ok", 100)
    _tick(paths, hooks)
    assert lifecycle.is_complete(paths.generation_dir(0))
    assert (paths.staging_dir / "bad.bad").exists()
    assert not (paths.staging_dir / "bad.slog").exists()


def test_mirror_replays_ingest(paths):
    hooks = Hooks()
    _stage(paths, "a", 100)
    _tick(paths, hooks)
    assert hooks.mirrored == [("a.slog", "generations/gen_000000")]


def test_committed_count_self_heals(paths):
    """A chunk landing in the open generation outside the ledger/manifest path
    (a crash between rename and manifest write) is still counted."""
    hooks = Hooks()
    _tick(paths, hooks)  # opens gen 0
    gen0 = paths.generation_dir(0)
    (gen0 / "stray.slog").write_text("100")
    _tick(paths, hooks)
    assert lifecycle.is_complete(gen0)
    assert lifecycle.read_manifest(gen0)["committed_games"] == 100
