"""Unit tests for ensure_generation (reuse-or-regenerate a generation).

The C++ game generation is faked: generate_fn writes a few empty .slog files and
count_fn counts them, so the reuse/regenerate/clean-slate decision logic is
exercised without the play_game binary.
"""

from pathlib import Path

import pytest
from scribblez.generational import generation, lifecycle
from scribblez.paths import POST_MOVE_VALUE, TagPaths


@pytest.fixture
def paths(tmp_path: Path) -> TagPaths:
    return TagPaths("t", POST_MOVE_VALUE, mount_root=tmp_path)


def _fake_generate(calls: list, n_files: int = 3):
    """A generate_fn that records its call and writes `n_files` empty .slog files."""

    def generate_fn(gen_dir: Path, num_games: int, seed: int) -> int:
        calls.append((gen_dir, num_games, seed))
        for i in range(n_files):
            (gen_dir / f"g{i}.slog").touch()
        return 0

    return generate_fn


def _count(gen_dir: Path) -> int:
    return len(list(gen_dir.glob("*.slog")))


def test_fills_and_marks_complete(paths):
    calls = []
    gen_dir = generation.ensure_generation(
        paths,
        0,
        target_games=100,
        seed=7,
        player_spec="--type=hastybot",
        generate_fn=_fake_generate(calls, n_files=3),
        count_fn=_count,
    )
    assert gen_dir == paths.generation_dir(0)
    assert calls[0][1:] == (100, 7)  # num_games, seed threaded through
    assert lifecycle.is_complete(gen_dir)
    assert lifecycle.read_manifest(gen_dir)["committed_games"] == 3


def test_reuses_already_complete(paths):
    generation.ensure_generation(
        paths,
        0,
        target_games=100,
        seed=1,
        player_spec="p",
        generate_fn=_fake_generate([], n_files=2),
        count_fn=_count,
    )
    calls = []
    generation.ensure_generation(
        paths,
        0,
        target_games=100,
        seed=1,
        player_spec="p",
        generate_fn=_fake_generate(calls, n_files=5),
        count_fn=_count,
    )
    assert calls == []  # not regenerated
    assert lifecycle.read_manifest(paths.generation_dir(0))["committed_games"] == 2


def test_regenerates_partial_from_clean_slate(paths):
    # Simulate a crash mid-generation: a `generating` manifest + a leftover file.
    lifecycle.open_generation(paths, 0, target_games=100, seed=1, player_spec="p")
    (paths.generation_dir(0) / "stale.slog").touch()

    calls = []
    generation.ensure_generation(
        paths,
        0,
        target_games=100,
        seed=2,
        player_spec="p",
        generate_fn=_fake_generate(calls, n_files=3),
        count_fn=_count,
    )
    gen_dir = paths.generation_dir(0)
    assert len(calls) == 1
    assert not (gen_dir / "stale.slog").exists()  # leftover discarded
    assert lifecycle.is_complete(gen_dir)
    assert lifecycle.read_manifest(gen_dir)["committed_games"] == 3


def test_raises_on_generate_failure(paths):
    def failing(gen_dir, num_games, seed):
        return 1

    with pytest.raises(RuntimeError, match="exit code 1"):
        generation.ensure_generation(
            paths,
            0,
            target_games=100,
            seed=1,
            player_spec="p",
            generate_fn=failing,
            count_fn=_count,
        )
    assert not lifecycle.is_complete(paths.generation_dir(0))
