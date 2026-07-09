"""Unit tests for the generation lifecycle bookkeeping.

Pure filesystem logic -- no C++ loader, no real .slog content.
"""

from pathlib import Path

import pytest
from scribblez.generational import lifecycle as lc
from scribblez.paths import POSITION_EVAL, TagPaths


@pytest.fixture
def paths(tmp_path: Path) -> TagPaths:
    return TagPaths("t", POSITION_EVAL, mount_root=tmp_path)


def _open(paths: TagPaths, index: int, target_games: int = 100) -> Path:
    return lc.open_generation(paths, index, target_games=target_games)


def test_open_generation_writes_generating_manifest(paths):
    gen_dir = _open(paths, 0, target_games=100)
    assert gen_dir == paths.generation_dir(0)
    manifest = lc.read_manifest(gen_dir)
    assert manifest["status"] == lc.GENERATING
    assert manifest["target_games"] == 100
    assert manifest["index"] == 0
    assert not lc.is_complete(gen_dir)


def test_mark_complete_records_committed_games(paths):
    gen_dir = _open(paths, 3)
    lc.mark_complete(gen_dir, committed_games=97)
    assert lc.is_complete(gen_dir)
    manifest = lc.read_manifest(gen_dir)
    assert manifest["status"] == lc.COMPLETE
    assert manifest["committed_games"] == 97


def test_mark_complete_without_manifest_raises(paths):
    with pytest.raises(FileNotFoundError):
        lc.mark_complete(paths.generation_dir(9), committed_games=1)


def test_read_manifest_absent_is_none(paths):
    assert lc.read_manifest(paths.generation_dir(0)) is None


def test_write_manifest_is_atomic_no_tmp_left(paths):
    gen_dir = _open(paths, 0)
    assert not (gen_dir / (lc.MANIFEST_NAME + ".tmp")).exists()


def test_list_generation_indices_sorted_and_filtered(paths):
    for i in (2, 0, 10):
        _open(paths, i)
    # A stray non-generation directory is ignored.
    (paths.generations_dir / "scratch").mkdir()
    assert lc.list_generation_indices(paths) == [0, 2, 10]


def test_list_generation_indices_empty_when_no_dir(paths):
    assert lc.list_generation_indices(paths) == []


def test_window_dirs_returns_recent_complete_oldest_first(paths):
    for i in range(5):
        lc.mark_complete(_open(paths, i), 100)
    dirs = lc.window_dirs(paths, latest_index=4, window=3)
    assert dirs == [paths.generation_dir(i) for i in (2, 3, 4)]


def test_window_dirs_excludes_partials_and_future(paths):
    lc.mark_complete(_open(paths, 0), 100)
    lc.mark_complete(_open(paths, 1), 100)
    _open(paths, 2)  # partial background refill beyond latest_index
    dirs = lc.window_dirs(paths, latest_index=1, window=0)  # window<=0 => all complete
    assert dirs == [paths.generation_dir(0), paths.generation_dir(1)]


def test_evict_beyond_window_removes_old_complete_only(paths):
    for i in range(4):
        lc.mark_complete(_open(paths, i), 100)
    _open(paths, 4)  # partial refill beyond latest_index -- must survive
    evicted = lc.evict_beyond_window(paths, latest_index=3, window=2)
    assert evicted == [0, 1]
    assert not paths.generation_dir(0).exists()
    assert not paths.generation_dir(1).exists()
    assert paths.generation_dir(2).exists()
    assert paths.generation_dir(3).exists()
    assert paths.generation_dir(4).exists()  # partial untouched


def test_evict_window_nonpositive_is_noop(paths):
    for i in range(3):
        lc.mark_complete(_open(paths, i), 100)
    assert lc.evict_beyond_window(paths, latest_index=2, window=0) == []
    assert lc.list_generation_indices(paths) == [0, 1, 2]
