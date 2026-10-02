"""The trainer's state as a named pair, and the cursor rule
(scribblez.generational.state_pair)."""

import json
from pathlib import Path

import pytest
from cloud.sinks import LocalSink
from scribblez.generational import state_pair
from scribblez.paths import POSITION_EVAL, TagPaths


@pytest.fixture
def paths(tmp_path: Path) -> TagPaths:
    return TagPaths("t", POSITION_EVAL, mount_root=tmp_path)


def _cursor(rows: int) -> str:
    return json.dumps({"generation_index": rows // 100, "rows_trained": rows})


def _pair(dir: Path, rows: int, weights: str = "w") -> Path:
    dir.mkdir(parents=True, exist_ok=True)
    (dir / state_pair.MODEL_NAME).write_text(weights)
    (dir / state_pair.CURSOR_NAME).write_text(_cursor(rows))
    return dir


def _installed(paths) -> tuple[str, int]:
    return paths.rolling_checkpoint.read_text(), state_pair.rows_trained(paths.train_state_path)


def test_a_pair_installs_only_if_it_has_trained_at_least_as_far(paths, tmp_path):
    assert state_pair.install(_pair(tmp_path / "a", 500, "w500"), paths)  # nothing installed yet
    assert _installed(paths) == ("w500", 500)
    assert not state_pair.install(_pair(tmp_path / "b", 400, "w400"), paths)  # stale
    assert _installed(paths) == ("w500", 500)
    assert not (tmp_path / "b").exists()  # consumed either way
    assert state_pair.install(_pair(tmp_path / "c", 600, "w600"), paths)
    assert _installed(paths) == ("w600", 600)
    assert state_pair.install(_pair(tmp_path / "d", 600, "w600'"), paths)  # a tie installs
    assert _installed(paths) == ("w600'", 600)


def test_a_torn_pair_never_installs(paths, tmp_path):
    """A pair without its cursor (the commit marker) or its weights is not a
    pair: copying stopped partway."""
    torn = _pair(tmp_path / "a", 500)
    (torn / state_pair.CURSOR_NAME).unlink()
    assert not state_pair.install(torn, paths)
    torn = _pair(tmp_path / "b", 500)
    (torn / state_pair.MODEL_NAME).unlink()
    assert not state_pair.install(torn, paths)
    assert not paths.rolling_checkpoint.exists()


def test_delivery_sends_the_cursor_last_and_keeps_only_the_newest_pair(paths, tmp_path):
    sink = LocalSink(paths.root)
    for gen in (3, 4):
        model, cursor = tmp_path / f"m{gen}", tmp_path / f"c{gen}"
        model.write_text(f"w{gen}")
        cursor.write_text(_cursor(gen * 100))
        state_pair.deliver(sink, model, cursor, gen)
    assert sink.list_dirs(state_pair.STATE_DIR) == ["gen_000004"]
    pair = paths.root / state_pair.pair_rel(4)
    assert (pair / state_pair.MODEL_NAME).read_text() == "w4"


def test_a_seed_installs_under_the_rule(paths):
    seed = _pair(paths.root / state_pair.SEED_DIR, 500, "w-controller")
    assert state_pair.take_seed(paths, expected=True)
    assert _installed(paths) == ("w-controller", 500) and not seed.exists()
    # A home volume already ahead of the controller's copy keeps its own.
    _pair(paths.root / state_pair.SEED_DIR, 400, "w-older")
    assert not state_pair.take_seed(paths, expected=True)
    assert _installed(paths) == ("w-controller", 500)


def test_a_fresh_container_waits_for_its_seed_and_a_restarted_one_does_not(paths, monkeypatch):
    slept = []

    def arrive(_):  # the controller's push lands while the trainer waits
        slept.append(1)
        _pair(paths.root / state_pair.SEED_DIR, 300, "w-seed")

    monkeypatch.setattr(state_pair.time, "sleep", arrive)
    assert state_pair.take_seed(paths, expected=True)
    assert slept == [1] and _installed(paths) == ("w-seed", 300)
    # Restarted: its own checkpoint is there, so nothing is awaited.
    monkeypatch.setattr(state_pair.time, "sleep", lambda _: pytest.fail("waited"))
    assert not state_pair.take_seed(paths, expected=True)
    # Not told a seed is coming: never waits.
    paths.rolling_checkpoint.unlink()
    assert not state_pair.take_seed(paths, expected=False)


def test_a_seed_that_never_comes_fails_the_trainer(paths, monkeypatch):
    clock = iter(range(0, 10_000, 1000))
    monkeypatch.setattr(state_pair.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(state_pair.time, "sleep", lambda _: None)
    with pytest.raises(AssertionError, match="no state seed"):
        state_pair.take_seed(paths, expected=True)
