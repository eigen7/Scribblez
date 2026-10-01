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


class _BucketSink:
    """A remote sink holding objects by root-relative path."""

    kind = "ssh"

    def __init__(self, objects: dict[str, str]):
        self.objects = dict(objects)
        self.fetched: list[str] = []

    def list_dirs(self, rel):
        return sorted(
            {k[len(rel) + 1 :].split("/")[0] for k in self.objects if k.startswith(rel + "/")}
        )

    def fetch_file(self, rel, dest: Path) -> bool:
        self.fetched.append(rel)
        if rel not in self.objects:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self.objects[rel])
        return True


def _bucket_pair(gen: int, rows: int) -> dict[str, str]:
    rel = state_pair.pair_rel(gen)
    return {f"{rel}/model.pt": f"w{rows}", f"{rel}/train_state.json": _cursor(rows)}


def test_a_restore_takes_the_newest_copy_and_reads_only_cursors_otherwise(paths):
    """Whichever holds the most rows wins: a pair, or the pre-pair layout an
    older bundle still writes. Losing candidates cost only their small
    cursor."""
    legacy = {"checkpoints/model.pt": "w-legacy", "train_state.json": _cursor(900)}
    sink = _BucketSink({**_bucket_pair(7, 700), **_bucket_pair(8, 800), **legacy})
    assert state_pair.restore(paths, sink)
    assert _installed(paths) == ("w-legacy", 900)
    models = [f for f in sink.fetched if not f.endswith(".json")]
    assert models == ["checkpoints/model.pt"]  # only the winner's weights

    sink.fetched.clear()
    assert not state_pair.restore(paths, sink)  # this machine already holds 900
    assert all(f.endswith(".json") for f in sink.fetched)
    assert _installed(paths) == ("w-legacy", 900)


def test_a_pair_pruned_after_its_cursor_was_read_is_skipped(paths):
    """The trainer can prune a listed pair between the restore's cursor read
    and its weights fetch; the restore installs nothing and the next one
    looks again."""
    sink = _BucketSink(_bucket_pair(7, 700))
    del sink.objects[f"{state_pair.pair_rel(7)}/model.pt"]
    assert not state_pair.restore(paths, sink)
    assert not paths.rolling_checkpoint.exists()


def test_a_fresher_machine_keeps_its_own_state(paths, tmp_path):
    """The live-test case: the bucket fell two generations behind the volume
    when the trainer was killed mid-upload."""
    state_pair.install(_pair(tmp_path / "local", 16832, "w-volume"), paths)
    sink = _BucketSink({**_bucket_pair(7, 12864)})
    assert not state_pair.restore(paths, sink)
    assert _installed(paths) == ("w-volume", 16832)


def test_a_local_sink_has_nothing_newer_to_offer(paths):
    assert not state_pair.restore(paths, LocalSink(paths.root))
