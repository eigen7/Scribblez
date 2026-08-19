"""Tests for the controller half of match eval: assignment and ingest."""

import json
from pathlib import Path

from scribblez.dashboard import db
from scribblez.dashboard.slot_files import LocalSlotFiles
from scribblez.match_eval import dispatch
from scribblez.paths import POSITION_EVAL, TagPaths
from scribblez.workloads.position_eval import SPEC, PositionEvalParams


class _Spec:
    """The position_eval spec with its tag tree rooted in the test's tmp dir --
    the only thing dispatch.tick asks of a spec."""

    name = SPEC.name
    params_cls = SPEC.params_cls

    def __init__(self, mount_root):
        self._mount_root = mount_root

    def paths(self, tag: str) -> TagPaths:
        return TagPaths(tag, POSITION_EVAL, mount_root=self._mount_root)


def _paths(tmp_path) -> TagPaths:
    paths = TagPaths("t", POSITION_EVAL, mount_root=tmp_path)
    paths.onnx_dir.mkdir(parents=True)
    return paths


def _result(epoch: int, **overrides) -> dict:
    record = {
        "epoch": epoch,
        "opponent": "--type=hastybot-endgame",
        "games": 100,
        "wins": 55,
        "draws": 2,
        "losses": 43,
        "pair_counts": [5, 10, 15, 12, 8],
        "score": 0.56,
        "ci_half_width": 0.04,
        "llr": 1.2,
        "llr_lower": -2.94,
        "llr_upper": 2.94,
        "decision": "continue",
        "elapsed_s": 60.0,
    }
    record.update(overrides)
    return record


def _deliver(paths: TagPaths, record: dict) -> Path:
    paths.match_results_dir.mkdir(parents=True, exist_ok=True)
    path = paths.match_results_dir / f"gen_{record['epoch']:06d}-w0.json"
    path.write_text(json.dumps(record))
    return path


def test_pending_generation_prefers_the_frontier(tmp_path):
    paths = _paths(tmp_path)
    conn = db.connect(paths.dashboard_db)
    for gen in (4, 5, 10, 15, 16):
        paths.onnx_path(gen).touch()

    assert dispatch.pending_generation(paths, conn, every=5) == 15
    db.write_match_eval(conn, 15, {**_result(15), "positions": 0})
    assert dispatch.pending_generation(paths, conn, every=5) == 10
    db.write_match_eval(conn, 10, {**_result(10), "positions": 0})
    db.write_match_eval(conn, 5, {**_result(5), "positions": 0})
    assert dispatch.pending_generation(paths, conn, every=5) is None
    # every=1 picks up the non-multiple stragglers too.
    assert dispatch.pending_generation(paths, conn, every=1) == 16


def test_ingest_writes_rows_and_metrics(tmp_path):
    paths = _paths(tmp_path)
    conn = db.connect(paths.dashboard_db)
    db.write_metrics(conn, 10, {"positions": 250_000})
    delivered = _deliver(paths, _result(10, score=0.61, decision="H1"))

    assert dispatch.ingest(paths, conn) == [10]
    rows = db.read_all_match_eval(conn)
    assert [r["epoch"] for r in rows] == [10]
    assert rows[0]["score"] == 0.61
    # The rows-clock label is the controller's to fill in, off its own metrics.
    assert rows[0]["positions"] == 250_000
    epochs, scores = db.read_metric_series(conn, "match_score")
    assert list(epochs) == [10] and list(scores) == [0.61]
    assert not delivered.exists()  # ingested files do not linger
    assert dispatch.ingest(paths, conn) == []


def test_ingest_quarantines_a_result_it_cannot_read(tmp_path):
    paths = _paths(tmp_path)
    conn = db.connect(paths.dashboard_db)
    paths.match_results_dir.mkdir(parents=True)
    bad = paths.match_results_dir / "gen_000005-w0.json"
    bad.write_text("{ not json")
    incomplete = _deliver(paths, {"epoch": 6, "score": 0.5})
    good = _deliver(paths, _result(7))

    assert dispatch.ingest(paths, conn) == [7]
    assert [r["epoch"] for r in db.read_all_match_eval(conn)] == [7]
    assert not good.exists()
    assert not bad.exists() and bad.with_suffix(".bad").is_file()
    assert not incomplete.exists() and incomplete.with_suffix(".bad").is_file()


def _slot(paths: TagPaths) -> LocalSlotFiles:
    return LocalSlotFiles("w0", paths.root)


def test_assign_gives_an_idle_slot_the_frontier_generation(tmp_path):
    paths = _paths(tmp_path)
    conn = db.connect(paths.dashboard_db)
    for gen in (5, 10):
        paths.onnx_path(gen).write_bytes(b"onnx")
    (paths.onnx_dir / "lexicon_frozen.bin").write_bytes(b"blob")

    dispatch._assign(paths, conn, 5, _slot(paths))
    inbox = paths.match_inbox_dir("w0")
    assert sorted(p.name for p in inbox.iterdir()) == [
        "lexicon_frozen.bin",
        paths.onnx_path(10).name,
    ]
    # The model is pointed at, not copied, and resolves to the export.
    assert (inbox / paths.onnx_path(10).name).read_bytes() == b"onnx"

    # A slot still holding a model is mid-match: nothing is assigned over it.
    paths.onnx_path(15).write_bytes(b"onnx")
    dispatch._assign(paths, conn, 5, _slot(paths))
    assert sorted(p.name for p in inbox.iterdir()) == [
        "lexicon_frozen.bin",
        paths.onnx_path(10).name,
    ]

    # Once the match is delivered the next one is assigned, and the sidecar
    # that is already there is left alone.
    (inbox / paths.onnx_path(10).name).unlink()
    db.write_match_eval(conn, 10, {**_result(10), "positions": 0})
    dispatch._assign(paths, conn, 5, _slot(paths))
    assert sorted(p.name for p in inbox.iterdir()) == [
        "lexicon_frozen.bin",
        paths.onnx_path(15).name,
    ]


def test_assign_does_nothing_without_a_due_generation(tmp_path):
    paths = _paths(tmp_path)
    conn = db.connect(paths.dashboard_db)
    paths.onnx_path(3).touch()  # not a multiple of the cadence
    dispatch._assign(paths, conn, 5, _slot(paths))
    assert not paths.match_inbox_dir("w0").exists()


def test_tick_ingests_before_assigning(tmp_path):
    """A result delivered for the frontier must not be reassigned: the tick
    ingests first, so the generation it just recorded is no longer pending."""
    paths = _paths(tmp_path)
    conn = db.connect(paths.dashboard_db)
    conn.close()
    paths.onnx_path(10).touch()
    _deliver(paths, _result(10))

    params = PositionEvalParams(match_every_generations=5)
    dispatch.tick(_Spec(tmp_path), "t", params, [_slot(paths)])
    assert not paths.match_inbox_dir("w0").exists()
    conn = db.connect(paths.dashboard_db)
    assert [r["epoch"] for r in db.read_all_match_eval(conn)] == [10]


def test_tick_is_a_no_op_before_the_trainer_has_run(tmp_path):
    paths = TagPaths("t", POSITION_EVAL, mount_root=tmp_path)
    paths.root.mkdir(parents=True)
    dispatch.tick(_Spec(tmp_path), "t", PositionEvalParams(), [_slot(paths)])
    assert not paths.dashboard_db.exists()
