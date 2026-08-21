"""Tests for the controller half of match eval: assignment and ingest."""

import json
from pathlib import Path

from scribblez.dashboard import db
from scribblez.dashboard.slot_files import LocalSlotFiles
from scribblez.match_eval import dispatch
from scribblez.paths import DONE_SUFFIX, POSITION_EVAL, TagPaths
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

    def pending(every: int):
        return dispatch.pending_generation(paths, dispatch.recorded_generations(conn), every)

    assert pending(5) == 15
    db.write_match_eval(conn, 15, {**_result(15), "positions": 0})
    assert pending(5) == 10
    db.write_match_eval(conn, 10, {**_result(10), "positions": 0})
    db.write_match_eval(conn, 5, {**_result(5), "positions": 0})
    assert pending(5) is None
    # every=1 picks up the non-multiple stragglers too.
    assert pending(1) == 16


def test_ingest_writes_rows_and_metrics(tmp_path):
    paths = _paths(tmp_path)
    conn = db.connect(paths.dashboard_db)
    db.write_metrics(conn, 10, {"positions": 250_000})
    delivered = _deliver(paths, _result(10, score=0.61))

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

    # Once the result is recorded the mark is cleared, the next generation is
    # assigned, and the sidecar that is already there is left alone.
    _mark_played(inbox, paths.onnx_path(10).name)
    db.write_match_eval(conn, 10, {**_result(10), "positions": 0})
    dispatch._assign(paths, conn, 5, _slot(paths))
    assert sorted(p.name for p in inbox.iterdir()) == [
        "lexicon_frozen.bin",
        paths.onnx_path(15).name,
    ]


def _mark_played(inbox: Path, name: str):
    """What a worker does to a model once its result is on its way out."""
    (inbox / name).rename(inbox / f"{name}{DONE_SUFFIX}")


def test_a_played_generation_holds_the_slot_until_its_result_is_recorded(tmp_path):
    """The window that matters is the one between a worker finishing and the
    controller having the result in hand -- for a container, a collection
    away, and collections fail. A slot freed on the worker's schedule would be
    handed back the generation it just played."""
    paths = _paths(tmp_path)
    conn = db.connect(paths.dashboard_db)
    for gen in (10, 15):
        paths.onnx_path(gen).write_bytes(b"onnx")

    dispatch._assign(paths, conn, 5, _slot(paths))
    inbox = paths.match_inbox_dir("w0")
    _mark_played(inbox, paths.onnx_path(15).name)  # played, result not yet ingested

    dispatch._assign(paths, conn, 5, _slot(paths))
    assert [p.name for p in inbox.iterdir()] == [f"{paths.onnx_path(15).name}{DONE_SUFFIX}"]

    # Recording it settles the generation; the mark is then cleared the next
    # time the slot is offered work, which is why gen 10 is still exported here.
    db.write_match_eval(conn, 15, {**_result(15), "positions": 0})
    dispatch._assign(paths, conn, 5, _slot(paths))
    assert [p.name for p in inbox.iterdir()] == [paths.onnx_path(10).name]


def test_tick_assigns_nothing_when_the_cadence_disables_match_eval(tmp_path):
    paths = _paths(tmp_path)
    db.connect(paths.dashboard_db).close()
    paths.onnx_path(10).touch()
    _deliver(paths, _result(10))

    dispatch.tick(
        _Spec(tmp_path), "t", PositionEvalParams(match_every_generations=0), [_slot(paths)]
    )
    assert not paths.match_inbox_dir("w0").exists()
    # Ingest still runs: a result already delivered belongs in the database
    # however the cadence has since been set.
    conn = db.connect(paths.dashboard_db)
    assert [r["epoch"] for r in db.read_all_match_eval(conn)] == [10]


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


def test_a_quarantined_result_does_not_strand_the_slot(tmp_path):
    """A result that cannot be read is a terminal outcome like any other: it
    frees the slot. A mark waiting for a row that will never be written would
    stop the tag's readout for good, recoverable only by hand."""
    paths = _paths(tmp_path)
    conn = db.connect(paths.dashboard_db)
    paths.onnx_path(25).write_bytes(b"onnx")
    dispatch._assign(paths, conn, 5, _slot(paths))
    inbox = paths.match_inbox_dir("w0")
    _mark_played(inbox, paths.onnx_path(25).name)

    paths.match_results_dir.mkdir(parents=True, exist_ok=True)
    (paths.match_results_dir / "gen_000025-w0.json").write_text("{ damaged in transit")
    assert dispatch.ingest(paths, conn) == []
    assert not db.read_all_match_eval(conn)  # nothing was recorded for it

    # The generation simply falls due again and is replayed.
    dispatch._assign(paths, conn, 5, _slot(paths))
    assert [p.name for p in inbox.iterdir()] == [paths.onnx_path(25).name]


def test_a_quarantined_file_that_is_not_a_result_is_ignored(tmp_path):
    """Ingest sets aside whatever it finds, so what a .bad name means cannot be
    assumed -- and one nobody can parse must not break every later pass."""
    paths = _paths(tmp_path)
    conn = db.connect(paths.dashboard_db)
    paths.match_results_dir.mkdir(parents=True)
    (paths.match_results_dir / "stray-notes.bad").write_text("x")
    paths.onnx_path(10).write_bytes(b"onnx")

    dispatch._assign(paths, conn, 5, _slot(paths))
    assert [p.name for p in paths.match_inbox_dir("w0").iterdir()] == [paths.onnx_path(10).name]


def test_tick_records_delivered_results_with_no_worker_left(tmp_path):
    """A slot can be paused or removed after its container's output was
    collected. The result is on this disk either way, and belongs in the
    database."""
    paths = _paths(tmp_path)
    db.connect(paths.dashboard_db).close()
    _deliver(paths, _result(10))

    dispatch.tick(_Spec(tmp_path), "t", PositionEvalParams(), [])
    conn = db.connect(paths.dashboard_db)
    assert [r["epoch"] for r in db.read_all_match_eval(conn)] == [10]


def test_tick_leaves_an_idle_tag_alone(tmp_path):
    """Every task of the workload is ticked every few seconds forever, so a
    tag with nothing to do must not be opened: applying the schema to an
    archived database recreates its write-ahead log on every pass."""
    paths = _paths(tmp_path)
    db.connect(paths.dashboard_db).close()
    for stray in paths.root.glob("dashboard.db-*"):
        stray.unlink()

    dispatch.tick(_Spec(tmp_path), "t", PositionEvalParams(), [])
    assert not list(paths.root.glob("dashboard.db-*"))
