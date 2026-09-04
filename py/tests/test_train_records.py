"""The trainer's record stream and the controller's ingest of it
(generational/records.py + train_ingest.py): what a trainer delivers must
land in dashboard.db exactly as the direct writes it replaced would have."""

import json

import numpy as np
import pytest
from cloud.sinks import LocalSink
from scribblez.dashboard import db
from scribblez.generational import train_ingest
from scribblez.generational.records import TrainRecorder, read_controls, write_controls_file
from scribblez.paths import POSITION_EVAL, TagPaths

PARAMS = {"trunk": "transformer", "lr": 0.001, "window": 4}
LOSS_WEIGHTS = {"loss_wld": 1.0, "loss_score_diff": 0.0002}
CONTROLS = {"dataloader_workers": 4, "torch_threads": 12}


class _Spec:
    """A workload spec as train_ingest.tick asks of one: the tag tree rooted
    in the test's tmp dir."""

    def __init__(self, mount_root):
        self._mount_root = mount_root

    def paths(self, tag: str) -> TagPaths:
        return TagPaths(tag, POSITION_EVAL, mount_root=self._mount_root)


@pytest.fixture
def paths(tmp_path) -> TagPaths:
    p = TagPaths("t", POSITION_EVAL, mount_root=tmp_path)
    p.root.mkdir(parents=True)
    train_ingest.forget(p)
    return p


def _metrics(gen: int) -> dict:
    return {
        "epoch": gen,
        "positions": 1000 * (gen + 1),
        "loss": 0.5 / (gen + 1),
        "wld_acc": np.float32(0.5),  # a tensor reduction, as trainers produce
        "lr": 1e-3,
        "flag": True,  # not a metric; write_metrics skips bools
        "label": "x",  # nor strings
    }


def _preds(gen: int, n: int = 3) -> dict:
    rng = np.random.default_rng(gen)
    return {
        "position_eval_pred": {
            "wld": rng.random((n, 3), dtype=np.float32),
            "sd_mean": rng.random(n, dtype=np.float32),
            "sd_std": rng.random(n, dtype=np.float32),
            "placement_logits": rng.random((n, 4, 5), dtype=np.float32),  # not a column
        },
        "lane_pred": {
            "occ": (rng.random((n, 30, 15, 27)) > 0.5).astype(np.uint8),
            "score_pmf": rng.random((n, 30, 100), dtype=np.float32),
            "has_move": rng.random((n, 30), dtype=np.float32),
        },
    }


def _dump(conn) -> dict:
    """Every table's rows, with the write-time stamps (meta's, the control
    table's) left out: they are when the write happened, which the two paths
    cannot share."""
    tables = [
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        if r["name"] != "train_record"
    ]
    out = {}
    for t in tables:
        rows = [dict(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY rowid")]
        for r in rows:
            for stamp in ("created_at", "updated_at"):
                r.pop(stamp, None)
        out[t] = rows
    return out


def test_records_ingest_exactly_as_direct_writes_would(paths, tmp_path):
    """The whole exchange: run record, control events, two generations with
    prediction tables, through a LocalSink and the ingest, against the same
    facts written straight into a second database."""
    recorder = TrainRecorder(LocalSink(paths.root))
    recorder.publish_run("t", PARAMS, 123, LOSS_WEIGHTS, CONTROLS)
    recorder.control_event(0, "dataloader_workers", 4)
    recorder.commit_generation(0, 1000, _metrics(0), _preds(0))
    recorder.control_event(1500, "lr", 5e-4)
    recorder.commit_generation(1, 2000, _metrics(1), _preds(1))
    events = [
        json.loads(paths.generation_record_path(g).read_text())["control_events"] for g in (0, 1)
    ]
    assert [[e["name"] for e in evs] for evs in events] == [["dataloader_workers"], ["lr"]]

    conn = db.connect(paths.dashboard_db)
    assert train_ingest.ingest(paths, conn) == ["run.json", "gen_000000.json", "gen_000001.json"]

    direct = db.connect(tmp_path / "direct.db")
    db.write_meta(direct, "t", PARAMS, 123)
    db.write_loss_weights(direct, LOSS_WEIGHTS)
    db.init_control(direct, CONTROLS)
    for g, evs in ((0, events[0]), (1, events[1])):
        # Written directly, a numpy scalar is not a metric (write_metrics
        # takes int/float); through a record it is a plain number. Same value.
        db.write_metrics(direct, g, {**_metrics(g), "wld_acc": 0.5})
        for e in evs:
            db.write_control_event(direct, e["positions"], e["name"], e["value"], t=e["t"])
        p = _preds(g)
        db.write_position_eval_preds(direct, g, 1000 * (g + 1), p["position_eval_pred"])
        db.write_lane_preds(direct, g, 1000 * (g + 1), p["lane_pred"])

    got, want = _dump(conn), _dump(direct)
    assert set(got) == set(want)
    for table in want:
        assert got[table] == want[table], table
    # The float32 reduction survives as a metric (json carries it as a number).
    assert list(db.read_metric_series(conn, "wld_acc")[1]) == [0.5, 0.5]


def test_ingest_is_incremental_and_follows_a_rewritten_run_record(paths):
    recorder = TrainRecorder(LocalSink(paths.root))
    recorder.publish_run("t", PARAMS, 0, LOSS_WEIGHTS, CONTROLS)
    recorder.commit_generation(0, 1000, _metrics(0))
    conn = db.connect(paths.dashboard_db)
    assert train_ingest.ingest(paths, conn) == ["run.json", "gen_000000.json"]
    assert train_ingest.ingest(paths, conn) == []
    # The operator moved a control meanwhile; a re-published run record (the
    # parameter count re-stamped) must not put the default back.
    db.write_control(conn, "dataloader_workers", 2)
    recorder.publish_run("t", PARAMS, 456, LOSS_WEIGHTS, CONTROLS)
    recorder.commit_generation(1, 2000, _metrics(1))
    assert train_ingest.ingest(paths, conn) == ["run.json", "gen_000001.json"]
    assert db.read_meta(conn)["model_params"] == 456
    assert db.read_control(conn, "dataloader_workers") == 2
    assert list(db.read_metric_series(conn, "loss")[0]) == [0, 1]


def test_unreadable_record_is_skipped_and_retried(paths):
    recorder = TrainRecorder(LocalSink(paths.root))
    recorder.publish_run("t", PARAMS, 0, LOSS_WEIGHTS, CONTROLS)
    bad = paths.generation_record_path(0)
    bad.write_text("{not json")
    conn = db.connect(paths.dashboard_db)
    assert train_ingest.ingest(paths, conn) == ["run.json"]
    recorder.commit_generation(0, 1000, _metrics(0))  # the trainer rewrites it whole
    assert train_ingest.ingest(paths, conn) == ["gen_000000.json"]


def test_tick_opens_the_database_only_when_something_landed(paths, monkeypatch):
    spec = _Spec(paths.mount_root)
    train_ingest.tick(spec, "t")  # no records dir: nothing, not even a database
    assert not paths.dashboard_db.exists()

    recorder = TrainRecorder(LocalSink(paths.root))
    recorder.publish_run("t", PARAMS, 0, LOSS_WEIGHTS, CONTROLS)
    train_ingest.tick(spec, "t")
    assert db.read_meta(db.connect(paths.dashboard_db))["tag"] == "t"

    def refuse(*a, **k):
        raise AssertionError("opened the database with nothing new")

    monkeypatch.setattr(db, "connect", refuse)
    train_ingest.tick(spec, "t")  # quiet pass
    monkeypatch.undo()
    recorder.commit_generation(0, 1000, _metrics(0))
    train_ingest.tick(spec, "t")
    assert list(db.read_metric_series(db.connect(paths.dashboard_db), "loss")[0]) == [0]


def test_controls_file_round_trips_and_is_seeded_from_an_older_database(paths):
    sink = LocalSink(paths.root)
    assert read_controls(sink) == {}
    write_controls_file(paths, {"dataloader_workers": 2})
    assert read_controls(sink) == {"dataloader_workers": 2.0}

    # A tag from before the file: its control table alone holds the
    # operator's values, and the first ingest pass publishes them.
    other = TagPaths("old", POSITION_EVAL, mount_root=paths.mount_root)
    train_ingest.forget(other)
    conn = db.connect(other.dashboard_db)
    db.write_control(conn, "torch_threads", 6)
    TrainRecorder(LocalSink(other.root)).publish_run("old", PARAMS, 0, LOSS_WEIGHTS, CONTROLS)
    train_ingest.tick(_Spec(paths.mount_root), "old")
    assert read_controls(LocalSink(other.root)) == {"torch_threads": 6.0}


def test_local_sink_writes_records_atomically(paths):
    sink = LocalSink(paths.root)
    sink.push_json("records/run.json", {"a": 1})
    assert json.loads(paths.run_record_path.read_text()) == {"a": 1}
    assert [p.name for p in paths.records_dir.iterdir()] == ["run.json"]  # no .tmp left behind
