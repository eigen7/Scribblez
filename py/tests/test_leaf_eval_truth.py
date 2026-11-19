"""The leaf_eval_truth workload (workloads/leaf_eval_truth.py) and the truth
files it writes (deep_bot_leaf_eval/truth.py): the leave inference model is
pinned at task creation, the slot computes each known position once, as the
network's analysis would ask it but over the monte-carlo leaf, the controller
finishes it once every position is in, and the copy to the repo keeps only the
truths of the current GCGs."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from cloud.sinks import LocalSink
from scribblez import params
from scribblez.deep_bot import known_positions
from scribblez.deep_bot_leaf_eval import truth
from scribblez.paths import TagPaths
from scribblez.workloads import leaf_eval_truth
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.leaf_eval_truth import (
    LEAVE_INFERENCE,
    LEAVE_INPUT,
    ROLE,
    SPEC,
    LeafEvalTruthParams,
)
from scribblez.workloads.worker import WorkerStopped

LEAVE_TAG = "li"
PARAMS = LeafEvalTruthParams(leave_inference_tag=LEAVE_TAG, leave_inference_generation=3)


def _leave_exports(mount: Path, gens: list[int]) -> TagPaths:
    paths = TagPaths(LEAVE_TAG, LEAVE_INFERENCE, mount)
    paths.onnx_dir.mkdir(parents=True, exist_ok=True)
    for g in gens:
        paths.onnx_path(g).write_bytes(b"onnx")
    return paths


def _tag(mount: Path) -> TagPaths:
    return SPEC.paths("t", mount)


def test_the_leave_model_pins_its_latest_export(tmp_path):
    _leave_exports(tmp_path, [3, 5])
    latest = replace(PARAMS, leave_inference_generation=-1)
    assert leaf_eval_truth.finalize(SPEC, _tag(tmp_path), latest).leave_inference_generation == 5


@pytest.mark.parametrize(
    "changes", [{"leave_inference_tag": ""}, {"leave_inference_generation": 4}, {"games": 0}]
)
def test_a_tag_without_a_usable_setup_is_refused_at_creation(tmp_path, changes):
    _leave_exports(tmp_path, [3, 5])
    with pytest.raises(params.ParamsError):
        leaf_eval_truth.finalize(SPEC, _tag(tmp_path), replace(PARAMS, **changes))


def test_the_leave_model_and_the_known_positions_are_the_inputs(tmp_path):
    got = leaf_eval_truth.inputs(PARAMS, tmp_path)
    assert got.pop(LEAVE_INPUT) == TagPaths(LEAVE_TAG, LEAVE_INFERENCE, tmp_path).onnx_path(3)
    assert set(got.values()) == set(known_positions.files())


def _ctx(mount: Path) -> WorkerContext:
    paths = _tag(mount)
    return WorkerContext(
        spec=SPEC,
        role=SPEC.role(ROLE),
        tag="t",
        params=PARAMS,
        worker_id="w0",
        threads=2,
        max_cycles=0,
        sink=LocalSink(paths.root),
        mount_root=mount,
    )


def _fake_compute(asked: list):
    def compute(position, leave, run, work, threads):
        asked.append(position.id)
        return {
            "id": position.id,
            "gcg_hash": truth.gcg_hash(position),
            "run": run,
            "seconds": 1.0,
            "values": {"m": 0.5},
        }

    return compute


def _stop(seconds):
    raise WorkerStopped


def test_the_slot_computes_each_position_once_then_idles(tmp_path, monkeypatch):
    _leave_exports(tmp_path, [3])
    asked: list[str] = []
    monkeypatch.setattr(truth, "compute", _fake_compute(asked))
    monkeypatch.setattr(leaf_eval_truth.time, "sleep", _stop)
    positions = known_positions.load()
    ctx = _ctx(tmp_path)
    paths = _tag(tmp_path)
    assert not leaf_eval_truth.complete(SPEC, paths, PARAMS)
    assert leaf_eval_truth.run_truth(ctx) == 0
    assert asked == [p.id for p in positions]
    assert leaf_eval_truth.complete(SPEC, paths, PARAMS)
    doc = json.loads((paths.data_dir / truth.rel(positions[0])).read_text())
    assert doc["run"] == leaf_eval_truth.run_settings(PARAMS)
    # A restarted slot has nothing left to compute.
    asked.clear()
    assert leaf_eval_truth.run_truth(ctx) == 0
    assert asked == []


class _Hooks:
    def __init__(self, paths: TagPaths):
        self.paths = paths
        self.finished: list[str] = []

    def finish(self, role: str):
        self.finished.append(role)


class _Task:
    params = {"leave_inference_tag": LEAVE_TAG, "leave_inference_generation": 3}


def test_the_controller_finishes_the_slot_once_every_truth_is_in(tmp_path, monkeypatch):
    _leave_exports(tmp_path, [3])
    hooks = _Hooks(_tag(tmp_path))
    leaf_eval_truth.tick(SPEC, _Task(), hooks)
    assert hooks.finished == []
    monkeypatch.setattr(truth, "compute", _fake_compute([]))
    monkeypatch.setattr(leaf_eval_truth.time, "sleep", _stop)
    leaf_eval_truth.run_truth(_ctx(tmp_path))
    leaf_eval_truth.tick(SPEC, _Task(), hooks)
    assert hooks.finished == [ROLE]


def _write_truth(data_dir: Path, position, run: dict, gcg_hash: str | None = None):
    path = data_dir / truth.rel(position)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "id": position.id,
        "gcg_hash": gcg_hash or truth.gcg_hash(position),
        "run": run,
        "seconds": 1.0,
        "values": {"m": 0.25},
    }
    path.write_text(json.dumps(doc))


def test_the_copy_keeps_the_truths_of_the_current_gcgs(tmp_path):
    a, b, c = known_positions.load()[:3]
    run = leaf_eval_truth.run_settings(PARAMS)
    _write_truth(tmp_path, a, run)
    _write_truth(tmp_path, b, run, gcg_hash="stale")
    doc, skipped = truth.collect(tmp_path, [a, b, c])
    kept = {a.id: {"gcg_hash": truth.gcg_hash(a), "values": {"m": 0.25}}}
    assert doc == {"run": run, "positions": kept}
    assert skipped == [b.id, c.id]


def test_the_copy_refuses_truths_of_different_runs(tmp_path):
    a, b = known_positions.load()[:2]
    _write_truth(tmp_path, a, leaf_eval_truth.run_settings(PARAMS))
    _write_truth(tmp_path, b, leaf_eval_truth.run_settings(replace(PARAMS, games=8)))
    with pytest.raises(ValueError):
        truth.collect(tmp_path, [a, b])
