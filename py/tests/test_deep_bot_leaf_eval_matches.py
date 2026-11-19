"""deep_bot_leaf_eval's match_eval role: the leave inference model the matches
draw opponent leaves from is pinned at task creation and staged for a remote
slot, and each assigned export plays as LeafBot under hidden leaves."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from cloud.sinks import LocalSink
from scribblez import params
from scribblez.deep_bot import known_positions
from scribblez.deep_bot_leaf_eval import positions as leaf_positions
from scribblez.match_eval import runner
from scribblez.match_eval.harness import RoundResult
from scribblez.paths import TagPaths
from scribblez.trunk_arms import TRUNK_CONV, TRUNK_TRANSFORMER
from scribblez.workloads import deep_bot_leaf_eval
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.deep_bot_leaf_eval import (
    LEAVE_INFERENCE,
    LEAVE_INPUT,
    SPEC,
    DeepBotLeafEvalParams,
)
from scribblez.workloads.known_position_inputs import INPUT_DIR as POSITIONS_INPUT_DIR

LEAVE_TAG = "li"
PARAMS = DeepBotLeafEvalParams(match_leave_inference_tag=LEAVE_TAG)


def _leave_exports(mount: Path, gens: list[int]) -> TagPaths:
    paths = TagPaths(LEAVE_TAG, LEAVE_INFERENCE, mount)
    paths.onnx_dir.mkdir(parents=True, exist_ok=True)
    for g in gens:
        paths.onnx_path(g).write_bytes(b"onnx")
    return paths


def _leaf_tag(mount: Path) -> TagPaths:
    return TagPaths("leaf", "deep_bot_leaf_eval", mount)


def test_the_leave_model_pins_its_latest_export(tmp_path):
    _leave_exports(tmp_path, [3, 5])
    pinned = deep_bot_leaf_eval.finalize(SPEC, _leaf_tag(tmp_path), PARAMS)
    assert pinned.match_leave_inference_generation == 5


@pytest.mark.parametrize(
    "changes",
    [
        {"match_leave_inference_tag": ""},
        {"match_leave_inference_generation": 4},  # no export
        {"match_leave_inference_tag": "missing"},
    ],
)
def test_matches_without_a_usable_leave_model_are_refused_at_task_creation(tmp_path, changes):
    _leave_exports(tmp_path, [3, 5])
    with pytest.raises(params.ParamsError):
        deep_bot_leaf_eval.finalize(SPEC, _leaf_tag(tmp_path), replace(PARAMS, **changes))


def test_without_matches_no_leave_model_is_needed(tmp_path):
    off = DeepBotLeafEvalParams(match_every_generations=0)
    assert deep_bot_leaf_eval.finalize(SPEC, _leaf_tag(tmp_path), off) == off


def test_the_pinned_leave_model_and_the_known_positions_are_the_match_slots_inputs(tmp_path):
    """A remote slot's mount has none of the exports, only the staged copy."""
    p = replace(PARAMS, match_leave_inference_generation=3)
    expected = TagPaths(LEAVE_TAG, LEAVE_INFERENCE, tmp_path).onnx_path(3)
    got = deep_bot_leaf_eval.inputs(p, tmp_path)
    assert got[LEAVE_INPUT] == expected
    staged = {rel: src for rel, src in got.items() if rel != LEAVE_INPUT}
    assert set(staged.values()) == set(known_positions.files())
    assert all(rel.startswith(POSITIONS_INPUT_DIR + "/") for rel in staged)


def _ctx(mount: Path, sink=None, **overrides) -> WorkerContext:
    return WorkerContext(
        spec=SPEC,
        role=SPEC.role("match_eval"),
        tag="leaf",
        params=replace(PARAMS, match_leave_inference_generation=3, **overrides),
        worker_id="w0",
        threads=2,
        max_cycles=1,
        sink=sink,
        mount_root=mount,
    )


def test_an_export_plays_as_leafbot_under_hidden_leaves(tmp_path):
    leave = _leave_exports(tmp_path, [3]).onnx_path(3)
    model = tmp_path / "model_epoch_0010.onnx"
    spec, face_up = deep_bot_leaf_eval.leafbot_player(
        _ctx(tmp_path, match_candidates=6, match_opp_leaves=8), model
    )
    assert spec == (
        f"--type=leafbot-endgame --leaf-model={model} --leave-inference={leave} "
        "--candidates=6 --opp-leaves=8 --name=model"
    )
    assert face_up is False


def test_the_match_role_plays_the_assigned_export_as_leafbot(tmp_path, monkeypatch):
    _leave_exports(tmp_path, [3])
    paths = _leaf_tag(tmp_path)
    inbox = paths.match_inbox_dir("w0")
    inbox.mkdir(parents=True)
    model = inbox / paths.onnx_path(20).name
    model.touch()
    played = []

    def fake_round(spec0, spec1, num_pairs, threads, seed, results_file, face_up_leaves):
        played.append((spec0, spec1, num_pairs, face_up_leaves))
        return RoundResult([0.5] * num_pairs, wins=num_pairs, draws=0, losses=num_pairs)

    analyzed = []

    def fake_analyze(positions, model_path, leave, opp_leaves, candidates, work, threads):
        analyzed.append((len(positions), model_path, opp_leaves, candidates))
        return {"candidates": candidates, "positions": {}}

    monkeypatch.setattr(runner.harness, "play_round", fake_round)
    monkeypatch.setattr(deep_bot_leaf_eval.leaf_positions, "analyze", fake_analyze)
    ctx = _ctx(tmp_path, sink=LocalSink(paths.root), match_pairs=3)
    assert deep_bot_leaf_eval.run_match(ctx) == 0
    ((spec0, spec1, pairs, face_up),) = played
    assert spec0.startswith(f"--type=leafbot-endgame --leaf-model={model} ")
    assert (spec1, pairs, face_up) == ("--type=hastybot-endgame", 3, False)
    assert [p.name for p in paths.match_results_dir.glob("*.json")] == ["gen_000020-w0.json"]
    # The same export analyzed the known positions, delivered apart from the
    # results the controller ingests.
    assert analyzed == [(len(known_positions.load()), model, 16, 10)]
    delivered = paths.data_dir / leaf_positions.rel(20)
    assert json.loads(delivered.read_text())["generation"] == 20


def test_a_match_slot_needs_its_trunks_measured_gpu_memory_plus_the_build_scratch():
    need = {
        trunk: deep_bot_leaf_eval.gpu_need(replace(PARAMS, trunk=trunk), "match_eval")
        for trunk in (TRUNK_CONV, TRUNK_TRANSFORMER)
    }
    assert deep_bot_leaf_eval.LEAF_TRT_WORKSPACE_GB < need[TRUNK_CONV] < need[TRUNK_TRANSFORMER]
