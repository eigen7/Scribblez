"""The positions backfill (workloads/positions_backfill.py) and its role in
both workloads: an analysis lacking positions of the manifest gains them,
analyzed under its own generation's exports and its own settings, with the
entries it held unchanged; the newest generation goes first, and a complete
analysis is left alone. The engine's analysis is stubbed: these check what
the backfill asks of it and what it writes."""

from pathlib import Path

import pytest
from cloud.sinks import LocalSink
from scribblez.deep_bot import LEAF, LEAVE, known_positions, release
from scribblez.deep_bot_leaf_eval import positions as lp
from scribblez.workloads import deep_bot_leaf_eval, leaf_bot_generation, positions_backfill
from scribblez.workloads.base import WorkerContext

MANIFEST = [p.id for p in known_positions.load()]
OLD = MANIFEST[:-2]  # what the analyses were made with: the last two were added since
KEPT = {"values": [], "pick": "kept", "reading": "other"}


@pytest.fixture
def asked(monkeypatch):
    """The stubbed engine's calls: (position ids, player spec) each."""
    calls = []

    def ask(positions, player, work, threads):
        calls.append(([p.id for p in positions], player))
        return [{"pick": f"pick-{p.id}"} for p in positions]

    monkeypatch.setattr(known_positions, "ask", ask)
    return calls


def _ctx(spec, tmp_path: Path, params, max_cycles: int = 0) -> WorkerContext:
    return WorkerContext(
        spec=spec,
        role=spec.role(positions_backfill.ROLE),
        tag="t",
        params=params,
        worker_id="local-0",
        threads=1,
        max_cycles=max_cycles,
        sink=LocalSink(spec.data_dir("t", tmp_path)),
        mount_root=tmp_path,
    )


def _write(ctx: WorkerContext, generation: int, ids: list[str]):
    doc = {
        "generation": generation,
        "candidates": 7,
        "opp_leaves": 32,
        "positions": dict.fromkeys(ids, KEPT),
    }
    lp.write(doc, ctx.tag_paths().data_dir / lp.rel(generation))


def _read(ctx: WorkerContext, generation: int) -> dict:
    return lp.read(ctx.tag_paths(), generation)


def _release_ctx(tmp_path: Path, max_cycles: int = 0) -> WorkerContext:
    return _ctx(
        leaf_bot_generation.SPEC,
        tmp_path,
        leaf_bot_generation.LeafBotGenerationParams(seed_leave_tag="a", seed_leaf_tag="b"),
        max_cycles,
    )


def test_an_analysis_gains_the_positions_it_lacks_and_keeps_the_rest(tmp_path, asked):
    ctx = _release_ctx(tmp_path)
    _write(ctx, 0, OLD)
    assert leaf_bot_generation.run_backfill(ctx) == 0
    paths = ctx.tag_paths()
    ((ids, player),) = asked
    assert ids == MANIFEST[-2:]
    # The release's own exports, the analysis's own settings.
    leaf, leave = (release.export_path(paths, 0, k) for k in (LEAF, LEAVE))
    assert player == lp.player_spec(leaf, leave, 32)
    doc = _read(ctx, 0)
    assert list(doc["positions"]) == MANIFEST
    assert all(doc["positions"][i] == KEPT for i in OLD)
    added = doc["positions"][MANIFEST[-1]]
    assert added["pick"] == f"pick-{MANIFEST[-1]}" and doc["candidates"] == 7


def test_the_newest_generation_goes_first_and_complete_ones_are_left_alone(tmp_path, asked):
    ctx = _release_ctx(tmp_path, max_cycles=1)
    _write(ctx, 0, OLD)
    _write(ctx, 10, MANIFEST)
    _write(ctx, 20, OLD)
    complete = (ctx.tag_paths().data_dir / lp.rel(10)).stat().st_mtime_ns
    leaf_bot_generation.run_backfill(ctx)
    assert len(asked) == 1 and MANIFEST[-1] in _read(ctx, 20)["positions"]
    assert MANIFEST[-1] not in _read(ctx, 0)["positions"]  # the cycle budget is spent
    leaf_bot_generation.run_backfill(ctx)
    assert len(asked) == 2 and MANIFEST[-1] in _read(ctx, 0)["positions"]
    assert (ctx.tag_paths().data_dir / lp.rel(10)).stat().st_mtime_ns == complete
    leaf_bot_generation.run_backfill(ctx)
    assert len(asked) == 2  # nothing left


def test_only_the_chosen_generations_are_backfilled(tmp_path, asked):
    ctx = _release_ctx(tmp_path)
    for g in (0, 10, 20):
        _write(ctx, g, OLD)
    list(positions_backfill.backfill_all(ctx, leaf_bot_generation.backfill_exports, [10]))
    assert len(asked) == 1
    assert [MANIFEST[-1] in _read(ctx, g)["positions"] for g in (0, 10, 20)] == [
        False,
        True,
        False,
    ]


def test_a_deep_bot_leaf_eval_analysis_uses_its_export_and_the_pinned_leave_model(tmp_path, asked):
    params = deep_bot_leaf_eval.DeepBotLeafEvalParams(
        match_leave_inference_tag="li", match_leave_inference_generation=292
    )
    ctx = _ctx(deep_bot_leaf_eval.SPEC, tmp_path, params)
    _write(ctx, 1270, OLD)
    deep_bot_leaf_eval.run_backfill(ctx)
    ((_, player),) = asked
    leave = deep_bot_leaf_eval.leave_model_path(params, tmp_path)
    assert player == lp.player_spec(ctx.tag_paths().onnx_path(1270), leave, 32)


def test_the_role_is_a_local_singleton_in_both_workloads():
    for spec in (deep_bot_leaf_eval.SPEC, leaf_bot_generation.SPEC):
        role = spec.role(positions_backfill.ROLE)
        assert role.singleton and role.kinds == ("local",) and role.gpu
    gpu = deep_bot_leaf_eval.gpu_need(
        deep_bot_leaf_eval.DeepBotLeafEvalParams(), positions_backfill.ROLE
    )
    assert gpu == deep_bot_leaf_eval.gpu_need(
        deep_bot_leaf_eval.DeepBotLeafEvalParams(), "match_eval"
    )


def test_the_role_records_a_cycle_per_generation_backfilled(tmp_path, asked):
    ctx = _release_ctx(tmp_path)
    for g in (0, 10):
        _write(ctx, g, OLD)
    _write(ctx, 20, MANIFEST)
    leaf_bot_generation.run_backfill(ctx)
    record = ctx.sink.read_json("stats/local-0.json")
    assert (record["cycles_total"], record["units_total"]) == (2, 4)
