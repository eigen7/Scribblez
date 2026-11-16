"""The deep_bot trainer (scribblez.deep_bot): units built from seeds with
this tag's windows and clocks, a generation scored under the newest release
and release 0, trained and published as a release, and the checkpoint the
three models resume from."""

import shutil
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)
if not LEAVES.exists():
    pytest.skip("HastyBot leave values not installed", allow_module_level=True)

import policy_set_chunk  # noqa: E402
from cloud.sinks import LocalSink  # noqa: E402
from scribblez.deep_bot import KINDS, LEAF, LEAVE, POLICY, release, trainer, units  # noqa: E402
from scribblez.generational import lifecycle  # noqa: E402
from scribblez.generational.records import TrainRecorder  # noqa: E402
from scribblez.leave_inference import forgone  # noqa: E402
from scribblez.paths import TagPaths  # noqa: E402
from scribblez.position_eval.trainer import OutputDeliverer  # noqa: E402
from scribblez.workloads.deep_bot import DeepBotParams  # noqa: E402
from scribblez.workloads.deep_bot_leaf_eval import DeepBotLeafEvalParams  # noqa: E402
from scribblez.workloads.leave_inference import LeaveInferenceParams  # noqa: E402
from scribblez.workloads.policy_set import PolicySetParams  # noqa: E402

CPU = torch.device("cpu")
# The smallest seeds, and a tag whose generation is the tests' one game.
SEED_PARAMS = {
    LEAVE: LeaveInferenceParams(trunk_channels=8, num_blocks=1, batch_size=2),
    LEAF: DeepBotLeafEvalParams(trunk="conv", width=8, depth=1, batch_size=2),
    POLICY: PolicySetParams(trunk_channels=16, num_blocks=1, num_heads=2, batch_size=2),
}
PARAMS = DeepBotParams(
    games_per_generation=1,
    pool_positions_per_game=2,
    leave_window=1,
    leaf_window=1,
    policy_window=1,
    keep_checkpoint_every=1,
)


@pytest.fixture(scope="module")
def chunk(tmp_path_factory) -> Path:
    """A one-game chunk with its pools and its forgone plays."""
    made = policy_set_chunk.surveyed_chunk(
        tmp_path_factory.mktemp("chunk"), rollouts=4, positions_per_game=2
    )
    forgone.write_sidecars(made, threads=2)
    return made


def _seed(paths: TagPaths, kind: str):
    """Save a tiny seed of `kind` where the tag keeps its seeds."""
    unit = units.UNIT_CLASSES[kind](SEED_PARAMS[kind], CPU)
    config = asdict(SEED_PARAMS[kind]) | units.unit_versions(kind, unit)
    path = units.seed_path(paths, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": unit.model.state_dict(), "config": config}, path)


@pytest.fixture
def tag(chunk, tmp_path, monkeypatch) -> TagPaths:
    """A tag with its three seeds and the chunk as its complete generation 0."""
    # Compiling the tiny models would take most of the suite's time.
    monkeypatch.setattr(torch, "compile", lambda model: model)
    paths = TagPaths("t", "deep_bot", tmp_path)
    torch.manual_seed(0)
    for kind in KINDS:
        _seed(paths, kind)
    gen_dir = lifecycle.open_generation(paths, 0, target_games=1)
    for f in chunk.iterdir():
        shutil.copy(f, gen_dir / f.name)
    lifecycle.mark_complete(gen_dir, 1)
    return paths


def _context(paths: TagPaths, monkeypatch):
    recorder = TrainRecorder(LocalSink(paths.root))
    staged = []
    monkeypatch.setattr(recorder, "deliver_staged", staged.append)
    ctx = {
        "tag": paths.tag,
        "sink": LocalSink(paths.root),
        "cpu": type("Cpu", (), {"dataloader_workers": 1})(),
        "stats": type("S", (), {"cycle_done": lambda self, phases, units, nbytes: None})(),
        "deliverer": OutputDeliverer(),
    }
    return recorder, ctx, staged


def test_a_units_params_are_its_seeds_with_this_tags_window_and_clocks():
    params = DeepBotParams(
        games_per_generation=100,
        leave_window=3,
        leave_turns_per_game=5,
        leave_lr=0.01,
        warmup_generations=2,
        averaging_generations=4,
    )
    seed = asdict(LeaveInferenceParams(trunk_channels=24, lr_warmup_rows=7, max_rows=9))
    got = units.unit_params(LEAVE, seed | {"not_a_param": 1}, params)
    rows = 100 * 3 * 5
    assert (got.trunk_channels, got.window, got.turns_per_game, got.lr) == (24, 3, 5, 0.01)
    assert (got.lr_warmup_rows, got.averaging_window_rows, got.max_rows) == (2 * rows, 4 * rows, -1)
    # The policy's clock counts pools; a unit that trains every turn, a game's.
    assert units.rows_per_generation(replace_pools(params, 6), POLICY) == 100 * 2 * 6
    every_turn = DeepBotParams(games_per_generation=10, leaf_window=1, leaf_turns_per_game=0)
    assert units.rows_per_generation(every_turn, LEAF) == 10 * units.TURNS_PER_GAME


def replace_pools(params: DeepBotParams, per_game: int) -> DeepBotParams:
    return DeepBotParams(**asdict(params) | {"pool_positions_per_game": per_game})


def test_a_seed_of_another_format_is_refused(tag):
    path = units.seed_path(tag, LEAF)
    seed = torch.load(path, weights_only=False)
    seed["config"]["leaf_eval_format_version"] = -1
    torch.save(seed, path)
    with pytest.raises(RuntimeError, match="leaf_eval_format_version=-1"):
        units.build_unit(LEAF, tag, PARAMS, CPU)


def test_a_generation_is_scored_trained_and_published_as_a_release(tag, monkeypatch):
    """The trainer's one end-to-end case, a few seconds: six units built, a
    generation scored twice and trained, and two releases exported."""
    by_kind = units.build_units(tag, PARAMS, CPU)
    baseline = units.build_units(tag, PARAMS, CPU)
    recorder, ctx, staged = _context(tag, monkeypatch)
    state = trainer.DeepBotState()
    trainer.ensure_newest_release(by_kind, tag, "t", ctx["sink"], state)
    assert release.newest(tag) == 0
    for unit in by_kind.values():
        unit.start(recorder, 0)

    trainer.run_one_generation(by_kind, baseline, recorder, tag, PARAMS, state, ctx)
    ctx["deliverer"].drain()

    assert (state.generation_index, state.rows_trained) == (1, 1)
    assert state.leave_rows > 0 and state.leaf_rows > 0 and state.policy_rows == 2
    assert release.newest(tag) == 1
    names = sorted(p.name for p in release.release_dir(tag, 1).iterdir())
    assert names == ["leaf.onnx", "leave.onnx", "manifest.json", "policy.onnx"]
    (record,) = [s.record["metrics"] for s in staged]
    for key in ("leave_eval_nll", "leaf_eval_model_logloss", "policy_eval_recall_8"):
        # Before any training the newest release is release 0.
        assert record[key] == record[f"r0_{key}"], key
    for kind in KINDS:
        assert record[f"{kind}_loss"] > 0, kind
    assert tag.checkpoint_path(1).exists()  # keep_checkpoint_every = 1

    resumed = units.build_units(tag, PARAMS, CPU)
    assert trainer.resume(resumed, tag, CPU) == state
    for kind in KINDS:
        want = by_kind[kind].model.state_dict()
        for name, value in resumed[kind].model.state_dict().items():
            assert torch.equal(value, want[name]), (kind, name)
    # The seeds did not move: release 0's models are still what the tag began with.
    trained = by_kind[LEAVE].model.state_dict()
    assert any(
        not torch.equal(v, trained[k]) for k, v in baseline[LEAVE].model.state_dict().items()
    )


def test_a_restored_trainer_republishes_the_release_its_checkpoint_holds(tag, monkeypatch):
    by_kind = units.build_units(tag, PARAMS, CPU)
    _, ctx, _ = _context(tag, monkeypatch)
    state = trainer.DeepBotState(generation_index=3, rows_trained=3)
    trainer.ensure_newest_release(by_kind, tag, "t", ctx["sink"], state)
    assert release.newest(tag) == 3
