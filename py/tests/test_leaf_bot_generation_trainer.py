"""The leaf_bot_generation trainer (scribblez/leaf_bot_generation/trainer.py):
release 0 from the seeds, a generation scored under the newest release and
release 0, trained and published as the next release, and the checkpoint
that both resumes the tag and seeds the next generation's tag."""

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

from cloud.sinks import LocalSink  # noqa: E402
from scribblez import workloads  # noqa: E402
from scribblez.deep_bot import LEAF, LEAVE, release, seeds, unit_set, units  # noqa: E402
from scribblez.generational import lifecycle  # noqa: E402
from scribblez.generational.records import TrainRecorder  # noqa: E402
from scribblez.leaf_bot_generation import KINDS, LAYOUT, trainer  # noqa: E402
from scribblez.leave_inference import forgone  # noqa: E402
from scribblez.paths import TagPaths  # noqa: E402
from scribblez.position_eval.trainer import OutputDeliverer  # noqa: E402
from scribblez.selfplay import run_games  # noqa: E402
from scribblez.workloads import leaf_bot_generation  # noqa: E402
from scribblez.workloads.deep_bot_leaf_eval import DeepBotLeafEvalParams  # noqa: E402
from scribblez.workloads.leaf_bot_generation import LeafBotGenerationParams  # noqa: E402
from scribblez.workloads.leave_inference import LeaveInferenceParams  # noqa: E402

CPU = torch.device("cpu")
SPEC = workloads.get("leaf_bot_generation")
# The smallest seeds, and a tag whose generation is the tests' one game.
SEED_PARAMS = {
    LEAVE: LeaveInferenceParams(trunk_channels=8, num_blocks=1, batch_size=2),
    LEAF: DeepBotLeafEvalParams(trunk="conv", width=8, depth=1, batch_size=2),
}
PARAMS = LeafBotGenerationParams(games_per_generation=1, leave_window=1, leaf_window=1)


@pytest.fixture(scope="module")
def chunk(tmp_path_factory) -> Path:
    """A one-game chunk of HastyBot self-play with its forgone plays: the
    trainer reads any policy's games alike."""
    made = tmp_path_factory.mktemp("chunk") / "chunk"
    assert run_games(made, num_games=1, threads=2, player_spec="--type=hastybot-endgame") == 0
    forgone.write_sidecars(made, threads=2)
    return made


def save_tiny_seed(paths: TagPaths, kind: str):
    """Save a tiny standalone seed of `kind` where the tag keeps its seeds."""
    unit = units.UNIT_CLASSES[kind](SEED_PARAMS[kind], CPU)
    config = asdict(SEED_PARAMS[kind]) | units.unit_versions(kind, unit)
    path = seeds.seed_path(paths, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": unit.model.state_dict(), "config": config}, path)


def _generation_0(paths: TagPaths, chunk: Path):
    gen_dir = lifecycle.open_generation(paths, 0, target_games=1)
    for f in chunk.iterdir():
        shutil.copy(f, gen_dir / f.name)
    lifecycle.mark_complete(gen_dir, 1)


@pytest.fixture
def tag(chunk, tmp_path, monkeypatch) -> TagPaths:
    """A tag with its two seeds and the chunk as its complete generation 0."""
    # Compiling the tiny models would take most of the suite's time.
    monkeypatch.setattr(torch, "compile", lambda model: model)
    paths = SPEC.paths("g1", tmp_path)
    torch.manual_seed(0)
    for kind in KINDS:
        save_tiny_seed(paths, kind)
    _generation_0(paths, chunk)
    return paths


def _context(paths: TagPaths, monkeypatch, remote: bool = False):
    recorder = TrainRecorder(LocalSink(paths.root))
    staged = []
    monkeypatch.setattr(recorder, "deliver_staged", staged.append)
    ctx = {
        "tag": paths.tag,
        "sink": LocalSink(paths.root),
        "remote": remote,
        "cpu": type("Cpu", (), {"dataloader_workers": 1})(),
        "stats": type("S", (), {"cycle_done": lambda self, phases, units, nbytes: None})(),
        "deliverer": OutputDeliverer(),
    }
    return recorder, ctx, staged


def _equal_weights(a, b) -> bool:
    sb = b.state_dict()
    return all(torch.equal(v, sb[k]) for k, v in a.state_dict().items())


def test_a_generation_is_scored_trained_and_published_and_seeds_the_next_tag(
    tag, chunk, monkeypatch
):
    """The trainer's one end-to-end case, a few seconds: four units built, a
    generation scored twice and trained, two releases exported, and a next
    tag built from the checkpoint."""
    by_kind = units.build_units(tag, PARAMS, CPU, KINDS)
    baseline = units.build_units(tag, PARAMS, CPU, KINDS)
    recorder, ctx, staged = _context(tag, monkeypatch)
    state = trainer.LeafBotGenerationState()
    trainer.ensure_newest_release(by_kind, tag, "g1", False, state)
    assert release.newest(tag) == 0
    assert all((tag.root / rel).is_file() for rel in LAYOUT.files(0))
    for unit in by_kind.values():
        unit.start(recorder, 0)

    trainer.run_one_generation(by_kind, baseline, recorder, tag, PARAMS, state, ctx)
    ctx["deliverer"].drain()

    assert (state.generation_index, state.rows_trained) == (1, 1)
    assert state.leave_rows > 0 and state.leaf_rows > 0
    names = sorted(p.name for p in release.release_dir(tag, 1).iterdir())
    assert names == ["leaf.onnx", "leave.onnx", "manifest.json"]
    assert release.read_manifest(tag, 1)["rows_trained"] == {
        LEAVE: state.leave_rows,
        LEAF: state.leaf_rows,
    }
    (record,) = [s.record["metrics"] for s in staged]
    for key in ("leave_eval_nll_vs_uniform", "leaf_eval_model_logloss", "leaf_activation_peak"):
        # Before any training the newest release is release 0.
        assert record[key] == record[f"r0_{key}"], key
    assert record["loss_leave"] > 0 and record["loss_leaf"] > 0

    resumed = units.build_units(tag, PARAMS, CPU, KINDS)
    assert unit_set.resume(resumed, tag, CPU, trainer.LeafBotGenerationState) == state
    assert all(_equal_weights(resumed[k].model, by_kind[k].model) for k in KINDS)
    assert not _equal_weights(baseline[LEAVE].model, by_kind[LEAVE].model)

    # The next generation's tag starts from this one's trained models, with
    # this tag's architecture and its own clocks.
    following = SPEC.paths("g2", tag.mount_root)
    params = LeafBotGenerationParams(seed_tag="g1", leave_window=3)
    leaf_bot_generation.finalize(SPEC, following, params)
    seeded = units.build_units(following, params, CPU, KINDS)
    assert all(_equal_weights(seeded[k].model, by_kind[k].model) for k in KINDS)
    assert seeded[LEAF].params.width == SEED_PARAMS[LEAF].width
    assert seeded[LEAVE].params.window == 3


def test_a_remote_trainers_release_reaches_the_controller(tag, monkeypatch):
    by_kind = units.build_units(tag, PARAMS, CPU, KINDS)
    trainer.ensure_newest_release(by_kind, tag, "g1", True, trainer.LeafBotGenerationState())

    home = TagPaths("g1", SPEC.name, tag.mount_root / "controller")
    pulled = home.root / release.OUTBOX
    shutil.move(tag.root / release.OUTBOX, pulled)
    release.install(pulled / "release_000000", home)

    assert release.newest(home) == 0
    assert [(home.root / rel).is_file() for rel in LAYOUT.files(0)] == [True] * 3
