"""The deep_bot_leaf_eval workload: the model's arms, the win table baseline's
features against the engine's own, the epoch and the held-out eval, the resume
gate, and the workload's registration."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)

from scribblez import ffi, workloads  # noqa: E402
from scribblez.dataset import SlogDataset  # noqa: E402
from scribblez.deep_bot_leaf_eval import evaluation, trainer  # noqa: E402
from scribblez.deep_bot_leaf_eval.model import (  # noqa: E402
    LAYOUT,
    SCALAR_FEATURES,
    LeafEvalModel,
    LeafEvalShape,
)
from scribblez.leave_inference import win_table  # noqa: E402
from scribblez.paths import TagPaths  # noqa: E402
from scribblez.position_eval.train_loop import run_epoch  # noqa: E402
from scribblez.trunk_arms import TRUNKS  # noqa: E402
from scribblez.workloads import deep_bot_leaf_eval  # noqa: E402

TASK = trainer.TASK


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    binary = _ENGINE_DIR / "test_slog_writer"
    if not binary.is_file():
        pytest.skip("test_slog_writer not built")
    root = tmp_path_factory.mktemp("corpus")
    subprocess.run([str(binary), str(root), "8", "2"], check=True, capture_output=True)
    return root


@pytest.fixture(scope="module")
def batches(corpus) -> list[dict]:
    ds = SlogDataset(corpus, task=TASK, apply_symmetry=False, num_workers=1)
    return list(ds.iter_batches(64, seed=0))


def _shape(trunk: str, join_planes: bool = True) -> LeafEvalShape:
    return LeafEvalShape(trunk=trunk, join_planes=join_planes, width=16, depth=1, num_heads=2)


@pytest.mark.parametrize("trunk", TRUNKS)
def test_without_join_planes_the_model_ignores_them(batches, trunk):
    torch.manual_seed(0)
    model = LeafEvalModel(_shape(trunk, join_planes=False)).eval()
    spatial, scalar = batches[0]["input_spatial"], batches[0]["input_scalar"]
    scrambled = spatial.clone()
    scrambled[:, LAYOUT["join_plane0"] :] = torch.rand_like(scrambled[:, LAYOUT["join_plane0"] :])
    with torch.no_grad():
        a, b = model(spatial, scalar), model(scrambled, scalar)
    for name in ("wld", "score_diff"):
        torch.testing.assert_close(a[name], b[name])
    with_planes = LeafEvalModel(_shape(trunk)).eval()
    with torch.no_grad():
        assert not torch.allclose(
            with_planes(spatial, scalar)["wld"], with_planes(scrambled, scalar)["wld"]
        )


def test_scalar_features_normalize_counts_and_flag_zeros(batches):
    from scribblez.deep_bot_leaf_eval.model import scalar_features
    from scribblez.supply_registers import TILE_COUNTS

    scalar = batches[0]["input_scalar"]
    features = scalar_features(scalar, torch.tensor(TILE_COUNTS, dtype=torch.float32))
    assert features.shape == (len(scalar), SCALAR_FEATURES)
    pool = scalar[:, LAYOUT["pool0"] : LAYOUT["pool0"] + 27]
    normalized_pool = features[:, 2 * 27 : 3 * 27]
    torch.testing.assert_close(normalized_pool * torch.tensor(TILE_COUNTS).float(), pool)
    zero_pool = features[:, 5 * 27 : 6 * 27]
    torch.testing.assert_close(zero_pool, (pool == 0).float())
    torch.testing.assert_close(features[:, -4:], scalar[:, LAYOUT["score_diff"] :])


def _game_metadata(path: Path) -> np.ndarray:
    header = ffi.struct_dtype("SlogFileHeader")
    num_games = int(np.fromfile(path, header, count=1)[0]["num_games"])
    return np.fromfile(
        path, ffi.struct_dtype("SlogGameMetadata"), count=num_games, offset=header.itemsize
    )


def test_win_table_features_match_the_engines_fitting_samples(corpus):
    """The leaf row at turn s and the engine's fitting sample at turn s + 1
    are the same position, seen by the side to move there; their win table
    cells must agree."""
    path = sorted(corpus.glob("*.slog"))[0]
    samples = ffi.win_table_samples(path)
    games, turns, expected = [], [], []
    offset = 0
    for g, meta in enumerate(_game_metadata(path)):
        first = max(2, int(meta["eligible_begin"]) + 1)  # sim/win_table_feature.cpp
        for s in range(int(meta["eligible_begin"]), int(meta["eligible_end"])):
            if first <= s + 1 < meta["num_turns"]:
                games.append(g)
                turns.append(s)
                expected.append(samples[offset + s + 1 - first]["feature"])
        offset += max(int(meta["num_turns"]) - first, 0)
    assert offset == len(samples) and len(games) >= 10
    rows = ffi.decode_rows(path, np.array(games), np.array(turns), post_move=True, task=TASK)
    spatial = int(np.prod(ffi.task_input_shapes(TASK)[0].dims))
    scalar = rows[:, spatial : spatial + ffi.task_input_shapes(TASK)[1].dims[0]]
    got = evaluation.win_table_features(scalar)
    expected = np.array(expected)
    np.testing.assert_array_equal(got["tiles_unseen"], expected["tiles_unseen"])
    np.testing.assert_array_equal(got["spread"], expected["spread"])


@pytest.mark.parametrize("trunk", TRUNKS)
def test_epoch_and_eval(batches, corpus, tmp_path, trunk):
    torch.manual_seed(0)
    model = LeafEvalModel(_shape(trunk))
    table = tmp_path / "table.csv"
    samples = win_table.read_samples(sorted(corpus.glob("*.slog")))
    win_table.write(table, win_table.fit(samples), {"test": True})
    cpu = torch.device("cpu")
    params = deep_bot_leaf_eval.DeepBotLeafEvalParams()
    # One batch, fitted for two steps: the conv trunk's CPU bf16 autocast makes
    # every step cost.
    batch = batches[:1]
    before = evaluation.evaluate(model, iter(batch), cpu, table, win_table.MACONDO_WIN_TABLE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(2):
        result = run_epoch(model, optimizer, iter(batch), cpu, trainer.loss_config(params))
        assert np.isfinite(result.losses["total"])
    after = evaluation.evaluate(model, iter(batch), cpu, table, win_table.MACONDO_WIN_TABLE)
    assert after["eval_wld_nll"] < before["eval_wld_nll"]
    # The tables do not train: their read is the same before and after.
    for key in ("eval_wintable_logloss", "eval_wintable_macondo_logloss"):
        assert after[key] == before[key] and 0 < after[key] < 1


def test_expected_score_logloss():
    outcome = np.array([1.0, 0.5, 0.0])
    loss = evaluation.expected_score_logloss(np.array([0.9, 0.5, 0.1]), outcome)
    np.testing.assert_allclose(loss, [-np.log(0.9), np.log(2), -np.log(0.9)])


def _paths(tmp_path) -> TagPaths:
    return TagPaths("t", TASK, tmp_path)


@pytest.mark.parametrize("key", [trainer.FORMAT_KEY, trainer.LEAVE_VALUES_KEY])
def test_resume_gate_refuses_another_format_or_leave_values(tmp_path, key):
    paths = _paths(tmp_path)
    saved = {trainer.FORMAT_KEY: 1, trainer.LEAVE_VALUES_KEY: "abc"}
    trainer.check_resumable(paths, saved)  # no checkpoint yet
    paths.rolling_checkpoint.parent.mkdir(parents=True)
    torch.save({"config": saved}, paths.rolling_checkpoint)
    trainer.check_resumable(paths, saved)
    with pytest.raises(RuntimeError, match=key):
        trainer.check_resumable(paths, {**saved, key: "other"})


def test_workload_registration_and_hidden_leaves_generation(monkeypatch):
    spec = workloads.get(TASK)
    assert {r.name for r in spec.roles} == {"generate", "train"}
    seen = {}

    def fake_generate(ctx, player_spec, face_up_leaves, random_opening_mean):
        seen.update(face_up_leaves=face_up_leaves, random_opening_mean=random_opening_mean)
        return 0

    monkeypatch.setattr(deep_bot_leaf_eval, "generate", fake_generate)

    class Ctx:
        params = deep_bot_leaf_eval.DeepBotLeafEvalParams()

    deep_bot_leaf_eval.run_generate(Ctx())
    assert seen == {"face_up_leaves": False, "random_opening_mean": 0.0}
