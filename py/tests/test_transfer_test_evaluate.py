"""M1a's evaluation harness (scribblez/transfer_test/evaluate.py) and the
trainer's best checkpoint, on synthetic corpora."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scribblez.paths import TagPaths
from scribblez.transfer_test import evaluate as ev
from scribblez.transfer_test import trainer
from scribblez.transfer_test.reader import Reader, ReaderConfig
from scribblez.transfer_test.rows import LEAF, RowConfig, assemble_row
from tests.test_transfer_test_reader_data import TEACHER_WIDTH, fake_file

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs CUDA")
PARAMS = {"max_held_out": 2, "max_probes": 8, "max_tokens": 2048, "graded_max": 0}


def _scored(label, prior, probe_sum, probe_count, held) -> ev.Scored:
    k = len(label)
    return ev.Scored(
        file=0,
        position=0,
        bag=5,
        label=np.asarray(label, float),
        label_var=np.full(k, 1e-4),
        prior=np.asarray(prior, float),
        probe_sum=np.asarray(probe_sum, float),
        probe_count=np.asarray(probe_count, float),
        held=np.asarray(held),
        plausible=np.ones(k, bool),
    )


def test_probe_evidence_sums_each_candidates_kept_outcomes(tmp_path):
    f = fake_file(tmp_path)
    row = assemble_row(f, 0, 0, RowConfig(), np.random.default_rng(0))
    total, count = ev.probe_evidence(row, len(row.held_out))
    leaves = row.kind == LEAF
    assert count.sum() == leaves.sum()
    assert np.all(count[row.held_out] == 0)
    expected = row.leaf[:, 0] + 0.5 * row.leaf[:, 1]
    assert total.sum() == pytest.approx(expected.sum())


def test_baselines_leave_held_out_moves_at_the_prior_save_a_common_shift():
    s = _scored(
        label=[0.6, 0.5, 0.4],
        prior=[0.5, 0.5, 0.5],
        probe_sum=[7.0, 0.0, 3.0],  # two probed candidates, ten probes each
        probe_count=[10, 0, 10],
        held=[False, True, False],
    )
    ev.baseline_arms(s, k=10)
    assert s.arms["shrinkage"].tolist() == pytest.approx([0.6, 0.5, 0.4])
    # The probed corrections, +0.1 and -0.1, average to no shift.
    assert s.arms["common_shift"][1] == pytest.approx(0.5)
    s.probe_sum = np.array([7.0, 0.0, 7.0])
    ev.baseline_arms(s, k=10)
    assert s.arms["common_shift"][1] == pytest.approx(0.6)


def test_only_pairs_resolved_beyond_the_label_noise_count():
    s = _scored([0.6, 0.59, 0.5, 0.3], [0.5] * 4, [0] * 4, [0] * 4, [True, True, True, False])
    s.label_var[:] = 0.001  # pair noise sqrt(0.002) = 0.045, threshold 0.089
    assert ev.resolved_pairs(s) == [(0, 2), (1, 2)]  # 0 vs 1 is too close; 3 is probed


def test_a_decided_position_is_not_scored():
    decided = _scored([0.5, 0.5], [0.4, 0.6], [0, 0], [0, 0], [True, False])
    live = _scored([0.7, 0.3], [0.4, 0.6], [0, 0], [0, 0], [True, False])
    live.position = 1
    for s in (decided, live):
        for arm in ev.ARMS:
            s.arms[arm] = s.prior
    (totals,) = ev.position_totals([decided, live])
    assert totals.held_n == 1


def _reader() -> Reader:
    cfg = ReaderConfig(
        width=64, depth=1, heads=4, kv_heads=2, teacher_width=TEACHER_WIDTH, max_slots=16
    )
    return Reader(cfg)


def test_the_best_checkpoint_round_trips(tmp_path):
    paths = TagPaths("r", "transfer_reader", tmp_path)
    model = _reader()
    state = trainer.ReaderTrainState(steps=500, best_heldout=0.03)
    ctx = SimpleNamespace(sink=SimpleNamespace(deliver_output=lambda *a, **k: None))
    trainer.save_best(ctx, paths, model, state, {**PARAMS, "reader": model.cfg.to_dict()})
    loaded, config = ev.load_reader(trainer.best_checkpoint_path(paths), "cpu")
    assert config["max_probes"] == 8 and loaded.cfg == model.cfg
    for a, b in zip(loaded.state_dict().values(), model.state_dict().values(), strict=True):
        assert torch.equal(a, b)


@needs_cuda
def test_an_untrained_reader_scores_as_the_prior(tmp_path):
    files = [fake_file(tmp_path, k=k, seed=k) for k in (5, 6)]
    for f in files:
        f.labels["wins"] = np.arange(len(f.labels)) % 7  # labels that differ
    result = ev.evaluate(_reader().cuda().eval(), PARAMS, files, "cuda", replicates=3)
    reader, prior = result["overall"]["reader"], result["overall"]["prior"]
    assert reader["heldout_rmse"] == pytest.approx(prior["heldout_rmse"], abs=1e-6)
    assert reader["pair_accuracy"] == pytest.approx(prior["pair_accuracy"])
