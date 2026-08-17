"""Tests for the evidence trainer (roadmap items 2-3): the sim-outcome targets,
the frozen-backbone model surface (proves-best head, freeze, student init),
the trajectory dataset's rows, and -- on the GPU e2e corpus of
test_evidence_trajectories -- a training pass with the plain-vs-conditioned
metrics and the prefix-0 exactness they rest on."""

import numpy as np
import pytest
import torch
from scribblez.evidence import dataset as ED
from scribblez.evidence.train_loop import LossConfig, conditioned_forward, evaluate, run_epoch
from scribblez.move_set_eval.model import MoveSetEvalModel
from test_evidence_trajectories import traj_corpus  # noqa: F401  (fixture)
from test_move_set_eval_evidence import _synthetic_sobs


def test_sim_targets_and_gain():
    _, obs = _synthetic_sobs(4)
    obs["wins"][:] = [20, 30, 10, 30]
    obs["draws"][:] = [0, 0, 20, 0]
    obs["losses"][:] = [20, 10, 10, 10]
    wld, delta, value = ED.sim_targets(obs)
    assert np.allclose(wld.sum(axis=1), 1.0)
    assert np.allclose(value, [0.5, 0.75, 0.5, 0.75])
    assert np.allclose(delta[:, 0], 12.0) and np.allclose(delta[:, 1], 5.0)
    # Prefix 0: the gain is the value itself. Prefix 2: best-so-far is 0.75, so
    # only a candidate strictly above it gains, and the tie gains nothing.
    assert np.allclose(ED.gain_targets(value, 0), value)
    assert np.allclose(ED.gain_targets(value, 2), [0.0, 0.0, 0.0, 0.0])
    assert np.allclose(ED.gain_targets(value, 1), [0.0, 0.25, 0.0, 0.25])


def _tiny_model() -> MoveSetEvalModel:
    torch.manual_seed(0)
    return MoveSetEvalModel(
        spatial_planes=6, scalar_size=4, trunk_channels=8, num_blocks=1, num_heads=2
    )


def test_freeze_backbone_leaves_only_the_evidence_modules_trainable():
    model = _tiny_model()
    model.freeze_backbone()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(n.startswith(("evidence_fusion.", "proves_best.")) for n in trainable)
    assert {p.data_ptr() for p in model.evidence_parameters()} == {
        p.data_ptr() for n, p in model.named_parameters() if n in trainable
    }
    # The frozen trunk stays in eval mode under train(): its BatchNorm must
    # neither use batch statistics nor drift its running ones.
    model.train()
    assert not model.trunk.training and model.evidence_fusion.training


def test_load_student_tolerates_only_the_evidence_modules_missing():
    student = _tiny_model()
    state = {k: v for k, v in student.state_dict().items() if not k.startswith("proves_best.")}
    fresh = _tiny_model()
    fresh.load_student(state)  # missing proves-best head: fine
    state.pop("head.0.weight")
    with pytest.raises(ValueError, match="mismatch"):
        _tiny_model().load_student(state)


# --- e2e over the GPU trajectory corpus ---


@pytest.fixture(scope="module")
def traj_datasets(traj_corpus):  # noqa: F811
    from scribblez.ffi import set_contingent_features

    files = ED.complete_pairs(traj_corpus.dir)
    assert len(files) >= 2
    set_contingent_features(True)
    ED.adopt_information_condition(files)
    return ED.TrajectoryDataset(files[:-1]), ED.TrajectoryDataset(files[-1:])


def test_dataset_rows_follow_the_prefix(traj_datasets):
    """Every simmed candidate is a scored row; the held-out ones are those at
    or past the prefix; the gain targets are the CRN-paired improvements over
    the prefix's best; prefixes never include the uniform tail."""
    train, _ = traj_datasets
    assert train.num_positions > 0 and train.max_trajectory <= 1 + 3 + 1
    seen_prefixes = set()
    epochs = (b for e in range(4) for b in train.iter_batches(4, seed=1, epoch_index=e))
    for batch in epochs:
        pos_id = batch["move_pos_id"].numpy()
        slot = batch["slot"].numpy()
        prefixes = batch["prefix_sizes"].numpy()
        held = batch["held_out"].numpy()
        assert (held == (slot >= prefixes[pos_id])).all()
        for p, pos in enumerate(batch["positions"]):
            k = int(prefixes[p])
            seen_prefixes.add(k)
            assert k in pos.evidence_prefix_sizes()
            rows = pos_id == p
            assert rows.sum() == len(pos.moves)
            _, _, value = ED.sim_targets(pos.obs)
            assert np.allclose(batch["target_gain"].numpy()[rows], ED.gain_targets(value, k))
    assert 0 in seen_prefixes and len(seen_prefixes) > 1


def test_training_pass_moves_only_the_evidence_path(traj_datasets):
    """One pass on the frozen-backbone model: the plain outputs are unchanged
    (metrics' plain_* identical before/after), prefix-0 rows stay exact, and
    the fusion + head parameters moved."""
    train, hold = traj_datasets
    device = torch.device("cuda")
    model = MoveSetEvalModel(
        spatial_planes=train.spatial_planes,
        scalar_size=train.scalar_size,
        trunk_channels=8,
        num_blocks=1,
        num_heads=2,
    ).to(device)
    model.freeze_backbone()
    before = evaluate(model, hold, device, positions_per_batch=4, max_e=8)
    assert before["exact_p0_maxdiff"] == 0.0
    assert before["rows"] > 0
    ev_params = [p.detach().clone() for p in model.evidence_parameters()]

    opt = torch.optim.AdamW(model.evidence_parameters(), lr=1e-2)
    cfg = LossConfig(0.004, 1.0, 10.0, 10.0, 0.05)
    result = run_epoch(model, opt, train.iter_batches(4, seed=0), device, cfg, max_e=8)
    assert result.rows > 0 and np.isfinite(result.losses["total"])
    after = evaluate(model, hold, device, positions_per_batch=4, max_e=8)
    assert after["plain_wld_ce"] == pytest.approx(before["plain_wld_ce"], abs=1e-6)
    assert after["exact_p0_maxdiff"] == 0.0
    pairs = zip(ev_params, model.evidence_parameters(), strict=True)
    assert any(not torch.equal(a, b.detach()) for a, b in pairs)
    # Conditioned outputs differ from plain on evidence-bearing rows now.
    batch = next(train.iter_batches(4, seed=3))
    plain, cond = conditioned_forward(model, batch, device, max_e=8)
    with_ev = batch["held_out"].to(device) & (
        batch["prefix_sizes"].to(device)[batch["move_pos_id"].to(device)] > 0
    )
    if bool(with_ev.any()):
        assert not torch.allclose(plain["wld"][with_ev], cond["wld"][with_ev])
