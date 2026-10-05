"""M1a's reader (scribblez/transfer_test/reader.py, loss.py, trainer.py) and
its workload (scribblez/workloads/transfer_reader.py), on synthetic corpora."""

import dataclasses
import math

import numpy as np
import pytest
import torch
from scribblez import params as params_mod
from scribblez import workloads
from scribblez.position_eval.model import PLACEMENT_HEAD_NAMES
from scribblez.transfer_test import loss as loss_mod
from scribblez.transfer_test import trainer
from scribblez.transfer_test.reader import (
    FOOTPRINT_HEADS,
    Reader,
    ReaderConfig,
    _same_slot_bias,
    mask_mod_of,
    prior_outputs,
)
from scribblez.transfer_test.rows import RowConfig, assemble_row
from scribblez.transfer_test.tokens import collate
from scribblez.workloads import transfer_reader
from tests.test_transfer_test_reader_data import TEACHER_WIDTH, fake_file
from torch.nn.attention.flex_attention import create_mask

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs CUDA")


def _batch(tmp_path, rows: int = 2):
    f = fake_file(tmp_path)
    rng = np.random.default_rng(0)
    return collate([assemble_row(f, 0, 0, RowConfig(), rng) for _ in range(rows)])


def test_mask_queries_see_their_prefix_and_nothing_sees_a_query(tmp_path):
    b = _batch(tmp_path)
    batch, t = b.kind.shape
    q = b.query_slot.shape[1]
    dense = create_mask(mask_mod_of(b), batch, None, t + q, t + q, device="cpu")[:, 0]
    length = (~b.pad).sum(dim=1)
    for i in range(batch):
        n = int(length[i])
        context = dense[i, :t, :t]
        assert torch.equal(context[:n, :n], torch.ones(n, n, dtype=torch.bool).tril())
        seen = dense[i, :, t:].clone()
        seen[t + torch.arange(q), torch.arange(q)] = False  # a query sees itself
        assert not seen.any()  # and nothing else sees it
        for j in range(q):
            if b.query_pad[i, j]:
                continue
            prefix = int(b.query_prefix[i, j])
            row = dense[i, t + j, :t]
            assert row[:prefix].all() and not row[prefix:].any()


def test_heads_lean_toward_their_own_candidate_by_their_fixed_bias():
    bias = torch.tensor([0.0, 2.0])
    slots = torch.tensor([[-1, 0, 0, 1]])  # a root token, two of candidate 0's, one of 1's
    mod = _same_slot_bias(bias, slots)

    def score(h, qi, kvi):
        return float(mod(torch.tensor(1.0), 0, h, torch.tensor(qi), torch.tensor(kvi)))

    assert score(1, 1, 2) == 3.0  # same candidate
    assert score(0, 1, 2) == 1.0  # a head without the bias
    assert score(1, 1, 3) == 1.0  # another candidate
    assert score(1, 0, 0) == 1.0  # root tokens have no candidate


def _reader(width: int = 64) -> Reader:
    return Reader(
        ReaderConfig(
            width=width, depth=1, heads=4, kv_heads=2, teacher_width=TEACHER_WIDTH, max_slots=16
        )
    )


@needs_cuda
def test_an_untrained_reader_answers_with_the_prior(tmp_path):
    b = _batch(tmp_path).to(torch.device("cuda"))
    out = _reader().cuda()(b)
    prior = prior_outputs(b)
    for name, value in out.items():
        torch.testing.assert_close(value, prior[name])


def test_the_prior_is_the_teachers_prediction_for_each_querys_candidate(tmp_path):
    b = _batch(tmp_path)
    k = len(b.held_out)
    gen = torch.Generator().manual_seed(0)
    wld = torch.softmax(torch.randn(k, 3, generator=gen), dim=1)
    mean_sd = torch.stack([torch.randn(k, generator=gen), torch.rand(k, generator=gen) + 0.1], 1)
    b.candidate["prior_value"] = torch.cat([wld, mean_sd], dim=1)
    b.candidate["prior_placement"] = torch.randn(
        b.candidate["prior_placement"].shape, generator=gen
    ).half()
    prior = prior_outputs(b)
    for i, j in [(0, 0), (1, int((~b.query_pad[1]).sum()) - 1)]:
        c = int(b.query_candidate[i, j])
        w, d, loss, mean, sd = b.candidate["prior_value"][c].tolist()
        torch.testing.assert_close(prior["wld"][i, j], torch.log(torch.tensor([w, d, loss])))
        torch.testing.assert_close(prior["score"][i, j], torch.tensor([mean, math.log(sd)]))
        torch.testing.assert_close(prior["expected"][i, j, 0], torch.tensor(w + d / 2))
        for head in FOOTPRINT_HEADS:
            logits = b.candidate["prior_placement"][
                c, PLACEMENT_HEAD_NAMES.index(f"{head}_placement")
            ]
            torch.testing.assert_close(prior[head][i, j], torch.log_softmax(logits.float(), dim=0))


def test_footprint_loss_skips_queries_with_no_next_move():
    logits = torch.zeros(3, 4)
    target = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    both = loss_mod._soft_ce(logits, target)
    assert torch.isfinite(both)
    # The empty row adds nothing and does not count toward the mean.
    torch.testing.assert_close(both, loss_mod._soft_ce(logits[[0, 2]], target[[0, 2]]))
    assert loss_mod._soft_ce(logits[[1]], target[[1]]) == 0


def test_beta_nll_gives_the_mean_a_squared_error_gradient_at_any_spread():
    grads = []
    for log_sd in (-4.0, 0.0):
        mean = torch.tensor([0.5], requires_grad=True)
        loss = loss_mod._beta_nll(
            mean, torch.tensor([log_sd]), torch.tensor([0.6]), torch.tensor([0.0]), scale=0.01
        )
        loss.backward()
        grads.append(float(mean.grad))
    # d/dmean of (0.6 - mean)^2 / (2 * 0.01), whatever the predicted spread.
    assert grads == pytest.approx([-10.0, -10.0])


def test_beta_nll_moves_the_spread_toward_the_misfit():
    def spread_grad(log_sd: float) -> float:
        log_sd_t = torch.tensor([log_sd], requires_grad=True)
        loss_mod._beta_nll(
            torch.tensor([0.5]), log_sd_t, torch.tensor([0.6]), torch.tensor([0.0]), 0.01
        ).backward()
        return float(log_sd_t.grad)

    # The misfit is 0.1: a narrower spread is pushed wider, a wider one narrower.
    assert spread_grad(np.log(0.05)) < 0 < spread_grad(np.log(0.2))


def test_rank_loss_weights_pairs_by_their_label_gap():
    def queries(mu):
        wld = torch.tensor([[0.6, 0.0, 0.4], [0.5, 0.0, 0.5], [0.5, 0.0, 0.5]])
        return loss_mod.Queries(
            out={"expected": torch.stack([torch.tensor(mu), torch.zeros(3)], dim=1)},
            target={"wld": wld, "n": torch.full((3,), 100.0)},
            row=torch.zeros(3, dtype=torch.long),
            prefix=torch.zeros(3, dtype=torch.long),
            candidate=torch.arange(3),
        )

    right = loss_mod._rank_loss(queries([0.6, 0.5, 0.0]), temperature=0.01)
    wrong = loss_mod._rank_loss(queries([0.4, 0.5, 0.5]), temperature=0.01)
    assert right < 0.01 < wrong
    # The tied pair (1, 2) costs nothing whichever way it is ordered.
    assert loss_mod._rank_loss(queries([0.6, 0.0, 0.5]), 0.01) < 0.01


@needs_cuda
def test_the_readout_scores_the_prior_and_an_untrained_reader_alike(tmp_path):
    b = _batch(tmp_path, rows=4).to(torch.device("cuda"))
    out = _reader().cuda()(b)
    result = loss_mod.finish_readout({k: float(v) for k, v in loss_mod.readout(out, b).items()})
    assert result["heldout_rmse_reader"] == pytest.approx(result["heldout_rmse_prior"], abs=1e-6)


@needs_cuda
def test_training_steps_lower_the_loss_on_a_fixed_batch(tmp_path):
    b = _batch(tmp_path, rows=4)
    params = dataclasses.replace(
        transfer_reader.TransferReaderParams(), lr=3e-3, warmup_steps=1, train_steps=30
    )
    model = _reader().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr)
    state = trainer.ReaderTrainState()
    batches = iter([b.apply(torch.Tensor.numpy)] * 30)
    first = trainer.run_steps(model, model.tower, optimizer, batches, "cuda", params, state, 1)
    last = trainer.run_steps(model, model.tower, optimizer, batches, "cuda", params, state, 29)
    assert state.steps == 30 and last["total"] < first["total"]


def test_split_is_fixed_by_file_and_position(tmp_path):
    files = [fake_file(tmp_path, seed=s) for s in range(3)]
    params = dataclasses.replace(transfer_reader.TransferReaderParams(), val_fraction=0.5)
    train, val = trainer.split_positions(files, params)
    assert sorted(train + val) == [(i, 0) for i in range(3)]
    assert trainer.split_positions(files[::-1], params)[1] == [
        (2 - i, p) for i, p in reversed(val)
    ]  # a position's side follows its file's name, not its index
    subset = dataclasses.replace(params, val_fraction=0.0, train_positions=2)
    assert len(trainer.split_positions(files, subset)[0]) == 2


def test_learning_rate_warms_up_then_decays_to_a_tenth_unless_unbounded():
    params = dataclasses.replace(
        transfer_reader.TransferReaderParams(), lr=1.0, warmup_steps=10, train_steps=110
    )
    assert trainer.lr_at(0, params) == pytest.approx(0.1)
    assert trainer.lr_at(9, params) == pytest.approx(1.0)
    assert trainer.lr_at(60, params) == pytest.approx(0.55)
    assert trainer.lr_at(110, params) == pytest.approx(0.1)
    unbounded = dataclasses.replace(params, train_steps=-1)
    assert trainer.lr_at(10**6, unbounded) == pytest.approx(1.0)


def test_every_profile_is_measured_and_placeable():
    spec = workloads.get("transfer_reader")
    for name, values in transfer_reader.PROFILES.items():
        params = params_mod.validate(spec.params_cls, values)
        (slot,) = transfer_reader.layout(params, vcpus=28, generator_threads=None)
        assert slot.role == "train" and slot.gpu_gb is not None, name


def test_finalize_refuses_a_corpus_without_prior_caches(tmp_path):
    paths = transfer_reader.TagPaths("reader", "transfer_reader", tmp_path)
    params = transfer_reader.TransferReaderParams(corpus_tag="c")
    store = transfer_reader.store_dir(params, tmp_path)
    store.mkdir(parents=True)
    with pytest.raises(params_mod.ParamsError, match="no probe files"):
        transfer_reader.finalize(transfer_reader.SPEC, paths, params)
    (store / "a.slog").touch()
    (store / "a.sprobe").touch()
    with pytest.raises(params_mod.ParamsError, match="prior cache"):
        transfer_reader.finalize(transfer_reader.SPEC, paths, params)
    (store / "a.sprior").touch()
    assert transfer_reader.finalize(transfer_reader.SPEC, paths, params) == params


@pytest.mark.parametrize("val_fraction", [0.0, 1.0])
def test_finalize_refuses_a_run_with_no_validation_or_no_training(tmp_path, val_fraction):
    """Every eval cycle validates; a run with no validation positions would
    crash at its first and restart forever."""
    paths = transfer_reader.TagPaths("reader", "transfer_reader", tmp_path)
    params = transfer_reader.TransferReaderParams(corpus_tag="c", val_fraction=val_fraction)
    with pytest.raises(params_mod.ParamsError, match="val_fraction"):
        transfer_reader.finalize(transfer_reader.SPEC, paths, params)


def test_the_scheduler_finishes_the_trainer_at_its_step_budget(tmp_path):
    spec = transfer_reader.SPEC
    paths = transfer_reader.TagPaths("reader", spec.name, tmp_path)
    params = transfer_reader.TransferReaderParams(train_steps=100)
    finished = []
    hooks = type("Hooks", (), {"paths": paths, "finish": staticmethod(finished.append)})
    task = type("Task", (), {"params": dataclasses.asdict(params)})
    transfer_reader.lifecycle.write_train_state(paths, {"steps": 99})
    transfer_reader.tick(spec, task, hooks)
    assert finished == []
    transfer_reader.lifecycle.write_train_state(paths, {"steps": 100})
    transfer_reader.tick(spec, task, hooks)
    assert finished == ["train"]
