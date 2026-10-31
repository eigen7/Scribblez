"""The policy_set workload's trainer side: the set metrics
(scribblez.policy_set.metrics), the slot form and its coverage loss
(scribblez.policy_set.model), a generation's training and eval end to end
(scribblez.policy_set.trainer), and the workload's registration."""

import math
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)

import policy_set_chunk  # noqa: E402
import torch  # noqa: E402
from cloud.sinks import LocalSink  # noqa: E402
from scribblez import ffi, workloads  # noqa: E402
from scribblez import params as params_mod  # noqa: E402
from scribblez.dashboard import plots  # noqa: E402
from scribblez.generational import lifecycle  # noqa: E402
from scribblez.generational.checkpoint import GenerationalState  # noqa: E402
from scribblez.generational.optim import build_optim_arm, build_optimizer  # noqa: E402
from scribblez.paths import TagPaths  # noqa: E402
from scribblez.policy_set import (  # noqa: E402
    dataset,
    forms,
    greedy,
    metrics,
    model,
    onnx_export,
    policy,
    pools,
    slot,
    train_loop,
    trainer,
)
from scribblez.position_eval.trainer import OutputDeliverer  # noqa: E402
from scribblez.workloads.policy_set import PolicySetParams  # noqa: E402
from scribblez.workloads.worker import WorkerFatal  # noqa: E402

LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")
EQUITY_FLOOR = ffi.format_layout()["constants"]["policy_set"]["equity_floor"]


def _pool(q, win, rank, values=None) -> pools.Pool:
    """A pool of len(q) candidates, each with `values` as its rollout values
    (default: 8 rollouts at its win probability), the last a pass, as every
    pool's is."""
    n = len(q)
    values = values or [[w] * 8 for w in win]
    moves = np.zeros(n, ffi.struct_dtype("Move"))
    moves["type"][-1] = pools.PASS
    return pools.Pool(
        game=0,
        turn=0,
        num_legal_moves=n,
        opp_rack=np.zeros((), ffi.struct_dtype("PoolOppRack"))[()],
        moves=moves,
        win_prob=np.array(win, np.float32),
        paired_se=np.zeros(n, np.float32),
        q=np.array(q, np.float32),
        equity_rank=np.array(rank, np.int32),
        rollouts=np.array([len(v) for v in values], np.uint16),
        rollout_values=np.array([x for v in values for x in v], np.float32),
    )


# The leader (largest q) is equity rank 3 in A, and rank 12 (the tail) in B.
POOL_A = _pool([0.1, 0.6, 0.3], [0.40, 0.50, 0.45], [0, 3, 1])
POOL_B = _pool([0.7, 0.3], [0.60, 0.55], [12, 0])


def test_the_equity_baseline_metrics_by_set_size():
    m = metrics.set_metrics([POOL_A, POOL_B], metrics.equity_top, "equity_")
    assert m["equity_eval_recall_1"] == 0
    assert m["equity_eval_recall_4"] == 0.5  # A's leader enters at rank 3; B's never by 4
    assert m["equity_eval_recall_16"] == 1
    assert m["equity_eval_cover_1"] == pytest.approx((0.1 + 0.3) / 2)
    assert m["equity_eval_regret_1"] == pytest.approx((0.10 + 0.05) / 2)
    assert m["equity_eval_regret_16"] == pytest.approx(0)
    assert (m["equity_eval_tail_recall_4"], m["equity_eval_tail_recall_16"]) == (0, 1)


@pytest.mark.parametrize("bad", [lambda p, n: np.array([], int), lambda p, n: np.arange(n + 1)])
def test_a_chooser_must_return_a_set_of_at_most_n(bad):
    with pytest.raises(ValueError):
        metrics.set_metrics([_pool([0.5] * 20, [0.5] * 20, list(range(20)))], bad, "")


def test_pool_stats():
    s = metrics.pool_stats([POOL_A, POOL_B], "eval_")
    assert (s["eval_pools"], s["eval_pool_size"], s["eval_tail_share"]) == (2, 2.5, 0.5)
    assert s["eval_label_entropy"] > 0
    # Each leader is 5 points of win probability over its pool's pass.
    assert s["eval_stakes"] == pytest.approx(0.05)
    assert s["eval_stakes_ess_share"] == pytest.approx(1.0)


def test_stakes_are_the_leaders_win_probability_over_the_pass():
    assert pools.stakes(_pool([0.2, 0.8, 0.0], [0.30, 0.70, 0.40], [0, 1, -1])) == pytest.approx(
        0.30
    )
    # A pass that out-sims the leader (the leader's noise) has no stakes, not negative ones.
    assert pools.stakes(_pool([0.9, 0.1], [0.50, 0.55], [0, -1])) == 0.0
    s = metrics.pool_stats([_pool([1.0, 0.0], [0.6, 0.2], [0, -1]), POOL_A], "")
    # Stakes 0.4 and 0.05: (0.45)^2 / (2 (0.16 + 0.0025)).
    assert s["stakes_ess_share"] == pytest.approx(0.45**2 / (2 * 0.1625))


def test_coverage_loss_is_the_log_probability_the_prefix_covers_the_best_move():
    # One position, 2 slots over 3 moves (the third padding); the best is move 0.
    p1, p2 = torch.tensor([0.5, 0.5]), torch.tensor([0.25, 0.75])
    logits = torch.stack([p1.log(), p2.log()]).unsqueeze(0)
    logits = torch.cat([logits, torch.full((1, 2, 1), float("-inf"))], dim=2)
    q = torch.tensor([[1.0, 0.0, 0.0]])
    loss = slot.coverage_loss(logits, q, sizes=(1, 2))
    assert loss[1].item() == pytest.approx(-np.log(0.5))
    assert loss[2].item() == pytest.approx(-np.log(1 - 0.5 * 0.75))


@pytest.mark.parametrize("gap", [20.0, 200.0])
def test_the_coverage_loss_keeps_its_gradient_on_a_confidently_missed_move(gap):
    # Both slots put move 0 `gap` logits above move 1, which holds 30% of q.
    logits = torch.tensor([[[gap, 0.0], [gap, 0.0]]], requires_grad=True)
    q = torch.tensor([[0.7, 0.3]])
    loss = slot.coverage_loss(logits, q, sizes=(2,))[2]
    loss.backward()
    # The cover of move 1 is about 2 exp(-gap): its loss term is 0.3 * (gap - log 2).
    assert loss.item() == pytest.approx(0.3 * (gap - np.log(2)), rel=1e-3)
    assert logits.grad[0, :, 1].min() < -0.1


def test_a_slot_whose_move_is_taken_falls_back_to_its_next_choice():
    logits = torch.tensor([[[3.0, 2.0, 1.0], [3.0, 1.0, 2.0], [3.0, 2.0, 1.0]]])
    pad = torch.tensor([[False, False, False]])
    assert slot.slot_picks(logits, pad) == [[0, 2, 1]]
    # Fewer candidates than slots: the set stops at every candidate.
    assert slot.slot_picks(logits[:, :, :2], pad[:, :2]) == [[0, 1]]


@pytest.fixture(scope="module")
def chunk(tmp_path_factory) -> Path:
    """A one-game chunk surveyed into 2 pools under uniform beliefs."""
    if not LEAVES.exists():
        pytest.skip("HastyBot leave values not installed")
    return policy_set_chunk.surveyed_chunk(
        tmp_path_factory.mktemp("chunk"), rollouts=4, positions_per_game=2
    )


# The smallest model that exercises every layer, on the CPU.
TINY = PolicySetParams(
    train_slot=True,
    train_greedy=True,
    train_policy=True,
    trunk_channels=16,
    num_blocks=1,
    num_heads=2,
    slots=4,
    slot_layers=1,
    batch_size=2,
)


def only(form: str, **changes) -> PolicySetParams:
    """TINY training `form` alone."""
    flags = {f"train_{name}": name == form for name in forms.FORMS}
    return replace(TINY, **flags, **changes)


def test_the_equity_input_is_relative_to_each_pools_best(chunk):
    items = dataset.read_items([chunk])
    batch = dataset.PoolBatcher().build(items)
    eq, pos_id = batch["move_equity"].numpy(), batch["move_pos_id"].numpy()
    for j in range(len(items)):
        mine = eq[pos_id == j]
        assert mine.max() == 0 and mine.min() >= EQUITY_FLOOR
        # The best equity is the pool's equity rank 0, where the pool has one.
        ranks = items[j].pool.equity_rank
        if (ranks == 0).any():
            assert mine[ranks == 0][0] == 0


def _transposed_square(sq: np.ndarray) -> np.ndarray:
    return (sq % 15) * 15 + sq // 15


def _cross_entries(cells: np.ndarray, letters: np.ndarray) -> set:
    """One move's cross-check entries as {(axis, square, legal letters)}."""
    out = set()
    for k, cell in enumerate(cells):
        if cell:
            axis, sq = divmod(int(cell) - 1, 225)
            out.add((axis, sq, letters[k * 26 : (k + 1) * 26].tobytes()))
    return out


def test_a_transposed_batch_is_the_same_position_in_the_transposed_frame(chunk):
    """Transposing a position transposes its board planes, its candidates'
    squares and cross-check entries (whose axes swap), and nothing else: the
    scalars, the equity input and the labels are the same game's."""
    items = dataset.read_items([chunk])
    batcher = dataset.PoolBatcher()
    plain = batcher.build(items)
    flipped = batcher.build(items, np.ones(len(items), dtype=bool))
    letters = slice(0, 26)  # the board block's letter planes, which name no axis
    assert torch.equal(
        flipped["input_spatial"][:, letters], plain["input_spatial"][:, letters].transpose(2, 3)
    )
    assert not torch.equal(flipped["input_spatial"], plain["input_spatial"])
    for key in ("input_scalar", "opp_rack", "move_equity", "move_letters", "move_scalars", "q"):
        assert torch.equal(flipped[key], plain[key]), key
    mask = (
        plain["move_tile_mask"].numpy().astype(bool) & (plain["move_scalars"][:, 2:3] > 0).numpy()
    )
    sq = plain["move_squares"].numpy()
    assert np.array_equal(flipped["move_squares"].numpy()[mask], _transposed_square(sq[mask]))
    assert plain["move_cross_cells"].count_nonzero() > 0
    for m in range(len(sq)):
        mine = _cross_entries(
            plain["move_cross_cells"][m].numpy(), plain["move_cross_letters"][m].numpy()
        )
        theirs = _cross_entries(
            flipped["move_cross_cells"][m].numpy(), flipped["move_cross_letters"][m].numpy()
        )
        assert theirs == {(1 - a, int(_transposed_square(np.int64(s))), L) for a, s, L in mine}


def test_a_partly_transposed_batch_transposes_only_its_flagged_positions(chunk):
    """Each position of a mixed batch, and each of its candidates, is encoded
    as in the plain batch or the all-transposed one, by its own flag."""
    items = dataset.read_items([chunk])
    batcher = dataset.PoolBatcher()
    plain = batcher.build(items)
    flipped = batcher.build(items, np.ones(len(items), dtype=bool))
    flags = np.arange(len(items)) % 2 == 0
    assert flags.any() and not flags.all()
    mixed = batcher.build(items, flags)
    per_position = {"input_spatial", "input_scalar", "opp_rack", "stakes"}
    pos_id = plain["move_pos_id"].numpy()
    for key, got in mixed.items():
        rows = flags if key in per_position else flags[pos_id]
        assert torch.equal(got[rows], flipped[key][rows]), key
        assert torch.equal(got[~rows], plain[key][~rows]), key


def test_the_slot_form_overfits_a_pool(chunk):
    items = dataset.read_items([chunk])[:1]
    batch = dataset.PoolBatcher().build(items)
    assert batch["opp_rack"].shape == (len(items), dataset.OPP_RACK_FLOATS)
    torch.manual_seed(0)
    m = trainer.build_models(only("slot", slots=16))["slot"]
    logits, pad = slot.slot_forward(m, batch)
    assert logits.shape == (len(items), 16, pad.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    # The loss sits on a plateau for the first few steps (moving by under
    # 0.05, either way), then falls: 12 steps clear the plateau, and 10% is
    # well inside the fall.
    for _ in range(12):
        total, _ = slot.slot_loss(slot.slot_forward(m, batch)[0], batch)
        opt.zero_grad()
        total.backward()
        opt.step()
        first = first if first is not None else total.item()
    assert total.item() < 0.9 * first


def test_the_policy_form_overfits_a_pool_and_ranks_its_top_n(chunk):
    items = dataset.read_items([chunk])[:1]
    batch = dataset.PoolBatcher().build(items)
    torch.manual_seed(0)
    m = trainer.build_models(only("policy"))["policy"]
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    losses = []
    # As for the slot form: a plateau for the first ~7 steps (5.73 to 5.62),
    # then the fall (3.17 at step 12).
    for _ in range(12):
        loss = policy.policy_loss(m, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.9 * losses[0]
    (picks,) = forms.FORMS["policy"].sets(m, batch, 8)
    logits, _ = policy.policy_forward(m, batch)
    assert len(picks) == min(8, len(items[0].pool.moves)) and len(set(picks)) == len(picks)
    assert torch.equal(logits[0, picks], logits[0, picks].sort(descending=True).values)


def test_the_stakes_weighted_policy_loss_scales_each_positions_cross_entropy(chunk):
    items = dataset.read_items([chunk])
    batch = dataset.PoolBatcher().build(items)
    assert batch["stakes"].shape == (len(items),) and (batch["stakes"] >= 0).all()
    torch.manual_seed(0)
    m = trainer.build_models(only("policy"))["policy"]
    ce = policy.cross_entropies(m, batch)
    assert policy.policy_loss(m, batch).item() == pytest.approx(ce.mean().item(), rel=1e-5)
    assert policy.policy_loss(m, batch, weighted=True).item() == pytest.approx(
        (batch["stakes"] * ce).mean().item(), rel=1e-5
    )
    batch["stakes"] = torch.zeros_like(batch["stakes"])
    assert policy.policy_loss(m, batch, weighted=True).item() == 0.0


def test_the_train_flags_select_the_forms_and_one_is_required():
    assert forms.selected(TINY) == ("slot", "greedy", "policy")
    assert forms.selected(replace(TINY, train_greedy=False)) == ("slot", "policy")
    assert forms.selected(PolicySetParams()) == ("policy",)
    with pytest.raises(params_mod.ParamsError, match="no form to train"):
        replace(TINY, train_slot=False, train_greedy=False, train_policy=False)


def _one_generation_setup(chunk, tmp_path, monkeypatch):
    """A tag with the chunk as generation 0 and the held-out set, tiny models
    of every form, and a trainer context: what train_one_generation takes."""
    paths = TagPaths("t", "policy_set", tmp_path)
    (slog,) = chunk.glob("*.slog")
    for dest in (paths.generation_dir(0), paths.held_out_chunks_dir):
        dest.mkdir(parents=True)
        for f in (slog, slog.with_suffix(pools.SUFFIX)):
            shutil.copy(f, dest / f.name)
    lifecycle.write_manifest(paths.generation_dir(0), {"status": lifecycle.GENERATING})
    lifecycle.mark_complete(paths.generation_dir(0), 1)
    device = torch.device("cpu")
    torch.manual_seed(0)
    m = trainer.build_models(TINY)  # every form
    opt = build_optimizer(m, TINY)
    recorder = trainer.TrainRecorder(LocalSink(paths.root))
    staged = []
    monkeypatch.setattr(recorder, "deliver_staged", staged.append)
    ctx = {
        "config": {**asdict(TINY), **trainer.versions()},
        "tag": paths.tag,
        "sink": LocalSink(paths.root),
        "device": device,
        "batcher": dataset.PoolBatcher(),
        "stats": type("S", (), {"cycle_done": lambda self, phases, units, nbytes: None})(),
        "deliverer": OutputDeliverer(),
    }
    state = GenerationalState()
    arm = build_optim_arm(recorder, TINY, opt, 0)
    return paths, m, opt, arm, recorder, state, ctx, staged


def test_a_generation_trains_evaluates_and_checkpoints(chunk, tmp_path, monkeypatch):
    paths, m, opt, arm, recorder, state, ctx, staged = _one_generation_setup(
        chunk, tmp_path, monkeypatch
    )
    trainer.train_one_generation(m, opt, arm, recorder, paths, TINY, state, ctx)
    ctx["deliverer"].drain()
    (record,) = [s.record["metrics"] for s in staged]
    assert (state.generation_index, state.rows_trained) == (1, 2)
    for key in (
        "loss",
        "loss_greedy_gain",
        "slot_eval_recall_8",
        "greedy_eval_recall_8",
        "equity_eval_recall_8",
        "slot_eval_set_ms",
        "greedy_eval_set_ms",
        "param_norm_slot",
        "slot_eval_spread",
        "greedy_eval_spread",
        "loss_policy_ce",
        "policy_eval_loss_ce",
        "policy_eval_recall_8",
        "policy_eval_entropy",
        "policy_eval_ce",
        "policy_eval_stakes_ce",
    ):
        assert np.isfinite(record[key]), key
    assert record["slot_eval_pointer_scale"] == pytest.approx(slot.POINTER_SCALE_INIT, rel=0.1)
    # The dashboard's figures read the record's keys; only set sizes past the
    # slots are absent.
    for _, keys, *_ in plots.POLICY_SET_QUALITY:
        for key in keys:
            assert key in record or int(key.rsplit("_", 1)[1]) > TINY.slots, key
    assert paths.rolling_checkpoint.exists()
    assert lifecycle.read_train_state(paths) == {"generation_index": 1, "rows_trained": 2}
    # The policy form's export for PolicyBot, stamped with what it must match.
    meta = {p.key: p.value for p in onnx.load(paths.onnx_path(0)).metadata_props}
    assert (meta["graph"], meta["tag"], meta["generation"]) == ("policy_set", "t", "0")
    assert meta["blank_prune_points"] == str(TINY.blank_prune_points)


# The export inputs whose engine dtype differs from the batch's.
_ENGINE_DTYPES = {
    "move_letters": np.int32,
    "move_squares": np.int32,
    "move_cross_cells": np.int32,
    "move_blanks": np.uint8,
    "move_tile_mask": np.uint8,
}


def _engine_feed(batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """One position's batch as the engine stages it: the export's inputs in
    their native dtypes (move_set_encoder.h), the equity a column."""
    feed = {name: batch[name].numpy() for name in onnx_export.input_names()}
    feed |= {name: feed[name].astype(dtype) for name, dtype in _ENGINE_DTYPES.items()}
    feed["move_equity"] = feed["move_equity"].reshape(-1, 1)
    return feed


def test_the_export_scores_a_pool_as_the_policy_form_does(chunk, tmp_path):
    """PolicyBot's graph, run on a pool's inputs as the engine stages them,
    gives the policy form's own logits."""
    torch.manual_seed(0)
    m = trainer.build_models(only("policy"))["policy"].eval()
    # The cross-check encoder's output starts at zero; perturbed, so its
    # inputs are read.
    with torch.no_grad():
        for p in m.encoder.cross_check_encoder.out.parameters():
            p.add_(0.1 * torch.randn_like(p))
    path = tmp_path / "policy.onnx"
    config = {**asdict(only("policy")), **trainer.versions()}
    onnx_export.export_onnx(m, path, config, {"tag": "t", "generation": 0})
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for item in dataset.read_items([chunk]):
        batch = dataset.PoolBatcher().build([item])
        with torch.no_grad():
            want, _ = m(batch)
        (got,) = session.run([onnx_export.OUTPUT], _engine_feed(batch))
        np.testing.assert_allclose(got[:, 0], want[0].numpy(), rtol=1e-4, atol=1e-4)


def test_a_checkpoint_exports_its_policy_form(chunk, tmp_path):
    """The script's path: a rolling checkpoint's policy form, rebuilt from its
    config, has the trained weights."""
    torch.manual_seed(0)
    models = trainer.build_models(only("policy"))
    ckpt = {
        "model_state_dict": models.state_dict(),
        "config": {**asdict(only("policy")), **trainer.versions()},
    }
    rebuilt = onnx_export.policy_model(ckpt)
    for name, value in models["policy"].state_dict().items():
        assert torch.equal(rebuilt.state_dict()[name], value), name


def test_a_non_finite_step_stops_the_trainer_before_its_checkpoint(chunk, tmp_path, monkeypatch):
    """A NaN weight makes the first step's loss non-finite: the trainer stops
    fatally, naming the form and the batch's pools, and writes no checkpoint,
    so the last good one stays the one to resume from."""
    paths, m, opt, arm, recorder, state, ctx, _ = _one_generation_setup(
        chunk, tmp_path, monkeypatch
    )
    with torch.no_grad():
        next(m["greedy"].gain.parameters()).view(-1)[0] = math.nan
    with pytest.raises(WorkerFatal, match=r"non-finite greedy loss.*at batch 0 \(chunk game"):
        trainer.train_one_generation(m, opt, arm, recorder, paths, TINY, state, ctx)
    ctx["deliverer"].drain()
    assert not paths.rolling_checkpoint.exists()


def test_the_pointer_logits_are_bounded_by_the_scale_cap():
    """However large the candidate vectors and however the scale trains, a
    logit is the capped scale times a cosine."""
    torch.manual_seed(0)
    dec = slot.SlotDecoder(channels=8, slots=3, layers=1, num_heads=2)
    with torch.no_grad():
        dec.log_scale.fill_(10.0)  # far past the cap
    h = 1e4 * torch.randn(5, 8)
    logits, pad = dec(h, torch.randn(2, 24), torch.tensor([0, 0, 0, 1, 1]))
    assert logits.dtype == torch.float32
    assert logits[~pad.unsqueeze(1).expand_as(logits)].abs().max() <= slot.POINTER_SCALE_MAX + 1e-3


def test_the_gain_head_passes_gradient_far_below_zero():
    """The softplus head the first run had stopped learning once the zero
    targets pushed it far negative; the linear head always passes gradient."""
    torch.manual_seed(0)
    enc = model.PolicySetEncoder(4, 2, 8, 1, 2)
    head = greedy.GreedyModel(enc, num_heads=2).gain
    with torch.no_grad():
        head[-1].bias.fill_(-400.0)
    out = head(torch.randn(6, 8))
    (out - 0.05).square().sum().backward()
    assert out.max() < 0 and head[-1].bias.grad.abs() > 1.0


def test_mean_spread_is_each_positions_deviation_over_its_candidates():
    grid = torch.tensor([[1.0, 3.0, 0.0], [5.0, 5.0, 5.0]])
    pad = torch.tensor([[False, False, True], [False, False, False]])
    assert model.mean_spread(grid, pad).item() == pytest.approx(0.5)


class _BestFirstModel(torch.nn.Module):
    """Every slot ranks a position's candidates by their q."""

    decoder = SimpleNamespace(scale=lambda: torch.tensor(1.0))  # what the diagnostics read

    def forward(self, batch):
        num_positions = batch["input_spatial"].shape[0]
        grid, pad = model.to_grid(batch["q"], batch["move_pos_id"], num_positions)
        logits = grid.masked_fill(pad, float("-inf")).unsqueeze(1).expand(-1, 2, -1)
        return logits, pad


class _LabelBatcher:
    """A PoolBatcher stand-in carrying only what evaluate reads."""

    def build(self, items):
        q = [torch.from_numpy(it.pool.q) for it in items]
        return {
            "input_spatial": torch.zeros(len(items), 1),
            "q": torch.cat(q),
            "move_pos_id": torch.cat([torch.full((len(x),), j) for j, x in enumerate(q)]),
        }


def test_evaluate_scores_each_pool_by_its_own_slots():
    # The leaders sit at different indices, and 3 pools span 2 batches, so a
    # pick read against the wrong pool misses.
    pool_c = _pool([0.1, 0.2, 0.7], [0.40, 0.45, 0.50], [0, 1, 2])
    items = [dataset.PoolItem(Path("x.slog"), p) for p in (POOL_A, POOL_B, pool_c)]
    models = torch.nn.ModuleDict({"slot": _BestFirstModel()})
    m = train_loop.evaluate(models, _LabelBatcher(), items, torch.device("cpu"), TINY)
    assert (m["slot_eval_recall_1"], m["slot_eval_regret_1"]) == (1, 0)


def test_a_checkpoint_from_another_version_is_refused(tmp_path):
    paths = TagPaths("t", "policy_set", tmp_path)
    paths.rolling_checkpoint.parent.mkdir(parents=True)
    config = trainer.versions()
    torch.save({"config": {**config, "policy_set_format_version": -1}}, paths.rolling_checkpoint)
    with pytest.raises(RuntimeError, match="policy_set_format_version"):
        trainer.check_resumable(paths, config)


def test_the_workload_registers_gpu_generators_and_a_gpu_data_home_trainer():
    spec = workloads.get("policy_set")
    assert spec.role("generate").gpu and spec.role("train").gpu
    assert spec.role("train").singleton and spec.role("train").ingest
    assert spec.finalize and spec.role("generate").inputs


def test_a_checkpoint_from_this_version_or_none_resumes(tmp_path):
    paths = TagPaths("t", "policy_set", tmp_path)
    config = trainer.versions()
    trainer.check_resumable(paths, config)  # no checkpoint
    paths.rolling_checkpoint.parent.mkdir(parents=True)
    torch.save({"config": dict(config)}, paths.rolling_checkpoint)
    trainer.check_resumable(paths, config)


def test_an_empty_chosen_set_leaves_the_board_unchanged():
    torch.manual_seed(0)
    stage = greedy.SetStage(16, 2)
    for p in stage.parameters():  # off the zero init, so a set would change it
        p.data.add_(0.1 * torch.randn_like(p))
    board, g, tokens = torch.randn(2, 225, 16), torch.randn(2, 48), torch.randn(2, 3, 16)
    mask = torch.tensor([[False] * 3, [True, True, False]])
    out_board, out_g = stage(board, g, tokens, mask)
    assert torch.equal(out_board[0], board[0]) and torch.equal(out_g[0], g[0])
    assert not torch.allclose(out_board[1], board[1])


def test_rows_list_every_candidate_of_their_position():
    rows = greedy.make_rows([0, 0, 1], [[0], [1], [4]])
    pos_id = torch.tensor([0, 0, 0, 1, 1])
    move_idx, row_id = greedy.row_candidates(pos_id, rows.position)
    assert move_idx.tolist() == [0, 1, 2, 0, 1, 2, 3, 4]
    assert row_id.tolist() == [0, 0, 0, 1, 1, 1, 2, 2]


def _expected_improvement(diffs) -> float:
    """E[max(0, X)], X normal about the mean of `diffs` with its standard error."""
    d = np.array(diffs, np.float64)
    mean, se = d.mean(), np.sqrt(d.var() / len(d))
    if se == 0:
        return max(0.0, mean)
    z = mean / se
    return mean * 0.5 * (1 + math.erf(z / math.sqrt(2))) + se * math.exp(-z * z / 2) / math.sqrt(
        2 * math.pi
    )


def test_gain_targets_are_expected_improvement_over_a_member():
    # Move 1 beats move 0 on one rollout of four; move 2 plays out exactly as
    # move 0; racing stopped move 3 after two rollouts.
    pool = _pool(
        [0.25] * 4,
        [0.5, 0.75, 0.5, 0.0],
        [0, 1, 2, 3],
        values=[[1, 0, 1, 0], [1, 1, 1, 0], [1, 0, 1, 0], [0, 0]],
    )
    grid = dataset.rollout_grid([pool])
    assert grid["rollout_values"][3].tolist() == [0, 0, 0, 0]
    assert grid["rollouts"].tolist() == [4, 4, 4, 2]
    rows = greedy.make_rows([0], [[0]])
    move_idx, row_id = greedy.row_candidates(torch.zeros(4, dtype=torch.int64), rows.position)
    target = greedy.gain_targets(grid, rows, move_idx, row_id)
    expected = [
        0,  # chosen
        _expected_improvement([0, 1, 0, 0]),
        0,  # identical outcomes on every rollout
        _expected_improvement([-1, 0]),  # over the two rollouts both ran
    ]
    np.testing.assert_allclose(target.numpy(), expected, rtol=1e-5, atol=1e-7)
    assert 0 < expected[3] < expected[1]


def test_gain_targets_read_the_right_pool_across_a_batch():
    # Two pools with different candidate and rollout counts: a wrong offset in
    # rollout_grid or the global move indices would read pool 0's values.
    first = _pool([0.5] * 2, [0.5, 0.5], [0, 1], values=[[1, 0, 1], [0, 1, 1]])
    a, b, c = [1, 1, 0, 1, 0], [0, 1, 1, 0, 0], [1, 1, 1, 1, 0]
    second = _pool([0.25] * 3, [0.6, 0.4, 0.8], [0, 1, 2], values=[a, b, c])
    grid = dataset.rollout_grid([first, second])
    assert grid["rollouts"].tolist() == [3, 3, 5, 5, 5]
    assert grid["rollout_values"][3].tolist() == b
    assert grid["rollout_values"][1].tolist() == [0, 1, 1, 0, 0]
    rows = greedy.make_rows([1], [[2]])
    pos_id = torch.tensor([0, 0, 1, 1, 1])
    move_idx, row_id = greedy.row_candidates(pos_id, rows.position)
    assert move_idx.tolist() == [2, 3, 4]
    target = greedy.gain_targets(grid, rows, move_idx, row_id).numpy()
    expected = [
        0,
        _expected_improvement(np.subtract(b, a)),
        _expected_improvement(np.subtract(c, a)),
    ]
    np.testing.assert_allclose(target, expected, rtol=1e-5, atol=1e-7)


def test_a_single_shared_rollout_gives_the_hard_difference():
    pool = _pool([0.5, 0.5], [0.5, 0.5], [0, 1], values=[[0, 1, 1], [1]])
    grid = dataset.rollout_grid([pool])
    rows = greedy.make_rows([0], [[0]])
    move_idx, row_id = greedy.row_candidates(torch.zeros(2, dtype=torch.int64), rows.position)
    target = greedy.gain_targets(grid, rows, move_idx, row_id)
    assert target.tolist() == [0.0, 1.0]  # 1 - 0 on the one shared rollout


def test_a_move_that_sims_as_a_member_does_gains_nothing():
    # A is the set's best. C plays out exactly as B, a weaker member, and D
    # plays out unlike either; against A alone, C and D look alike.
    a, b = [1, 1, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0]
    d = [1, 0, 1, 0, 1, 0, 1, 0]
    pool = _pool([0.25] * 4, [0.625, 0.5, 0.5, 0.5], [0, 1, 2, 3], values=[a, b, b, d])
    grid = dataset.rollout_grid([pool])
    rows = greedy.make_rows([0, 0], [[0], [0, 1]])
    move_idx, row_id = greedy.row_candidates(torch.zeros(4, dtype=torch.int64), rows.position)
    target = greedy.gain_targets(grid, rows, move_idx, row_id).numpy()
    c_vs_a = _expected_improvement(np.subtract(b, a))
    d_vs_a = _expected_improvement(np.subtract(d, a))
    # The set {A}: C and D are each measured against A.
    np.testing.assert_allclose(target[[2, 3]], [c_vs_a, d_vs_a], rtol=1e-5)
    # The set {A, B}: B covers C entirely; D keeps the lesser of its two.
    d_vs_b = _expected_improvement(np.subtract(d, b))
    np.testing.assert_allclose(target[[4, 5, 6, 7]], [0, 0, 0, min(d_vs_a, d_vs_b)], atol=1e-7)
    assert target[2] > 0.05 and target[7] > 0.05


def test_subset_assembly_draws_proper_subsets_holding_each_pools_seed():
    # Position 0's seed (the top equity) is move 2; position 2 has one candidate.
    win = np.array([0.5, 0.49, 0.1, 0.05, 0.6, 0.2, 0.3])
    equity = np.array([-1.0, -2.0, 0.0, -3.0, 0.0, -1.0, 0.0])
    pos_id = np.array([0, 0, 0, 0, 1, 1, 2])
    seeds = {0: 2, 1: 4, 2: 6}
    position, chosen = greedy.sample_rows(win, equity, pos_id, 50, np.random.default_rng(0))
    for p, c in zip(position, chosen, strict=True):
        members = np.flatnonzero(pos_id == p)
        assert c[0] == seeds[p] and len(set(c)) == len(c)
        assert set(c) <= set(members) and (len(c) < len(members) or len(members) == 1)
    # Toward high value: of position 0's sets of the seed and two more, most
    # add both near-best moves.
    triples = [
        frozenset(c) for p, c in zip(position, chosen, strict=True) if p == 0 and len(c) == 3
    ]
    assert triples.count(frozenset({2, 0, 1})) > len(triples) / 2


def test_a_drawn_seed_share_draws_some_seeds_toward_high_value():
    # The top equity is move 2, but moves 0 and 1 hold the high win probability.
    win = np.array([0.5, 0.49, 0.1, 0.05])
    equity = np.array([-1.0, -2.0, 0.0, -3.0])
    pos_id = np.zeros(4, dtype=np.int64)
    _, chosen = greedy.sample_rows(win, equity, pos_id, 200, np.random.default_rng(0), 0.5)
    seeds = [c[0] for c in chosen]
    drawn = sum(s in (0, 1) for s in seeds)
    assert 60 < drawn < 140 and seeds.count(2) > 60
    assert all(len(set(c)) == len(c) and len(c) < 4 for c in chosen)


def test_the_greedy_form_overfits_a_pool_and_builds_distinct_sets(chunk):
    items = dataset.read_items([chunk])[:1]
    batch = dataset.PoolBatcher().build(items)
    torch.manual_seed(0)
    m = trainer.build_models(only("greedy"))["greedy"]
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    form = forms.FORMS["greedy"]
    losses = []
    for _ in range(4):
        # The same chosen sets every step: a set's targets set its loss's
        # scale, so fresh draws would not compare.
        total = sum(form.loss(m, batch, TINY, np.random.default_rng(0)).values())
        opt.zero_grad()
        total.backward()
        opt.step()
        losses.append(total.item())
    assert losses[-1] < losses[0]
    (picks,) = form.sets(m, batch, 8)
    assert len(picks) == min(8, len(items[0].pool.moves)) and len(set(picks)) == len(picks)


class _TrueGainModel:
    """A GreedyModel stand-in whose gains are the true margins over the chosen
    set, plus a tiebreak toward higher value."""

    def __init__(self, value: torch.Tensor):
        self.value = value

    def cache(self, batch):
        return None, None

    def gains(self, enc, h, pos_id, rows):
        move_idx, row_id = greedy.row_candidates(pos_id, rows.position)
        chosen_v = torch.where(rows.mask, self.value[rows.chosen], -torch.inf)
        margin = (self.value[move_idx] - chosen_v.max(dim=1).values[row_id]).clamp(min=0)
        return margin + 1e-3 * self.value[move_idx], move_idx, row_id


@pytest.mark.parametrize(
    "n, expected", [(8, [[2, 1, 0], [0], [0, 2, 4, 3, 1]]), (2, [[2, 1], [0], [0, 2]])]
)
def test_greedy_picks_are_each_positions_own_local_indices(n, expected):
    # Pools of 3, 1 and 5 candidates: each set starts from the pool's seed
    # (its top equity), then follows the value order; picks are local to each
    # pool and stop at its size.
    value = torch.tensor([0.3, 0.9, 0.5, 0.2, 0.1, 0.4, 0.8, 0.6, 0.7])
    batch = {
        "move_pos_id": torch.tensor([0, 0, 0, 1, 2, 2, 2, 2, 2]),
        "move_equity": torch.tensor([-1.0, -2.0, 0.0, 0.0, 0.0, -1.0, -2.0, -3.0, -4.0]),
        "input_spatial": torch.zeros(3, 1),
    }
    assert greedy.greedy_picks(_TrueGainModel(value), batch, n) == expected
