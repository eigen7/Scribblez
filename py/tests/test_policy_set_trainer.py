"""The policy_set workload's trainer side: the set metrics
(scribblez.policy_set.metrics), the slot form and its coverage loss
(scribblez.policy_set.model), a generation's training and eval end to end
(scribblez.policy_set.trainer), and the workload's registration."""

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)

import torch  # noqa: E402
from cloud.sinks import LocalSink  # noqa: E402
from scribblez import ffi, workloads  # noqa: E402
from scribblez.dashboard import plots  # noqa: E402
from scribblez.generational import lifecycle  # noqa: E402
from scribblez.generational.checkpoint import GenerationalState  # noqa: E402
from scribblez.generational.optim import build_optim_arm, build_optimizer  # noqa: E402
from scribblez.leave_inference import win_table  # noqa: E402
from scribblez.paths import TagPaths  # noqa: E402
from scribblez.policy_set import (  # noqa: E402
    dataset,
    forms,
    greedy,
    metrics,
    model,
    pools,
    slot,
    train_loop,
    trainer,
)
from scribblez.position_eval.trainer import OutputDeliverer  # noqa: E402
from scribblez.selfplay import run_games  # noqa: E402
from scribblez.workloads import policy_set  # noqa: E402
from scribblez.workloads.policy_set import BELIEFS_UNIFORM, PolicySetParams  # noqa: E402

LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")


def _pool(q, win, rank) -> pools.Pool:
    n = len(q)
    return pools.Pool(
        game=0,
        turn=0,
        num_legal_moves=n,
        opp_rack_presence=np.zeros(27, np.float32),
        opp_leave_modelled=False,
        opp_leave_value_mean=0.0,
        opp_leave_value_sd=0.0,
        moves=np.zeros(n, ffi.struct_dtype("Move")),
        win_prob=np.array(win, np.float32),
        paired_se=np.zeros(n, np.float32),
        q=np.array(q, np.float32),
        equity_rank=np.array(rank, np.int32),
        rollouts=np.full(n, 8, np.uint16),
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
    tmp = tmp_path_factory.mktemp("chunk")
    chunk = tmp / "chunk"
    assert run_games(chunk, num_games=1, threads=2, player_spec="--type=hastybot-endgame") == 0
    table = tmp / "table.csv"
    spread = np.arange(-win_table.max_spread(), win_table.max_spread() + 1)
    grid = np.repeat(1 / (1 + np.exp(-spread / 30))[:, None], win_table.max_unseen() + 1, axis=1)
    win_table.write(table, grid, {"tag": "li", "generation": 3})
    p = PolicySetParams(leave_beliefs=BELIEFS_UNIFORM, rollouts=4, positions_per_game=2)
    policy_set.write_pools(p, table, None, chunk, threads=4)
    return chunk


# The smallest model that exercises every layer, on the CPU.
TINY = PolicySetParams(
    trunk_channels=16, num_blocks=1, num_heads=2, slots=4, slot_layers=1, batch_size=2
)


def test_the_equity_input_is_relative_to_each_pools_best(chunk):
    items = dataset.read_items([chunk])
    batch = dataset.PoolBatcher().build(items)
    eq, pos_id = batch["move_equity"].numpy(), batch["move_pos_id"].numpy()
    for j in range(len(items)):
        mine = eq[pos_id == j]
        assert mine.max() == 0 and mine.min() >= dataset.EQUITY_FLOOR
        # The best equity is the pool's equity rank 0, where the pool has one.
        ranks = items[j].pool.equity_rank
        if (ranks == 0).any():
            assert mine[ranks == 0][0] == 0


def test_the_slot_form_overfits_a_pool(chunk):
    items = dataset.read_items([chunk])[:1]
    batch = dataset.PoolBatcher().build(items)
    assert batch["opp_rack"].shape == (len(items), dataset.OPP_RACK_FLOATS)
    torch.manual_seed(0)
    m = trainer.build_models(replace(TINY, forms="slot", slots=16))["slot"]
    logits, pad = slot.slot_forward(m, batch)
    assert logits.shape == (len(items), 16, pad.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(4):
        total, _ = slot.slot_loss(slot.slot_forward(m, batch)[0], batch)
        opt.zero_grad()
        total.backward()
        opt.step()
        first = first if first is not None else total.item()
    assert total.item() < first


def test_a_generation_trains_evaluates_and_checkpoints(chunk, tmp_path, monkeypatch):
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
    m = trainer.build_models(TINY)  # both forms
    opt = build_optimizer(m, TINY)
    recorder = trainer.TrainRecorder(LocalSink(paths.root))
    staged = []
    monkeypatch.setattr(recorder, "deliver_staged", staged.append)
    ctx = {
        "config": {**trainer.versions()},
        "sink": LocalSink(paths.root),
        "device": device,
        "batcher": dataset.PoolBatcher(),
        "stats": type("S", (), {"cycle_done": lambda self, phases, units, nbytes: None})(),
        "deliverer": OutputDeliverer(),
    }
    state = GenerationalState()
    arm = build_optim_arm(recorder, TINY, opt, 0)
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
    ):
        assert np.isfinite(record[key]), key
    # The dashboard's figures read the record's keys; only set sizes past the
    # slots are absent.
    for _, keys in plots.POLICY_SET_QUALITY:
        for key in keys:
            assert key in record or int(key.rsplit("_", 1)[1]) > TINY.slots, key
    assert paths.rolling_checkpoint.exists()
    assert lifecycle.read_train_state(paths) == {"generation_index": 1, "rows_trained": 2}


class _BestFirstModel(torch.nn.Module):
    """Every slot ranks a position's candidates by their q."""

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


def test_gain_targets_are_the_margin_over_the_chosen_sets_best():
    win = torch.tensor([0.5, 0.4, 0.45, 0.3, 0.35])  # position 0: 0..2, position 1: 3..4
    rows = greedy.make_rows([0, 0, 1], [[], [1], [4]])
    pos_id = torch.tensor([0, 0, 0, 1, 1])
    move_idx, row_id = greedy.row_candidates(pos_id, rows.position)
    assert move_idx.tolist() == [0, 1, 2, 0, 1, 2, 3, 4]
    assert row_id.tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
    target = greedy.gain_targets(win, rows, move_idx, row_id)
    # Empty set: the value itself; else the margin over the set's best, or 0.
    assert torch.allclose(target, torch.tensor([0.5, 0.4, 0.45, 0.1, 0, 0.05, 0, 0]))


def test_subset_assembly_draws_proper_subsets_of_each_pool():
    win = np.array([0.5, 0.49, 0.1, 0.6, 0.2])
    pos_id = np.array([0, 0, 0, 1, 1])
    position, chosen = greedy.sample_rows(win, pos_id, 50, np.random.default_rng(0))
    for p, c in zip(position, chosen, strict=True):
        members = np.flatnonzero(pos_id == p)
        assert set(c) <= set(members) and len(c) < len(members) and len(set(c)) == len(c)
    # Toward high value: of the pairs from position 0, most hold both near-best moves.
    pairs = [frozenset(c) for p, c in zip(position, chosen, strict=True) if p == 0 and len(c) == 2]
    assert pairs.count(frozenset({0, 1})) > len(pairs) / 2


def test_the_greedy_form_overfits_a_pool_and_builds_distinct_sets(chunk):
    items = dataset.read_items([chunk])[:1]
    batch = dataset.PoolBatcher().build(items)
    torch.manual_seed(0)
    m = trainer.build_models(replace(TINY, forms="greedy"))["greedy"]
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    form = forms.FORMS["greedy"]
    rng = np.random.default_rng(0)
    losses = []
    for _ in range(4):
        total = sum(form.loss(m, batch, TINY, rng).values())
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
        target = greedy.gain_targets(self.value, rows, move_idx, row_id)
        return target + 1e-3 * self.value[move_idx], move_idx, row_id


@pytest.mark.parametrize(
    "n, expected", [(8, [[1, 2, 0], [0], [2, 4, 3, 1, 0]]), (2, [[1, 2], [0], [2, 4]])]
)
def test_greedy_picks_are_each_positions_own_local_indices(n, expected):
    # Pools of 3, 1 and 5 candidates: picks are local to each pool, stop at
    # its size, and follow the value order.
    value = torch.tensor([0.3, 0.9, 0.5, 0.2, 0.1, 0.4, 0.8, 0.6, 0.7])
    batch = {
        "move_pos_id": torch.tensor([0, 0, 0, 1, 2, 2, 2, 2, 2]),
        "input_spatial": torch.zeros(3, 1),
    }
    assert greedy.greedy_picks(_TrueGainModel(value), batch, n) == expected
