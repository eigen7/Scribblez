"""The second stage of a self-played turn (scribblez.policy_set.survivors):
the .survivors sidecar read back, and a pool's labels under the turn's
decision."""

from pathlib import Path

import numpy as np
import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)

from scribblez import ffi  # noqa: E402
from scribblez.policy_set import dataset, survivors  # noqa: E402
from scribblez.policy_set.pools import Pool  # noqa: E402
from scribblez.policy_set.survivors import SURVIVORS, Survivors  # noqa: E402
from scribblez.selfplay import run_games  # noqa: E402

SOLVED = SURVIVORS["source_solved"]
SIMS = SURVIVORS["source_sims"]


def _write(path: Path, positions: list[tuple], solved_racks: int = 100, version=None):
    """A .survivors file of `positions`, each (game, turn, records, values)
    with records as (candidate, source, win_prob, q, rollouts)."""
    hdr = np.zeros(1, ffi.struct_dtype("SurvivorFileHeader"))
    hdr["magic"], hdr["num_positions"] = SURVIVORS["magic"], len(positions)
    hdr["version"] = SURVIVORS["version"] if version is None else version
    hdr["rollouts"], hdr["solved_racks"] = 1000, solved_racks
    out = [hdr.tobytes()]
    for game, turn, records, values in positions:
        ph = np.zeros(1, ffi.struct_dtype("SurvivorPositionHeader"))
        ph["game_index"], ph["turn_index"], ph["num_survivors"] = game, turn, len(records)
        r = np.zeros(len(records), ffi.struct_dtype("SurvivorRecord"))
        for i, (candidate, source, win_prob, q, rollouts) in enumerate(records):
            r[i]["candidate"], r[i]["source"], r[i]["rollouts"] = candidate, source, rollouts
            r[i]["win_prob"], r[i]["q"] = win_prob, q
        out += [ph.tobytes(), r.tobytes(), np.asarray(values, np.float32).tobytes()]
    path.write_bytes(b"".join(out))


def _survivors(records: list[tuple], values=()) -> Survivors:
    """Survivors of (candidate, solved, win_prob, q, rollouts) records."""
    c, solved, win, q, n = zip(*records, strict=True) if records else ([], [], [], [], [])
    return Survivors(
        game=0,
        turn=0,
        candidate=np.array(c, np.int64),
        solved=np.array(solved, bool),
        win_prob=np.array(win, np.float32),
        q=np.array(q, np.float32),
        rollouts=np.array(n, np.uint16),
        values=np.array(values, np.float32),
    )


def _pool(game: int, turn: int, size: int) -> Pool:
    """A pool whose screen spreads its mass evenly."""
    return Pool(
        game=game,
        turn=turn,
        num_legal_moves=size,
        opp_rack=np.zeros(1)[0],
        moves=np.zeros(size),
        win_prob=np.full(size, 0.5, np.float32),
        paired_se=np.zeros(size, np.float32),
        q=np.full(size, 1 / size, np.float32),
        equity_rank=np.arange(size, dtype=np.int32),
        rollouts=np.zeros(size, np.uint16),
        rollout_values=np.zeros(0, np.float32),
    )


def test_a_survivors_file_reads_back(tmp_path):
    path = tmp_path / "c.survivors"
    simmed = [(4, SIMS, 0.5, 0.75, 2), (1, SOLVED, 0.625, 0.0, 0), (0, SIMS, 0.25, 0.25, 1)]
    _write(path, [(2, 9, simmed, [1.0, 0.0, 0.25]), (2, 11, [], [])], solved_racks=64)
    hdr, (first, second) = survivors.read_survivors(path)
    assert (int(hdr["solved_racks"]), int(hdr["rollouts"])) == (64, 1000)
    assert (first.game, first.turn) == (2, 9)
    assert first.candidate.tolist() == [4, 1, 0]
    assert first.solved.tolist() == [False, True, False]
    assert first.win_prob.tolist() == [0.5, 0.625, 0.25]
    assert first.values.tolist() == [1.0, 0.0, 0.25]
    assert (second.turn, len(second.candidate)) == (11, 0)

    _write(path, [], version=SURVIVORS["version"] + 1)
    with pytest.raises(ValueError, match="version"):
        survivors.read_survivors(path)
    _write(path, [(0, 0, simmed, [1.0])])  # two values short
    with pytest.raises(ValueError, match="truncated"):
        survivors.read_survivors(path)


def test_a_raced_turns_decision_is_the_races_label():
    s = _survivors([(3, False, 0.6, 0.7, 2), (5, False, 0.5, 0.3, 2)], [1, 0, 1, 0])
    assert survivors.decision_q(s, solved_racks=100).tolist() == pytest.approx([0.7, 0.3])


def test_a_solved_value_is_set_beside_the_races_winner():
    """The agent compares the race's winner with each solved survivor by
    value; the race's share of the mass then splits by its label."""
    winner_values = [1, 0] * 50  # mean 0.5, a standard error of 0.05
    race = [(3, False, 0.5, 0.8, 100), (5, False, 0.4, 0.2, 2)]
    values = [*winner_values, 1, 0]

    level = survivors.decision_q(_survivors([*race, (7, True, 0.5, 0.0, 0)], values), 100)
    assert level.sum() == pytest.approx(1.0)
    assert level[2] == pytest.approx(0.5)  # equal values: an even chance
    assert level[:2].tolist() == pytest.approx([0.4, 0.1])

    ahead = survivors.decision_q(_survivors([*race, (7, True, 0.9, 0.0, 0)], values), 100)
    assert ahead[2] > 0.99
    behind = survivors.decision_q(_survivors([*race, (7, True, 0.1, 0.0, 0)], values), 100)
    assert behind[2] < 0.01 and behind[0] == pytest.approx(0.8, abs=0.01)
    # Fewer racks make a solved value less sure, so a gap counts for less.
    few = survivors.decision_q(_survivors([*race, (7, True, 0.6, 0.0, 0)], values), 4)
    many = survivors.decision_q(_survivors([*race, (7, True, 0.6, 0.0, 0)], values), 400)
    assert 0.5 < few[2] < many[2]


def test_solved_survivors_alone_split_the_mass_by_value():
    s = _survivors([(1, True, 0.75, 0.0, 0), (2, True, 0.75, 0.0, 0), (3, True, 0.0, 0.0, 0)])
    assert survivors.decision_q(s, 100).tolist() == pytest.approx([0.5, 0.5, 0.0])


def test_a_decided_pool_puts_its_mass_on_the_survivors_and_takes_their_values():
    pool = _pool(0, 0, size=6)
    s = _survivors([(3, False, 0.6, 0.7, 2), (5, False, 0.4, 0.3, 2)], [1, 0, 1, 0])
    decided = survivors.decided(pool, s, solved_racks=100)
    assert decided.q.tolist() == pytest.approx([0, 0, 0, 0.7, 0, 0.3])
    assert decided.win_prob.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.6, 0.5, 0.4])
    assert pool.q.tolist() == pytest.approx([1 / 6] * 6)  # the screen's pool is not touched
    # An agent that records no second stage leaves the screen's labels.
    assert survivors.decided(pool, _survivors([]), 100) is pool


def test_reading_a_self_played_chunk_gives_its_pools_their_decisions(tmp_path, monkeypatch):
    pool_path = tmp_path / "c.pool"
    pool_path.touch()
    screen = [_pool(0, 4, size=3), _pool(1, 2, size=3)]
    monkeypatch.setattr(dataset.pools, "read_pools", lambda path: (None, screen))
    # The pool tool's chunk has no .survivors: its pools are the screen's.
    assert [it.pool for it in dataset.read_items([tmp_path])] == screen

    one = [(2, SIMS, 0.5, 1.0, 1)]
    _write(tmp_path / "c.survivors", [(0, 4, one, [1.0]), (1, 2, one, [0.0])])
    items = dataset.read_items([tmp_path])
    assert [it.pool.q.tolist() for it in items] == [[0, 0, 1], [0, 0, 1]]
    assert items[0].slog == tmp_path / "c.slog"

    _write(tmp_path / "c.survivors", [(0, 4, one, [1.0]), (1, 3, one, [0.0])])
    with pytest.raises(ValueError, match="not those of c.pool"):
        dataset.read_items([tmp_path])


LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")
PASS = ffi.format_layout()["constants"]["move_type"]["pass"]
# DeepBot on the CPU reference with the synthetic leaf, at the smallest
# search that has both stages.
TINY_DEEPBOT = (
    "--type=deepbot --leaf-model synthetic --reply hasty --screen-rollouts 1 --rollouts 4 "
    "--candidates 2 --late-candidates 2"
)


@pytest.mark.skipif(not LEAVES.exists(), reason="HastyBot leave values not installed")
def test_a_self_played_game_gives_the_trainer_its_decisions(tmp_path):
    """The one case in which the engine writes what the trainer reads: a
    game of play_game's, two seconds of CPU sims, its pools batched as a
    training step takes them."""
    played = run_games(
        tmp_path, num_games=1, threads=2, player_spec=TINY_DEEPBOT, seed=1,
        pool_positions_per_game=3,
    )  # fmt: skip
    assert played == 0
    items = dataset.read_items([tmp_path])
    assert len(items) == 3
    for it in items:
        mass = np.flatnonzero(it.pool.q)
        assert 1 <= len(mass) <= 2  # the screen's two survivors at most
        assert it.pool.q.sum() == pytest.approx(1.0)
        assert not (it.pool.moves["type"][mass] == PASS).any()
        assert (it.pool.moves["type"] == PASS).sum() == 1  # the pool has the pass, as the tool's
    batch = dataset.PoolBatcher().build(items)
    assert len(batch["q"]) == sum(len(it.pool.moves) for it in items)
    assert batch["input_spatial"].shape[0] == 3
