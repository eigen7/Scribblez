import math

import numpy as np
from scribblez.match_eval.arms_report import Game, arm_game, compare, summarize

META_DTYPE = np.dtype(
    [
        ("seed", "<u8"),
        ("seat0_player", "i1"),
        ("seat1_player", "i1"),
        ("final_score_p0", "<i4"),
        ("final_score_p1", "<i4"),
    ]
)


def games(spreads: dict[tuple[int, int], int]) -> dict:
    return {k: Game(result=(float(np.sign(s)) + 1) / 2, spread=s) for k, s in spreads.items()}


def test_the_arms_seat_decides_whose_spread_it_is():
    m = np.array([(9, 1, 0, 400, 350)], dtype=META_DTYPE)[0]
    assert arm_game(m) == ((9, 1), Game(result=0.0, spread=-50))
    m = np.array([(9, 0, 1, 400, 400)], dtype=META_DTYPE)[0]
    assert arm_game(m) == ((9, 0), Game(result=0.5, spread=0))


def test_summary_counts_every_game_but_intervals_only_complete_pairs():
    s = summarize(games({(1, 0): 10, (1, 1): -10, (2, 0): 30, (2, 1): 0, (3, 0): 50}))
    assert (s.games, s.wins, s.draws, s.losses) == (5, 3, 1, 1)
    assert s.score.mean == 0.625  # pairs 1 and 2: 0.5 and 0.75
    assert s.spread.mean == 7.5  # 0 and 15
    assert math.isclose(s.spread.half_width, 1.96 * np.std([0, 15], ddof=1) / math.sqrt(2))


def test_comparison_differences_shared_pairs_and_counts_identical_games():
    arm = games({(1, 0): 10, (1, 1): 20, (2, 0): -5, (2, 1): 5, (3, 0): 1, (3, 1): 1})
    base = games({(1, 0): 10, (1, 1): -20, (2, 0): -5, (2, 1): 5})
    c = compare(arm, base)
    assert c.pairs == 2
    assert c.score.mean == 0.25  # pair 1: 1.0 vs 0.5; pair 2 even
    assert c.spread.mean == 10.0  # pair 1: 15 vs -5
    assert c.identical == 3
