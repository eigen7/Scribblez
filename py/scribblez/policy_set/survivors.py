"""The second stage of a self-played turn: a .survivors sidecar (engine
data/survivor_log.h, which owns the format), position for position beside the
.pool of the agent's screen, and the policy's target from the two.

The screen labels every move of the turn; its survivors then get the sims the
agent decides by, raced, or the endgame solver's value where the move empties
the bag (docs/plans/deep_bot_generations.md, "Self-play writes the pools").
The agent plays the best value: a solved survivor's, or that of the race's
likeliest best. `decided` gives a pool the labels of that decision.
"""

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scribblez import ffi
from scribblez.policy_set import SURVIVORS_SUFFIX
from scribblez.policy_set.pools import Pool

SURVIVORS = ffi.format_layout()["constants"]["survivors"]
SUFFIX = SURVIVORS_SUFFIX


@dataclass
class Survivors:
    """One position's survivors (parallel arrays): each one's index among the
    pool's candidates, its value, and for a simmed one its label among the
    simmed (engine sim/pool_labels.h) and its rollouts' win values."""

    game: int
    turn: int
    candidate: np.ndarray  # (S,) int64
    solved: np.ndarray  # (S,) bool: the endgame solver's value, not sims'
    win_prob: np.ndarray  # (S,) float32
    q: np.ndarray  # (S,) float32, over the simmed; 0 for a solved one
    rollouts: np.ndarray  # (S,) uint16, the values kept; 0 for a solved one
    values: np.ndarray  # (sum of rollouts,) float32, each survivor's in turn


def read_survivors(path: str | Path) -> tuple[np.void, list[Survivors]]:
    """A .survivors file's header and its positions, in file order. Raises
    ValueError for another format version or a malformed file."""
    hdr_t = ffi.struct_dtype("SurvivorFileHeader")
    pos_t = ffi.struct_dtype("SurvivorPositionHeader")
    rec_t = ffi.struct_dtype("SurvivorRecord")
    buf = np.fromfile(path, np.uint8)
    if len(buf) < hdr_t.itemsize:
        raise ValueError(f"{path}: truncated header")
    hdr = np.frombuffer(buf, hdr_t, count=1)[0]
    if hdr["magic"] != SURVIVORS["magic"] or hdr["version"] != SURVIVORS["version"]:
        raise ValueError(
            f"{path}: .survivors version {hdr['version']}, but this code reads version "
            f"{SURVIVORS['version']}"
        )
    off, out = hdr_t.itemsize, []
    for _ in range(int(hdr["num_positions"])):
        if off + pos_t.itemsize > len(buf):
            raise ValueError(f"{path}: truncated at position {len(out)}")
        ph = np.frombuffer(buf, pos_t, count=1, offset=off)[0]
        off += pos_t.itemsize
        n = int(ph["num_survivors"])
        if off + n * rec_t.itemsize > len(buf):
            raise ValueError(f"{path}: truncated at position {len(out)}")
        r = np.frombuffer(buf, rec_t, count=n, offset=off)
        off += n * rec_t.itemsize
        k = int(r["rollouts"].sum())
        if off + 4 * k > len(buf):
            raise ValueError(f"{path}: truncated at position {len(out)}")
        values = np.frombuffer(buf, np.float32, count=k, offset=off)
        off += 4 * k
        out.append(
            Survivors(
                game=int(ph["game_index"]),
                turn=int(ph["turn_index"]),
                candidate=r["candidate"].astype(np.int64),
                solved=r["source"] == SURVIVORS["source_solved"],
                win_prob=r["win_prob"].copy(),
                q=r["q"].copy(),
                rollouts=r["rollouts"].copy(),
                values=values.copy(),
            )
        )
    if off != len(buf):
        raise ValueError(f"{path}: {len(buf) - off} bytes past the last position")
    return hdr, out


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _best_odds(value: float, se: float, leader: float, leader_se: float) -> float:
    """The odds that a value estimated at `value` with standard error `se`
    exceeds one estimated at `leader` with `leader_se`, the two independent:
    the pool label's odds (engine sim/pool_labels.h) without the pairing."""
    spread = math.hypot(se, leader_se)
    p = _normal_cdf((value - leader) / spread) if spread > 0 else 0.5 * (value == leader)
    return p / max(1.0 - p, 1e-9)


def race_winner(s: Survivors) -> tuple[int, float]:
    """The simmed survivor likeliest the best of the race, as an index into
    `s`, and its value's standard error over its rollouts. Raises ValueError
    if none was simmed."""
    simmed = np.flatnonzero(~s.solved)
    if len(simmed) == 0:
        raise ValueError("no survivor was simmed")
    winner = int(simmed[np.argmax(s.q[simmed])])
    start = int(s.rollouts[:winner].sum())
    values = s.values[start : start + int(s.rollouts[winner])].astype(np.float64)
    return winner, float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def decision_q(s: Survivors, solved_racks: int) -> np.ndarray:
    """(S,) float64, summing to 1: each survivor's chance of being the best,
    as the agent compares them.

    Among the simmed it is the race's label, paired on shared rollouts. The
    agent sets the race's winner beside each solved survivor by value, so
    where some are solved, those contenders split the mass by the label's
    rule on independent estimates (_best_odds): the winner's standard error
    from its rollouts, a solved value's from its `solved_racks` endgames, each
    worth 1, 1/2 or 0. The race's share then goes to the simmed by their
    label."""
    q = np.where(s.solved, 0.0, s.q.astype(np.float64))
    solved = np.flatnonzero(s.solved)
    if len(solved) == 0:
        return q / q.sum()
    v = s.win_prob.astype(np.float64)
    se = np.sqrt(v * (1.0 - v) / max(solved_racks, 1))
    contenders = list(solved)
    if len(solved) < len(q):
        winner, winner_se = race_winner(s)
        contenders.append(winner)
        se[winner] = winner_se
    leader = max(contenders, key=lambda i: v[i])
    odds = {
        i: 1.0 if i == leader else _best_odds(v[i], se[i], v[leader], se[leader])
        for i in contenders
    }
    total = sum(odds.values())
    race_share = odds[winner] / total if len(solved) < len(q) else 0.0
    out = q / q.sum() * race_share if race_share > 0 else np.zeros_like(q)
    for i in solved:
        out[i] = odds[i] / total
    return out


def decided(pool: Pool, s: Survivors, solved_racks: int) -> Pool:
    """`pool` under the turn's decision: q is decision_q over its survivors
    and 0 for the moves the screen dropped, and a survivor's win probability
    is its second stage's. The rollout values stay the screen's, the same
    sims on other racks under the hasty reply, which is what the value head's
    variance target and the regret metric read. A position with no survivors
    (an agent that records no second stage) keeps the screen's labels."""
    if len(s.candidate) == 0:
        return pool
    q = np.zeros(len(pool.moves), dtype=np.float32)
    q[s.candidate] = decision_q(s, solved_racks)
    win_prob = pool.win_prob.copy()
    win_prob[s.candidate] = s.win_prob
    return dataclasses.replace(pool, q=q, win_prob=win_prob)


def decided_pools(pool_path: Path, pool_list: list[Pool]) -> list[Pool]:
    """`pool_list`, the pools of the .pool at `pool_path`, each under its
    turn's decision where the file has a .survivors beside it; unchanged
    where it has none (the pool tool's). Raises ValueError if the two files'
    positions do not line up."""
    path = pool_path.with_suffix(SUFFIX)
    if not path.exists():
        return pool_list
    hdr, survivors = read_survivors(path)
    if [(p.game, p.turn) for p in pool_list] != [(s.game, s.turn) for s in survivors]:
        raise ValueError(f"{path}: its positions are not those of {pool_path.name}")
    racks = int(hdr["solved_racks"])
    return [decided(p, s, racks) for p, s in zip(pool_list, survivors, strict=True)]
