"""The policy-set eval (docs/plans/deep_bot_v0.md, "Eval metrics"): how well a
chooser of candidate sets covers each pool's best move, at each set size N.

A chooser maps a pool and N to the indices of at most N of its candidates.
The static-equity top N is the baseline chooser, the set every DeepBot set has
to beat; the policy-set model's forms are choosers too (step 8).

Per N, over the pools:
  recall       how often the pool's sim-best move (the leader, the largest q)
               is in the set
  cover        the q the set holds: the soft recall, P(best in the set)
  regret       the leader's win probability minus the set's best
  tail_recall  recall on the pools whose leader is outside the static-equity
               top TAIL_RANK
"""

from collections.abc import Callable

import numpy as np

from scribblez.policy_set.pools import Pool

SET_SIZES = (1, 2, 4, 8, 16)
TAIL_RANK = 10

Chooser = Callable[[Pool, int], np.ndarray]


def equity_top(pool: Pool, n: int) -> np.ndarray:
    """The baseline: the candidates ranked in the static-equity top `n`."""
    return np.flatnonzero((pool.equity_rank >= 0) & (pool.equity_rank < n))


def _leader(pool: Pool) -> int:
    return int(np.argmax(pool.q))


def _in_tail(pool: Pool) -> bool:
    rank = pool.equity_rank[_leader(pool)]
    return not 0 <= rank < TAIL_RANK


def set_metrics(pools: list[Pool], choose: Chooser, prefix: str) -> dict[str, float]:
    """Each metric at each N in SET_SIZES, keyed "<prefix>eval_<metric>_<N>".
    A chooser must return a non-empty set."""
    out = {}
    tail = [_in_tail(p) for p in pools]
    for n in SET_SIZES:
        hits, covers, regrets = [], [], []
        for p in pools:
            s = choose(p, n)
            if len(s) == 0 or len(s) > n:
                raise ValueError(f"a set of {len(s)} candidates at N={n}")
            best = _leader(p)
            hits.append(best in s)
            covers.append(float(p.q[s].sum()))
            regrets.append(float(p.win_prob[best] - p.win_prob[s].max()))
        hits = np.array(hits)
        out[f"{prefix}eval_recall_{n}"] = float(hits.mean())
        out[f"{prefix}eval_cover_{n}"] = float(np.mean(covers))
        out[f"{prefix}eval_regret_{n}"] = float(np.mean(regrets))
        if any(tail):
            out[f"{prefix}eval_tail_recall_{n}"] = float(hits[np.array(tail)].mean())
    return out


def pool_stats(pools: list[Pool], prefix: str) -> dict[str, float]:
    """What a set of pools is like: how many, their mean size, the mean
    entropy of their labels (nats; higher is noisier or more evenly
    contested), and the share whose leader is outside the static-equity top
    TAIL_RANK."""
    entropy = [float(-(p.q * np.log(np.clip(p.q, 1e-12, None))).sum()) for p in pools]
    return {
        f"{prefix}pools": len(pools),
        f"{prefix}pool_size": float(np.mean([len(p.moves) for p in pools])),
        f"{prefix}label_entropy": float(np.mean(entropy)),
        f"{prefix}tail_share": float(np.mean([_in_tail(p) for p in pools])),
    }
