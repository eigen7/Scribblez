"""Training rows for M1a's reader (docs/plans/supreme_bot_m1a.md, PR 3).

A row is one position's context and the questions asked of it. The context
is a token sequence:

    root (226)  candidates (K)  probe  probe  ...

and each kept probe is its own run of tokens:

    deal (root mover's refill)  deal (opponent's rack)
    action  [chance]  action  [chance]  ...  leaf

with a chance token after each turn that drew tiles. A row holds out one to
four candidates: their probes are dropped (or, in the graded variant, cut to
a few), and the reader must answer for them from the other candidates'
evidence. The kept candidates keep a random subset of their probes, and the
kept probes are interleaved in a random order.

Pick queries ask, for every candidate, what its label is given a prefix of
the context. They are listed apart from the context, each with the number of
context tokens it may see (always at a probe boundary), so the reader can
append them after the context under a mask: a query sees its prefix and
itself, and nothing sees a query.

The root tokens are the frozen teacher's trunk on the root position, one per
board cell plus its scalar projection. Every token type has its own feature
table; a token's `ref` indexes its type's table. Scores and bag counts are
scaled to about unit range here, so the token encoder only projects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scribblez.sim_evidence.sobs import MOVE_PLAY
from scribblez.transfer_test.corpus import CorpusFile
from scribblez.transfer_test.probes import tile_codes, tile_counts

# Token kinds.
ROOT, CANDIDATE, CHANCE, ACTION, LEAF = range(5)
NUM_KINDS = 5
NO_SLOT = -1

# A probe's two opening deals: the root mover's refill, then the opponent's rack.
MOVER_DEAL, OPP_DEAL = 0, 1

SCORE_SCALE = 100.0  # points per feature unit
BAG_SCALE = 100.0  # tiles per feature unit
# The teacher's trunk tokens on the root position: one per board cell
# (r * 15 + c), then its scalar projection.
ROOT_TOKENS = 226


@dataclass(frozen=True)
class RowConfig:
    max_held_out: int = 4
    max_probes: int = 32  # per kept candidate; each keeps a uniform 0..max_probes
    # Context tokens, root board included; a held-out candidate's graded probes
    # are kept even past it.
    max_tokens: int = 2048
    query_points: int = 4  # prefix lengths queried, the full context always one of them
    graded_max: int = 0  # graded variant: each held-out candidate keeps 1..graded_max probes


@dataclass
class Row:
    """One assembled row; see the module docstring. K candidates, T context
    tokens, Q queries."""

    file: int
    position: int
    # The context, (T,) each.
    kind: np.ndarray  # int8 token kind
    slot: np.ndarray  # int8 candidate slot, NO_SLOT for root tokens
    ply: np.ndarray  # int8: 0 before the reply, k for turn k, turns + 1 for a leaf
    ref: np.ndarray  # int32 row in the kind's feature table
    # Feature tables.
    root: np.ndarray  # (ROOT_TOKENS, Ct) float16 teacher trunk tokens
    candidate: dict[str, np.ndarray]  # (K, ...) each
    chance: dict[str, np.ndarray]
    action: dict[str, np.ndarray]
    leaf: np.ndarray  # (L, 6) float32
    # The questions.
    query_slot: np.ndarray  # (Q,) int8
    query_prefix: np.ndarray  # (Q,) int32 context tokens visible to the query
    held_out: np.ndarray  # (K,) bool
    target: dict[str, np.ndarray]  # (K, ...) each


def choose_held_out(k: int, cfg: RowConfig, rng: np.random.Generator) -> np.ndarray:
    """(k,) bool: a uniform 1..max_held_out of the candidates, leaving at least
    one with evidence."""
    held = np.zeros(k, dtype=bool)
    h = min(int(rng.integers(1, cfg.max_held_out + 1)), k - 1)
    held[rng.choice(k, size=h, replace=False)] = True
    return held


def choose_probes(
    held: np.ndarray, probes: int, cfg: RowConfig, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """The kept (slot, probe) pairs in context order: each kept candidate's
    random subset, each held-out candidate's graded few, all interleaved."""
    kept = []
    for slot, is_held in enumerate(held):
        if is_held:
            n = int(rng.integers(1, cfg.graded_max + 1)) if cfg.graded_max else 0
        else:
            n = int(rng.integers(0, cfg.max_probes + 1))
        kept += [(slot, int(i)) for i in rng.choice(probes, size=min(n, probes), replace=False)]
    return [kept[i] for i in rng.permutation(len(kept))]


def _probe_tokens(f: CorpusFile, turns: np.ndarray) -> int:
    """Tokens a probe takes: two deals, its actions and draws, and the leaf."""
    return 3 + len(turns) + int(f.drew[turns].sum())


def _fit_budget(
    f: CorpusFile,
    c0: int,
    picks: list[tuple[int, int]],
    held: np.ndarray,
    header: int,
    cfg: RowConfig,
) -> list[tuple[int, int, np.ndarray]]:
    """The picks that fit the token budget, as (slot, record, turns), in pick
    order. A held-out candidate's graded probes are always kept, their tokens
    reserved first; the kept candidates' picks fill the rest until one does
    not fit."""
    probes = []
    for slot, i in picks:
        record = (c0 + slot) * f.probes.probes + i
        turns = np.arange(f.probes.turn_start[record], f.probes.turn_start[record + 1])
        probes.append((slot, record, turns))
    used = header + sum(_probe_tokens(f, t) for slot, _, t in probes if held[slot])
    kept, full = [], False
    for slot, record, turns in probes:
        if not held[slot]:
            full = full or used + _probe_tokens(f, turns) > cfg.max_tokens
            if full:
                continue
            used += _probe_tokens(f, turns)
        kept.append((slot, record, turns))
    return kept


def assemble_row(
    f: CorpusFile,
    file_index: int,
    p: int,
    cfg: RowConfig,
    rng: np.random.Generator,
    held: np.ndarray | None = None,
) -> Row:
    """A random row for position `p` of `f`, holding out `held` ((K,) bool)
    when given, else a random choice (choose_held_out)."""
    c0, c1 = int(f.probes.candidate_start[p]), int(f.probes.candidate_start[p + 1])
    k = c1 - c0
    held = choose_held_out(k, cfg, rng) if held is None else np.asarray(held, dtype=bool)
    picks = choose_probes(held, f.probes.probes, cfg, rng)
    ctx = _Context(k)
    for slot, record, turns in _fit_budget(f, c0, picks, held, len(ctx), cfg):
        ctx.add_probe(f, slot, record, turns)
    query_prefix = _query_prefixes(ctx.boundaries, cfg, rng)
    return Row(
        file=file_index,
        position=p,
        **ctx.columns(),
        root=np.concatenate([f.prior.root_board[p], f.prior.root_summary[p][None]]),
        candidate=_candidate_table(f, p, c0, c1),
        chance=_chance_table(f, np.asarray(ctx.chance, dtype=np.int64).reshape(-1, 3)),
        action=_action_table(f, np.asarray(ctx.actions, dtype=np.int64)),
        leaf=_leaf_table(f, np.asarray(ctx.leaves, dtype=np.int64)),
        query_slot=np.tile(np.arange(k, dtype=np.int8), len(query_prefix)),
        query_prefix=np.repeat(query_prefix, k).astype(np.int32),
        held_out=held,
        target=_targets(f, c0, c1),
    )


class _Context:
    """A row's context as it is built: the per-token columns, the rows of each
    kind's feature table they point at, and the probe boundaries."""

    def __init__(self, k: int):
        self.cols: dict[str, list[int]] = {"kind": [], "slot": [], "ply": [], "ref": []}
        self.chance: list[tuple[int, int, int]] = []  # (turn, or -1 for a deal; record; deal)
        self.actions: list[int] = []  # turns
        self.leaves: list[int] = []  # records
        for cell in range(ROOT_TOKENS):
            self._add(ROOT, NO_SLOT, 0, cell)
        for slot in range(k):
            self._add(CANDIDATE, slot, 0, slot)
        self.boundaries = [len(self)]

    def __len__(self) -> int:
        return len(self.cols["kind"])

    def _add(self, kind: int, slot: int, ply: int, ref: int):
        for name, value in zip(self.cols, (kind, slot, ply, ref), strict=True):
            self.cols[name].append(value)

    def add_probe(self, f: CorpusFile, slot: int, record: int, turns: np.ndarray):
        for deal in (MOVER_DEAL, OPP_DEAL):
            self._add(CHANCE, slot, 0, len(self.chance))
            self.chance.append((-1, record, deal))
        for ply, t in enumerate(turns, start=1):
            self._add(ACTION, slot, ply, len(self.actions))
            self.actions.append(int(t))
            if f.drew[t]:
                self._add(CHANCE, slot, ply, len(self.chance))
                self.chance.append((int(t), record, MOVER_DEAL))
        self._add(LEAF, slot, len(turns) + 1, len(self.leaves))
        self.leaves.append(record)
        self.boundaries.append(len(self))

    def columns(self) -> dict[str, np.ndarray]:
        dtypes = {"kind": np.int8, "slot": np.int8, "ply": np.int8, "ref": np.int32}
        return {name: np.asarray(v, dtype=dtypes[name]) for name, v in self.cols.items()}


def _query_prefixes(boundaries: list[int], cfg: RowConfig, rng: np.random.Generator) -> np.ndarray:
    """Up to cfg.query_points probe boundaries, the whole context among them,
    ascending."""
    last = len(boundaries) - 1
    others = rng.choice(last, size=min(cfg.query_points - 1, last), replace=False) if last else []
    return np.asarray(sorted({boundaries[last], *(boundaries[int(i)] for i in others)}))


def _candidate_table(f: CorpusFile, p: int, c0: int, c1: int) -> dict[str, np.ndarray]:
    state = f.replay.candidates[c0:c1]
    return {
        "move": f.probes.candidates["move"][c0:c1],
        "pre_move_diff": np.full(c1 - c0, f.replay.roots["score_diff"][p], dtype=np.int32),
        "stratum": f.probes.candidates["stratum"][c0:c1].astype(np.int64),
        "leave": tile_counts(tile_codes(state["leave"])),
        "scalars": np.stack(
            [state["bag_size"] / BAG_SCALE, state["score_diff"] / SCORE_SCALE], axis=1
        ).astype(np.float32),
        "prior_value": np.concatenate(
            [f.prior.wld[c0:c1], f.prior.score[c0:c1] / SCORE_SCALE], axis=1
        ).astype(np.float32),
        # Raw float16 logits: the encoder normalizes them on the device.
        "prior_placement": f.prior.placement[c0:c1],
    }


def _chance_table(f: CorpusFile, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Deals and draws: the tiles dealt, the rack they made, whether the rack
    is the root mover's, and the bag after. `rows` is (N, 3): the turn whose
    draw it is (-1 for a probe's opening deal), the record, and which deal."""
    turn, record, deal = rows.T
    opening = turn < 0
    s = f.replay.starts[record]
    t = f.replay.turns[np.where(opening, 0, turn)]
    mover_deal = deal == MOVER_DEAL
    drawn = np.where(
        opening[:, None],
        np.where(mover_deal[:, None], tile_codes(s["mover_drawn"]), tile_codes(s["opp_drawn"])),
        tile_codes(t["drawn"]),
    )
    rack = np.where(
        opening[:, None],
        np.where(mover_deal[:, None], tile_codes(s["mover_rack"]), tile_codes(s["opp_rack"])),
        tile_codes(t["rack_after"]),
    )
    drawn_counts = tile_counts(drawn)
    # A play's draw drains the bag; an exchange's swaps tiles with it.
    played = f.probes.turns["move"]["type"][np.where(opening, 0, turn)] == MOVE_PLAY
    turn_bag = t["bag_size"].astype(np.int32) - played * drawn_counts.sum(axis=1, dtype=np.int32)
    opening_bag = f.replay.candidates["bag_size"][record // f.probes.probes]
    return {
        "drawn": drawn_counts,
        "rack": tile_counts(rack),
        "scalars": np.stack(
            [
                np.where(opening, mover_deal, t["root_mover"] == 1),
                np.where(opening, opening_bag, turn_bag) / BAG_SCALE,
            ],
            axis=1,
        ).astype(np.float32),
    }


def _action_table(f: CorpusFile, turns: np.ndarray) -> dict[str, np.ndarray]:
    """Moves and their surroundings, the score difference before the move
    from the root mover's side, and the mover's own for the move encoder."""
    t = f.replay.turns[turns]
    root_mover = t["root_mover"].astype(np.int32)
    diff = t["score_diff"].astype(np.int32)
    return {
        "move": f.probes.turns["move"][turns],
        "pre_move_diff": np.where(root_mover == 1, diff, -diff).astype(np.int32),
        "leave": tile_counts(tile_codes(t["leave"])),
        "scalars": np.stack(
            [root_mover, t["bag_size"] / BAG_SCALE, diff / SCORE_SCALE], axis=1
        ).astype(np.float32),
    }


def _leaf_table(f: CorpusFile, records: np.ndarray) -> np.ndarray:
    """(L, 6): win/draw/loss, the score difference's mean and spread, and
    whether the leaf model scored it (else the game ended)."""
    r = f.probes.records[records]
    spread = np.sqrt(np.maximum(r["delta_sq"] - np.square(r["delta"]), 0))
    return np.stack(
        [
            r["p_win"],
            r["p_draw"],
            r["p_loss"],
            r["delta"] / SCORE_SCALE,
            spread / SCORE_SCALE,
            r["truncated"],
        ],
        axis=1,
    ).astype(np.float32)


def _targets(f: CorpusFile, c0: int, c1: int) -> dict[str, np.ndarray]:
    """Each candidate's label: outcome frequencies, score moments and the
    footprint frequencies of the opponent's reply and the root mover's next
    move, all per label rollout."""
    lab = f.labels[c0:c1]
    n = lab["n"].astype(np.float32)
    mean = lab["delta_sum"] / n
    return {
        "wld": (np.stack([lab["wins"], lab["draws"], lab["losses"]], axis=1) / n[:, None]).astype(
            np.float32
        ),
        "score_mean": mean.astype(np.float32),
        "score_var": np.maximum(lab["delta_sq_sum"] / n - np.square(mean), 0).astype(np.float32),
        "opp_next": (lab["opp_next_count"] / n[:, None]).astype(np.float32),
        "self_next": (lab["self_next_count"] / n[:, None]).astype(np.float32),
        "n": n,
    }
