"""The M1a reader's token encoder (docs/plans/supreme_bot_m1a.md, PR 3): a
batch of rows (scribblez.transfer_test.rows) into width-C embeddings for the
context and the pick queries.

Each token kind has its own input projection:

    root       the teacher's trunk token, projected from its width
    candidate  the move, its leave and post-move scalars, the teacher's
               predictions at its post-move position, and its stratum
    chance     the tiles dealt or drawn and the rack they made, as counts,
               whose rack, and the bag after
    action     the move, the mover's leave as counts, whose move, the bag and
               the score difference before it
    leaf       the probe's outcome: win/draw/loss and the score difference's
               mean and spread, and whether the leaf model scored it

Moves go through the move set model's MoveEncoder, each placed tile reading
the root token of the square it lands on, so a move is tied to the board
the teacher saw. Every token then adds embeddings of its kind, its candidate
slot and its ply. A query is the query kind's embedding, its slot's, and its
candidate's content.

Collation flattens each kind's feature table across the batch, shifting
`ref` to index the flat table, and pads the context and the queries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from scribblez.ffi import encode_moves, format_layout, move_encoding_dims
from scribblez.move_set_eval.model import MoveEncoder
from scribblez.position_eval.model import FOOTPRINT_CLASSES, PLACEMENT_HEAD_NAMES
from scribblez.transfer_test.probes import TILE_KINDS
from scribblez.transfer_test.rows import (
    ACTION,
    CANDIDATE,
    CHANCE,
    LEAF,
    NO_SLOT,
    NUM_KINDS,
    ROOT,
    ROOT_TOKENS,
    Row,
)

NUM_STRATA = format_layout()["constants"]["sprobe"]["strata"]
QUERY = NUM_KINDS  # the query's kind embedding, after the context kinds
MAX_PLY = 15  # larger plies share the last ply embedding
# Each placement head's footprint distribution is compressed to this many
# features, by one projection shared across the heads.
PLACEMENT_FEATURES = 32


@dataclass
class MoveBatch:
    """Encoded moves (engine training/move_set_encoder.h), with each move's
    batch row so its tiles can read that row's root tokens."""

    letters: torch.Tensor  # (M, max_placed) int64
    blanks: torch.Tensor  # (M, max_placed) bool
    squares: torch.Tensor  # (M, max_placed) int64
    tile_mask: torch.Tensor  # (M, max_placed) bool
    scalars: torch.Tensor  # (M, num_scalars) float32
    row: torch.Tensor  # (M,) int64

    def to(self, device: torch.device) -> MoveBatch:
        return MoveBatch(**{k: v.to(device) for k, v in vars(self).items()})


@dataclass
class TokenBatch:
    """A collated batch of B rows: T context tokens and Q queries at most per
    row, padded; flat per-kind feature tables; flat per-candidate targets."""

    kind: torch.Tensor  # (B, T) int64
    slot: torch.Tensor  # (B, T) int64, NO_SLOT for root tokens and padding
    ply: torch.Tensor  # (B, T) int64
    ref: torch.Tensor  # (B, T) int64 into the kind's flat table
    pad: torch.Tensor  # (B, T) bool, True on padding
    root: torch.Tensor  # (B, ROOT_TOKENS, Ct) float32
    candidate_moves: MoveBatch
    candidate: dict[str, torch.Tensor]  # flat over the batch's candidates
    action_moves: MoveBatch
    action: dict[str, torch.Tensor]
    chance: dict[str, torch.Tensor]
    leaf: torch.Tensor  # (L, 6)
    query_candidate: torch.Tensor  # (B, Q) int64 flat candidate index
    query_slot: torch.Tensor  # (B, Q) int64
    query_prefix: torch.Tensor  # (B, Q) int64 context tokens the query sees
    query_pad: torch.Tensor  # (B, Q) bool
    held_out: torch.Tensor  # (Kf,) bool, flat over candidates
    target: dict[str, torch.Tensor]  # (Kf, ...) each

    def to(self, device: torch.device) -> TokenBatch:
        moved = {}
        for name, value in vars(self).items():
            if isinstance(value, dict):
                moved[name] = {k: v.to(device) for k, v in value.items()}
            else:
                moved[name] = value.to(device)
        return TokenBatch(**moved)


def _encode_moves(moves: np.ndarray, pre_move_diffs: np.ndarray, rows: np.ndarray) -> MoveBatch:
    enc = encode_moves(moves, pre_move_diffs)
    return MoveBatch(
        letters=torch.from_numpy(enc["letters"]),
        blanks=torch.from_numpy(enc["blanks"]),
        squares=torch.from_numpy(enc["squares"]),
        tile_mask=torch.from_numpy(enc["tile_mask"]),
        scalars=torch.from_numpy(enc["scalars"]),
        row=torch.from_numpy(rows.astype(np.int64)),
    )


def _flat(tables: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}


def _offsets(tables: list) -> np.ndarray:
    """Where each row's table starts in the flat table."""
    lengths = np.array([len(t) for t in tables], dtype=np.int64)
    return np.concatenate([[0], np.cumsum(lengths)[:-1]])


def _padded(columns: list[np.ndarray], fill: int) -> torch.Tensor:
    width = max(len(c) for c in columns)
    out = np.full((len(columns), width), fill, dtype=np.int64)
    for i, c in enumerate(columns):
        out[i, : len(c)] = c
    return torch.from_numpy(out)


def _flat_refs(rows: list[Row]) -> list[np.ndarray]:
    """Each row's refs shifted to index the flat tables of the batch (ROOT
    refs to the flat (B * ROOT_TOKENS) root table)."""
    offsets = {
        ROOT: np.arange(len(rows)) * ROOT_TOKENS,
        CANDIDATE: _offsets([r.candidate["move"] for r in rows]),
        CHANCE: _offsets([r.chance["drawn"] for r in rows]),
        ACTION: _offsets([r.action["move"] for r in rows]),
        LEAF: _offsets([r.leaf for r in rows]),
    }
    out = []
    for i, r in enumerate(rows):
        shift = np.zeros(len(r.ref), dtype=np.int64)
        for kind, starts in offsets.items():
            shift[r.kind == kind] = starts[i]
        out.append(r.ref + shift)
    return out


def _moves(rows: list[Row], table: str) -> MoveBatch:
    tables = [getattr(r, table) for r in rows]
    return _encode_moves(
        np.concatenate([t["move"] for t in tables]),
        np.concatenate([t["pre_move_diff"] for t in tables]),
        np.concatenate([np.full(len(t["move"]), i) for i, t in enumerate(tables)]),
    )


def _features(tables: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    """A kind's non-move features, flat over the batch."""
    flat = _flat(tables)
    return {k: torch.from_numpy(v) for k, v in flat.items() if k not in ("move", "pre_move_diff")}


def collate(rows: list[Row]) -> TokenBatch:
    """Stack rows into a TokenBatch (see the module docstring)."""
    candidate_start = _offsets([r.candidate["move"] for r in rows])
    return TokenBatch(
        kind=_padded([r.kind for r in rows], 0),
        slot=_padded([r.slot for r in rows], NO_SLOT),
        ply=_padded([r.ply for r in rows], 0),
        ref=_padded(_flat_refs(rows), 0),
        pad=_padded([np.zeros(len(r.kind)) for r in rows], 1).bool(),
        root=torch.from_numpy(np.stack([r.root for r in rows]).astype(np.float32)),
        candidate_moves=_moves(rows, "candidate"),
        candidate=_features([r.candidate for r in rows]),
        action_moves=_moves(rows, "action"),
        action=_features([r.action for r in rows]),
        chance=_features([r.chance for r in rows]),
        leaf=torch.from_numpy(np.concatenate([r.leaf for r in rows])),
        query_candidate=_padded([candidate_start[i] + r.query_slot for i, r in enumerate(rows)], 0),
        query_slot=_padded([r.query_slot for r in rows], 0),
        query_prefix=_padded([r.query_prefix for r in rows], 0),
        query_pad=_padded([np.zeros(len(r.query_slot)) for r in rows], 1).bool(),
        held_out=torch.from_numpy(np.concatenate([r.held_out for r in rows])),
        target={k: torch.from_numpy(v) for k, v in _flat([r.target for r in rows]).items()},
    )


class TokenEncoder(nn.Module):
    """TokenBatch -> (context (B, T, C), queries (B, Q, C)); see the module
    docstring. `teacher_width` is the root tokens' width; `max_slots` bounds
    the candidates per position."""

    def __init__(self, width: int, teacher_width: int, max_slots: int):
        super().__init__()
        _, num_scalars, letter_vocab, _ = move_encoding_dims()
        self.root = nn.Linear(teacher_width, width)
        self.moves = MoveEncoder(width, letter_vocab, num_scalars)
        self.placement = nn.Linear(FOOTPRINT_CLASSES, PLACEMENT_FEATURES)
        candidate_in = TILE_KINDS + 2 + 5 + len(PLACEMENT_HEAD_NAMES) * PLACEMENT_FEATURES
        self.candidate = nn.Linear(candidate_in, width)
        self.stratum = nn.Embedding(NUM_STRATA, width)
        self.chance = nn.Linear(2 * TILE_KINDS + 2, width)
        self.action = nn.Linear(TILE_KINDS + 3, width)
        self.leaf = nn.Linear(6, width)
        self.kind = nn.Embedding(NUM_KINDS + 1, width)
        self.slot = nn.Embedding(max_slots + 1, width)  # slot + 1; 0 is NO_SLOT
        self.ply = nn.Embedding(MAX_PLY + 1, width)

    def _moves(self, moves: MoveBatch, root: torch.Tensor) -> torch.Tensor:
        board = root[moves.row.unsqueeze(1).expand_as(moves.squares), moves.squares]
        return self.moves(moves.letters, moves.blanks, moves.tile_mask, moves.scalars, board)

    def _candidates(self, b: TokenBatch, root: torch.Tensor) -> torch.Tensor:
        c = b.candidate
        placement = self.placement(c["prior_placement"]).flatten(1)
        features = torch.cat([c["leave"].float(), c["scalars"], c["prior_value"], placement], dim=1)
        return (
            self._moves(b.candidate_moves, root)
            + self.candidate(features)
            + self.stratum(c["stratum"])
        )

    def _chances(self, b: TokenBatch) -> torch.Tensor:
        c = b.chance
        return self.chance(torch.cat([c["drawn"].float(), c["rack"].float(), c["scalars"]], 1))

    def _actions(self, b: TokenBatch, root: torch.Tensor) -> torch.Tensor:
        a = b.action
        features = torch.cat([a["leave"].float(), a["scalars"]], 1)
        return self._moves(b.action_moves, root) + self.action(features)

    def _kind_tables(self, b: TokenBatch, root: torch.Tensor) -> dict[int, torch.Tensor]:
        """Each kind's flat content embeddings, which `ref` indexes."""
        return {
            ROOT: root.flatten(0, 1),
            CANDIDATE: self._candidates(b, root),
            CHANCE: self._chances(b),
            ACTION: self._actions(b, root),
            LEAF: self.leaf(b.leaf),
        }

    def forward(self, b: TokenBatch) -> tuple[torch.Tensor, torch.Tensor]:
        root = self.root(b.root)  # (B, ROOT_TOKENS, C)
        tables = self._kind_tables(b, root)
        context = root.new_zeros((*b.kind.shape, root.shape[-1]))
        for kind, table in tables.items():
            at = (b.kind == kind) & ~b.pad
            context[at] = table[b.ref[at]].to(context.dtype)
        context = (
            context + self.kind(b.kind) + self.slot(b.slot + 1) + self.ply(b.ply.clamp(max=MAX_PLY))
        )
        queries = (
            tables[CANDIDATE][b.query_candidate]
            + self.kind.weight[QUERY]
            + self.slot(b.query_slot + 1)
        )
        return context, queries
