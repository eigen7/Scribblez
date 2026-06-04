"""Torch ``Dataset`` over Scribblez JSON game logs.

Each *example* is a single turn from a game. The dataset replays each game
forward, so that for every turn we can produce:

* ``board_before``: an integer tensor of shape ``(15, 15)`` with values in
  ``[0, 26]`` -- ``0..25`` are A..Z, ``26`` denotes an empty square. Blanks
  that have been placed on the board are stored as the letter they represent
  (their blank-ness is tracked separately).
* ``blank_mask_before``: bool tensor ``(15, 15)`` -- True iff the tile at
  that square was originally a blank.
* ``rack_before``: integer tensor ``(27,)`` -- count of each tile type
  (indices 0..25 = A..Z, index 26 = blank).
* ``scores``: ``(2,)`` cumulative scores at the start of the turn, current
  player first.
* ``bag_size``: int.
* ``player``: 0 or 1 (the player about to move).
* ``move``: dict describing what the agent did (raw from the log).
* ``score_delta``: int, score added by this move.
* ``final_scores``: ``(2,)`` final scores for the game, current player first.
* ``game_path``: source file path (useful for debugging).

The dataset is intentionally light on tensor encoding: it produces a small,
self-describing dict per turn. Higher layers (loss heads, networks) decide
how to consume it.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

EMPTY = 26  # sentinel value for an empty board cell in the integer board tensor


@dataclass
class TurnSample:
    board_before: torch.Tensor          # (15, 15) int8
    blank_mask_before: torch.Tensor     # (15, 15) bool
    rack_before: torch.Tensor           # (27,) int8
    scores: torch.Tensor                # (2,) int32, current player first
    bag_size: int
    player: int
    move: dict
    score_delta: int
    final_scores: torch.Tensor          # (2,) int32, current player first
    game_path: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _letter_idx(ch: str) -> int:
    if ch == "?":
        return 26
    c = ch.upper()
    assert "A" <= c <= "Z", f"unexpected tile char {ch!r}"
    return ord(c) - ord("A")


def _rack_to_counts(rack_str: str) -> torch.Tensor:
    counts = torch.zeros(27, dtype=torch.int8)
    for c in rack_str:
        counts[_letter_idx(c)] += 1
    return counts


def _replay_game(game: dict, game_path: str) -> List[TurnSample]:
    """Walk a single game log forward, producing one TurnSample per turn."""
    board = torch.full((15, 15), EMPTY, dtype=torch.int8)
    blanks = torch.zeros((15, 15), dtype=torch.bool)
    final = torch.tensor(game["final_scores"], dtype=torch.int32)
    samples: List[TurnSample] = []

    for turn in game["turns"]:
        player = int(turn["player"])
        rack_before = _rack_to_counts(turn["rack_before"])
        scores_before = list(turn["cumulative_scores"])
        # Convert to "current player first".
        score_delta = int(turn["score_delta"])
        scores_curr_first_before = [scores_before[player] - score_delta,
                                    scores_before[1 - player]]
        final_curr_first = [int(final[player].item()), int(final[1 - player].item())]

        samples.append(
            TurnSample(
                board_before=board.clone(),
                blank_mask_before=blanks.clone(),
                rack_before=rack_before,
                scores=torch.tensor(scores_curr_first_before, dtype=torch.int32),
                bag_size=int(turn["bag_size_before"]),
                player=player,
                move=turn["move"],
                score_delta=score_delta,
                final_scores=torch.tensor(final_curr_first, dtype=torch.int32),
                game_path=game_path,
            )
        )

        # Apply the move to the in-memory board state for the next sample.
        mv = turn["move"]
        if mv["type"] == "play":
            for t in mv["tiles"]:
                r, c = int(t["row"]), int(t["col"])
                board[r, c] = _letter_idx(t["letter"])
                blanks[r, c] = bool(t["is_blank"])
        # EXCHANGE/PASS: board unchanged.

    return samples


class GameLogDataset(Dataset):
    """Dataset of per-turn examples drawn from a directory or list of JSON logs.

    The whole corpus is loaded and replayed eagerly into memory. This is fine
    for development-scale corpora (thousands of games); the eventual binary
    format + streaming loader will replace this.
    """

    def __init__(self, sources: Iterable[str | os.PathLike]):
        paths: List[Path] = []
        for s in sources:
            p = Path(s)
            if p.is_dir():
                paths.extend(sorted(p.glob("*.json")))
            else:
                paths.append(p)
        self._paths = paths
        self._samples: List[TurnSample] = []
        for p in paths:
            with p.open() as f:
                game = json.load(f)
            self._samples.extend(_replay_game(game, str(p)))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> TurnSample:
        return self._samples[idx]

    @property
    def game_paths(self) -> Sequence[Path]:
        return self._paths


def _default_collate(batch: List[TurnSample]) -> dict:
    """Stack tensor fields and keep variable-length / nested fields as lists."""
    return {
        "board_before": torch.stack([s.board_before for s in batch]),
        "blank_mask_before": torch.stack([s.blank_mask_before for s in batch]),
        "rack_before": torch.stack([s.rack_before for s in batch]),
        "scores": torch.stack([s.scores for s in batch]),
        "bag_size": torch.tensor([s.bag_size for s in batch], dtype=torch.int32),
        "player": torch.tensor([s.player for s in batch], dtype=torch.int8),
        "score_delta": torch.tensor([s.score_delta for s in batch], dtype=torch.int32),
        "final_scores": torch.stack([s.final_scores for s in batch]),
        "move": [s.move for s in batch],
        "game_path": [s.game_path for s in batch],
    }


def build_dataloader(
    sources: Iterable[str | os.PathLike],
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    ds = GameLogDataset(sources)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_default_collate,
    )
