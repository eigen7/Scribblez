"""The search network's continuation heads on the policy form
(docs/plans/deep_bot.md, "The search network"), pretrained on game records:
at a pool position the move the game played is one of the candidates, and
the game's rest labels what follows it, from the mover's side on the board
after it (engine training/played_continuation.h, which the batch carries).

Per played candidate, decoded from its vector and its position's board
tokens: each square's ownership (five classes) and the signed points of the
move through it (mean and variance), from the tokens with the candidate's
vector added to every one; and who goes out and the net end-of-game points
(mean and variance), from the vector alone. The targets, classes, scale and
moment losses are the leaf network's (deep_bot_leaf_eval.model), so the two
networks' maps read alike. A position whose played move is not a candidate
(a blank designation the pool pruned) has no row here.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from scribblez.deep_bot_leaf_eval.model import (
    FILLED,
    GOING_OUT_CLASSES,
    POINTS_SCALE,
    SQUARE_FATE_CLASSES,
    UNLABELLED,
    mean_and_variance,
    moment_losses,
)

# The end head's hidden width.
END_HIDDEN = 64


class ContinuationOutputs(NamedTuple):
    rows: torch.Tensor  # (R,) the batch positions decoded, those with a played candidate
    ownership: torch.Tensor  # (R, 5, 225) FP32 logits
    square_points: torch.Tensor  # (R, 2, 225) FP32 [mean, variance], POINTS_SCALE units
    going_out: torch.Tensor  # (R, 3) FP32 logits
    end_points: torch.Tensor  # (R, 2) FP32 [mean, variance], POINTS_SCALE units


class ContinuationHeads(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.candidate = nn.Linear(channels, channels)
        self.square = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, SQUARE_FATE_CLASSES + 2),
        )
        self.end = nn.Sequential(
            nn.Linear(channels, END_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(END_HIDDEN, GOING_OUT_CLASSES + 2),
        )

    def forward(
        self, h: torch.Tensor, board: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> ContinuationOutputs:
        """The heads at the batch's played candidates: `h` (M, C) the
        candidates' vectors, `board` (P, 225, C) the positions' tokens."""
        rows = torch.nonzero(batch["played"] >= 0).squeeze(1)
        vectors = h[batch["played"][rows]]  # (R, C)
        tokens = board[rows] + self.candidate(vectors).unsqueeze(1)  # (R, 225, C)
        square = self.square(tokens).transpose(1, 2).float()  # (R, 7, 225)
        end = self.end(vectors).float()  # (R, 5)
        return ContinuationOutputs(
            rows,
            square[:, :SQUARE_FATE_CLASSES],
            mean_and_variance(square[:, SQUARE_FATE_CLASSES:]),
            end[:, :GOING_OUT_CLASSES],
            mean_and_variance(end[:, GOING_OUT_CLASSES:]),
        )


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, unlabelled: int):
    """The mean cross-entropy over the labels that are not `unlabelled`; 0,
    not NaN, when none is."""
    ce = F.cross_entropy(logits, labels, ignore_index=unlabelled, reduction="sum")
    return ce / (labels != unlabelled).sum().clamp(min=1)


def continuation_losses(
    out: ContinuationOutputs, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """The heads' losses over the decoded rows, keyed "ownership" and
    "going_out" (cross-entropies over the empty squares and the rows whose
    game reached its end) and "square_points_mean/var" and
    "end_points_mean/var" (moment losses, the same masks); each 0 with no
    row to fit."""
    rows = out.rows
    ownership = batch["ownership"][rows]
    going_out = batch["going_out"][rows]
    empty = (ownership != FILLED).to(out.square_points.dtype)
    labelled = (going_out != UNLABELLED).to(out.end_points.dtype)
    points_mean, points_var = moment_losses(
        *out.square_points.unbind(dim=1), batch["square_points"][rows] / POINTS_SCALE, empty
    )
    end_mean, end_var = moment_losses(
        *out.end_points.unbind(dim=1), batch["end_points"][rows] / POINTS_SCALE, labelled
    )
    return {
        "ownership": masked_cross_entropy(out.ownership, ownership, FILLED),
        "square_points_mean": points_mean,
        "square_points_var": points_var,
        "going_out": masked_cross_entropy(out.going_out, going_out, UNLABELLED),
        "end_points_mean": end_mean,
        "end_points_var": end_var,
    }
