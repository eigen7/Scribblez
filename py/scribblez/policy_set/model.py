"""The policy-set model's shared encoder (docs/plans/deep_bot.md, "The
policy-set model"), which both forms build on: slot.py's slot decoder and
greedy.py's set stage.

The encoder is the move set model's: the board trunk once per position, each
candidate embedded by the move and cross-check encoders against the board
tokens and cross-attending into them, fused with the position's global
summary. The trunk reads the opponent-rack input, the chooser's belief about
the opponent's rack (dataset.opp_rack_input), as a scalar block beside the
board row's, and each candidate's static equity relative to its pool's best
(dataset.relative_equity) joins the candidate's embedding, so a form can learn
the correction to the static-equity ranking rather than the ranking itself.
Its output is one vector per candidate, layer-normalized, which a form reads:
unnormalized, the vectors' scale ran away in training, driving the slot form's
pointer logits to divergence and the greedy form's gain head into saturation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from scribblez.move_set_eval.model import (
    CrossCheckEncoder,
    MoveEncoder,
    attend_by_position,
    encode_moves_on_board,
    rank_within_position,
)
from scribblez.move_set_eval.moves import move_cross_slots, move_encoding_dims
from scribblez.policy_set.dataset import OPP_RACK_FLOATS
from scribblez.spatial_trunk import SpatialTrunk, mean_max_pool


@dataclass
class Encoded:
    """A batch's encoder outputs before the candidates attend to the board."""

    board: torch.Tensor  # (P, 225, C) board token maps
    g: torch.Tensor  # (P, 3C) summaries
    e: torch.Tensor  # (M, C) candidate embeddings


def trunk_scalar(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """The trunk's scalar input: the board row's scalars, then the
    opponent-rack block."""
    return torch.cat([batch["input_scalar"], batch["opp_rack"]], dim=1)


class PolicySetEncoder(nn.Module):
    def __init__(
        self,
        spatial_planes: int,
        scalar_size: int,
        channels: int,
        num_blocks: int,
        num_heads: int,
    ):
        super().__init__()
        # The opponent-rack input rides beside the board row's scalars.
        self.trunk = SpatialTrunk(
            spatial_planes, scalar_size + OPP_RACK_FLOATS, channels, num_blocks
        )
        _, num_scalars, letter_vocab, cells = move_encoding_dims()
        self.board_pos_emb = nn.Parameter(torch.zeros(cells, channels))
        nn.init.normal_(self.board_pos_emb, std=0.02)
        self.move_encoder = MoveEncoder(channels, letter_vocab, num_scalars)
        self.cross_check_encoder = CrossCheckEncoder(channels, move_cross_slots(), cells)
        self.equity = nn.Sequential(
            nn.Linear(1, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels)
        )
        self.cross_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(4 * channels, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels)
        )
        self.vector_norm = nn.LayerNorm(channels)

    def encode(self, batch: dict[str, torch.Tensor]) -> Encoded:
        """A batch (dataset.PoolBatcher.build) -> its board token maps,
        summaries and candidate embeddings."""
        x, s = self.trunk(batch["input_spatial"], trunk_scalar(batch))
        board = x.flatten(2).transpose(1, 2) + self.board_pos_emb  # (P, 225, C)
        g = torch.cat([mean_max_pool(x), s], dim=1)
        e = encode_moves_on_board(
            self.move_encoder,
            self.cross_check_encoder,
            board,
            batch["move_letters"],
            batch["move_blanks"],
            batch["move_squares"],
            batch["move_tile_mask"],
            batch["move_scalars"],
            batch["move_cross_cells"],
            batch["move_cross_letters"],
            batch["move_pos_id"],
        ) + self.equity(batch["move_equity"].unsqueeze(1))
        return Encoded(board, g, e)

    def candidate_vectors(
        self, e: torch.Tensor, board: torch.Tensor, g: torch.Tensor, pos_id: torch.Tensor
    ) -> torch.Tensor:
        """Candidate embeddings (M, C) attending into their positions' board
        maps (P, 225, C), fused with the summaries (P, 3C) and normalized ->
        (M, C). The greedy form passes a conditioned map and summary."""
        attended = attend_by_position(self.cross_attn, e, board, pos_id)
        return self.vector_norm(self.fuse(torch.cat([attended, g[pos_id]], dim=1)))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """A batch -> the candidates' vectors (M, C) and the positions'
        summaries (P, 3C)."""
        enc = self.encode(batch)
        return self.candidate_vectors(enc.e, enc.board, enc.g, batch["move_pos_id"]), enc.g


def mean_spread(grid: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """The mean over positions of the standard deviation of a padded grid's
    values (P, maxK) across each position's candidates: about 0 for a head
    that scores every candidate alike."""
    valid = (~pad).to(grid.dtype)
    n = valid.sum(dim=1).clamp(min=1)
    values = torch.where(pad, 0.0, grid)
    mean = values.sum(dim=1) / n
    var = (((values - mean.unsqueeze(1)) * valid) ** 2).sum(dim=1) / n
    return var.sqrt().mean()


def to_grid(values: torch.Tensor, pos_id: torch.Tensor, num_positions: int, fill=0.0):
    """Flattened per-candidate values (M, ...) -> a padded (P, maxK, ...) grid
    and its padding mask (P, maxK), True where there is no candidate."""
    rank, max_k = rank_within_position(pos_id, num_positions)
    grid = values.new_full((num_positions, max_k, *values.shape[1:]), fill)
    grid[pos_id, rank] = values
    pad = torch.ones(num_positions, max_k, dtype=torch.bool, device=values.device)
    pad[pos_id, rank] = False
    return grid, pad
