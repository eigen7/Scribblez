"""The policy-set model (docs/plans/deep_bot.md, "The policy-set model"): the
shared encoder both forms build on, and form 2, the slot decoder.

The encoder is the move set model's: the board trunk once per position, each
candidate embedded by the move and cross-check encoders against the board
tokens and cross-attending into them, fused with the position's global
summary. The trunk reads the opponent-rack input, the chooser's belief about
the opponent's rack (dataset.opp_rack_input), as a scalar block beside the
board row's. Its output is one vector per candidate, which a form reads.

The slot decoder (form 2) has `slots` learned query slots. Conditioned on the
position's summary, they self-attend and cross-attend to the position's
candidate vectors, and each points to one candidate through a softmax over
the dot products. Trained with the coverage loss over slot prefixes, the
first N slots serve as the set for any N up to `slots`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
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
from scribblez.policy_set.metrics import SET_SIZES
from scribblez.spatial_trunk import SpatialTrunk, mean_max_pool

# Below this cover probability the coverage loss uses its first-order form.
NEARLY_MISSED = 1e-4
# log p floor in the coverage loss: finite at padding, and far below any
# real candidate's log-probability.
LOG_P_FLOOR = -1e4


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
        self.cross_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(4 * channels, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """A batch (dataset.PoolBatcher.build) -> the candidates' vectors
        (M, C) and the positions' summaries (P, 3C)."""
        x, s = self.trunk(batch["input_spatial"], trunk_scalar(batch))
        board = x.flatten(2).transpose(1, 2) + self.board_pos_emb  # (P, 225, C)
        g = torch.cat([mean_max_pool(x), s], dim=1)
        pos_id = batch["move_pos_id"]
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
            pos_id,
        )
        attended = attend_by_position(self.cross_attn, e, board, pos_id)
        return self.fuse(torch.cat([attended, g[pos_id]], dim=1)), g


def to_grid(values: torch.Tensor, pos_id: torch.Tensor, num_positions: int, fill=0.0):
    """Flattened per-candidate values (M, ...) -> a padded (P, maxK, ...) grid
    and its padding mask (P, maxK), True where there is no candidate."""
    rank, max_k = rank_within_position(pos_id, num_positions)
    grid = values.new_full((num_positions, max_k, *values.shape[1:]), fill)
    grid[pos_id, rank] = values
    pad = torch.ones(num_positions, max_k, dtype=torch.bool, device=values.device)
    pad[pos_id, rank] = False
    return grid, pad


class SlotDecoder(nn.Module):
    def __init__(self, channels: int, slots: int, layers: int, num_heads: int):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(slots, channels) * 0.02)
        self.g_proj = nn.Linear(3 * channels, channels)
        layer = nn.TransformerDecoderLayer(
            channels,
            num_heads,
            dim_feedforward=4 * channels,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)

    def forward(self, h: torch.Tensor, g: torch.Tensor, pos_id: torch.Tensor):
        """Candidate vectors (M, C), summaries (P, 3C) -> pointer logits
        (P, slots, maxK), -inf at padding, and the padding mask (P, maxK)."""
        memory, pad = to_grid(h, pos_id, g.shape[0])
        tgt = self.slots.unsqueeze(0) + self.g_proj(g).unsqueeze(1)  # (P, slots, C)
        out = self.decoder(tgt, memory, memory_key_padding_mask=pad)
        logits = torch.einsum("psc,pkc->psk", self.query(out), self.key(memory))
        logits = logits / math.sqrt(h.shape[1])
        return logits.masked_fill(pad.unsqueeze(1), float("-inf")), pad


class SlotModel(nn.Module):
    """Form 2: the encoder and the slot decoder."""

    def __init__(self, encoder: PolicySetEncoder, slots: int, layers: int, num_heads: int):
        super().__init__()
        self.encoder = encoder
        channels = encoder.board_pos_emb.shape[1]
        self.decoder = SlotDecoder(channels, slots, layers, num_heads)

    def forward(self, batch: dict[str, torch.Tensor]):
        h, g = self.encoder(batch)
        return self.decoder(h, g, batch["move_pos_id"])


def coverage_loss(
    logits: torch.Tensor, q_grid: torch.Tensor, sizes: tuple[int, ...] | None = None
) -> dict[int, torch.Tensor]:
    """The coverage loss per set size N in `sizes` (default: the SET_SIZES the
    model's slots reach), each a mean over positions:
    L_N = -sum_a q(a) log(1 - prod_{k<=N} (1 - p_k(a))), the negative
    log-probability that the first N slots cover the best move. `logits`
    (P, slots, maxK), `q_grid` (P, maxK) with 0 at padding."""
    if sizes is None:
        sizes = tuple(n for n in SET_SIZES if n <= logits.shape[1])
    # Floored so padding (-inf) stays finite; q is 0 there.
    log_p = F.log_softmax(logits.float(), dim=-1).clamp(min=LOG_P_FLOOR)
    # log prod (1 - p_k), accumulated over slots; p clamped below 1 so that a
    # move no slot can miss stays finite.
    miss = torch.cumsum(torch.log1p(-log_p.exp().clamp(max=1 - 1e-6)), dim=1)
    cum_log_p = torch.logcumsumexp(log_p, dim=1)  # log sum_{k<=n} p_k
    out = {}
    for n in sizes:
        log_cover = _log_cover(miss[:, n - 1], cum_log_p[:, n - 1])  # (P, maxK)
        out[n] = -(q_grid * log_cover).sum(dim=1).mean()
    return out


def _log_cover(miss: torch.Tensor, log_p_sum: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(miss)), the log-probability some slot of the prefix picks
    the move. Where the prefix nearly misses it, log(-expm1(miss)) loses its
    gradient to rounding, so the first-order form log sum p_k stands in, off
    by a relative O(sum p_k) there."""
    nearly_missed = miss > -NEARLY_MISSED
    # The unused branch's input is kept finite: torch.where would otherwise
    # pass its NaN gradient through.
    exact = torch.log(-torch.expm1(torch.where(nearly_missed, -1.0, miss)))
    return torch.where(nearly_missed, log_p_sum, exact)


def slot_picks(logits: torch.Tensor, pad: torch.Tensor) -> list[list[int]]:
    """Each position's ordered set, as candidate indices within the position:
    slot k takes its most probable move not taken by an earlier slot."""
    picks = []
    for pos_logits, pos_pad in zip(logits.float().cpu(), pad.cpu(), strict=True):
        n_moves = int((~pos_pad).sum())
        taken: list[int] = []
        # A slot passes over at most the earlier slots' picks, so its top
        # `slots` candidates hold its choice.
        k = min(pos_logits.shape[0], n_moves)
        for order in torch.topk(pos_logits[:, :n_moves], k, dim=1).indices.tolist():
            taken.append(next(m for m in order if m not in taken))
            if len(taken) == n_moves:
                break
        picks.append(taken)
    return picks
