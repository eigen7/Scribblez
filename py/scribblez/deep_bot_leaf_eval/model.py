"""The DeepBot leaf evaluation model.

DeepBot's rollouts end after one player moves from a full rack and the other
replies, neither drawing. This model values that leaf from the side of the
player who just moved: both leaves are known, and the pool (every tile on
neither the board nor a leave) is where both refills come from. DeepBot
evaluates one leaf per rollout, so the model is small.

The input is the engine's leaf eval row (encoding/leaf_eval_input_encoder.h).
The scalars reach either trunk as features: each seat's per-tile counts
normalized by the tile's total, a zero flag per count (so "none left" need not
be learned as the small end of a line), then the score differential, both
leave values and the pool size.

`trunk` picks the tower:

  transformer  one token per cell (a linear embedding of its planes plus a
               learned per-cell embedding, which carries position and the
               premium squares), one register per tile kind (its three seats'
               features plus a learned tile embedding), and one global token
               (the four scalars). Pre-norm layers of LayerNorm, attention
               without position encoding, and a GELU MLP: under TensorRT at
               this width these run ~1.3x faster than transformer_tower.py's
               RoPE / RMSNorm / SwiGLU blocks, which are bound by memory
               traffic rather than FLOPs.
  conv         spatial_trunk.py's conv tower, the features injected at the
               stem. ~3x the transformer's TensorRT throughput at equal FLOPs.

`join_planes` False drops the row's join planes, for measuring what they give
each trunk.

Outputs, with position_eval's heads and losses: wld (B, 3) logits, the value
used at inference, and score_diff (B, 2) [mean, std].
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.ffi import format_layout
from scribblez.position_eval.model import HeadedModel, ScoreDiffHead, WldHead
from scribblez.spatial_trunk import SpatialTrunk, mean_max_pool
from scribblez.supply_registers import N_TILES, TILE_COUNTS
from scribblez.trunk_arms import TRUNK_TRANSFORMER

LAYOUT = format_layout()["constants"]["deep_bot_leaf_eval"]
BOARD_CELLS = 15 * 15
NUM_SEATS = 3  # my leave, the opponent's, the pool
# The scalars after the seats' counts: score diff, both leave values, pool size.
GLOBAL_SCALARS = 4
SEAT_FEATURES = 2 * NUM_SEATS * N_TILES  # normalized counts, then zero flags
SCALAR_FEATURES = SEAT_FEATURES + GLOBAL_SCALARS
# The transformer's MLP hidden width, as a multiple of its width.
MLP_RATIO = 3


@dataclass(frozen=True)
class LeafEvalShape:
    trunk: str
    join_planes: bool
    width: int  # channels
    depth: int  # transformer layers, or conv residual blocks
    num_heads: int  # transformer only

    def spatial_planes_used(self) -> int:
        return LAYOUT["join_plane0"] + (LAYOUT["join_planes"] if self.join_planes else 0)


def scalar_features(input_scalar: torch.Tensor, tile_counts: torch.Tensor) -> torch.Tensor:
    """(B, row scalars) -> (B, SCALAR_FEATURES): per seat and tile, the count
    normalized by the tile's total, then the zero flags, then the global
    scalars as the row scales them."""
    seats = input_scalar[:, LAYOUT["my_leave0"] : LAYOUT["pool0"] + N_TILES]
    seats = seats.unflatten(1, (NUM_SEATS, N_TILES))
    normalized = (seats / tile_counts).flatten(1)
    zero = (seats == 0).to(seats.dtype).flatten(1)
    global_scalars = input_scalar[:, LAYOUT["score_diff"] : LAYOUT["pool_size"] + 1]
    return torch.cat([normalized, zero, global_scalars], dim=1)


class TransformerLayer(nn.Module):
    """Pre-norm attention then a GELU MLP, each on its own residual."""

    def __init__(self, width: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.attn_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.mlp_norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, MLP_RATIO * width)
        self.down = nn.Linear(MLP_RATIO * width, width)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        b, s, c = x.shape
        qkv = self.qkv(self.attn_norm(x)).reshape(b, s, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        context = F.scaled_dot_product_attention(q, k, v)
        return self.out(context.transpose(1, 2).reshape(b, s, c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attention(x)
        return x + self.down(F.gelu(self.up(self.mlp_norm(x))))


class TransformerTrunk(nn.Module):
    """(spatial, features) -> (B, 3C): the mean and max of the cell tokens, and
    the global token."""

    def __init__(self, shape: LeafEvalShape):
        super().__init__()
        c = shape.width
        self.cell_proj = nn.Linear(shape.spatial_planes_used(), c)
        self.cell_embed = nn.Parameter(torch.randn(BOARD_CELLS, c) * 0.02)
        self.tile_embed = nn.Parameter(torch.randn(N_TILES, c) * 0.02)
        self.register_proj = nn.Linear(2 * NUM_SEATS, c)
        self.global_embed = nn.Parameter(torch.randn(c) * 0.02)
        self.global_proj = nn.Linear(GLOBAL_SCALARS, c)
        self.layers = nn.ModuleList(
            TransformerLayer(c, shape.num_heads) for _ in range(shape.depth)
        )
        self.final_norm = nn.LayerNorm(c)

    def _registers(self, features: torch.Tensor) -> torch.Tensor:
        """(B, N_TILES, C): each tile's three normalized counts and zero flags."""
        seats = features[:, :SEAT_FEATURES].unflatten(1, (2 * NUM_SEATS, N_TILES))
        return self.tile_embed + self.register_proj(seats.transpose(1, 2))

    def _tokens(self, spatial: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        cells = self.cell_proj(spatial.flatten(2).transpose(1, 2)) + self.cell_embed
        global_token = self.global_embed + self.global_proj(features[:, SEAT_FEATURES:])
        return torch.cat([cells, self._registers(features), global_token[:, None]], dim=1)

    def forward(self, spatial: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = self._tokens(spatial, features)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        cells = x[:, :BOARD_CELLS]
        # max(dim).values, not amax: see spatial_trunk.mean_max_pool.
        return torch.cat([cells.mean(dim=1), cells.max(dim=1).values, x[:, -1]], dim=1)


class ConvTrunk(nn.Module):
    """(spatial, features) -> (B, 3C): the conv tower's pooled map and its
    projection of the features."""

    def __init__(self, shape: LeafEvalShape):
        super().__init__()
        self.tower = SpatialTrunk(
            shape.spatial_planes_used(), SCALAR_FEATURES, shape.width, shape.depth
        )

    def forward(self, spatial: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x, s = self.tower(spatial, features)
        return torch.cat([mean_max_pool(x), s], dim=1)


class LeafEvalModel(HeadedModel):
    def __init__(self, shape: LeafEvalShape):
        super().__init__()
        self.shape = shape
        self.register_buffer(
            "tile_counts", torch.tensor(TILE_COUNTS, dtype=torch.float32), persistent=False
        )
        trunk_cls = TransformerTrunk if shape.trunk == TRUNK_TRANSFORMER else ConvTrunk
        self.trunk = trunk_cls(shape)
        value_in = 3 * shape.width
        self.heads = nn.ModuleDict(
            {h.name: h for h in (WldHead(value_in), ScoreDiffHead(value_in))}
        )

    def forward(
        self, input_spatial: torch.Tensor, input_scalar: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        spatial = input_spatial[:, : self.shape.spatial_planes_used()]
        value_in = self.trunk(spatial, scalar_features(input_scalar, self.tile_counts))
        return {name: head(None, value_in) for name, head in self.heads.items()}
