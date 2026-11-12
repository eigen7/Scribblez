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
used at inference, and score_diff (B, 2) [mean, std]. Two per-square heads read
the trunk's cell features: ownership (B, 5, 15, 15) logits over what becomes of
each empty square (engine encoding/square_fates.h), and square_points
(B, 2, 15, 15), the mean and variance of the signed points of the move that
fills it. Two end-of-game heads read the pooled vector: going_out (B, 3)
logits over who goes out (the mover, the opponent, nobody), and end_points
(B, 2), the mean and variance of the net end-of-game rack adjustment from the
mover's side (engine training/game_end_targets.h). docs/plans/deep_bot.md
("The search network") gives their purpose: sims complete their maps and
end-of-game outcomes with them, and residuals against them are evidence.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.ffi import format_layout
from scribblez.position_eval.model import (
    Head,
    HeadedModel,
    HeadLoss,
    LossConfig,
    ScoreDiffHead,
    WldHead,
)
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
SQUARE_FATE_CLASSES = 5
# A filled square's ownership label: it has no fate to predict.
FILLED = -1
# Points are learned in units of 100, the row's score-differential scale.
POINTS_SCALE = 100.0
GOING_OUT_CLASSES = 3
# A game cut short has no end to predict: its going-out label.
UNLABELLED = -1


@dataclass(frozen=True)
class LeafEvalShape:
    trunk: str
    join_planes: bool
    width: int  # channels
    depth: int  # transformer layers, or conv residual blocks
    num_heads: int  # transformer only
    normalize_scalar: bool = True  # conv only (SpatialTrunk's)

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
    """(spatial, features) -> the cell tokens as a (B, C, 15, 15) map, and
    (B, 3C): the mean and max of the cell tokens, and the global token."""

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

    def forward(
        self, spatial: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._tokens(spatial, features)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        cells = x[:, :BOARD_CELLS]
        # max(dim).values, not amax: see spatial_trunk.mean_max_pool.
        pooled = torch.cat([cells.mean(dim=1), cells.max(dim=1).values, x[:, -1]], dim=1)
        return cells.transpose(1, 2).unflatten(2, spatial.shape[2:]), pooled


class ConvTrunk(nn.Module):
    """(spatial, features) -> the conv tower's (B, C, 15, 15) map, and
    (B, 3C): that map pooled and the tower's projection of the features."""

    def __init__(self, shape: LeafEvalShape):
        super().__init__()
        self.tower = SpatialTrunk(
            shape.spatial_planes_used(),
            SCALAR_FEATURES,
            shape.width,
            shape.depth,
            normalize_scalar=shape.normalize_scalar,
        )

    def forward(
        self, spatial: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, s = self.tower(spatial, features)
        return x, torch.cat([mean_max_pool(x), s], dim=1)


@dataclass
class LeafLossConfig(LossConfig):
    """position_eval's loss config plus the per-square and end-of-game heads'
    weights."""

    lambda_ownership: float
    lambda_points: float
    lambda_going_out: float
    lambda_end_points: float


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The mean of `values` where `mask` (same shape, 0/1) is set."""
    return (values * mask).sum() / mask.sum().clamp(min=1)


def moment_losses(
    mean: torch.Tensor, var: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fits of two moments by squared error over `mask`: the mean against
    `target`, the variance against the squared deviation from the detached
    mean. Each is consistent whatever the target's shape, a spike at zero plus
    scattered large values, where a Gaussian likelihood is not."""
    loss_mean = _masked_mean((mean - target) ** 2, mask)
    loss_var = _masked_mean((var - (target - mean.detach()) ** 2) ** 2, mask)
    return loss_mean, loss_var


def mean_and_variance(out: torch.Tensor) -> torch.Tensor:
    """(B, 2, ...) raw head output -> [mean, softplus variance] along dim 1."""
    return torch.stack([out[:, 0], F.softplus(out[:, 1])], dim=1)


class OwnershipHead(Head):
    """(B, 5, 15, 15) logits over each square's fate, trained by cross-entropy
    on the empty squares."""

    name = "ownership"
    loss_keys = ("ownership",)
    target_keys = ("ownership",)

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, SQUARE_FATE_CLASSES, 1)

    def forward(self, x: torch.Tensor, value_in: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def loss(self, outputs, targets, cfg):
        ce = F.cross_entropy(outputs["ownership"], targets["ownership"].long(), ignore_index=FILLED)
        return HeadLoss(cfg.lambda_ownership * ce, {"ownership": ce})


class SquarePointsHead(Head):
    """(B, 2, 15, 15): per square, the mean and variance of the signed points
    of the move that fills it (zero if none does), in POINTS_SCALE units, by
    moment_losses on the empty squares."""

    name = "square_points"
    loss_keys = ("square_points", "square_points_mean", "square_points_var")
    target_keys = ("square_points", "ownership")

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2, 1)

    def forward(self, x: torch.Tensor, value_in: torch.Tensor) -> torch.Tensor:
        return mean_and_variance(self.conv(x))

    def loss(self, outputs, targets, cfg):
        empty = (targets["ownership"] != FILLED).to(outputs["square_points"].dtype)
        mean, var = outputs["square_points"].unbind(dim=1)
        loss_mean, loss_var = moment_losses(
            mean, var, targets["square_points"] / POINTS_SCALE, empty
        )
        total = loss_mean + loss_var
        reported = {
            "square_points": total,
            "square_points_mean": loss_mean,
            "square_points_var": loss_var,
        }
        return HeadLoss(cfg.lambda_points * total, reported)


def _end_head(value_in: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(value_in, 64), nn.ReLU(inplace=True), nn.Linear(64, outputs))


def _labelled(targets: dict[str, torch.Tensor]) -> torch.Tensor:
    """(B,) 1.0 for a row whose game reached its end."""
    return (targets["going_out"][:, 0] != UNLABELLED).float()


class GoingOutHead(Head):
    """(B, 3) logits over who goes out: the mover, the opponent, or nobody;
    cross-entropy on the rows whose game reached its end."""

    name = "going_out"
    loss_keys = ("going_out",)
    target_keys = ("going_out",)

    def __init__(self, value_in: int):
        super().__init__()
        self.fc = _end_head(value_in, GOING_OUT_CLASSES)

    def forward(self, x: torch.Tensor, value_in: torch.Tensor) -> torch.Tensor:
        return self.fc(value_in)

    def loss(self, outputs, targets, cfg):
        labels = targets["going_out"][:, 0].long()
        ce = F.cross_entropy(outputs["going_out"], labels, ignore_index=UNLABELLED)
        return HeadLoss(cfg.lambda_going_out * ce, {"going_out": ce})


class EndPointsHead(Head):
    """(B, 2): the mean and variance of the net end-of-game rack adjustment
    from the mover's side, in POINTS_SCALE units, by moment_losses on the rows
    whose game reached its end."""

    name = "end_points"
    loss_keys = ("end_points", "end_points_mean", "end_points_var")
    target_keys = ("end_points", "going_out")

    def __init__(self, value_in: int):
        super().__init__()
        self.fc = _end_head(value_in, 2)

    def forward(self, x: torch.Tensor, value_in: torch.Tensor) -> torch.Tensor:
        return mean_and_variance(self.fc(value_in))

    def loss(self, outputs, targets, cfg):
        labelled = _labelled(targets).to(outputs["end_points"].dtype)
        mean, var = outputs["end_points"].unbind(dim=1)
        points = targets["end_points"][:, 0] / POINTS_SCALE
        loss_mean, loss_var = moment_losses(mean, var, points, labelled)
        total = loss_mean + loss_var
        reported = {"end_points": total, "end_points_mean": loss_mean, "end_points_var": loss_var}
        return HeadLoss(cfg.lambda_end_points * total, reported)


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
        heads = (
            WldHead(value_in),
            ScoreDiffHead(value_in),
            OwnershipHead(shape.width),
            SquarePointsHead(shape.width),
            GoingOutHead(value_in),
            EndPointsHead(value_in),
        )
        self.heads = nn.ModuleDict({h.name: h for h in heads})

    def forward(
        self, input_spatial: torch.Tensor, input_scalar: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        spatial = input_spatial[:, : self.shape.spatial_planes_used()]
        x, value_in = self.trunk(spatial, scalar_features(input_scalar, self.tile_counts))
        return {name: head(x, value_in) for name, head in self.heads.items()}
