"""Shared spatial trunk for Scribblez models.

Every Scribblez model encodes the board with the same front end: a conv stem,
an injection of the scalar (non-spatial) features, and a tower of residual
blocks -- some of them KataGo-style global-pooling blocks that re-broadcast
board-global context through the tower. The result is a (B, C, 15, 15) feature
map that each model's task-specific heads (and, for the per-lane model, a lane
sequence model) build on. This module owns that trunk and its building blocks so
the post-move and max-move-per-lane models share one implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# The board's letter planes lead the spatial input (BoardPlanes lays the 26 letter
# one-hots first), so input_spatial[:, :N_LETTERS] is the per-cell one-hot a lexicon
# module reads.
N_LETTERS = 26


def mean_max_pool(x: torch.Tensor) -> torch.Tensor:
    """Concatenate the channel-wise mean and max over the spatial dims:
    (B, C, H, W) -> (B, 2C). Mean captures average board texture; max captures
    the strongest local activation (e.g. a high-value square or threat)."""
    return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)


class ResBlock(nn.Module):
    """Pre-activation residual block: BN -> ReLU -> conv -> BN -> ReLU -> conv -> skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + residual


class GlobalPoolingResBlock(nn.Module):
    """Residual block that injects board-global context (KataGo-style).

    The first conv's output is split into a spatial branch and a pooling branch.
    The pooling branch is mean+max pooled over the whole board and projected to a
    per-channel bias that is broadcast-added to the spatial branch before the
    second conv. This lets the block re-read global state (tiles remaining,
    overall board openness, and -- for the post-move model -- the score
    differential) instead of relying on it surviving unchanged from the stem
    injection through every preceding conv.
    """

    def __init__(self, channels: int, pool_channels: int | None = None):
        super().__init__()
        if pool_channels is None:
            pool_channels = channels // 2
        self.pool_channels = pool_channels
        self.spatial_channels = channels - pool_channels

        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        # mean + max over the pooling branch -> per-(spatial-channel) bias.
        self.pool_fc = nn.Linear(2 * pool_channels, self.spatial_channels)
        self.bn2 = nn.BatchNorm2d(self.spatial_channels)
        self.conv2 = nn.Conv2d(self.spatial_channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(F.relu(self.bn1(x)))
        spatial = out[:, : self.spatial_channels]
        pool = out[:, self.spatial_channels :]
        bias = self.pool_fc(mean_max_pool(pool))  # (B, spatial_channels)
        spatial = spatial + bias[:, :, None, None]
        out = self.conv2(F.relu(self.bn2(spatial)))
        return out + residual


def make_block(channels: int, index: int) -> nn.Module:
    """Every third block is a global-pooling block; the rest are plain residual
    blocks. Interleaving keeps the cost modest while periodically re-broadcasting
    global context through the tower."""
    return GlobalPoolingResBlock(channels) if index % 3 == 2 else ResBlock(channels)


class SpatialTrunk(nn.Module):
    """Conv stem + scalar injection + residual tower -> (B, C, 15, 15) features.

    forward returns (features, scalar_proj): the post-tower feature map and the
    C-dimensional projection of the scalar input. The projection is broadcast-
    added at the stem and also returned so a model's heads can read the scalar
    features directly (the post-move value heads do).
    """

    def __init__(
        self,
        spatial_planes: int,
        scalar_size: int,
        trunk_channels: int,
        num_blocks: int,
        lexicon_module: nn.Module | None = None,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spatial_planes, trunk_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(inplace=True),
        )
        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_size, trunk_channels),
            nn.ReLU(inplace=True),
            nn.Linear(trunk_channels, trunk_channels),
        )
        # Optional compiled-lexicon tool (see scribblez.lexical_tool.modules).
        # It is a per-lane DAWG walker, so it is queried once per row and once per column
        # and its per-cell residual is summed into the board feature map after the stem,
        # giving the tower per-cell word-legality signal in both orientations.
        self.lexicon_module = lexicon_module
        self.blocks = nn.Sequential(*[make_block(trunk_channels, i) for i in range(num_blocks)])
        self.trunk_bn = nn.BatchNorm2d(trunk_channels)

    def _lexicon_residual(self, x: torch.Tensor, letters: torch.Tensor) -> torch.Tensor:
        """Run the per-lane lexicon module over rows and columns and lay its per-cell
        residuals back onto the (B, C, 15, 15) board feature map. `x` is the current
        feature map (queried features); `letters` is the (B, N_LETTERS, 15, 15) one-hot."""
        b, c, s, _ = x.shape
        # Rows: each (batch, row) is a lane of `s` cells running along the columns.
        rows = x.permute(0, 2, 3, 1).reshape(b * s, s, c)
        row_letters = letters.permute(0, 2, 3, 1).reshape(b * s, s, N_LETTERS)
        row_res = self.lexicon_module(rows, row_letters).cell_residual
        # Columns: each (batch, col) is a lane of `s` cells running along the rows.
        cols = x.permute(0, 3, 2, 1).reshape(b * s, s, c)
        col_letters = letters.permute(0, 3, 2, 1).reshape(b * s, s, N_LETTERS)
        col_res = self.lexicon_module(cols, col_letters).cell_residual
        res = x.new_zeros(b, c, s, s)
        if row_res is not None:
            res = res + row_res.reshape(b, s, s, c).permute(0, 3, 1, 2)
        if col_res is not None:
            res = res + col_res.reshape(b, s, s, c).permute(0, 3, 2, 1)
        return res

    def forward(
        self, input_spatial: torch.Tensor, input_scalar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(input_spatial)
        s = self.scalar_proj(input_scalar)  # (B, C)
        x = x + s[:, :, None, None]  # broadcast-add over spatial dims
        if self.lexicon_module is not None:
            x = x + self._lexicon_residual(x, input_spatial[:, :N_LETTERS])
        x = self.blocks(x)
        x = F.relu(self.trunk_bn(x))
        return x, s
