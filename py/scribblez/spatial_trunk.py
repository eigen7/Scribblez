"""The board trunk shared by every Scribblez model.

The trunk is a conv stem, an injection of the scalar (non-spatial) input
features, and a tower, producing a (B, C, 15, 15) feature map for each model's
task-specific heads. Two towers are available (trunk_arms.py):
- conv: residual conv blocks, every third a KataGo-style global-pooling block
  that re-broadcasts board-global context;
- transformer: attention over the 225 cells as tokens, optionally with extra
  register tokens (transformer_tower.py).

docs/model_architectures.md diagrams this trunk; keep it in sync.
"""

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.transformer_tower import TransformerConfig, TransformerTower
from scribblez.trunk_arms import TRUNK_TRANSFORMER

# The spatial input starts with the 26 letter one-hot planes
# (engine/include/encoding/board_planes.h), which the lexicon module reads.
N_LETTERS = 26


def mean_max_pool(x: torch.Tensor) -> torch.Tensor:
    """Channel-wise mean and max over the spatial dims: (B, C, H, W) -> (B, 2C)."""
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
    """KataGo-style residual block that re-injects board-global context.

    The first conv's output splits into a spatial branch and a pooling branch.
    The pooling branch is pooled over the whole board and projected to a
    per-channel bias (or, with use_film, a gain and bias) on the spatial branch
    before the second conv. This lets deep blocks read global state, such as
    tiles remaining or the score differential, instead of relying on it
    surviving from the stem injection through every preceding conv.
    """

    def __init__(self, channels: int, pool_channels: int | None = None, use_film: bool = False):
        super().__init__()
        if pool_channels is None:
            pool_channels = channels // 2
        self.pool_channels = pool_channels
        self.spatial_channels = channels - pool_channels
        self.use_film = use_film

        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        # Emits beta, plus gamma under FiLM. The gamma half is zero-initialized
        # so (1 + gamma) starts at 1 and the block starts out additive.
        out_features = (2 * self.spatial_channels) if use_film else self.spatial_channels
        self.pool_fc = nn.Linear(2 * pool_channels, out_features)
        if use_film:
            nn.init.zeros_(self.pool_fc.weight[self.spatial_channels :])
            nn.init.zeros_(self.pool_fc.bias[self.spatial_channels :])
        self.bn2 = nn.BatchNorm2d(self.spatial_channels)
        self.conv2 = nn.Conv2d(self.spatial_channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(F.relu(self.bn1(x)))
        spatial = out[:, : self.spatial_channels]
        pool = out[:, self.spatial_channels :]
        proj = self.pool_fc(mean_max_pool(pool))
        if self.use_film:
            beta, gamma = proj[:, : self.spatial_channels], proj[:, self.spatial_channels :]
            spatial = (1 + gamma[:, :, None, None]) * spatial + beta[:, :, None, None]
        else:
            spatial = spatial + proj[:, :, None, None]  # beta (per-channel bias)
        out = self.conv2(F.relu(self.bn2(spatial)))
        return out + residual


def make_block(channels: int, index: int, use_film: bool = False) -> nn.Module:
    """The conv tower's block `index`: every third is a global-pooling block,
    which re-broadcasts global context at modest cost; the rest are plain
    residual blocks. use_film affects only the global-pooling blocks."""
    if index % 3 == 2:
        return GlobalPoolingResBlock(channels, use_film=use_film)
    return ResBlock(channels)


class SpatialTrunk(nn.Module):
    """Conv stem + scalar injection + tower -> (B, C, 15, 15) features.

    forward returns (features, scalar_proj), where scalar_proj is the (B, C)
    projection of the scalar input that is broadcast-added at the stem. It is
    returned so heads can also read the scalar features directly.

    `transformer` None selects the conv tower; a TransformerConfig selects the
    transformer tower. `registers`, transformer only, maps the scalar input to
    (B, registers.num_tokens, C) extra tokens appended to the cell sequence.

    use_film turns the scalar injection sites (the stem, and the conv tower's
    global-pooling blocks) from additive into FiLM: (1 + gamma) * x + beta. The
    multiplicative term lets a scalar, such as an opponent-leave letter count,
    gate a board feature, such as that letter's cross-check plane, rather than
    only shift it. The gamma projections are zero-initialized, so a FiLM trunk
    starts out identical to the additive one.
    """

    def __init__(
        self,
        spatial_planes: int,
        scalar_size: int,
        trunk_channels: int,
        num_blocks: int,
        lexicon_module: nn.Module | None = None,
        use_film: bool = False,
        transformer: TransformerConfig | None = None,
        registers: nn.Module | None = None,
        board_size: int = 15,
    ):
        super().__init__()
        self.use_film = use_film
        self.num_cells = board_size * board_size
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
        if use_film:
            self.stem_gamma = nn.Linear(scalar_size, trunk_channels)
            nn.init.zeros_(self.stem_gamma.weight)
            nn.init.zeros_(self.stem_gamma.bias)
        # Optional compiled-lexicon module (scribblez.lexical_tool.modules), a
        # per-lane DAWG walker. It runs over every row and column after the stem,
        # adding a per-cell word-legality residual in both orientations.
        self.lexicon_module = lexicon_module
        # Exactly one tower is built; `tower` is None for the conv tower. The
        # attribute names are checkpoint parameter names, so renaming them breaks
        # loading existing checkpoints.
        self.registers = registers
        if transformer is None:
            if registers is not None:
                raise ValueError("register tokens need the transformer tower")
            self.blocks = nn.Sequential(
                *[make_block(trunk_channels, i, use_film=use_film) for i in range(num_blocks)]
            )
            self.trunk_bn = nn.BatchNorm2d(trunk_channels)
            self.tower = None
        else:
            num_registers = registers.num_tokens if registers is not None else 0
            self.tower = TransformerTower(
                trunk_channels, num_blocks, transformer, board_size, num_registers
            )

    def _lexicon_residual(self, x: torch.Tensor, letters: torch.Tensor) -> torch.Tensor:
        """The lexicon module's per-cell residual, summed over rows and columns:
        x (B, C, 15, 15) features, letters (B, N_LETTERS, 15, 15) -> (B, C, 15, 15)."""
        b, c, s, _ = x.shape
        rows = x.permute(0, 2, 3, 1).reshape(b * s, s, c)
        row_letters = letters.permute(0, 2, 3, 1).reshape(b * s, s, N_LETTERS)
        row_res = self.lexicon_module(rows, row_letters).cell_residual
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
        if self.use_film:
            gamma = self.stem_gamma(input_scalar)  # (B, C)
            x = (1 + gamma[:, :, None, None]) * x + s[:, :, None, None]
        else:
            x = x + s[:, :, None, None]
        if self.lexicon_module is not None:
            x = x + self._lexicon_residual(x, input_spatial[:, :N_LETTERS])
        if self.tower is None:
            x = F.relu(self.trunk_bn(self.blocks(x)))
        else:
            x = self._transformer_tower(x, input_scalar)
        return x, s

    def _transformer_tower(self, x: torch.Tensor, input_scalar: torch.Tensor) -> torch.Tensor:
        """Run the transformer tower over the stem features as a row-major cell
        sequence, plus any register tokens, and return the cells as (B, C, H, W)."""
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        if self.registers is not None:
            tokens = torch.cat([tokens, self.registers(input_scalar)], dim=1)
        tokens = self.tower(tokens)
        cells = tokens[:, : self.num_cells].transpose(1, 2).reshape(b, c, h, w)
        return F.relu(cells)


def transformer_config(cfg: Mapping) -> TransformerConfig | None:
    """The TransformerConfig a workload's `trunk` and `transformer_*` params
    select, or None for the conv tower. `cfg` is the params as a dict or a
    checkpoint's config; sharing this function keeps trainers and checkpoint
    loaders in agreement."""
    if cfg["trunk"] != TRUNK_TRANSFORMER:
        return None
    return TransformerConfig(
        mid_channels=cfg["transformer_mid_channels"],
        num_heads=cfg["transformer_heads"],
        ffn_channels=cfg["transformer_ffn_channels"],
    )
