"""Post-move value model for Scrabble position evaluation.

Architecture:
    - Spatial trunk: (85, 15, 15) -> conv 3x3 -> 128 channels -> N residual blocks
  - Scalar injection: (936,) -> FC -> 128 -> broadcast-add to spatial features
  - Pooling: global average pool -> 128-d trunk vector
  - Three heads:
    * WLD (inference): FC -> 3 logits (win/draw/loss)
    * ScoreDiff (aux): FC -> 2 = [mean, std] of the final score-differential
      Gaussian (std via softplus), trained by Gaussian NLL
    * OppNextPlacement (aux): 1x1 conv -> (1, 15, 15) -> sigmoid mask

The two model inputs (33 spatial planes, 936 scalars) and the three head output
shapes are fixed by the training pipeline and the C++ inference contract; the
trunk between them is free to change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mean_max_pool(x: torch.Tensor) -> torch.Tensor:
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
    second conv. This lets the block re-read global state (score differential,
    tiles remaining, overall board openness) instead of relying on it surviving
    unchanged from the stem injection through every preceding conv.
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
        bias = self.pool_fc(_mean_max_pool(pool))  # (B, spatial_channels)
        spatial = spatial + bias[:, :, None, None]
        out = self.conv2(F.relu(self.bn2(spatial)))
        return out + residual


def _make_block(channels: int, index: int) -> nn.Module:
    """Every third block is a global-pooling block; the rest are plain residual
    blocks. Interleaving keeps the cost modest while periodically re-broadcasting
    global context through the tower."""
    return GlobalPoolingResBlock(channels) if index % 3 == 2 else ResBlock(channels)


class ScribblezModel(nn.Module):
    """Post-move value network with 3 heads."""

    def __init__(
        self,
        spatial_planes: int,
        scalar_size: int,
        trunk_channels: int = 192,
        num_blocks: int = 10,
        board_size: int = 15,
    ):
        super().__init__()
        self.board_size = board_size

        # Spatial stem.
        self.stem = nn.Sequential(
            nn.Conv2d(spatial_planes, trunk_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(inplace=True),
        )

        # Scalar injection: project scalars to trunk_channels, broadcast-add. The
        # same projection is also fed to the value heads (see forward()).
        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_size, trunk_channels),
            nn.ReLU(inplace=True),
            nn.Linear(trunk_channels, trunk_channels),
        )

        # Residual tower (some blocks re-inject global context).
        self.blocks = nn.Sequential(
            *[_make_block(trunk_channels, i) for i in range(num_blocks)]
        )
        self.trunk_bn = nn.BatchNorm2d(trunk_channels)

        # --- Heads ---
        # The value heads read a 3C summary: mean+max board pooling (2C) plus the
        # scalar projection (C), so the score-diff scalar reaches them directly.
        value_in = 3 * trunk_channels

        # WLD head (inference head): FC -> 3.
        self.wld_fc = nn.Sequential(
            nn.Linear(value_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

        # Score-diff head (aux): FC -> 2 = [mean, raw_std] of the final
        # score-differential Gaussian. forward() maps raw_std through softplus to
        # a positive std, so the exported output is [mean, std].
        self.sd_fc = nn.Sequential(
            nn.Linear(value_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

        # Opp-next-placement head (aux): 1x1 conv -> (1, 15, 15).
        self.opp_conv = nn.Sequential(
            nn.Conv2d(trunk_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(
        self, input_spatial: torch.Tensor, input_scalar: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_spatial: (B, 85, 15, 15)
            input_scalar:  (B, 936)

        Returns:
            Dict with keys: "wld" (B,3 logits), "score_diff" (B,2 = [mean, std]),
            "opp_next_placement" (B,15,15 logits).
        """
        # Spatial stem.
        x = self.stem(input_spatial)

        # Scalar injection (s is reused by the value heads below).
        s = self.scalar_proj(input_scalar)  # (B, C)
        x = x + s[:, :, None, None]  # broadcast add over spatial dims

        # Residual tower.
        x = self.blocks(x)
        x = F.relu(self.trunk_bn(x))

        # Value summary: mean+max board pooling concatenated with the scalar
        # projection, so the heads see global board context and the raw scalars.
        value_in = torch.cat([_mean_max_pool(x), s], dim=1)  # (B, 3C)

        wld = self.wld_fc(value_in)
        opp = self.opp_conv(x).squeeze(1)  # (B, 15, 15)

        # Score-diff Gaussian: [mean, std]. softplus + floor keeps std positive
        # and away from 0 so the Gaussian NLL stays finite.
        sd_raw = self.sd_fc(value_in)  # (B, 2): [mean, raw_std]
        sd_mean = sd_raw[:, 0:1]
        sd_std = F.softplus(sd_raw[:, 1:2]) + 1e-3
        sd = torch.cat([sd_mean, sd_std], dim=1)  # (B, 2): [mean, std]

        return {"wld": wld, "score_diff": sd, "opp_next_placement": opp}


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    lambda_sd: float = 1.0,
    lambda_opp: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute combined loss for all heads.

    Args:
        outputs: model forward() result. "wld" (B,3 logits), "score_diff"
                 (B,2 = [mean, std]), "opp_next_placement" (B,15,15 logits).
        targets: dict with "wld" (B,3) one-hot, "score_diff" (B,1) the observed
                 final differential, "opp_next_placement" (B,15,15) binary mask.

    Returns:
        Dict with "total", "wld", "score_diff", "opp_next_placement" losses.
    """
    # WLD: cross-entropy against one-hot target.
    wld_target_idx = targets["wld"].argmax(dim=1)
    loss_wld = F.cross_entropy(outputs["wld"], wld_target_idx)

    # Score-diff: Gaussian negative log-likelihood of the observed differential
    # under the head's predicted (mean, std).
    sd_mean = outputs["score_diff"][:, 0]
    sd_var = outputs["score_diff"][:, 1] ** 2
    sd_target = targets["score_diff"].squeeze(1)
    loss_sd = F.gaussian_nll_loss(sd_mean, sd_target, sd_var)

    # Opp-next-placement: binary cross-entropy per cell.
    loss_opp = F.binary_cross_entropy_with_logits(
        outputs["opp_next_placement"], targets["opp_next_placement"]
    )

    total = loss_wld + lambda_sd * loss_sd + lambda_opp * loss_opp

    return {
        "total": total,
        "wld": loss_wld,
        "score_diff": loss_sd,
        "opp_next_placement": loss_opp,
    }
