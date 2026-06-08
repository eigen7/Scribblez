"""Post-move value model for Scrabble position evaluation.

Architecture:
  - Spatial trunk: (32, 15, 15) -> conv 3x3 -> 128 channels -> N residual blocks
  - Scalar injection: (58,) -> FC -> 128 -> broadcast-add to spatial features
  - Pooling: global average pool -> 128-d trunk vector
  - Three heads:
    * WLD (inference): FC -> 3 logits (win/draw/loss)
    * ScoreDiff (aux): FC -> 801 logits (one-hot score-diff bins)
    * OppNextPlacement (aux): 1x1 conv -> (1, 15, 15) -> sigmoid mask
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ScribblezModel(nn.Module):
    """Post-move value network with 3 heads."""

    def __init__(
        self,
        spatial_planes: int = 32,
        scalar_size: int = 58,
        trunk_channels: int = 128,
        num_blocks: int = 8,
        score_diff_bins: int = 801,
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

        # Scalar injection: project scalars to trunk_channels, broadcast-add.
        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_size, trunk_channels),
            nn.ReLU(inplace=True),
            nn.Linear(trunk_channels, trunk_channels),
        )

        # Residual tower.
        self.blocks = nn.Sequential(*[ResBlock(trunk_channels) for _ in range(num_blocks)])
        self.trunk_bn = nn.BatchNorm2d(trunk_channels)

        # --- Heads ---

        # WLD head (inference head): global pool -> FC -> 3.
        self.wld_fc = nn.Sequential(
            nn.Linear(trunk_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

        # Score-diff head (aux): global pool -> FC -> 801.
        self.sd_fc = nn.Sequential(
            nn.Linear(trunk_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, score_diff_bins),
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
            input_spatial: (B, 32, 15, 15)
            input_scalar:  (B, 58)

        Returns:
            Dict with keys: "wld" (B,3), "score_diff" (B,801),
            "opp_next_placement" (B,15,15).
        """
        # Spatial stem.
        x = self.stem(input_spatial)

        # Scalar injection.
        s = self.scalar_proj(input_scalar)  # (B, C)
        x = x + s[:, :, None, None]  # broadcast add over spatial dims

        # Residual tower.
        x = self.blocks(x)
        x = F.relu(self.trunk_bn(x))

        # Global pool for value heads.
        pooled = x.mean(dim=(2, 3))  # (B, C)

        wld = self.wld_fc(pooled)
        sd = self.sd_fc(pooled)
        opp = self.opp_conv(x).squeeze(1)  # (B, 15, 15)

        return {"wld": wld, "score_diff": sd, "opp_next_placement": opp}


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    lambda_sd: float = 1.0,
    lambda_opp: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute combined loss for all heads.

    Args:
        outputs: model forward() result (logits).
        targets: dict with "wld" (B,3) one-hot, "score_diff" (B,801) one-hot,
                 "opp_next_placement" (B,15,15) binary mask.

    Returns:
        Dict with "total", "wld", "score_diff", "opp_next_placement" losses.
    """
    # WLD: cross-entropy against one-hot target.
    wld_target_idx = targets["wld"].argmax(dim=1)
    loss_wld = F.cross_entropy(outputs["wld"], wld_target_idx)

    # Score-diff: cross-entropy against one-hot target.
    sd_target_idx = targets["score_diff"].argmax(dim=1)
    loss_sd = F.cross_entropy(outputs["score_diff"], sd_target_idx)

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
