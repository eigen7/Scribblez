"""Post-move value model for Scrabble position evaluation.

Architecture:
    - Spatial trunk: (85, 15, 15) -> conv 3x3 -> 128 channels -> N residual blocks
  - Scalar injection: (936,) -> FC -> 128 -> broadcast-add to spatial features
  - Pooling: global average pool -> 128-d trunk vector
  - Three heads:
    * WLD (inference): FC -> 3 logits (win/draw/loss)
    * ScoreDiff (aux): FC -> 2 = [mean, std] of the final score differential.
      The mean is regressed against the observed differential and the std
      against the absolute residual of the mean, both with Huber loss in score
      points. The std stack reads a detached copy of the trunk summary, so its
      loss trains only that stack and never perturbs the trunk or the mean. The
      exported second value is the std directly.
    * OppNextPlacement (aux): 1x1 conv -> (1, 15, 15) -> sigmoid mask

The two model inputs (85 spatial planes, 936 scalars) and the three head output
shapes are fixed by the training pipeline and the C++ inference contract; the
trunk between them is free to change.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# For r ~ N(0, sigma), E|r| = sqrt(2/pi)*sigma. Regressing the std against the
# absolute residual would otherwise converge to ~0.8*sigma; this rescales the
# target so the optimum matches a Gaussian sigma (what consumers assume).
MAD_TO_STD = math.sqrt(math.pi / 2)  # ~1.2533


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
        # TODO: make the set of heads modular so experimenting with additional
        # auxiliary heads touches as few places as possible. Right now each head
        # is hardcoded in several spots that must stay in sync: its submodule here
        # in __init__, its output entry in forward(), its loss term and weight in
        # compute_loss(), and the class/forward docstrings. A registry of head
        # objects (each owning its modules, forward, and loss) iterated over in
        # these spots would let a new head be added in one place.
        # The value heads read a 3C summary: mean+max board pooling (2C) plus the
        # scalar projection (C), so the score-diff scalar reaches them directly.
        value_in = 3 * trunk_channels

        # WLD head (inference head): FC -> 3.
        self.wld_fc = nn.Sequential(
            nn.Linear(value_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

        # Score-diff head (aux): two independent FC stacks over the value
        # summary. The mean stack regresses the final score differential; the
        # std stack regresses the absolute residual of the mean and reads a
        # detached copy of the value summary (see forward()), so the std loss
        # updates only this stack -- never the shared trunk or the mean.
        # forward() maps the std stack's output through softplus to a positive
        # std, so the exported second value is the std directly.
        self.sd_mean_fc = nn.Sequential(
            nn.Linear(value_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.sd_std_fc = nn.Sequential(
            nn.Linear(value_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
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

        # Score-diff head: [mean, std] in score points. The std stack reads a
        # detached copy of the value summary, so its loss never flows into the
        # trunk or the mean. softplus + floor keeps the std positive and away
        # from zero.
        sd_mean = self.sd_mean_fc(value_in)  # (B, 1)
        sd_std = F.softplus(self.sd_std_fc(value_in.detach())) + 1e-3  # (B, 1)
        sd = torch.cat([sd_mean, sd_std], dim=1)  # (B, 2): [mean, std]

        return {"wld": wld, "score_diff": sd, "opp_next_placement": opp}


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    lambda_sd: float = 1.0,
    lambda_opp: float = 0.5,
    huber_delta_mean: float = 10.0,
    huber_delta_std: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Compute combined loss for all heads.

    Args:
        outputs: model forward() result. "wld" (B,3 logits), "score_diff"
                 (B,2 = [mean, std]), "opp_next_placement" (B,15,15 logits).
        targets: dict with "wld" (B,3) one-hot, "score_diff" (B,1) the observed
                 final differential, "opp_next_placement" (B,15,15) binary mask.
        huber_delta_mean: Huber transition point (points) for the mean.
        huber_delta_std: Huber transition point (points) for the std.

    Returns:
        Dict with "total", "wld", "score_diff", "opp_next_placement" losses.
    """
    # WLD: cross-entropy against one-hot target.
    wld_target_idx = targets["wld"].argmax(dim=1)
    loss_wld = F.cross_entropy(outputs["wld"], wld_target_idx)

    # Score-diff: two Huber regressions in score points. The mean regresses the
    # observed differential. The std regresses the absolute residual of the mean
    # (scaled by MAD_TO_STD so its optimum is a Gaussian sigma). The std
    # prediction and the residual target are both detached from the trunk and
    # the mean, so the std loss trains only the std stack.
    sd_mean = outputs["score_diff"][:, 0]
    sd_std = outputs["score_diff"][:, 1]
    sd_target = targets["score_diff"].squeeze(1)
    loss_sd_mean = F.huber_loss(sd_mean, sd_target, delta=huber_delta_mean)
    std_target = (sd_mean.detach() - sd_target).abs() * MAD_TO_STD
    loss_sd_std = F.huber_loss(sd_std, std_target, delta=huber_delta_std)
    loss_sd = loss_sd_mean + loss_sd_std

    # Opp-next-placement: binary cross-entropy per cell.
    loss_opp = F.binary_cross_entropy_with_logits(
        outputs["opp_next_placement"], targets["opp_next_placement"]
    )

    total = loss_wld + lambda_sd * loss_sd + lambda_opp * loss_opp

    return {
        "total": total,
        "wld": loss_wld,
        "score_diff": loss_sd,
        "score_diff_mean": loss_sd_mean,
        "score_diff_std": loss_sd_std,
        "opp_next_placement": loss_opp,
    }
