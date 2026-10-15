"""The leave inference network (docs/plans/deep_bot_v0.md, "Python work"): a
conv tower over the row's planes and an MLP over its scalars, fused into a
context vector, then the tilted head's 29 floats.

The head emits theta (27 per-letter presence weights), a, and b >= 0 (by a
softplus) for the tilt in loss.py, which acts on leave values in units of
loss.VALUE_SCALE points.
"""

import torch
import torch.nn.functional as F
from torch import nn

from scribblez.spatial_trunk import SpatialTrunk, mean_max_pool

TILE_KINDS = 27
B_INIT_LOGIT = -8.0


class LeaveInferenceModel(nn.Module):
    def __init__(self, spatial_planes: int, scalar_size: int, trunk_channels: int, num_blocks: int):
        super().__init__()
        self.trunk = SpatialTrunk(spatial_planes, scalar_size, trunk_channels, num_blocks)
        # Pooled board features (mean and max) plus the scalar projection.
        self.context = nn.Sequential(
            nn.Linear(3 * trunk_channels, trunk_channels), nn.ReLU(inplace=True)
        )
        self.head = nn.Linear(trunk_channels, TILE_KINDS + 2)
        # Start at the uniform prior: zero tilt, and b = softplus(-8) ~ 3e-4.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.constant_(self.head.bias[TILE_KINDS + 1 :], B_INIT_LOGIT)

    def forward(
        self, input_spatial: torch.Tensor, input_scalar: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        features, scalar_proj = self.trunk(input_spatial, input_scalar)
        # Pooled in FP32. With the pool in bf16, the compiled (torch.compile,
        # bf16 autocast) model's trunk gradients came out NaN from the first
        # step, while eager was fine; the pool alone, compiled, does not
        # reproduce it, so the cause is somewhere in the fused graph.
        pooled = mean_max_pool(features.float())
        context = self.context(torch.cat([pooled, scalar_proj.float()], dim=1))
        out = self.head(context)
        return {
            "theta": out[:, :TILE_KINDS],
            "a": out[:, TILE_KINDS],
            "b": F.softplus(out[:, TILE_KINDS + 1]),
        }
