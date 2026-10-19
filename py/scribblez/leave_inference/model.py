"""The leave inference network (docs/plans/deep_bot_v0.md, "Python work"): a
conv tower over the row's planes and an MLP over its scalars, fused into a
context vector, then one of two heads.

- `tilted` (the design's): 29 floats, theta (27 per-letter presence weights),
  a, and b >= 0 (by a softplus), for the tilt in loss.py, which acts on leave
  values in units of loss.VALUE_SCALE points.
- `autoregressive` (the comparison): a decoder over the leave's tiles in
  sorted order (autoregressive.py). It is flexible where the tilted prior is
  constrained, so the NLL gap between the heads measures what the compact
  output gives up. Trained and evaluated in Python only, never exported.
"""

import torch
import torch.nn.functional as F
from torch import nn

from scribblez.leave_inference.autoregressive import AutoregressiveHead
from scribblez.leave_inference.heads import HEAD_AUTOREGRESSIVE, HEAD_TILTED, HEADS
from scribblez.leave_inference.loss import TILE_KINDS
from scribblez.spatial_trunk import SpatialTrunk, mean_max_pool

B_INIT_LOGIT = -8.0


class LeaveInferenceModel(nn.Module):
    def __init__(
        self,
        spatial_planes: int,
        scalar_size: int,
        trunk_channels: int,
        num_blocks: int,
        head: str = HEAD_TILTED,
    ):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"unknown head {head!r}; expected one of {HEADS}")
        self.head_kind = head
        self.trunk = SpatialTrunk(spatial_planes, scalar_size, trunk_channels, num_blocks)
        # Pooled board features (mean and max) plus the scalar projection.
        self.context = nn.Sequential(
            nn.Linear(3 * trunk_channels, trunk_channels), nn.ReLU(inplace=True)
        )
        if head == HEAD_TILTED:
            self.head = nn.Linear(trunk_channels, TILE_KINDS + 2)
            # Start at the uniform prior: zero tilt, and b = softplus(-8) ~ 3e-4.
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            nn.init.constant_(self.head.bias[TILE_KINDS + 1 :], B_INIT_LOGIT)
        else:
            self.head = AutoregressiveHead(trunk_channels)

    def forward(
        self, input_spatial: torch.Tensor, input_scalar: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """The tilted head's theta, a and b, or the autoregressive head's
        context vector, which its decoder reads (autoregressive.py)."""
        features, scalar_proj = self.trunk(input_spatial, input_scalar)
        # Pooled in FP32. With the pool in bf16, the compiled (torch.compile,
        # bf16 autocast) model's trunk gradients came out NaN from the first
        # step, while eager was fine; the pool alone, compiled, does not
        # reproduce it, so the cause is somewhere in the fused graph.
        pooled = mean_max_pool(features.float())
        context = self.context(torch.cat([pooled, scalar_proj.float()], dim=1))
        if self.head_kind == HEAD_AUTOREGRESSIVE:
            return {"context": context}
        out = self.head(context)
        return {
            "theta": out[:, :TILE_KINDS],
            "a": out[:, TILE_KINDS],
            "b": F.softplus(out[:, TILE_KINDS + 1]),
        }
