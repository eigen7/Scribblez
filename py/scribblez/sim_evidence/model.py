"""Evidence-conditioned position evaluation model for the kill test
(py/scripts/kill_test.py, docs/plans/sim_residual_feedback.md).

This is PositionEvalModel with a fusion stage between the trunk and the heads.
Each simmed candidate contributes a token (from its scalar sim summary) and
its spatial observation planes (sobs.evidence_features). Tokens self-attend,
because cross-candidate contrasts ("A left this spot open, B blocked it") are
pairwise, then FiLM-modulate their plane features. The modulated planes are
mean-pooled over candidates into a residual added to the trunk output, and a
pooled token summary is added to the value-head input.

Both output projections are zero-initialized, so a fresh model computes
exactly the plain PositionEvalModel, and an empty evidence set keeps it so.
The kill test's evidence-free baseline arm is therefore this same
architecture with zeroed evidence inputs, with identical parameter counts.

The move proposal model's fusion stage is a separate design
(scribblez.evidence_fusion).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from scribblez.footprint_spatial import SLOTS_PER_CELL
from scribblez.position_eval.model import PositionEvalModel
from scribblez.sim_evidence.sobs import NUM_EVIDENCE_SCALARS
from scribblez.spatial_trunk import mean_max_pool

# Per-candidate spatial channels in sobs.evidence_features' layout: the four
# observed footprint histograms (opp/self next placement, and each conjoined
# with that player winning), then the candidate's own footprint one-hot.
NUM_EVIDENCE_PLANES = 5 * SLOTS_PER_CELL


class EvidenceEncoder(nn.Module):
    """Encodes a padded evidence set into a spatial residual for the trunk
    output and a pooled vector for the value heads."""

    def __init__(self, trunk_channels: int, d_token: int = 96, d_planes: int = 16):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(NUM_EVIDENCE_SCALARS, d_token),
            nn.ReLU(inplace=True),
            nn.Linear(d_token, d_token),
        )
        self.token_attn = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=4, dim_feedforward=2 * d_token, dropout=0.0, batch_first=True
        )
        # A 3x3 layer lets a hot square influence its neighborhood (a blocker
        # need not occupy the hot square itself, only disturb its lane).
        self.plane_conv = nn.Sequential(
            nn.Conv2d(NUM_EVIDENCE_PLANES, d_planes, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_planes, d_planes, 3, padding=1),
        )
        self.film = nn.Linear(d_token, 2 * d_planes)
        # Zero-initialized (see the module docstring).
        self.spatial_out = nn.Conv2d(d_planes, trunk_channels, 1)
        nn.init.zeros_(self.spatial_out.weight)
        nn.init.zeros_(self.spatial_out.bias)
        self.d_token = d_token

    def forward(
        self, planes: torch.Tensor, scalars: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """planes (B,K,NUM_EVIDENCE_PLANES,15,15), scalars (B,K,S), mask
        (B,K) bool -> spatial residual (B,C,15,15), pooled tokens (B,d_token)."""
        b, k = mask.shape
        denom = mask.sum(dim=1).clamp(min=1).float()  # (B,)

        tokens = self.token_mlp(scalars)  # (B,K,d)
        # TransformerEncoderLayer returns NaN for a row with every key masked.
        # Empty sets are zeroed downstream anyway, so unmask one key for them.
        attn_pad = ~mask
        attn_pad = attn_pad & ~(attn_pad.all(dim=1, keepdim=True))
        tokens = self.token_attn(tokens, src_key_padding_mask=attn_pad)
        tokens = tokens * mask.unsqueeze(-1)

        feats = self.plane_conv(planes.flatten(0, 1).float())  # (B*K, dp, 15, 15)
        feats = feats.view(b, k, -1, 15, 15)
        gamma, beta = self.film(tokens).chunk(2, dim=-1)  # (B,K,dp) each
        feats = feats * (1.0 + gamma[..., None, None]) + beta[..., None, None]
        feats = feats * mask[..., None, None, None]
        evidence_map = feats.sum(dim=1) / denom[:, None, None, None]  # masked mean over K

        pooled = tokens.sum(dim=1) / denom[:, None]  # (B,d_token)
        return self.spatial_out(evidence_map), pooled


class EvidencePositionEvalModel(PositionEvalModel):
    """PositionEvalModel whose trunk output and value summary are conditioned
    on a sim-evidence set before reaching the (inherited) heads."""

    def __init__(
        self,
        spatial_planes: int,
        scalar_size: int,
        trunk_channels: int = 192,
        num_blocks: int = 10,
        board_size: int = 15,
        d_token: int = 96,
    ):
        super().__init__(spatial_planes, scalar_size, trunk_channels, num_blocks, board_size)
        self.evidence = EvidenceEncoder(trunk_channels, d_token=d_token)
        self.value_proj = nn.Linear(d_token, 3 * trunk_channels)
        nn.init.zeros_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)

    def forward(  # type: ignore[override]
        self,
        input_spatial: torch.Tensor,
        input_scalar: torch.Tensor,
        ev_planes: torch.Tensor,
        ev_scalars: torch.Tensor,
        ev_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # The parent forward with the fusion stage between trunk and heads.
        x, s = self.trunk(input_spatial, input_scalar)
        ev_spatial, ev_pooled = self.evidence(ev_planes, ev_scalars, ev_mask)
        x = x + ev_spatial

        value_in = torch.cat([mean_max_pool(x), s], dim=1) + self.value_proj(ev_pooled)
        return self._run_heads(x, value_in)
