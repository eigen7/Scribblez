"""Evidence-conditioned position evaluation model (docs/sim_residual_feedback.md).

The model is PositionEvalModel with a fusion stage inserted between the trunk
and the heads. Each simmed candidate contributes one evidence token (its
scalar sim summary) paired with its spatial observation planes; tokens
self-attend (cross-candidate contrasts -- "A left this spot open, B blocked
it" -- are pairwise computations), FiLM-modulate their plane features, and the
modulated planes are mean-pooled over candidates into a spatial residual added
to the trunk output. A pooled token summary is likewise projected into the
value-head input.

Both fusion projections are zero-initialized, so at initialization the model
computes exactly the plain PositionEvalModel, and an empty evidence set (all
mask entries False) keeps it that way -- the evidence-free baseline arm of the
kill-test is this same architecture with the evidence inputs zeroed, making
the two arms parameter-identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from scribblez.position_eval.model import PLACEMENT_HEAD_NAMES, PositionEvalModel
from scribblez.sim_evidence.sobs import NUM_EVIDENCE_SCALARS
from scribblez.spatial_trunk import mean_max_pool

# Per-candidate spatial evidence channels: the four rollout-frequency planes
# (opp/self placement, each also conjoined with that player winning) plus the
# candidate's own footprint.
NUM_EVIDENCE_PLANES = 5


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
        # Zero-initialized: fusion starts as (and, with no evidence, stays) a
        # no-op, so the conditioned model's initialization equals the baseline.
        self.spatial_out = nn.Conv2d(d_planes, trunk_channels, 1)
        nn.init.zeros_(self.spatial_out.weight)
        nn.init.zeros_(self.spatial_out.bias)
        self.d_token = d_token

    def forward(
        self, planes: torch.Tensor, scalars: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """planes (B,K,5,15,15), scalars (B,K,S), mask (B,K) bool ->
        spatial residual (B,C,15,15), pooled tokens (B,d_token)."""
        b, k = mask.shape
        denom = mask.sum(dim=1).clamp(min=1).float()  # (B,)

        tokens = self.token_mlp(scalars)  # (B,K,d)
        # TransformerEncoderLayer NaNs on rows whose key-padding mask is all
        # True (no keys). Padding rows are zeroed out downstream anyway, so
        # mark one key valid for empty sets.
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
        # The parent forward with the fusion stage spliced in between the
        # trunk and the heads (the head submodules are inherited unchanged).
        x, s = self.trunk(input_spatial, input_scalar)
        ev_spatial, ev_pooled = self.evidence(ev_planes, ev_scalars, ev_mask)
        x = x + ev_spatial

        value_in = torch.cat([mean_max_pool(x), s], dim=1) + self.value_proj(ev_pooled)

        wld = self.wld_fc(value_in)
        sd_mean = self.sd_mean_fc(value_in)
        sd_std = nn.functional.softplus(self.sd_std_fc(value_in.detach())) + 1e-3
        sd = torch.cat([sd_mean, sd_std], dim=1)

        out = {"wld": wld, "score_diff": sd}
        for name in PLACEMENT_HEAD_NAMES:
            out[name] = self.placement_heads[name](x, value_in)
        return out
