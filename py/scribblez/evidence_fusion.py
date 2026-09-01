"""Evidence-set fusion stage (docs/sim_residual_feedback.md, roadmap item 5).

Conditions a model's board token map on the sims run so far at a decision
point. Each simmed candidate contributes one evidence token built from three
parts: the candidate's move encoding (the move set evaluation model's move
encoder, reused), its raw sim observations, and the model's own evidence-free
predictions for that candidate -- observed and predicted placement planes
concatenated channel-wise, so the encoder sees observation and prediction for
the same square side by side and forms the residual itself. Feeding the
predictions in as inputs is load-bearing: an encoder reading observations
alone can express `posterior = prior + g(obs)` but never
`posterior = prior + k*(obs - prior)`, because nothing downstream of an
additive merge can separate the summands again.

Tokens self-attend (cross-candidate contrasts are pairwise computations), then
the board's 225 tokens cross-attend into them, producing the
evidence-conditioned map the per-move scoring reads in place of the plain one.
The attention value each token delivers carries, besides its vector, the
token's own spatial feature at the querying square -- which is what lets a hot
square discovered by one candidate's rollouts land on that square rather than
being pooled away.

The stage is late fusion by design: it reads the trunk's outputs and never
modulates the trunk's own layers, so at one decision point the trunk output,
the move encodings, and the per-candidate tokens are computed once and cached,
and only the set stage (self-attention + fusion) plus re-scoring run per loop
iteration. Per-token encodings are independent of the rest of the set, so a
token never changes once created.

All three output projections are zero-initialized, and an empty evidence set
hard-gates the stage to a no-op -- so a fresh model computes exactly the plain
one-pass model, and an evidence-free forward stays bit-identical to it at any
weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.footprint_spatial import SLOTS_PER_CELL
from scribblez.spatial_trunk import mean_max_pool

# Per-token spatial channels -- THE canonical evidence-plane layout; the C++
# staging port (agent/evidence_staging.h) mirrors it and must change in
# lockstep. Placement is footprint-categorical: each head is SLOTS_PER_CELL
# board-shaped channels, anchored class (cell, slot) at channel
# (head * SLOTS_PER_CELL + slot). First the four observed rollout-frequency
# histograms (SimObservation's counts / rollouts) and the model's four
# predicted footprint distributions (both in the placement-head order the FFI
# serves -- targets.PLANE_NAMES), then the candidate's own footprint as a
# one-hot in a final SLOTS_PER_CELL block. The two catch-all classes
# (pass / not-win) are dropped and nothing is renormalized: the conv sees the
# raw per-class frequencies/probabilities.
_PLANE_HEADS = ("opp_next", "self_next", "opp_win", "self_win")
EVIDENCE_PLANE_NAMES = tuple(
    f"{kind}_{head}_s{slot}"
    for kind in ("obs", "pred")
    for head in _PLANE_HEADS
    for slot in range(SLOTS_PER_CELL)
) + tuple(f"footprint_s{slot}" for slot in range(SLOTS_PER_CELL))
# The three channel blocks, in order: observed, predicted, candidate footprint.
NUM_OBSERVED_PLANES = len(_PLANE_HEADS) * SLOTS_PER_CELL
NUM_PREDICTED_PLANES = len(_PLANE_HEADS) * SLOTS_PER_CELL
NUM_EVIDENCE_PLANES = len(EVIDENCE_PLANE_NAMES)  # 117
assert NUM_EVIDENCE_PLANES == NUM_OBSERVED_PLANES + NUM_PREDICTED_PLANES + SLOTS_PER_CELL

# Per-token scalars: the sim's value estimate with its rollout count (the
# confidence signal -- maps built from 40 rollouts and from 2000 warrant
# different updates), beside the model's own evidence-free value prediction
# for the same candidate, so the value residual is formable like the spatial
# one. Delta moments are in score points scaled to ~unit range.
EVIDENCE_SCALAR_NAMES = (
    "win_freq",
    "draw_freq",
    "loss_freq",
    "delta_mean_100",
    "delta_std_100",
    "log1p_rollouts",
    "pred_p_win",
    "pred_p_draw",
    "pred_p_loss",
    "pred_sd_mean_100",
    "pred_sd_std_100",
)
NUM_EVIDENCE_SCALARS = len(EVIDENCE_SCALAR_NAMES)


@dataclass
class EvidenceInputs:
    """One padded evidence set per position in the batch.

    The move half uses the engine's move-input layout (move_set_encoder.h),
    shaped (P, E, ...) rather than the candidate set's flattened (M, ...):
    `letters`/`blanks`/`squares`/`tile_mask` (P, E, T) and `scalars`
    (P, E, kMoveScalars). The observation half is `obs_planes`
    (P, E, NUM_EVIDENCE_PLANES, 15, 15) and `obs_scalars`
    (P, E, NUM_EVIDENCE_SCALARS), both in the orders named above. `mask`
    (P, E) bool marks real tokens; padded rows may hold anything.
    """

    letters: torch.Tensor
    blanks: torch.Tensor
    squares: torch.Tensor
    tile_mask: torch.Tensor
    scalars: torch.Tensor
    obs_planes: torch.Tensor
    obs_scalars: torch.Tensor
    mask: torch.Tensor


def _zero_init(linear: nn.Linear) -> nn.Linear:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


class EvidenceFusion(nn.Module):
    """Encodes an evidence set and fuses it into the board token map.

    Two-phase API mirroring the deployment loop's caching split:
    `encode_tokens` is per-candidate work whose results are cacheable across
    loop iterations; `forward` is the per-iteration set stage, producing the
    conditioned board map and position summary the unchanged scoring machinery
    then reads.
    """

    def __init__(self, channels: int, num_heads: int = 4, d_spatial: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        # A 3x3 layer lets a hot square influence its neighborhood (a blocker
        # need not occupy the hot square itself, only disturb its lane).
        self.plane_conv = nn.Sequential(
            nn.Conv2d(NUM_EVIDENCE_PLANES, d_spatial, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_spatial, d_spatial, 3, padding=1),
        )
        self.scalar_mlp = nn.Sequential(
            nn.Linear(NUM_EVIDENCE_SCALARS, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )
        self.token_fuse = nn.Linear(2 * channels + 2 * d_spatial, channels)
        # Normalization at the stage's seams. Nothing upstream of these points
        # is normalized (the token encoder is linear, the trunk map arrives as
        # is), so under a peak-LR Adam run their scales drift up without bound
        # -- the first evidence run's tokens grew 40x while its loss stood
        # still, and its board rewrite outgrew the trunk map it corrects; the
        # frozen scoring attention reading that inflated map is where the
        # gradients blew up. Each norm sits before a zero-init projection, so
        # the empty-set exactness contract is untouched.
        self.token_norm = nn.LayerNorm(channels)
        self.attended_norm = nn.LayerNorm(channels)
        self.local_norm = nn.LayerNorm(d_spatial)
        self.pooled_norm = nn.LayerNorm(channels)
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=2 * channels,
            dropout=0.0,
            batch_first=True,
        )
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        # QK-norm: the queries come straight off the trunk's board map and the
        # keys off the token encoder, neither normalized, so without this the
        # attention logits scale with the projection weights -- which Adam
        # grows at a steady per-step rate whatever the gradient says, until a
        # near-saturated softmax's transition gradients blow the run up (seen
        # at the peak LR of the first evidence run). Normalizing per head
        # bounds the logits; the learned affine keeps the temperature free.
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.out_proj = _zero_init(nn.Linear(channels, channels))
        self.spatial_out = _zero_init(nn.Linear(d_spatial, channels))
        self.summary_out = _zero_init(nn.Linear(channels, 3 * channels))

    def encode_tokens(
        self, move_enc: torch.Tensor, obs_planes: torch.Tensor, obs_scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-candidate token vectors and spatial features -- independent of
        the rest of the set, so cacheable per candidate.

        move_enc (P, E, C), obs_planes (P, E, planes, 15, 15), obs_scalars
        (P, E, S) -> tokens (P, E, C), spatial features (P, E, d_spatial, 225).
        """
        p, e = move_enc.shape[:2]
        feats = self.plane_conv(obs_planes.flatten(0, 1))  # (P*E, d, 15, 15)
        pooled = mean_max_pool(feats)  # (P*E, 2d)
        parts = [move_enc.flatten(0, 1), self.scalar_mlp(obs_scalars.flatten(0, 1)), pooled]
        tokens = self.token_norm(self.token_fuse(torch.cat(parts, dim=1)))
        return tokens.view(p, e, -1), feats.flatten(2).view(p, e, feats.shape[1], -1)

    def forward(
        self,
        board: torch.Tensor,
        g: torch.Tensor,
        tokens: torch.Tensor,
        spatial_feats: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The per-iteration set stage: evidence self-attention, then the
        board tokens cross-attend into the set.

        board (P, 225, C), g (P, 3C), tokens/spatial_feats from encode_tokens,
        mask (P, E) bool -> (conditioned board (P, 225, C), conditioned g
        (P, 3C)). Positions whose mask row is all-False pass through exactly
        unchanged.
        """
        denom = mask.sum(dim=1).clamp(min=1).to(board.dtype)  # (P,)
        # TransformerEncoderLayer NaNs on rows whose key-padding mask is all
        # True (no keys); empty rows are hard-gated to zero below anyway, so
        # mark one key valid for them.
        attn_pad = ~mask
        attn_pad = attn_pad & ~attn_pad.all(dim=1, keepdim=True)
        t = self.self_attn(tokens, src_key_padding_mask=attn_pad)
        t = t * mask.unsqueeze(-1)

        attended, weights = self._cross_attention(board, t, mask)
        # The value each square receives is the attended token mix plus the
        # tokens' own spatial features at that square, weighted the same way --
        # the per-square half of the evidence, which pooling into a token
        # vector would erase.
        local = torch.einsum("pne,pedn->pnd", weights, spatial_feats)
        delta = self.out_proj(self.attended_norm(attended)) + self.spatial_out(
            self.local_norm(local)
        )

        pooled = self.pooled_norm(t.sum(dim=1) / denom.unsqueeze(-1))  # (P, C)
        # Empty sets take the plain path bit-exactly, whatever the weights.
        has_evidence = mask.any(dim=1).to(board.dtype)
        board = board + delta * has_evidence.view(-1, 1, 1)
        g = g + self.summary_out(pooled) * has_evidence.view(-1, 1)
        return board, g

    def _cross_attention(
        self, board: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-head attention with the board tokens as queries and the
        evidence tokens as keys/values -> attended (P, 225, C) and the
        head-averaged weights (P, 225, E) that also mix the spatial features.
        Padded tokens are masked out of the softmax. A position with no real
        tokens is left attending to token 0: its rows are discarded by the
        empty-set gate regardless, and an all-masked softmax row would be NaN
        -- which the gate's multiply-by-zero does not scrub from the
        *gradients*, so empty rows must stay finite rather than be patched
        after the fact."""
        p, n, c = board.shape
        h, d = self.num_heads, self.head_dim
        attend_to = mask.clone()
        attend_to[:, 0] |= ~mask.any(dim=1)
        q = self.q_norm(self.q_proj(board).view(p, n, h, d)).transpose(1, 2)  # (P, H, 225, d)
        k = self.k_norm(self.k_proj(tokens).view(p, -1, h, d)).transpose(1, 2)  # (P, H, E, d)
        v = self.v_proj(tokens).view(p, -1, h, d).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / math.sqrt(d)  # (P, H, 225, E)
        logits = logits.masked_fill(~attend_to[:, None, None, :], -torch.inf)
        weights = F.softmax(logits, dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape(p, n, c)
        return attended, weights.mean(dim=1)
