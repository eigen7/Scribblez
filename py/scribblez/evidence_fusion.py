"""Evidence-set fusion: conditioning the move set model on the sims run so far.

At a decision point the agent sims some candidates. This stage folds those
results into the model's board token map so it can re-score every candidate
(docs/plans/sim_residual_feedback.md).

Each simmed candidate becomes one evidence token, built from its move encoding
(the move set model's move encoder, reused), its raw sim observations, and the
model's own evidence-free predictions for it. Observed and predicted placement
planes are concatenated channel-wise, so the encoder sees both for the same
square and can form the residual itself. The predictions must be inputs: an
encoder that reads only observations can express `posterior = prior + g(obs)`
but not `posterior = prior + k*(obs - prior)`, because nothing downstream of an
additive merge can separate the summands again.

The tokens self-attend, since comparing candidates is a pairwise computation.
Then the 225 board tokens cross-attend into them, and the per-move scoring reads
the conditioned map in place of the plain one. Each token delivers, besides its
vector, its own spatial feature at the querying square, so a hot square found by
one candidate's rollouts lands on that square instead of being pooled away.

The stage is late fusion: it reads the trunk's outputs and never modulates the
trunk. At one decision point the trunk output, move encodings and tokens are
therefore computed once and cached, and only the set stage plus re-scoring run
per loop iteration. A token depends only on its own candidate, so it never
changes once created.

The three output projections are zero-initialized and an empty evidence set
gates the stage off, so a fresh model computes exactly the plain model, and an
evidence-free forward is bit-identical to it at any weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.footprint_spatial import SLOTS_PER_CELL
from scribblez.spatial_trunk import mean_max_pool

# Per-token spatial channels. engine/include/agent/evidence_staging.h mirrors
# this layout and must change in lockstep. Each placement head contributes
# SLOTS_PER_CELL board-shaped channels, anchored footprint class (cell, slot) at
# channel head * SLOTS_PER_CELL + slot. The blocks, in order:
#   - observed: the four heads' rollout frequencies (SimObservation counts / rollouts)
#   - predicted: the model's four evidence-free footprint distributions
#   - the candidate's own footprint, one-hot
# Heads follow the FFI's placement-head order (move_set_eval.targets.PLANE_NAMES).
# The catch-all classes (pass, not-win) are dropped without renormalizing.
_PLANE_HEADS = ("opp_next", "self_next", "opp_win", "self_win")
EVIDENCE_PLANE_NAMES = tuple(
    f"{kind}_{head}_s{slot}"
    for kind in ("obs", "pred")
    for head in _PLANE_HEADS
    for slot in range(SLOTS_PER_CELL)
) + tuple(f"footprint_s{slot}" for slot in range(SLOTS_PER_CELL))
NUM_OBSERVED_PLANES = len(_PLANE_HEADS) * SLOTS_PER_CELL
NUM_PREDICTED_PLANES = len(_PLANE_HEADS) * SLOTS_PER_CELL
NUM_EVIDENCE_PLANES = len(EVIDENCE_PLANE_NAMES)  # 117
assert NUM_EVIDENCE_PLANES == NUM_OBSERVED_PLANES + NUM_PREDICTED_PLANES + SLOTS_PER_CELL

# Per-token scalars: the sim's value estimate and rollout count (evidence from
# 40 rollouts and from 2000 warrants different updates), beside the model's own
# evidence-free value prediction, so the value residual can be formed like the
# spatial one. The "_100" moments are score points / 100.
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
    """One padded evidence set per position: P positions, E token slots.

    Move encoding (training/move_set_encoder.h layout, T = max placed tiles):
        letters, blanks, squares, tile_mask   (P, E, T)
        scalars                               (P, E, kMoveScalars)
    Observations, in the orders defined above:
        obs_planes    (P, E, NUM_EVIDENCE_PLANES, 15, 15)
        obs_scalars   (P, E, NUM_EVIDENCE_SCALARS)
    mask (P, E) bool marks real tokens; padded slots may hold anything.
    """

    letters: torch.Tensor
    blanks: torch.Tensor
    squares: torch.Tensor
    tile_mask: torch.Tensor
    scalars: torch.Tensor
    obs_planes: torch.Tensor
    obs_scalars: torch.Tensor
    mask: torch.Tensor


def best_so_far(obs_scalars: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(P,) best observed win value (win freq + draw freq / 2) over each
    position's real tokens, 0 for an empty set.

    This is the baseline the proves-best gain label subtracts
    (evidence.dataset.gain_targets), fed to the gain head as an input. The
    exported step graph computes it in-graph, so it uses float ops only: values
    are >= 0, so multiplying by the mask is an exact select."""
    value = obs_scalars[..., 0] + 0.5 * obs_scalars[..., 1]  # (P, E)
    return (value * mask.to(value.dtype)).amax(dim=1)


def _zero_init(linear: nn.Linear) -> nn.Linear:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


class EvidenceFusion(nn.Module):
    """Encodes an evidence set and fuses it into the board token map.

    The API splits along the agent loop's caching: `encode_tokens` is
    per-candidate and cacheable across iterations; `forward` is the
    per-iteration set stage.
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
        # Norms at the stage's seams. Nothing upstream is normalized (the token
        # encoder is linear, the trunk map arrives as is), and at peak LR Adam
        # lets these scales grow without bound: in training, tokens grew 40x
        # with flat loss, and the board update outgrew the trunk map until the
        # scoring attention's gradients blew up. Each norm feeds a zero-init
        # projection, so the empty-set exactness is unaffected.
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
        # QK-norm. Neither the queries (trunk board map) nor the keys (token
        # encoder) are normalized, so without it the logits scale with the
        # projection weights, which Adam grows at a steady rate until the
        # softmax saturates and its gradients blow up (observed at peak LR).
        # Per-head norms bound the logits; the learned affine keeps the
        # temperature trainable.
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.out_proj = _zero_init(nn.Linear(channels, channels))
        self.spatial_out = _zero_init(nn.Linear(d_spatial, channels))
        self.summary_out = _zero_init(nn.Linear(channels, 3 * channels))

    def encode_tokens(
        self, move_enc: torch.Tensor, obs_planes: torch.Tensor, obs_scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-candidate token vectors and spatial features.

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
        """Evidence self-attention, then board-to-evidence cross-attention.

        board (P, 225, C), g (P, 3C), tokens/spatial_feats from encode_tokens,
        mask (P, E) bool -> (conditioned board (P, 225, C), conditioned g
        (P, 3C)). Positions with no evidence pass through unchanged.
        """
        denom = mask.sum(dim=1).clamp(min=1).to(board.dtype)  # (P,)
        # TransformerEncoderLayer returns NaN for a row with every key masked.
        # Empty rows are gated off below anyway, so unmask one key for them.
        attn_pad = ~mask
        attn_pad = attn_pad & ~attn_pad.all(dim=1, keepdim=True)
        t = self.self_attn(tokens, src_key_padding_mask=attn_pad)
        t = t * mask.unsqueeze(-1)

        attended, weights = self._cross_attention(board, t, mask)
        # Each square also receives the tokens' spatial features at that square,
        # mixed by the same attention weights: the per-square evidence that
        # pooling into a token vector would erase.
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
        """Board tokens attend over the evidence tokens -> attended (P, 225, C)
        and the head-averaged weights (P, 225, E), which also mix the spatial
        features.

        A position with no real tokens attends to token 0. The empty-set gate
        discards its output anyway, but an all-masked softmax row would be NaN,
        and multiplying by the zero gate does not remove NaN from the
        gradients."""
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
