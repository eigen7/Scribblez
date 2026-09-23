"""Max-move-per-lane model: predict the highest-scoring play in each lane.

A lane is one board row read horizontally or one column read vertically, 30 in
all. For each lane the model predicts which tiles the lane's best play places
(per cell, multi-label over 27 tile kinds, the union over plays tied for best),
the best play's score as a 100-bin distribution, and whether the lane has any
legal play. The best move on the board is the max over lanes. See
docs/lexical_nn.md for the task and label layout.

Architecture: a conv trunk encodes the board, then one transformer runs along
every lane, rows and columns sharing weights. The conv handles where things are
(premium squares, geometry, placed tiles). The lane transformer handles which
words fit: attention binds the cells a word threads through, and the FFN is
where the network memorizes the lexicon.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.lexical_tool.compiler import N_LETTERS
from scribblez.lexical_tool.modules import LexiconModule
from scribblez.spatial_trunk import SpatialTrunk

# Label dimensions; must match the C++ lane-target layout (training/lane_targets.h).
BOARD_SIZE = 15
LANE_LEN = 15  # cells along a lane
N_LANES = 2 * BOARD_SIZE  # 15 rows + 15 columns
N_TILE_KINDS = 27  # 26 letters + 1 for a blank, whatever letter it designates
N_SCORE_BINS = 100  # bin k == score k; top bin is the catch-all score >= 99


class LaneModel(nn.Module):
    """One transformer encoder, run on every lane.

    The input is a lane's 15 per-cell trunk vectors, prefixed with rack tokens so
    cells can attend to the rack. Rows and columns share these weights: scoring a
    main word and checking a cross-word are the same operation on two axes.
    """

    def __init__(
        self,
        channels: int,
        n_layers: int,
        n_heads: int,
        ffn_mult: int,
        n_rack_tokens: int,
        n_lex_tokens: int = 0,
    ):
        super().__init__()
        # Prefix tokens prepended to every lane: rack tokens, then any lexicon-
        # module tokens. Both are dropped from the output.
        self.n_prefix = n_rack_tokens + n_lex_tokens
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            # Clamp to >= 1 so a lexicon-replace FFN multiple of 0 (attention-only)
            # still builds a valid layer.
            dim_feedforward=max(1, ffn_mult * channels),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.pos = nn.Parameter(torch.randn(1, self.n_prefix + LANE_LEN, channels) * 0.02)

    def forward(
        self,
        lanes: torch.Tensor,
        rack_tokens: torch.Tensor,
        lex_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(M, LANE_LEN, C) lanes, with prefix tokens (M, n, C) -> (M, LANE_LEN, C)."""
        prefix = [rack_tokens] if lex_tokens is None else [rack_tokens, lex_tokens]
        x = torch.cat([*prefix, lanes], dim=1)  # (M, n_prefix + LANE_LEN, C)
        x = x + self.pos[:, : x.size(1)]
        x = self.encoder(x)
        return x[:, self.n_prefix :]  # drop prefix tokens


def _lane_pool(feat: torch.Tensor, cell_dim: int) -> torch.Tensor:
    """Mean and max over a lane's cells: (B, S, S, C) -> (B, S, 2C)."""
    return torch.cat([feat.mean(dim=cell_dim), feat.amax(dim=cell_dim)], dim=-1)


class MaxMovePerLaneModel(nn.Module):
    """CNN trunk + transpose-shared lane transformer + per-lane heads."""

    def __init__(
        self,
        spatial_planes: int,
        scalar_size: int,
        trunk_channels: int = 128,
        num_blocks: int = 8,
        lane_layers: int = 4,
        lane_heads: int = 4,
        ffn_mult: int = 4,
        n_rack_tokens: int = 4,
        n_score_bins: int = N_SCORE_BINS,
        lexicon_module: LexiconModule | None = None,
        lane_ffn_mult: int | None = None,
    ):
        super().__init__()
        self.n_rack_tokens = n_rack_tokens
        self.n_score_bins = n_score_bins

        self.trunk = SpatialTrunk(spatial_planes, scalar_size, trunk_channels, num_blocks)

        # Rack tokens, prepended to every lane. They read the raw per-tile rack
        # counts (the whole scalar input) because exact counts matter: "can I play
        # two R's?"
        self.rack_tokens = nn.Sequential(
            nn.Linear(scalar_size, trunk_channels),
            nn.GELU(),
            nn.Linear(trunk_channels, n_rack_tokens * trunk_channels),
        )

        # Optional compiled-lexicon tool (lexical_tool.modules), queried per lane.
        self.lexicon_module = lexicon_module
        n_lex_tokens = lexicon_module.n_tokens if lexicon_module is not None else 0

        # `lane_ffn_mult` overrides the lane FFN width so a lexicon tool can take
        # over the memorization the FFN would otherwise do. Attention keeps its
        # full width, since that is what the network needs to use the tool. The
        # policy that picks the value is lexical_tool.modules.resolve_lane_ffn_mult.
        effective_ffn_mult = ffn_mult if lane_ffn_mult is None else lane_ffn_mult

        self.lane = LaneModel(
            trunk_channels, lane_layers, lane_heads, effective_ffn_mult, n_rack_tokens, n_lex_tokens
        )

        # Heads are shared across the two axes.
        self.occ_head = nn.Linear(trunk_channels, N_TILE_KINDS)  # per cell, per kind
        self.score_head = nn.Sequential(  # per lane (pooled), score PMF logits
            nn.Linear(2 * trunk_channels, trunk_channels),
            nn.GELU(),
            nn.Linear(trunk_channels, n_score_bins),
        )
        self.has_move_head = nn.Sequential(  # per lane, "any legal move?" logit
            nn.Linear(2 * trunk_channels, trunk_channels),
            nn.GELU(),
            nn.Linear(trunk_channels, 1),
        )

    def _encode_axis(
        self, lane_feats: torch.Tensor, lane_letters: torch.Tensor, rack_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Run one axis's lanes through the lexicon tool, if any, and the lane transformer."""
        lex_tokens = None
        if self.lexicon_module is not None:
            out = self.lexicon_module(lane_feats, lane_letters)
            if out.cell_residual is not None:
                lane_feats = lane_feats + out.cell_residual
            lex_tokens = out.tokens
        return self.lane(lane_feats, rack_tokens, lex_tokens)

    def _run_lanes(self, h: torch.Tensor, rack_tokens: torch.Tensor, input_spatial: torch.Tensor):
        """(B, C, S, S) trunk output -> (row_feat, col_feat), each (B, row, col, C).

        The lexicon tool reads each lane's board letters from the first 26 planes
        of input_spatial, which are the one-hot letter planes."""
        b, c, s, _ = h.shape
        rt = rack_tokens.repeat_interleave(s, dim=0)  # one rack-token set per lane
        letters = input_spatial[:, :N_LETTERS]  # (B, 26, S, S) one-hot board letters
        # Rows: each row is a sequence over columns -> indexed [b, row, col].
        rows = h.permute(0, 2, 3, 1).reshape(b * s, s, c)
        row_letters = letters.permute(0, 2, 3, 1).reshape(b * s, s, N_LETTERS)
        row_feat = self._encode_axis(rows, row_letters, rt).reshape(b, s, s, c)
        # Cols: each column is a sequence over rows; permute back to [b, row, col].
        cols = h.permute(0, 3, 2, 1).reshape(b * s, s, c)
        col_letters = letters.permute(0, 3, 2, 1).reshape(b * s, s, N_LETTERS)
        col_feat = self._encode_axis(cols, col_letters, rt).reshape(b, s, s, c).permute(0, 2, 1, 3)
        return row_feat, col_feat

    def forward(
        self, input_spatial: torch.Tensor, input_scalar: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        b = input_spatial.size(0)
        h, _ = self.trunk(input_spatial, input_scalar)  # (B, C, 15, 15)
        rack_tokens = self.rack_tokens(input_scalar).view(b, self.n_rack_tokens, -1)

        row_feat, col_feat = self._run_lanes(h, rack_tokens, input_spatial)  # (B,row,col,C) each

        # Occupancy: axis 0 (horizontal lanes) from row_feat [lane=row, cell=col];
        # axis 1 (vertical lanes) from col_feat, transposed to [lane=col, cell=row].
        occ_h = self.occ_head(row_feat)  # (B, 15, 15, 27)
        occ_v = self.occ_head(col_feat).transpose(1, 2)  # (B, 15, 15, 27)
        occupancy = torch.stack([occ_h, occ_v], dim=1).reshape(b, N_LANES, LANE_LEN, N_TILE_KINDS)

        # Per-lane score / has-move from the pooled lane (over its cells).
        pooled = torch.cat([_lane_pool(row_feat, 2), _lane_pool(col_feat, 1)], dim=1)  # (B,30,2C)
        score_logits = self.score_head(pooled)  # (B, 30, bins)
        has_move_logits = self.has_move_head(pooled).squeeze(-1)  # (B, 30)

        # Board-level estimate: max over lanes of the expected score, gated by the
        # has-move probability. The gate matters because a lane with no legal play
        # gets no score supervision, so its score head output is arbitrary.
        bins = torch.arange(self.n_score_bins, device=score_logits.device, dtype=score_logits.dtype)
        expected = (F.softmax(score_logits, dim=-1) * bins).sum(-1)  # (B, 30)
        gated = expected * torch.sigmoid(has_move_logits)
        global_expected_score = gated.amax(dim=1)  # (B,)

        return {
            "lane_occupancy_logits": occupancy,
            "lane_score_logits": score_logits,
            "lane_has_move_logits": has_move_logits,
            "global_expected_score": global_expected_score,
        }


def _masked_mean(per_lane: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of a (B, 30) per-lane loss over the lanes the mask selects."""
    return (per_lane * mask).sum() / mask.sum().clamp_min(1.0)


def score_pdf_loss(logits: torch.Tensor, target_bin: torch.Tensor, mask: torch.Tensor):
    """Per-lane cross-entropy on the score PMF, averaged over legal lanes.
    logits (B,30,bins); target_bin (B,30) in [0,bins-1]; mask (B,30)."""
    n = logits.size(-1)
    ce = F.cross_entropy(logits.reshape(-1, n), target_bin.reshape(-1), reduction="none")
    return _masked_mean(ce.reshape_as(mask), mask)


def score_cdf_loss(logits: torch.Tensor, target_bin: torch.Tensor, mask: torch.Tensor):
    """Per-lane discrete CRPS, sum_k (CDF_hat(k) - 1[k >= target])^2, masked.

    Unlike cross-entropy it is distance-aware: mass is penalized by how far it
    sits from the true bin."""
    n = logits.size(-1)
    cdf_hat = torch.cumsum(F.softmax(logits, dim=-1), dim=-1)  # (B,30,bins)
    ks = torch.arange(n, device=logits.device)
    cdf_tgt = (ks.view(1, 1, n) >= target_bin.unsqueeze(-1)).to(cdf_hat.dtype)
    return _masked_mean(((cdf_hat - cdf_tgt) ** 2).sum(-1), mask)


def occupancy_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Per-cell BCE on the lane occupancy, masked to legal lanes.
    logits/target (B,30,15,27); mask (B,30)."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")  # (B,30,15,27)
    m = mask[:, :, None, None]
    return (bce * m).sum() / (mask.sum().clamp_min(1.0) * LANE_LEN * N_TILE_KINDS)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    lambda_cdf: float = 1.0,
    lambda_occ: float = 1.0,
    lambda_has_move: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Combined per-lane loss.

    targets: "lane_occupancy" (B,30,15,27) binary, "lane_score" (B,30) bin index,
    "lane_mask" (B,30) 1.0 where the lane has a legal move. The score and
    occupancy terms are masked to legal lanes; the has-move term supervises all
    30 lanes (it is what makes a lane "legal").
    """
    mask = targets["lane_mask"]
    target_bin = targets["lane_score"].long()

    loss_pdf = score_pdf_loss(outputs["lane_score_logits"], target_bin, mask)
    loss_cdf = score_cdf_loss(outputs["lane_score_logits"], target_bin, mask)
    loss_occ = occupancy_loss(outputs["lane_occupancy_logits"], targets["lane_occupancy"], mask)
    loss_has_move = F.binary_cross_entropy_with_logits(outputs["lane_has_move_logits"], mask)

    total = (
        loss_pdf + lambda_cdf * loss_cdf + lambda_occ * loss_occ + lambda_has_move * loss_has_move
    )
    return {
        "total": total,
        "score_pdf": loss_pdf,
        "score_cdf": loss_cdf,
        "move": loss_occ,
        "has_move": loss_has_move,
    }
