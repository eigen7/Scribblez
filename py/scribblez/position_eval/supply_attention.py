"""Tile-supply cross-attention for the position-evaluation model.

The placement heads must gate a square's cross-check letters by whether those
letters are actually *available* -- present in the bag, the opponent's known
leave, or the mover's own rack. The convolutional trunk struggles to learn this:
cross-checks are a per-square, per-letter spatial signal while availability is a
global per-letter scalar, and the two meet only through the trunk's blunt
global-context injection (a per-channel bias/FiLM). That composition is
sample-expensive -- the model learns availability-gating for common tiles and
falls back to a fixed frequency prior for rare ones (see the I13 analysis: a
live model holds a ~0.17 Y-hook belief with zero Y's unseen).

This module makes the letter-indexed cross a first-class attention operation.
Each of the 27 tiles (A..Z, blank) becomes a *supply token* carrying that
letter's availability to each seat; every board square attends to the supply
tokens, its query built from the square's trunk features *and* its raw
cross-check legality vector (so "which letters are legal here" is explicit).
A square hooking on S/Y can then read S-supply and Y-supply directly and gate
its placement belief on them -- graded by the actual counts, and distinguishing
"available to me" (my rack) from "available to the opponent" (unseen pool /
opp leave), which a single gated input plane cannot.

The output projection is zero-initialised, so at init the block is the identity
and the network starts numerically identical to the no-attention baseline -- a
strict superset, matching the use_film convention in SpatialTrunk.

docs/model_architectures.md diagrams this block; any change to it belongs in the
same commit as the corresponding change there.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Spatial-plane layout (engine/include/encoding/input_encoder.h): the 52
# cross-check planes (26 horizontal-play + 26 vertical-play legality masks)
# follow the 31 board planes and the two placement planes.
CROSS_CHECK_PLANE0 = 33
CROSS_CHECK_PLANES = 52
EXPECTED_SPATIAL_PLANES = 85

# Scalar-block layout (same header): rack counts, the unseen-pool thermometer,
# score diff, move meta, then the open-leaves opponent-leave counts. Supply
# attention needs the opp-leave block, so it requires the open-leaves arm.
N_TILES = 27  # A..Z + blank
RACK0 = 0
UNSEEN_THERMO0 = 27
UNSEEN_THERMO_LEN = 100
OPP_LEAVE0 = 136
EXPECTED_SCALAR_SIZE = 163

# English Scrabble tile counts (engine/src/game/tile.cpp), used to normalise the
# per-letter availability counts to [0, 1] and to lay out the unseen thermometer.
TILE_COUNTS = (9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, 2)

# One supply feature per seat's view of a letter's availability.
N_SUPPLY_FEATURES = 3  # mover rack, unseen pool (bag + opp), opp known leave


def _thermometer_to_count_matrix() -> torch.Tensor:
    """A fixed (100, 27) 0/1 matrix summing each letter's thermometer region back
    to a raw count: `unseen_thermo @ M` recovers the per-letter unseen counts.
    Letter i owns a contiguous width-TILE_COUNTS[i] region (input_encoder.h)."""
    m = torch.zeros(UNSEEN_THERMO_LEN, N_TILES)
    offset = 0
    for i, width in enumerate(TILE_COUNTS):
        m[offset : offset + width, i] = 1.0
        offset += width
    assert offset == UNSEEN_THERMO_LEN
    return m


class TileSupplyAttention(nn.Module):
    """Cross-attention from board squares to per-letter tile-supply tokens.

    forward(x, input_spatial, input_scalar) -> refined feature map, same shape as
    x. Adds a zero-initialised residual, so it is the identity at init.
    """

    def __init__(
        self,
        trunk_channels: int,
        scalar_size: int,
        token_dim: int = 64,
        attn_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        if scalar_size != EXPECTED_SCALAR_SIZE:
            raise ValueError(
                f"supply attention requires the open-leaves arm "
                f"(scalar_size {EXPECTED_SCALAR_SIZE}), got {scalar_size}"
            )
        if attn_dim % num_heads != 0:
            raise ValueError(f"attn_dim {attn_dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads

        # Fixed decode of the unseen thermometer to per-letter counts, and the
        # per-letter normalisers -- registered as buffers so they move with the
        # module and export as constants.
        self.register_buffer("thermo_to_count", _thermometer_to_count_matrix())
        self.register_buffer("tile_counts", torch.tensor(TILE_COUNTS, dtype=torch.float32))

        # Supply tokens: a learned identity embedding per tile plus a learned
        # "null" token (index N_TILES) that squares with no relevant supply
        # concern can attend to. Availability counts are projected into the
        # token content and added to the identity embedding.
        self.token_embed = nn.Parameter(torch.randn(N_TILES + 1, token_dim) * 0.02)
        self.supply_proj = nn.Linear(N_SUPPLY_FEATURES, token_dim)
        self.token_norm = nn.LayerNorm(token_dim)

        # Query from each square's trunk features plus its raw cross-check
        # legality vector (so the letters legal at the square are explicit in the
        # query, matching the tokens' letter identities). Keys/values from tokens.
        self.query_norm = nn.LayerNorm(trunk_channels)
        self.q_proj = nn.Linear(trunk_channels + CROSS_CHECK_PLANES, attn_dim, bias=False)
        self.k_proj = nn.Linear(token_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(token_dim, attn_dim, bias=False)

        # Zero-initialised output projection -> identity at init (strict superset
        # of the no-attention model, like the FiLM gain in SpatialTrunk).
        self.out_proj = nn.Linear(attn_dim, trunk_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _supply_tokens(self, input_scalar: torch.Tensor) -> torch.Tensor:
        """Build (B, N_TILES + 1, token_dim) supply tokens from the scalar input."""
        rack = input_scalar[:, RACK0 : RACK0 + N_TILES]
        unseen = input_scalar[:, UNSEEN_THERMO0 : UNSEEN_THERMO0 + UNSEEN_THERMO_LEN]
        unseen = unseen @ self.thermo_to_count
        opp_leave = input_scalar[:, OPP_LEAVE0 : OPP_LEAVE0 + N_TILES]
        # (B, N_TILES, N_SUPPLY_FEATURES), each availability count normalised to [0, 1].
        feats = torch.stack([rack, unseen, opp_leave], dim=-1) / self.tile_counts[None, :, None]
        content = self.supply_proj(feats)  # (B, N_TILES, token_dim)

        b = input_scalar.shape[0]
        tokens = self.token_embed.unsqueeze(0).expand(b, -1, -1).clone()
        tokens[:, :N_TILES] = tokens[:, :N_TILES] + content
        return self.token_norm(tokens)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(B, L, attn_dim) -> (B, num_heads, L, head_dim)."""
        b, length, _ = t.shape
        return t.reshape(b, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(
        self, x: torch.Tensor, input_spatial: torch.Tensor, input_scalar: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = h * w

        x_flat = x.permute(0, 2, 3, 1).reshape(b, seq, c)
        cross = input_spatial[:, CROSS_CHECK_PLANE0 : CROSS_CHECK_PLANE0 + CROSS_CHECK_PLANES]
        cross_flat = cross.permute(0, 2, 3, 1).reshape(b, seq, CROSS_CHECK_PLANES)
        query_in = torch.cat([self.query_norm(x_flat), cross_flat], dim=-1)
        q = self._split_heads(self.q_proj(query_in))  # (B, nh, seq, hd)

        tokens = self._supply_tokens(input_scalar)
        k = self._split_heads(self.k_proj(tokens))  # (B, nh, N_TILES + 1, hd)
        v = self._split_heads(self.v_proj(tokens))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)  # (B, nh, seq, hd)

        context = context.permute(0, 2, 1, 3).reshape(b, seq, self.num_heads * self.head_dim)
        out = self.out_proj(context)  # (B, seq, C), zero at init
        out = out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x + out
