"""Tile-supply register tokens for the position-evaluation model's transformer trunk.

The placement heads must gate a square's cross-check letters by whether those
letters are actually *available* -- present in the bag, the opponent's known
leave, or the mover's own rack. A conv trunk struggles to learn this:
cross-checks are a per-square, per-letter spatial signal while availability is a
global per-letter scalar, and the two meet only through the trunk's blunt
global-context injection (a per-channel bias/FiLM). That composition is
sample-expensive -- a conv model learns availability-gating for common tiles and
falls back to a fixed frequency prior for rare ones (the I13 analysis: a live
model holds a ~0.17 Y-hook belief with zero Y's unseen).

This module gives the letter-indexed cross a first-class token. Each of the 27
tiles (A..Z, blank) becomes a *register token* -- a learned identity embedding
plus that letter's per-seat availability (mover rack, unseen pool, and under the
open-leaves arm the opponent's known leave) -- appended to the board-cell
sequence the transformer tower attends over (transformer_tower.py). A square
hooking on S/Y then reads S- and Y-supply directly in every attention layer,
graded by the actual counts and distinguishing "available to me" from
"available to the opponent", which a single gated input plane cannot.

docs/model_architectures.md diagrams these tokens; any change here belongs in
the same commit as the corresponding change there.
"""

import torch
import torch.nn as nn

# Scalar-block layout (engine/include/encoding/input_encoder.h): rack counts, the
# unseen-pool thermometer, score diff, move meta, then -- open-leaves arm only --
# the opponent-leave counts. The two arms are told apart by the scalar width.
N_TILES = 27  # A..Z + blank
RACK0 = 0
UNSEEN_THERMO0 = 27
UNSEEN_THERMO_LEN = 100
OPP_LEAVE0 = 136
SCALAR_SIZE_HIDDEN_LEAVES = 136
SCALAR_SIZE_OPEN_LEAVES = 163

# English Scrabble tile counts (engine/src/game/tile.cpp), used to normalise the
# per-letter availability counts to [0, 1] and to lay out the unseen thermometer.
TILE_COUNTS = (9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, 2)


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


class TileSupplyRegisters(nn.Module):
    """Builds the (B, N_TILES, channels) register tokens from the scalar input:
    a learned per-tile identity embedding plus a projection of the tile's
    per-seat availability counts."""

    num_tokens = N_TILES

    def __init__(self, channels: int, scalar_size: int):
        super().__init__()
        if scalar_size not in (SCALAR_SIZE_HIDDEN_LEAVES, SCALAR_SIZE_OPEN_LEAVES):
            raise ValueError(
                f"tile-supply registers expect scalar_size {SCALAR_SIZE_HIDDEN_LEAVES} or "
                f"{SCALAR_SIZE_OPEN_LEAVES}, got {scalar_size}"
            )
        self.has_opp_leave = scalar_size == SCALAR_SIZE_OPEN_LEAVES
        # Fixed decode of the unseen thermometer to per-letter counts, and the
        # per-letter normalisers -- buffers so they move with the module and
        # export as constants.
        self.register_buffer("thermo_to_count", _thermometer_to_count_matrix())
        self.register_buffer("tile_counts", torch.tensor(TILE_COUNTS, dtype=torch.float32))
        self.token_embed = nn.Parameter(torch.randn(N_TILES, channels) * 0.02)
        self.supply_proj = nn.Linear(3 if self.has_opp_leave else 2, channels)

    def _features(self, input_scalar: torch.Tensor) -> torch.Tensor:
        """(B, N_TILES, seats): each seat's availability count per tile, normalised
        by the tile's total count."""
        rack = input_scalar[:, RACK0 : RACK0 + N_TILES]
        unseen = input_scalar[:, UNSEEN_THERMO0 : UNSEEN_THERMO0 + UNSEEN_THERMO_LEN]
        seats = [rack, unseen @ self.thermo_to_count]
        if self.has_opp_leave:
            seats.append(input_scalar[:, OPP_LEAVE0 : OPP_LEAVE0 + N_TILES])
        return torch.stack(seats, dim=-1) / self.tile_counts[None, :, None]

    def forward(self, input_scalar: torch.Tensor) -> torch.Tensor:
        return self.token_embed + self.supply_proj(self._features(input_scalar))
