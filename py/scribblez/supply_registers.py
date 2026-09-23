"""Tile-supply register tokens for the transformer trunk (spatial_trunk.py).

The placement heads must weigh a square's cross-check letters by whether those
letters are actually available: in the mover's rack, in the unseen pool, or in
the opponent's known leave. A conv trunk learns this poorly. Cross-checks are a
per-square, per-letter spatial signal, availability is a global per-letter
scalar, and a conv trunk combines them only through a per-channel global
bias/FiLM. In practice a conv model learns the gating for common tiles but
falls back to a fixed frequency prior for rare ones: one model predicted a
~0.17 chance of a Y hook with no Y left unseen.

The transformer trunk instead appends one register token per tile (A..Z,
blank) to the board-cell sequence (transformer_tower.py). Each token is a
learned tile embedding plus that tile's normalized count in each seat: mover
rack, unseen pool, and, under the open-leaves arm, the opponent's leave. Every
attention layer can then read a hooking square's letter supply directly, graded
by count and split by who holds it.

docs/model_architectures.md diagrams these tokens; keep it in sync.
"""

import torch
import torch.nn as nn

# Scalar-block offsets (engine/include/encoding/input_encoder.h): rack counts
# (27), unseen-pool thermometer (100), score diff (1), move meta (8), then, in the
# open-leaves arm only, opponent-leave counts (27). The scalar width tells the
# two arms apart.
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
    """A fixed (100, 27) 0/1 matrix M with `unseen_thermo @ M` = per-letter
    unseen counts. Letter i owns a contiguous TILE_COUNTS[i]-wide region of the
    thermometer."""
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
        # Buffers, so they follow .to(device) and export as constants.
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
