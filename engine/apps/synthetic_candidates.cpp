#include "synthetic_candidates.h"

namespace scribblez::move_set {

MoveFeatureArrays synthetic_candidates(int num_moves) {
  MoveFeatureArrays moves;
  moves.resize(num_moves);

  for (int m = 0; m < num_moves; ++m) {
    const bool is_play = m % 5 != 0;
    const int tiles = m % kMoveMaxPlaced + 1;
    for (int t = 0; t < tiles; ++t) {
      const size_t slot = size_t(m) * kMoveMaxPlaced + t;
      moves.letters[slot] = (m + t) % 26 + 1;
      moves.tile_mask[slot] = 1;
      if (is_play) moves.squares[slot] = (m * kMoveMaxPlaced + t) % kMoveCells;
    }
    if (is_play) {
      for (int e = 0; e < m % 3 + 1; ++e) {
        const size_t slot = size_t(m) * kMoveCrossSlots + e;
        moves.cross_cells[slot] = 1 + (e % 2) * kMoveCells + (m * 7 + e * 31) % kMoveCells;
        for (int l = 0; l < 26; ++l) {
          moves.cross_letters[slot * 26 + l] = (m + e + l) % 4 == 0;
        }
      }
    }
    float* scalars = moves.scalars.data() + size_t(m) * kMoveScalars;
    scalars[0] = float(m - num_moves / 2) / 100.0f;
    scalars[1] = float(tiles) / kMoveMaxPlaced;
    scalars[2] = is_play ? 1.0f : 0.0f;
  }
  return moves;
}

}  // namespace scribblez::move_set
