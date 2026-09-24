#pragma once

#include "encoding/board_planes.h"
#include "game/board.h"
#include "game/rack.h"

namespace scribblez {

// Input encoder for the max-move-per-lane model: the shared BoardPlanes block
// plus the rack as raw per-tile counts. It leaves out two things the position
// evaluation encoder carries:
//
//   * cross-check planes, which encode which letters are legal where: the very
//     lexicon knowledge this model is meant to learn.
//   * game-context features (scores, unseen pool, move history), which do not
//     affect which play scores best.
//
// Rack counts are small exact integers ("can I play two R's"), so they go in
// raw rather than unary-coded.
struct MaxMovePerLaneInputEncoder {
  static constexpr int kSpatialPlanes = BoardPlanes::kPlanes;          // 31
  static constexpr int kBoardCells = BOARD_SIZE * BOARD_SIZE;          // 225
  static constexpr int kSpatialFloats = kSpatialPlanes * kBoardCells;  // 6975

  static constexpr int kRackCountFloats = TILE_KINDS;
  static constexpr int kScalarFloats = kRackCountFloats;               // 27
  static constexpr int kInputFloats = kSpatialFloats + kScalarFloats;  // 7002

  // Writes kInputFloats: the spatial planes, then the rack counts.
  static void encode(const Board& board, const Rack& rack, float* out);
};

}  // namespace scribblez
