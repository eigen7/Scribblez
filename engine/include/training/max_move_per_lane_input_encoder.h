#pragma once

#include "encoding/board_planes.h"
#include "game/board.h"
#include "game/rack.h"

namespace scribblez {

// Stateless input encoder for the "highest-scoring move per lane" model: a lean
// board + rack tensor. It deliberately OMITS two things the post-move encoder
// carries:
//
//   * cross-check planes -- they encode which letters are legal where, i.e. the
//     very lexicon knowledge this model is meant to learn; feeding them in would
//     defeat the experiment.
//   * post-move-specific features (scores, unseen pool, move history) -- a
//     single-move query does not depend on game context.
//
// All-static, so it composes as a policy type alongside the target structs in
// training_targets.h. The spatial half is the shared BoardPlanes block and
// nothing else; the scalar half is the POV rack as raw per-tile counts -- raw
// rather than unary because rack counts are tiny, and exact ("can I play two
// R's").
struct MaxMovePerLaneInputEncoder {
  static constexpr int kSpatialPlanes = BoardPlanes::kPlanes;          // 31
  static constexpr int kBoardCells = BOARD_SIZE * BOARD_SIZE;          // 225
  static constexpr int kSpatialFloats = kSpatialPlanes * kBoardCells;  // 6975

  static constexpr int kRackCountFloats = 27;
  static constexpr int kScalarFloats = kRackCountFloats;               // 27
  static constexpr int kInputFloats = kSpatialFloats + kScalarFloats;  // 7002

  // Encode `board` plus the POV player's `rack` into the kInputFloats-long `out`
  // (spatial planes, then rack scalars). `flip` applies the diagonal symmetry
  // (r,c) -> (c,r) to the spatial planes; the rack scalars are flip-invariant.
  static void encode(const Board& board, const Rack& rack, bool flip, float* out);
};

}  // namespace scribblez
