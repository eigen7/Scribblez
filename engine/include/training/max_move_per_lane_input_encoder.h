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
// All-static (it carries no state), so it composes as a policy type: its
// constants and encode() plug into the training-task machinery alongside the
// target structs in training_targets.h.
//
// Spatial -- kSpatialPlanes planes, channel-major, each 15x15: the shared
// BoardPlanes block (letters, blank-marker, premiums) and nothing else.
// Scalar -- kScalarFloats floats:
//   [0..26]  the POV rack as raw per-tile counts (A..Z, blank). Raw (not unary)
//            because rack counts are tiny, and exact ("can I play two R's").
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
