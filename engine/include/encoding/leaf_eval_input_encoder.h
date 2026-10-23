#pragma once

// The DeepBot leaf evaluation model's input (py/scribblez/deep_bot_leaf_eval/
// model.py): a rollout's leaf, after one player moved from a full rack and the
// other replied, neither drawing. The row is from the side of the player who
// just moved. Both leaves are known; the pool is every tile on neither the
// board nor a leave, which both refills come from.
//
// A row is kSpatialPlanes 15x15 planes, channel-major, then kScalarFloats
// scalars.
//
//   Spatial
//     board         BoardPlanes (31)
//     cross-checks  horizontal then vertical (52)
//     join planes   per seat (mine, the opponent's, the pool), per play axis
//                   (horizontal, vertical): how well the seat's tiles fit the
//                   square's cross-check, then the highest face value among
//                   the fitting tiles / kFaceValueScale (12)
//   Scalars
//     my leave's per-tile counts (27), the opponent's (27), the pool's (27),
//     unscaled; the score differential / kScoreDiffInputScale (1); my and the
//     opponent's static leave values / kLeaveValueScale (2); the pool size /
//     kCountScale (1)
//
// The join planes exist because a conv trunk joins a square's cross-check
// letters with who holds those letters poorly: it needs one gate per letter
// and seat, and learns them only for common tiles. They are set only on hook
// squares (empty, with a neighbor across the play axis), the squares whose
// cross-check constrains a play. A leave's fit is 1 if any of its tiles fits;
// the pool's is the fraction of its tiles that fit, the chance a draw fits. A
// blank fits any square that allows some letter, at face value 0.

#include "encoding/board_planes.h"
#include "encoding/input_encoder.h"
#include "game/board.h"
#include "game/rack.h"

namespace scribblez {

class Dictionary;
class LeaveValues;

// The format version of the leaf eval model's input. Bump it whenever a row's
// layout or meaning changes; checkpoints and models stamp it.
inline constexpr int kLeafEvalFormatVersion = 1;

// Everything the encoder reads, from the side of the player who just moved.
struct LeafEvalInput {
  const Board* board;
  Rack my_leave;
  Rack opp_leave;
  int score_diff;  // my score minus the opponent's
};

struct LeafEvalInputEncoder {
  static constexpr int kSeats = 3;  // my leave, the opponent's, the pool
  static constexpr int kAxes = 2;
  static constexpr int kJoinPlanesPerAxis = 2;  // fit, highest fitting face value
  static constexpr int kJoinPlanes = kSeats * kAxes * kJoinPlanesPerAxis;       // 12
  static constexpr int kJoinPlane0 = BoardPlanes::kPlanes + kCrossCheckPlanes;  // 83
  static constexpr int kSpatialPlanes = kJoinPlane0 + kJoinPlanes;              // 95

  static constexpr int kMyLeave0 = 0;
  static constexpr int kOppLeave0 = kMyLeave0 + TILE_KINDS;
  static constexpr int kPool0 = kOppLeave0 + TILE_KINDS;
  static constexpr int kScoreDiff = kPool0 + TILE_KINDS;
  static constexpr int kMyLeaveValue = kScoreDiff + 1;
  static constexpr int kOppLeaveValue = kMyLeaveValue + 1;
  static constexpr int kPoolSize = kOppLeaveValue + 1;
  static constexpr int kScalarFloats = kPoolSize + 1;  // 85

  static constexpr int kSpatialFloats = kSpatialPlanes * kBoardCells;
  static constexpr int kInputFloats = kSpatialFloats + kScalarFloats;

  static constexpr float kFaceValueScale = 10.0f;
  static constexpr float kLeaveValueScale = 25.0f;
  static constexpr float kCountScale = 100.0f;

  // Writes kInputFloats into `out`. Builds the board's
  // move-generation caches from `dict` if needed.
  static void encode(const LeafEvalInput& in, const Dictionary& dict,
                     const LeaveValues& leave_values, float* out);
};

}  // namespace scribblez
