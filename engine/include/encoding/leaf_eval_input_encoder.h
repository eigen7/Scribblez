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
//     kCountScale (1); the bag after both refills, one-hot (kBagBuckets)
//
// The join planes exist because a conv trunk joins a square's cross-check
// letters with who holds those letters poorly: it needs one gate per letter
// and seat, and learns them only for common tiles. They are set only on hook
// squares (empty, with a neighbor across the play axis), the squares whose
// cross-check constrains a play. A leave's fit is 1 if any of its tiles fits;
// the pool's is the fraction of its tiles that fit, the chance a draw fits. A
// blank fits any square that allows some letter, at face value 0.
//
// The bag after both refills holds the tiles on neither the board nor two
// full racks: the pool less what refills both leaves, and the bag the opponent
// moves with, since their own draw is in the pool. Its buckets are 0 (the
// refills exhaust it) through 7, each exact, then 8 or more: a play of up to 7
// tiles empties a bag of 7 or fewer, and whether the opponent's next move can
// empty it, so that a bingo ends the game, turns on this count. The row's
// counts imply it, but a trained trunk placed that step a tile late when the
// opponent threatened a bingo, valuing a play that leaves one tile in the bag
// like one that empties it, so the row states it outright.

#include "encoding/board_planes.h"
#include "encoding/input_encoder.h"
#include "game/board.h"
#include "game/rack.h"

namespace scribblez {

class Dictionary;
class LeaveValues;

// The format version of the leaf eval model's input. Bump it whenever a row's
// layout or meaning changes; checkpoints and models stamp it.
inline constexpr int kLeafEvalFormatVersion = 2;

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
  static constexpr int kBagAfterRefills0 = kPoolSize + 1;
  static constexpr int kBagBuckets = 9;  // 0 through 7, then 8 or more
  static constexpr int kScalarFloats = kBagAfterRefills0 + kBagBuckets;  // 94

  static constexpr int kSpatialFloats = kSpatialPlanes * kBoardCells;
  static constexpr int kInputFloats = kSpatialFloats + kScalarFloats;

  static constexpr float kFaceValueScale = 10.0f;
  static constexpr float kLeaveValueScale = 25.0f;
  static constexpr float kCountScale = 100.0f;

  // The bag bucket of a board with `off_board` tiles not on it.
  static int bag_bucket(int off_board);

  // Writes kInputFloats into `out`. Builds the board's
  // move-generation caches from `dict` if needed.
  static void encode(const LeafEvalInput& in, const Dictionary& dict,
                     const LeaveValues& leave_values, float* out);
};

}  // namespace scribblez
