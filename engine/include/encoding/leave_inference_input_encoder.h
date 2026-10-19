#pragma once

// The leave inference model's input (docs/plans/deep_bot_v0.md, "Workload 1:
// leave_inference"): what the player to move (the observer) can see of a
// position and of the last few moves, from which the model infers the tiles
// the opponent kept at their last move. A standalone encoder, so the position
// model's InputEncodingSpec and version are untouched.
//
// A row is kSpatialPlanes 15x15 planes, channel-major, then kScalarFloats
// scalars.
//
//   Spatial
//     current board       BoardPlanes (31)
//     current checks      cross-check planes, horizontal then vertical (52)
//     per history slot    the squares its move placed tiles on (1), then the
//                         cross-check planes as they stood before it (52)
//   Scalars
//     current             the observer's rack counts (27), the unseen-pool
//                         thermometer (100), the score differential (1), the
//                         bag count / kCountScale (1), and the predicted
//                         leave's length one-hot over 1..kMaxLeaveLength (6)
//     per history slot    present (1), move type one-hot (3), the tiles it
//                         placed or exchanged as counts (27), how many (1), its
//                         score / kScoreDiffInputScale (1), and the bag count
//                         before it / kCountScale (1)
//     per opponent slot   its forgone plays (belief/forgone_plays.h): per tile
//                         kind, whether the margin applies (1) and the margin /
//                         kScoreDiffInputScale (1), tile kind-major (54)
//
// A margin applies when the slot's move has one for the tile kind and the
// observer, just after that move, could not see every copy of it, so the
// opponent may have kept one. Otherwise both floats are 0.
//
// Hidden information never reaches a row: for an exchange in an opponent
// slot, the tile counts stay zero and only the number shows.

#include "belief/forgone_plays.h"
#include "encoding/board_planes.h"
#include "encoding/input_encoder.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <array>

namespace scribblez {

class Dictionary;

// The format version of the leave inference model's input and output
// (docs/plans/deep_bot_v0.md, "Versions"). Bump it whenever a row's layout or
// meaning changes; models stamp it and loaders accept only this version.
inline constexpr int kLeaveInferenceFormatVersion = 2;

// One past move as the observer saw it.
struct LeaveHistorySlot {
  bool present = false;          // false before the game had this many moves
  Move move;                     // in the position's board frame
  int score = 0;                 // what it scored; 0 for an exchange or a pass
  int bag_before = 0;            // tiles in the bag before it
  CrossCheckMasks cross_checks;  // the board's, just before it
  // The move's forgone plays, from the .forgone sidecar. Frame-invariant.
  belief::ForgonePlays forgone = belief::ForgonePlays::none();
  // Bit t set iff the player who did not make the move could not see every
  // tile of kind t just after it: not on the board, not on their own rack.
  uint32_t unseen_by_other = 0;
};

// The kSlots moves before a position, most recent first. Players alternate,
// so slot 0 is the opponent's last move, slot 1 the observer's, and so on.
struct LeaveHistory {
  static constexpr int kSlots = 5;
  static constexpr int kOpponentSlots = (kSlots + 1) / 2;

  static constexpr bool is_opponent_slot(int slot) { return slot % 2 == 0; }

  // The history of the transposed position (Board::transpose).
  LeaveHistory transpose() const;

  std::array<LeaveHistorySlot, kSlots> slots;
};

// Everything the encoder reads, from the observer's side.
struct LeaveInferenceInput {
  const Board* board;  // the current position
  Rack rack;           // the observer's
  int score_diff;      // the observer's lead
  int bag_size;
  int leave_length;  // of the opponent's leave; outside 1..kMaxLeaveLength encodes as none
  const LeaveHistory* history;
};

struct LeaveInferenceInputEncoder {
  static constexpr int kMaxLeaveLength = 6;
  static constexpr float kCountScale = 100.0f;

  static constexpr int kCurrentPlanes = BoardPlanes::kPlanes + kCrossCheckPlanes;             // 83
  static constexpr int kSlotPlanes = 1 + kCrossCheckPlanes;                                   // 53
  static constexpr int kSpatialPlanes = kCurrentPlanes + LeaveHistory::kSlots * kSlotPlanes;  // 348

  static constexpr int kCurrentScalars = kRackCountFloats + kUnseenPoolThermoFloats +
                                         kScoreDiffInputFloats + 1 + kMaxLeaveLength;  // 135
  static constexpr int kSlotScalars = 1 + kMoveMetaTypeFloats + TILE_KINDS + 3;        // 34
  static constexpr int kForgoneScalars = 2 * TILE_KINDS;                               // 54
  static constexpr int kScalarFloats = kCurrentScalars + LeaveHistory::kSlots * kSlotScalars +
                                       LeaveHistory::kOpponentSlots * kForgoneScalars;  // 467

  static constexpr int kSpatialFloats = kSpatialPlanes * kBoardCells;
  static constexpr int kInputFloats = kSpatialFloats + kScalarFloats;

  // Writes kInputFloats into `out`. Builds the current board's move-generation
  // caches from `dict` if needed.
  static void encode(const LeaveInferenceInput& in, const Dictionary& dict, float* out);
};

}  // namespace scribblez
