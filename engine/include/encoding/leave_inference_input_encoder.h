#pragma once

// The leave inference model's input (docs/plans/deep_bot_v0.md, "Workload 1:
// leave_inference"): what the player to move (the observer) can see of a
// position and of the last few moves, from which the model infers the tiles
// the opponent kept at their last move. A standalone encoder, so the position
// model's InputEncodingSpec and version are untouched.
//
// A row is kSpatialPlanes 15x15 planes, channel-major, then kScalarFloats
// scalars. The history is the opponent's last LeaveHistory::kSlots moves: the
// observer's own moves tell nothing about the opponent's leave beyond the
// board, except their exchanges on their last two turns (below).
//
//   Spatial
//     current board       BoardPlanes (31)
//     current checks      cross-check planes, horizontal then vertical (52)
//     per history slot    the squares its move placed tiles on (1), then the
//                         cross-check planes as they stood before it (52)
//   Scalars
//     current             the observer's rack counts (27), the unseen-pool
//                         thermometer (100), the score differential (1), the
//                         bag count / kCountScale (1), the predicted leave's
//                         length one-hot over 1..kMaxLeaveLength (6), the
//                         tiles the observer exchanged on their previous turn,
//                         then on the turn before, as counts (27 each), and
//                         whether the observer's private history is hidden (1)
//     per history slot    present (1), move type one-hot (3), the tiles it
//                         placed as counts (27), how many it played or
//                         exchanged (1), its score / kScoreDiffInputScale (1),
//                         the bag count before it / kCountScale (1), then its
//                         forgone plays (belief/forgone_plays.h): per tile
//                         kind, whether the margin applies (1) and the margin /
//                         kScoreDiffInputScale (1), tile kind-major (54)
//
// A margin applies when the slot's move has one for the tile kind and that
// kind could be in the leave the move kept (LeaveHistorySlot::leave_kinds).
// Otherwise both floats are 0.
//
// Hidden information never reaches a row: for an exchange in a slot, the tile
// counts stay zero and only the number shows.
//
// A row whose observer's private history is hidden (LeaveHistory::hide_private)
// is the row of an observer whose past someone else is modelling: DeepBot's
// Alice, asking what Bob believes about her rack after her candidate
// (belief/reply_query.h), knows the rack she samples for him but not his past
// racks or exchanges. Its indicator is set, the observer's exchanges are zero,
// and the history slots before the newest apply their margins for every kind.
// Training hides it in a fraction of rows (DataLoader::EpochConfig), so the
// model learns both readings.
//
// A public row (LeaveInferenceView::kPublic) holds only what both players see,
// for a model that infers the leave from public information alone: the
// observer's rack counts and exchanges stay zero, the unseen pool is every
// tile off the board, and a forgone-play margin applies whenever it has one.
// The layout is the same, so either view's model serves through one engine.

#include "belief/forgone_plays.h"
#include "encoding/board_planes.h"
#include "encoding/input_encoder.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile_counts.h"

#include <array>

namespace scribblez {

class Dictionary;

// The format version of the leave inference model's input and output
// (docs/plans/deep_bot_v0.md, "Versions"). Bump it whenever a row's layout or
// meaning changes; models stamp it and loaders accept only this version.
inline constexpr int kLeaveInferenceFormatVersion = 4;

// One of the opponent's past moves as the observer saw it.
struct LeaveHistorySlot {
  bool present = false;          // false before the game had this many moves
  Move move;                     // in the position's board frame
  int score = 0;                 // what it scored; 0 for an exchange or a pass
  int bag_before = 0;            // tiles in the bag before it
  CrossCheckMasks cross_checks;  // the board's, just before it
  // The move's forgone plays, from the .forgone sidecar. Frame-invariant.
  belief::ForgonePlays forgone = belief::ForgonePlays::none();
  // Bit t set iff the leave this move kept may hold a tile of kind t, as the
  // observer saw it just after the move (leave_pool).
  uint32_t leave_kinds = 0;
};

// The opponent's kSlots moves before a position, most recent first, and the
// tiles the observer exchanged on their last two turns.
struct LeaveHistory {
  static constexpr int kSlots = 3;
  static constexpr int kObserverExchanges = 2;

  // The history of the transposed position (Board::transpose).
  LeaveHistory transpose() const;

  // Hides what only the observer could know of their own past: their
  // exchanges, and the leave-kind masks of the slots before the newest, which
  // read their racks at the time. The newest slot's mask reads the observer's
  // current rack, which the row holds anyway, but no longer subtracts their
  // previous turn's exchange, as the leave pool no longer does.
  void hide_private();

  std::array<LeaveHistorySlot, kSlots> slots;
  // The observer's exchanges, most recent turn first; each is empty unless
  // that move exchanged. The previous turn's exchange shrinks the leave pool
  // (leave_pool). The one before only caps the leave, at most as many of its
  // tiles as the opponent drew since, which the row leaves to the network.
  std::array<Rack, kObserverExchanges> observer_exchanges;
  bool private_hidden = false;  // hide_private() ran
};

// The tiles the leave an opponent's move kept may hold, as the observer saw it
// just after that move: those unseen to the observer (`unseen`, which counts
// the bag and the opponent's rack), less what the observer exchanged just
// before the move. The opponent's rack was drawn before that exchange, so the
// returned tiles cannot be in the leave, though they are unseen. An older
// exchange is not subtracted: the opponent has drawn since, so its tiles cap
// the leave rather than leave it, which a pool cannot express.
TileCounts leave_pool(TileCounts unseen, const Rack& observer_exchange);

// LeaveHistorySlot::leave_kinds for a leave drawn from `pool`: the kinds it
// holds at least one of.
uint32_t leave_kinds_of(const TileCounts& pool);

// LeaveHistorySlot::leave_kinds that reads nothing of the observer's rack: every
// kind may be in the leave, as a public row has it.
inline constexpr uint32_t kAnyLeaveKind = ~0u;

// A present history slot for `move`, about to be played on `before` with
// `bag_before` tiles in the bag; its forgone plays are attached separately.
// Builds `before`'s move-generation caches from `dict` if needed.
LeaveHistorySlot history_slot(const Board& before, const Dictionary& dict, const Move& move,
                              int bag_before, uint32_t leave_kinds);

// Whose information a row holds: the observer's, or only the public part of
// it (see the layout above).
enum class LeaveInferenceView : uint8_t { kObserver, kPublic };

// Everything the encoder reads, from the observer's side.
struct LeaveInferenceInput {
  const Board* board;  // the current position
  Rack rack;           // the observer's
  int score_diff;      // the observer's lead
  int bag_size;
  int leave_length;  // of the opponent's leave; outside 1..kMaxLeaveLength encodes as none
  const LeaveHistory* history;
  LeaveInferenceView view = LeaveInferenceView::kObserver;
};

struct LeaveInferenceInputEncoder {
  static constexpr int kMaxLeaveLength = 6;
  static constexpr float kCountScale = 100.0f;

  static constexpr int kCurrentPlanes = BoardPlanes::kPlanes + kCrossCheckPlanes;             // 83
  static constexpr int kSlotPlanes = 1 + kCrossCheckPlanes;                                   // 53
  static constexpr int kSpatialPlanes = kCurrentPlanes + LeaveHistory::kSlots * kSlotPlanes;  // 242

  static constexpr int kCurrentScalars = kRackCountFloats + kUnseenPoolThermoFloats +
                                         kScoreDiffInputFloats + 1 + kMaxLeaveLength +
                                         LeaveHistory::kObserverExchanges * TILE_KINDS + 1;  // 190
  static constexpr int kForgoneScalars = 2 * TILE_KINDS;                                     // 54
  static constexpr int kSlotScalars =
    1 + kMoveMetaTypeFloats + TILE_KINDS + 3 + kForgoneScalars;  // 88
  static constexpr int kScalarFloats =
    kCurrentScalars + LeaveHistory::kSlots * kSlotScalars;  // 454

  static constexpr int kSpatialFloats = kSpatialPlanes * kBoardCells;
  static constexpr int kInputFloats = kSpatialFloats + kScalarFloats;

  // Writes kInputFloats into `out`. Builds the current board's move-generation
  // caches from `dict` if needed.
  static void encode(const LeaveInferenceInput& in, const Dictionary& dict, float* out);
};

}  // namespace scribblez
