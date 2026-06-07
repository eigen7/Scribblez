#pragma once

// Training-target (label) encoder for the win-probability model. The model
// has three heads; each head writes into its own caller-provided buffer.
//
//   Head 0 -- WLD (kWldFloats = 3):
//     [win, draw, loss] from the active player's POV. Mutually exclusive
//     one-hot, sums to 1.0.
//
//   Head 1 -- ScoreDiff (kScoreDiffFloats = 1):
//     final_score_active - final_score_opp (signed, can exceed 100).
//
//   Head 2 -- OppNextPlacement (kOppNextPlacementFloats = 225):
//     A 15x15 binary mask. Cell (r,c) is 1.0 iff the OPPONENT placed a tile
//     on (r,c) on their next move (the move at moves[turn_index + 1]).
//     All zeros if (a) there is no next move (this is the last turn of the
//     game), or (b) the next move is EXCHANGE / PASS.
//     The mask is transposed across the main diagonal when view.apply_flip
//     is true (so it stays aligned with the InputEncoder's spatial planes).
//
// Labels are produced from a GameLogView -- a lightweight slice over a
// per-game moves array plus the sampled turn index, position kind, active
// player, final scores, and the per-row symmetry flip flag.

#include "scribblez/move.h"

#include <cstdint>

namespace scribblez {
namespace binlog {

enum class PositionKind : uint8_t;  // forward; defined in binary_log.h

inline constexpr int kWldFloats = 3;
inline constexpr int kScoreDiffFloats = 1;
inline constexpr int kOppNextPlacementSide = 15;
inline constexpr int kOppNextPlacementFloats =
  kOppNextPlacementSide * kOppNextPlacementSide;  // 225
inline constexpr int kNumLabelHeads = 3;
inline constexpr int kLabelFloats = kWldFloats + kScoreDiffFloats + kOppNextPlacementFloats;  // 229

// Sizes of each head's buffer, indexed by head id (0..kNumLabelHeads-1).
inline constexpr int kHeadFloats[kNumLabelHeads] = {
  kWldFloats,
  kScoreDiffFloats,
  kOppNextPlacementFloats,
};

// A lightweight view into one sampled position's slice of a GameLog. The
// caller (the DataLoader's decode path) constructs this from the per-game
// moves blob in the .slog file and a sampled PositionRecord.
struct GameLogView {
  const Move* moves;  // pointer into the per-game moves array
  int num_turns;      // length of `moves`
  int turn_index;     // sampled k in [0, num_turns); the "current" turn
  PositionKind kind;  // kPreMove / kPostMove (informational)
  int active_player;  // 0 or 1 -- the POV for WLD / score_diff
  int final_score_p0;
  int final_score_p1;
  bool apply_flip;  // transpose spatial output for head 2 when true

  int final_active() const { return active_player == 0 ? final_score_p0 : final_score_p1; }
  int final_opp() const { return active_player == 0 ? final_score_p1 : final_score_p0; }

  bool has_next_move() const { return turn_index + 1 < num_turns; }
  const Move& next_move() const { return moves[turn_index + 1]; }
};

// Encode all kNumLabelHeads heads. `out` is an array of kNumLabelHeads float
// pointers; out[k] must point to at least kHeadFloats[k] writeable floats.
// Previous contents of each buffer are overwritten.
void encode_labels(const GameLogView& view, float** out);

}  // namespace binlog
}  // namespace scribblez
