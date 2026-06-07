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
// Labels are produced from a GameLogView -- a tiny POD describing the
// active player's POV, the final scores, the opponent's next move (if any),
// and the per-row symmetry flip flag.

#include "scribblez/move.h"

namespace scribblez {
namespace binlog {

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

// A snapshot of the per-sample inputs needed to compute the three label
// heads. The caller (the DataLoader's decode path, or a test) populates
// this from the replay state at the sampled position.
struct GameLogView {
  Move next_move{};  // the opponent's response to the sampled position;
                     // only consulted when has_next_move is true
  bool has_next_move = false;
  int active_player = 0;  // 0 or 1 -- the POV for WLD / score_diff
  int final_score_p0 = 0;
  int final_score_p1 = 0;
  bool apply_flip = false;  // transpose spatial output for head 2 when true

  int final_active() const { return active_player == 0 ? final_score_p0 : final_score_p1; }
  int final_opp() const { return active_player == 0 ? final_score_p1 : final_score_p0; }
};

// Encode all kNumLabelHeads heads. `out` is an array of kNumLabelHeads float
// pointers; out[k] must point to at least kHeadFloats[k] writeable floats.
// Previous contents of each buffer are overwritten.
void encode_labels(const GameLogView& view, float** out);

}  // namespace binlog
}  // namespace scribblez
