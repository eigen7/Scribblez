#pragma once

#include "scribblez/board.h"
#include "scribblez/dictionary.h"
#include "scribblez/rack.h"

#include <array>
#include <cstdint>

namespace scribblez {

// Per-lane maximal-move targets for the lexical training task. A "lane" is a
// single row (read horizontally) or a single column (read vertically); there
// are 15 + 15 = 30 lanes, and each is a sub-task: predict the highest-scoring
// play that lies in that lane, plus the union of all plays tied for that score.
// See docs/lexical_nn.md for the framing and the single-tile classification
// rule.

// Tile kinds in a lane union: 0..25 are letters A..Z; 26 is "blank" with every
// blank designation collapsed into this single kind.
inline constexpr int kLaneTileKinds = 27;
inline constexpr int kLaneBlankKind = 26;
inline constexpr int kLaneLen = BOARD_SIZE;       // cells along one lane
inline constexpr int kLanesPerAxis = BOARD_SIZE;  // 15 rows, 15 cols
inline constexpr int kNumLanes = 2 * BOARD_SIZE;  // 30 sub-tasks

// The maximal-move target for one lane.
struct LaneBest {
  bool has_move = false;  // true iff at least one legal play lies in this lane
  int max_score = 0;      // raw score of the lane's best play(s); valid iff has_move
  // placed[pos] has bit t set iff some play tied for max_score places tile-kind
  // t (see kLaneTileKinds) at lane cell `pos`. Only newly placed tiles are
  // marked; tiles the word threads through stay 0 (they are already on the
  // board). All zero when !has_move.
  std::array<uint32_t, kLaneLen> placed{};
};

// All 30 per-lane targets for one (board, rack) position.
struct LaneTargets {
  std::array<LaneBest, kLanesPerAxis> rows{};  // horizontal plays, indexed by row
  std::array<LaneBest, kLanesPerAxis> cols{};  // vertical plays, indexed by column
};

// Enumerate every legal play for `rack` on `board` and reduce it to the 30
// per-lane maximal-move targets. A multi-tile play is assigned to the single
// lane it lies along; a single-tile play is assigned to its row iff it forms a
// horizontal word and to its column iff it forms a vertical word (so a crossing
// tile contributes to both lanes, with the same score).
LaneTargets compute_lane_targets(const Board& board, const Rack& rack, const Dictionary& dict);

// ---- Flat label layout for the lexical training row -------------------------
//
// A row's label region is three contiguous blocks, all indexed by a flat lane
// id `axis * 15 + lane`, where axis 0 is the horizontal lanes (a lane is a row)
// and axis 1 is the vertical lanes (a lane is a column):
//
//   occupancy  (2, 15, 15, 27)  BCE targets: [axis][lane][cell][kind] is 1.0
//                               iff some maximal play in that lane places
//                               tile-kind `kind` at lane `cell`. For axis 0,
//                               (lane, cell) = (row, col); for axis 1,
//                               (lane, cell) = (col, row). Equivalently two
//                               (15, 15, 27) per-cell tensors.
//   score      (30,)            per-lane max score as a bin index in
//                               [0, kLaneScoreBins-1] (top bin is the catch-all
//                               score >= kLaneScoreBins-1); 0 when !has_move.
//   mask       (30,)            1.0 iff the lane has a legal move, else 0.0.
//                               Empty lanes are excluded from every loss.
inline constexpr int kLaneScoreBins = 100;
inline constexpr int kLaneOccupancyFloats = kNumLanes * kLaneLen * kLaneTileKinds;  // 12150
inline constexpr int kLaneScoreFloats = kNumLanes;                                  // 30
inline constexpr int kLaneMaskFloats = kNumLanes;                                   // 30
inline constexpr int kLaneLabelFloats =
  kLaneOccupancyFloats + kLaneScoreFloats + kLaneMaskFloats;  // 12210

// Flatten `t` into the kLaneLabelFloats-long label region at `out` (occupancy,
// then score bins, then mask), zero-filling the occupancy of empty lanes.
//
// `flip` applies the diagonal symmetry (r,c) -> (c,r) so the labels stay aligned
// with a flipped input. A transpose turns each horizontal lane into the vertical
// lane of the same index (and vice versa) while preserving the along-lane cell
// position, so it is exactly a rows<->cols swap: the horizontal-lane axis is
// read from `t.cols` and the vertical-lane axis from `t.rows`.
void encode_lane_targets(const LaneTargets& t, bool flip, float* out);

}  // namespace scribblez
