#pragma once

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"

#include <array>
#include <cstdint>
#include <vector>

namespace scribblez {

// Per-lane maximal-move targets for the max-move-per-lane training task. A
// "lane" is a single row read horizontally or column read vertically, and each
// of the 30 is a sub-task: predict the highest-scoring play lying in it, plus
// the union of all plays tied for that score. See docs/lexical_nn.md for the
// framing and the single-tile classification rule.

// Tile kinds in a lane union: 0..25 are letters A..Z; 26 is "blank", every
// blank designation collapsing into this one kind.
inline constexpr int kLaneTileKinds = 27;
inline constexpr int kLaneBlankKind = 26;
inline constexpr int kLaneLen = BOARD_SIZE;       // cells along one lane
inline constexpr int kLanesPerAxis = BOARD_SIZE;  // 15 rows, 15 cols
inline constexpr int kNumLanes = 2 * BOARD_SIZE;  // 30 sub-tasks

// ---- Play decomposition (shared by every lane-reduction consumer) ----------

// A single newly placed tile of a PLAY, in absolute board coordinates.
struct PlacedTile {
  int r;
  int c;
  int kind;  // the lane tile kind
};

// In word order. `out` must hold RACK_SIZE entries.
int decode_placements(const Move& m, PlacedTile* out);

// Whether a tile at (r, c) forms a word of length >= 2 along the axis, i.e.
// whether the square has an occupied neighbor there. `board` must be as it
// stood before the play.
bool forms_word_along_axis(const Board& board, int r, int c, bool horizontal);

// Which lane(s) a play contributes to: the single lane a multi-tile play lies
// along; for a single tile, each axis in which it forms a word (so a crossing
// tile contributes to both its row and its column, with the same score).
struct LaneAssignment {
  bool horizontal;
  int lane_index;
};
struct LaneAssignments {
  int count = 0;
  std::array<LaneAssignment, 2> items{};
};
LaneAssignments compute_lane_assignments(const Board& board, const Move& m,
                                         const PlacedTile* placed, int num_placed);

// The maximal-move target for one lane.
struct LaneBest {
  bool has_move = false;
  int max_score = 0;  // the lane's best play(s); valid iff has_move
  // placed[pos] has bit t set iff some play tied for max_score places tile-kind
  // t at lane cell `pos`. Only newly placed tiles are marked -- the tiles a
  // word threads through are already on the board.
  std::array<uint32_t, kLaneLen> placed{};
};

// All 30 per-lane targets for one (board, rack) position.
struct LaneTargets {
  std::array<LaneBest, kLanesPerAxis> rows{};  // horizontal plays, indexed by row
  std::array<LaneBest, kLanesPerAxis> cols{};  // vertical plays, indexed by column
};

// A multi-tile play is assigned to the single lane it lies along; a single-tile
// play to its row iff it forms a horizontal word and to its column iff it forms
// a vertical one, so a crossing tile contributes to both at the same score.
LaneTargets compute_lane_targets(const Board& board, const Rack& rack, const Dictionary& dict);

// Every move tied for one lane's best score. compute_lane_targets keeps only
// their placed-tile union; this keeps the moves, so a consumer can recover each
// play's word and coordinates. `moves` is empty iff !has_move.
struct LaneBestMoves {
  bool has_move = false;
  int max_score = 0;
  std::vector<Move> moves;
};

// Indexed like LaneTargets.
struct LaneBestMovesSet {
  std::array<LaneBestMoves, kLanesPerAxis> rows;
  std::array<LaneBestMoves, kLanesPerAxis> cols;
};

// compute_lane_targets under the identical assignment rule, collecting the
// moves tied for each lane's maximum instead of their union.
LaneBestMovesSet compute_lane_best_moves(const Board& board, const Rack& rack,
                                         const Dictionary& dict);

// ---- Flat label layout for the max-move-per-lane training row --------------
//
// Three contiguous blocks, all indexed by a flat lane id `axis * 15 + lane`,
// where axis 0's lanes are rows and axis 1's are columns:
//
//   occupancy  (2, 15, 15, 27)  BCE targets: [axis][lane][cell][kind] is 1.0
//                               iff some maximal play in that lane places
//                               tile-kind `kind` at lane `cell`. Equivalently
//                               two (15, 15, 27) per-cell tensors.
//   score      (30,)            per-lane max score as a bin index, the top bin
//                               a catch-all; 0 when !has_move.
//   mask       (30,)            1.0 iff the lane has a legal move. Empty lanes
//                               are excluded from every loss.
inline constexpr int kLaneScoreBins = 100;
inline constexpr int kLaneOccupancyFloats = kNumLanes * kLaneLen * kLaneTileKinds;  // 12150
inline constexpr int kLaneScoreFloats = kNumLanes;                                  // 30
inline constexpr int kLaneMaskFloats = kNumLanes;                                   // 30
inline constexpr int kLaneLabelFloats =
  kLaneOccupancyFloats + kLaneScoreFloats + kLaneMaskFloats;  // 12210

// Flatten `t` into the kLaneLabelFloats-long label region at `out`, zero-filling
// the occupancy of empty lanes.
//
// The targets of a transposed board (Board::transpose) come out with rows and
// cols exchanged, so the symmetry augmentation needs nothing here.
void encode_lane_targets(const LaneTargets& t, float* out);

}  // namespace scribblez
