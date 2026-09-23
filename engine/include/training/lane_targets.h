#pragma once

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"

#include <array>
#include <cstdint>
#include <vector>

namespace scribblez {

// Per-lane maximal-move targets for the max-move-per-lane training task. A lane
// is a row read horizontally or a column read vertically. Each of the 30 is a
// sub-task: predict the score of the highest-scoring play in it, and the union
// of the tiles placed by every play tied for that score. docs/lexical_nn.md
// has the framing.

// Tile kinds in a lane union: 0..25 are letters A..Z; 26 is any blank,
// whatever letter it designates.
inline constexpr int kLaneTileKinds = 27;
inline constexpr int kLaneBlankKind = 26;
inline constexpr int kLaneLen = BOARD_SIZE;       // cells along one lane
inline constexpr int kLanesPerAxis = BOARD_SIZE;  // 15 rows, 15 cols
inline constexpr int kNumLanes = 2 * BOARD_SIZE;  // 30 sub-tasks

// ---- Play decomposition ------------------------------------------------------

// One newly placed tile of a PLAY, in board coordinates.
struct PlacedTile {
  int r;
  int c;
  int kind;  // the lane tile kind
};

// A PLAY's newly placed tiles in word order, returning their count. `out` must
// hold RACK_SIZE entries.
int decode_placements(const Move& m, PlacedTile* out);

// Whether a lone tile at (r, c) forms a word of length >= 2 along the axis,
// i.e. has an occupied neighbor on it. `board` is the pre-move board.
bool forms_word_along_axis(const Board& board, int r, int c, bool horizontal);

// The lane(s) a play counts toward. A multi-tile play counts toward the one lane
// it lies along. A single tile has no direction, so it counts toward its row iff
// it forms a horizontal word and its column iff it forms a vertical one; a
// crossing tile counts toward both, at the same score.
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
  int max_score = 0;  // valid iff has_move
  // placed[pos] has bit t set iff some play tied for max_score newly places
  // tile kind t at lane cell `pos`. Tiles already on the board are not marked.
  std::array<uint32_t, kLaneLen> placed{};
};

// All 30 per-lane targets for one (board, rack) position.
struct LaneTargets {
  std::array<LaneBest, kLanesPerAxis> rows{};  // horizontal plays, indexed by row
  std::array<LaneBest, kLanesPerAxis> cols{};  // vertical plays, indexed by column
};

// Lanes per compute_lane_assignments.
LaneTargets compute_lane_targets(const Board& board, const Rack& rack, const Dictionary& dict);

// Every move tied for one lane's best score, for consumers that need the plays
// themselves rather than their placed-tile union. `moves` is empty iff
// !has_move.
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

// compute_lane_targets, collecting the tied moves instead of their union.
LaneBestMovesSet compute_lane_best_moves(const Board& board, const Rack& rack,
                                         const Dictionary& dict);

// ---- Flat label layout for the max-move-per-lane training row ----------------
//
// Three contiguous blocks, indexed by the flat lane id `axis * 15 + lane`
// (axis 0: rows, axis 1: columns):
//
//   occupancy  (2, 15, 15, 27)  BCE targets: [axis][lane][cell][kind] is 1.0
//                               iff some maximal play in that lane places
//                               tile kind `kind` at lane `cell`.
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

// Flatten `t` into the kLaneLabelFloats-long label region at `out`. Symmetry
// augmentation needs nothing here: targets computed on a transposed board
// (Board::transpose) already come out with rows and columns exchanged.
void encode_lane_targets(const LaneTargets& t, float* out);

}  // namespace scribblez
