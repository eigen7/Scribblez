#pragma once

// Feature-vector layout constants for the win-probability model's input
// tensor. The encoding itself is performed by GameStateEncoder (see
// scribblez/game_state_encoder.h); this header is just the layout
// description -- the single source of truth for input width and offsets,
// shared between the engine and the Python side (re-exported through
// data_loader.h).
//
// Layout (in order, contiguous floats):
//
//   Spatial features -- 85 planes, channel-major, each 15x15. Shape-compatible
//   with PyTorch (C=85, H=15, W=15) via a zero-copy reshape on the Python
//   side.
//
//     planes  [0..25]   letter A..Z presence on the board (1.0 where that
//                       letter is played, including as a designated blank).
//     plane    26       blank-marker: 1.0 where the board square holds a
//                       designated blank (regardless of which letter).
//     planes  [27..30]  premium-square mask: DLS, TLS, DWS, TWS. Although the
//                       premium pattern is board-static for vanilla Scrabble,
//                       the trunk is fully convolutional + global pool with NO
//                       positional encoding, so these planes are the model's
//                       only absolute-position anchor (a weight-shared conv
//                       cannot synthesize a per-cell constant on its own).
//                       Always emitted from the canonical Board::PREMIUM table
//                       -- i.e. the premium under a played tile is still
//                       reported (the model can subtract the letter planes to
//                       know which premiums have already been consumed).
//     plane    31       self last-placement: 1.0 on each square the POV player
//                       placed a tile on in THEIR most recent turn.
//     plane    32       opponent last-placement: 1.0 on each square the
//                       opponent placed a tile on in their most recent turn.
//                       The two placement planes encode the last K=2 plies;
//                       together they let the model reconstruct the board as it
//                       stood when the opponent moved (needed to reason about
//                       the opponent's leave, and to interpret a candidate move
//                       whose leave is the POV rack). A placement plane is
//                       all-zero for an EXCHANGE / PASS / game-start move.
//     planes  [33..58]  horizontal cross-check letters A..Z: plane L marks empty
//                       squares where placing letter L would fuse with an
//                       existing left and/or right neighbor and satisfy the
//                       horizontal-pass cross-check mask.
//     planes  [59..84]  vertical cross-check letters A..Z: plane L marks empty
//                       squares where placing letter L would fuse with an
//                       existing above and/or below neighbor and satisfy the
//                       vertical-pass cross-check mask.
//
//   Scalar features -- 936 floats. All values reflect ONLY information the
//   active player would have at the table -- in particular, the opponent's
//   rack CONTENTS are not encoded. Every scalar feature is in [0, 1] (counts
//   are unary/thermometer-encoded), so no input normalization is required.
//
//     [0..26]      active player's rack: per-tile RAW counts (A..Z, blank).
//                  Kept raw (not unary) because rack counts are tiny (0..2)
//                  and a 7-tile rack would waste a 100-wide unary block.
//     [27..126]    "unseen pool" per-tile UNARY (thermometer) counts: a fixed
//                  100-slot vector, one slot per physical tile, grouped into
//                  per-letter regions of width TILE_COUNTS[i] (A..Z, blank,
//                  concatenated in that order). Within a letter's region the
//                  first `count` slots are 1.0 and the holes sit at the tail.
//                  The unseen pool is TILE_COUNTS - board - active_rack: the
//                  tiles the active player has not observed -- the union of the
//                  bag and the opponent's rack, indistinguishable from the
//                  active player's POV. Scrabble's refill-to-7 rule pins the
//                  partition: opp_rack_size == min(unseen_pool_size, 7) and
//                  bag_size == unseen_pool_size - opp_rack_size, and
//                  unseen_pool_size == sum of this block, so neither size is
//                  emitted as a separate scalar. Unary (vs raw counts) lets the
//                  model key sharply on the presence of scarce critical tiles
//                  (blanks, S, J/Q/X/Z).
//     [127..927]   score_active - score_opp, UNARY (thermometer) over the
//                  clipped range [-kScoreDiffClip, +kScoreDiffClip]
//                  (kScoreDiffThermoBins slots): slot i is 1.0 iff the clipped
//                  diff >= (i - kScoreDiffClip). Thermometer (not one-hot)
//                  preserves ordinality; unary (not a raw int) bounds the
//                  feature to [0,1] and lets the model fit the sharply
//                  nonlinear win-prob / score-diff relationship.
//     [928..935]   last-2-move metadata, self-move first then opponent-move,
//                  4 floats each: a 3-way move-type one-hot
//                  (PLAY / EXCHANGE / PASS, indexed by MoveType) followed by
//                  num_glyphs (raw 0..7). The type one-hot removes the need to
//                  derive the move type from the placement plane.
//
// Symmetry: Scrabble boards (and the standard premium pattern) are invariant
// under the diagonal flip (r,c) -> (c,r). When the encoder is asked to
// `apply_flip`, each spatial plane (including both placement planes) is
// transposed. All scalar features are flip-invariant under this layout.

#include "scribblez/training_targets.h"  // kScoreDiffClip

namespace scribblez {
namespace binlog {

inline constexpr int kBoardSide = 15;
inline constexpr int kBoardCells = kBoardSide * kBoardSide;  // 225

inline constexpr int kLetterPlanes = 26;
inline constexpr int kBlankMarkerPlanes = 1;
inline constexpr int kPremiumPlanes = 4;
inline constexpr int kPlacementPlanes = 2;  // self + opponent most-recent placements
inline constexpr int kHorizontalCrossCheckPlanes = 26;
inline constexpr int kVerticalCrossCheckPlanes = 26;
inline constexpr int kSpatialPlanes = kLetterPlanes + kBlankMarkerPlanes + kPremiumPlanes +
                                      kPlacementPlanes + kHorizontalCrossCheckPlanes +
                                      kVerticalCrossCheckPlanes;     // 85
inline constexpr int kSpatialFloats = kSpatialPlanes * kBoardCells;  // 19125

// Plane offsets within the spatial block (single source of truth).
inline constexpr int kBlankMarkerPlane = kLetterPlanes;                        // 26
inline constexpr int kPremiumPlane0 = kBlankMarkerPlane + kBlankMarkerPlanes;  // 27
inline constexpr int kSelfPlacementPlane = kPremiumPlane0 + kPremiumPlanes;    // 31
inline constexpr int kOppPlacementPlane = kSelfPlacementPlane + 1;             // 32
inline constexpr int kHorizontalCrossCheckPlane0 = kOppPlacementPlane + 1;     // 33
inline constexpr int kVerticalCrossCheckPlane0 =
  kHorizontalCrossCheckPlane0 + kHorizontalCrossCheckPlanes;  // 59

inline constexpr int kRackCountFloats = 27;
inline constexpr int kUnseenPoolThermoFloats = 100;  // == sum(TILE_COUNTS) for English Scrabble
// The input score-diff thermometer spans the same clipped differential range as
// the regression target (kScoreDiffClip), one unary slot per integer
// differential. Owned here: the value head predicts a Gaussian (mean/std), not
// bins, so the input feature no longer borrows the head's binning.
inline constexpr int kScoreDiffThermoBins = 2 * kScoreDiffClip + 1;  // 801
inline constexpr int kScoreDiffThermoFloats = kScoreDiffThermoBins;  // 801
inline constexpr int kMoveMetaTypeFloats = 3;  // PLAY / EXCHANGE / PASS one-hot
inline constexpr int kMoveMetaFloatsPerMove = kMoveMetaTypeFloats + 1;  // + num_glyphs
inline constexpr int kMoveMetaFloats = 2 * kMoveMetaFloatsPerMove;      // self + opp = 8

inline constexpr int kScalarFloats =
  kRackCountFloats + kUnseenPoolThermoFloats + kScoreDiffThermoFloats + kMoveMetaFloats;  // 936

// Scalar-block offsets (relative to the start of the scalar block).
inline constexpr int kRackCountOffset = 0;
inline constexpr int kUnseenPoolOffset = kRackCountOffset + kRackCountFloats;         // 27
inline constexpr int kScoreDiffOffset = kUnseenPoolOffset + kUnseenPoolThermoFloats;  // 127
inline constexpr int kMoveMetaOffset = kScoreDiffOffset + kScoreDiffThermoFloats;     // 928

inline constexpr int kInputFloats = kSpatialFloats + kScalarFloats;  // 20061

}  // namespace binlog
}  // namespace scribblez
