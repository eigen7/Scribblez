#pragma once

// Feature encoder for the win-probability model: turns a self-contained
// PositionRecord into a flat float vector consumable by PyTorch.
//
// Layout (in order, contiguous floats):
//
//   Spatial features -- 32 planes, channel-major, each 15x15. The block is
//   shape-compatible with PyTorch (C=32, H=15, W=15) via a zero-copy
//   reshape on the Python side.
//
//     planes  [0..25]   letter A..Z presence on the board (1.0 where that
//                       letter is played, including as a designated blank).
//     plane    26       blank-marker: 1.0 where the board square holds a
//                       designated blank (regardless of which letter).
//     planes  [27..30]  premium-square mask: DLS, TLS, DWS, TWS. Always
//                       emitted from the canonical Board::PREMIUM table --
//                       i.e. the premium under a played tile is still
//                       reported (the model can subtract the letter planes
//                       to know which premiums have already been consumed).
//     plane    31       last-opponent-move placement: 1.0 on each square the
//                       opponent placed a tile on in their most recent turn.
//                       All zero for EXCHANGE / PASS / game-start.
//
//   Scalar features -- 58 floats, raw counts / values (the model can learn
//   its own normalization in the first FC layer):
//
//     [0..26]    active player's rack: per-tile counts (A..Z, blank).
//     [27..53]   bag composition: per-tile counts (A..Z, blank), i.e. how
//                many of each tile type are still in the bag.
//     [54]       score_active - score_opp
//     [55]       bag_size   -- total tiles still in the bag
//     [56]       active_rack_size
//     [57]       last opponent move: num_glyphs (0 = PASS; >0 with all-zero
//                placement plane = EXCHANGE; >0 with nonzero placement plane
//                = PLAY -- the model can derive the move type from those two
//                signals, so we don't emit a separate type one-hot).
//
// Symmetry: Scrabble boards (and the standard premium pattern) are invariant
// under the diagonal flip (r,c) -> (c,r). When `apply_flip == true`, each
// spatial plane (including the last-opp-placement plane) is transposed.
// All scalar features are flip-invariant under this layout.
//
// `kInputFloats` is the single source of truth for the input width and is
// re-exported by data_loader.h.

#include "scribblez/binary_log.h"

namespace scribblez {
namespace binlog {

inline constexpr int kBoardSide = 15;
inline constexpr int kBoardCells = kBoardSide * kBoardSide;  // 225

inline constexpr int kLetterPlanes = 26;
inline constexpr int kBlankMarkerPlanes = 1;
inline constexpr int kPremiumPlanes = 4;
inline constexpr int kLastOppPlanes = 1;
inline constexpr int kSpatialPlanes =
  kLetterPlanes + kBlankMarkerPlanes + kPremiumPlanes + kLastOppPlanes;  // 32
inline constexpr int kSpatialFloats = kSpatialPlanes * kBoardCells;      // 7200

inline constexpr int kRackCountFloats = 27;
inline constexpr int kBagCountFloats = 27;
inline constexpr int kScalarMiscFloats = 3;   // score_diff, bag_size, rack_size
inline constexpr int kLastOppMoveFloats = 1;  // num_glyphs
inline constexpr int kScalarFloats =
  kRackCountFloats + kBagCountFloats + kScalarMiscFloats + kLastOppMoveFloats;  // 58

inline constexpr int kInputFloats = kSpatialFloats + kScalarFloats;  // 7258

// Encode `record` into the kInputFloats-long buffer pointed to by `out`.
// `out` must have room for at least kInputFloats floats.
//
// If `apply_flip` is true, the input is rendered as if the board (and the
// last opponent move) had been reflected across the main diagonal -- a
// label-preserving augmentation.
void encode_input(const PositionRecord& record, bool apply_flip, float* out);

}  // namespace binlog
}  // namespace scribblez
