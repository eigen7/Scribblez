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
//   Spatial features -- 32 planes, channel-major, each 15x15. Shape-compatible
//   with PyTorch (C=32, H=15, W=15) via a zero-copy reshape on the Python
//   side.
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
//   Scalar features -- 59 floats. All values reflect ONLY information the
//   active player would have at the table -- in particular, the opponent's
//   rack CONTENTS are not encoded; only the SIZE of the opponent's rack is.
//
//     [0..26]    active player's rack: per-tile counts (A..Z, blank).
//     [27..53]   "unseen pool" per-tile counts (A..Z, blank): the tiles the
//                active player has not observed, computed as
//                    TILE_COUNTS - board - active_rack.
//                This pool is the union of the bag and the opponent's rack;
//                they are indistinguishable from the active player's POV.
//     [54]       score_active - score_opp
//     [55]       unseen_pool_size -- total tiles in the unseen pool
//                (== bag size + opponent's rack size; sum of [27..53])
//     [56]       active_rack_size
//     [57]       opp_rack_size -- the player tracks every draw, so they
//                know how many tiles the opponent holds (just not which)
//     [58]       last opponent move: num_glyphs (0 = PASS; >0 with all-zero
//                placement plane = EXCHANGE; >0 with nonzero placement plane
//                = PLAY -- the model can derive the move type from those two
//                signals, so we don't emit a separate type one-hot).
//
// Symmetry: Scrabble boards (and the standard premium pattern) are invariant
// under the diagonal flip (r,c) -> (c,r). When the encoder is asked to
// `apply_flip`, each spatial plane (including the last-opp-placement plane)
// is transposed. All scalar features are flip-invariant under this layout.

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
inline constexpr int kUnseenPoolCountFloats = 27;
inline constexpr int kScalarMiscFloats =
  4;  // score_diff, unseen_pool_size, active_rack_size, opp_rack_size
inline constexpr int kLastOppMoveFloats = 1;  // num_glyphs
inline constexpr int kScalarFloats =
  kRackCountFloats + kUnseenPoolCountFloats + kScalarMiscFloats + kLastOppMoveFloats;  // 59

inline constexpr int kInputFloats = kSpatialFloats + kScalarFloats;  // 7259

}  // namespace binlog
}  // namespace scribblez
