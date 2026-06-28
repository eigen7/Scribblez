#pragma once

#include "scribblez/board.h"

namespace scribblez {

// The static board-content planes shared, identically, by the post-move and
// lexical input encoders: 26 letter planes, a blank-marker plane, and 4 premium
// planes. A stateless, composable sub-encoder -- both full input encoders place
// this 31-plane block at the very start of their spatial features, so it owns
// the canonical plane offsets rather than taking them as parameters.
struct BoardPlanes {
  static constexpr int kLetterPlanes = 26;
  static constexpr int kBlankMarkerPlane = kLetterPlanes;       // 26
  static constexpr int kPremiumPlane0 = kBlankMarkerPlane + 1;  // 27
  static constexpr int kPremiumPlanes = 4;
  static constexpr int kPlanes = kPremiumPlane0 + kPremiumPlanes;  // 31

  // Write the 31 board-content planes for `board` into the channel-major
  // `planes_out` (each plane BOARD_SIZE*BOARD_SIZE floats, laid out row-major,
  // or diagonally transposed when `flip`). Letter plane L (0..25) is 1.0 where
  // that letter sits (a designated blank renders its letter, so its letter plane
  // is set); the blank-marker plane is 1.0 where a designated blank sits; the
  // four premium planes mark the canonical Board::PREMIUM pattern (the premium
  // under a played tile is still reported).
  static void encode(const Board& board, bool flip, float* planes_out);
};

}  // namespace scribblez
