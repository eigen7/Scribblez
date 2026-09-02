#pragma once

#include "game/board.h"

namespace scribblez {

// The static board-content planes shared, identically, by the post-move and
// max-move-per-lane input encoders. Both place this block at the very start of
// their spatial features, so it owns the canonical plane offsets rather than
// taking them as parameters.
struct BoardPlanes {
  static constexpr int kLetterPlanes = 26;
  static constexpr int kBlankMarkerPlane = kLetterPlanes;       // 26
  static constexpr int kPremiumPlane0 = kBlankMarkerPlane + 1;  // 27
  static constexpr int kPremiumPlanes = 4;
  static constexpr int kPlanes = kPremiumPlane0 + kPremiumPlanes;  // 31

  // Into the channel-major `planes_out`, each plane row-major. Letter plane L is 1.0 where that
  // letter sits, a designated blank rendering its letter and also setting the blank-marker plane;
  // the premium planes mark the canonical Board::PREMIUM pattern, still reporting the premium under
  // a played tile.
  static void encode(const Board& board, float* planes_out);
};

}  // namespace scribblez
