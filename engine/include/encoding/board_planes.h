#pragma once

#include "game/board.h"

namespace scribblez {

// The board-content planes shared by the position evaluation input encoder
// (input_encoder.h) and the max-move-per-lane input encoder. Both put this
// block first among their spatial planes, so the plane offsets below are
// absolute.
struct BoardPlanes {
  static constexpr int kLetterPlanes = 26;
  static constexpr int kBlankMarkerPlane = kLetterPlanes;       // 26
  static constexpr int kPremiumPlane0 = kBlankMarkerPlane + 1;  // 27
  static constexpr int kPremiumPlanes = 4;
  static constexpr int kPlanes = kPremiumPlane0 + kPremiumPlanes;  // 31

  // Writes 1.0s into the zeroed, channel-major `planes_out`. A designated blank
  // sets both its letter's plane and the blank-marker plane. The premium planes
  // (DLS, TLS, DWS, TWS) mark premium squares whether or not they are covered.
  static void encode(const Board& board, float* planes_out);
};

}  // namespace scribblez
