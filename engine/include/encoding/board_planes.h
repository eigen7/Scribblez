#pragma once

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"

#include <cstdint>

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

// Block writers shared by the input encoders (input_encoder.h documents each
// block). Each writes into zeroed memory at `out` and returns the number of
// planes or floats written.

// One plane marking the squares `m` placed tiles on.
int encode_placement_plane(const Move& m, float* out);

// The cross-check planes: 26 horizontal, then 26 vertical. Reads the board's
// move-generation caches, which must be built.
int encode_cross_check_planes(const Board& board, float* out);

// Per-tile counts of `rack`, unscaled.
int encode_rack_counts(const Rack& rack, float* out);

// The unseen pool as a thermometer, one slot per physical tile.
int encode_unseen_pool_thermometer(const uint8_t unseen[TILE_KINDS], float* out);

// The score differential, scaled by kScoreDiffInputScale.
int encode_score_diff_scalar(int score_diff, float* out);

}  // namespace scribblez
