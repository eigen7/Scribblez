#pragma once

// The "contingent draw potential map" lexical input feature for the value
// models (see docs/lexical_features_for_value.md). Per drawable tile kind X and
// each of the 30 lanes, the best play `rack ∪ {X}` can make along that lane
// among plays that NEED the drawn X, plus each lane's best play with the rack
// alone. On a post-move snapshot the rack is the mover's leave and the board
// already includes the move, so the entries are exactly "what the kept tiles
// could do next turn, per replenishment draw".
//
// Everything comes from ONE move generation over rack ∪ {blank}. A play whose
// blanks the rack covers is a rack-alone play; one consuming the extra blank
// needs a drawn blank. Independently, ANY placed blank designated as letter L
// reads as the real drawn L (rescored with L's face value at that cell) -- a
// contingent "drew L" play, provided the placement needs more copies of L than
// the rack holds. Kinds absent from the unseen pool are undrawable and stay
// empty. A full rack draws no replenishment tile, so it gets no phantom: its
// contingent columns are empty and only the rack-alone lanes are populated.

#include "game/board.h"
#include "game/rack.h"
#include "training/lane_targets.h"

#include <array>
#include <cstdint>

namespace scribblez {

class Dictionary;

// Raw scores are clipped to [0, kContingentScoreClip] and divided by it, so
// every encoded float lands in [0, 1]. Draw-weighted values divide by the
// smaller kContingentWeightedScale instead, a typical P(draw) * score product
// being an order of magnitude below a raw score.
inline constexpr int kContingentScoreClip = 150;
inline constexpr int kContingentWeightedScale = 30;

class ContingentMap {
 public:
  // The best qualifying play for one (tile kind, lane) or rack-alone slot: its
  // raw score and its newly placed cells as a mask over the lane's 15
  // positions (bit = column for a row lane, row for a column lane).
  struct Entry {
    int16_t score = -1;  // -1: no qualifying play
    uint16_t placed_mask = 0;
  };

  // `unseen` is the position's unseen-pool composition, the draw distribution
  // the map is gated and weighted by.
  static ContingentMap compute(const Board& board, const Rack& rack, const uint8_t unseen[27],
                               const Dictionary& dict);

  // Max, draw-weighted, and rack-alone contingent score, each painted onto the
  // best plays' placed cells (per-cell max), into `planes_out`
  // (kContingentPlanes * 225 floats, assumed zeroed).
  void encode_planes(bool flip, float* planes_out) const;

  // Per-kind best score over lanes, the same draw-weighted, then the expected
  // best under the unseen-pool draw distribution and the rack-alone best.
  void encode_scalars(float* out) const;

  // Lane ids are row lanes 0..14 then column lanes 15..29.
  const Entry& best(int kind, int lane) const { return best_[kind][lane]; }
  const Entry& rack_best(int lane) const { return rack_best_[lane]; }

 private:
  std::array<std::array<Entry, kNumLanes>, kLaneTileKinds> best_{};
  std::array<Entry, kNumLanes> rack_best_{};
  std::array<uint8_t, kLaneTileKinds> unseen_{};
  int unseen_total_ = 0;
};

}  // namespace scribblez
