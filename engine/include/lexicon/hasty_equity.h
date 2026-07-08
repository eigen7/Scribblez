#pragma once

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "lexicon/leave_values.h"

#include <array>
#include <bitset>
#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {

// Per-turn leave cache for one mover's rack. Owns the sorted-rack bit layout
// (bit i == the i-th sorted rack tile) and, for each candidate move, derives the
// leave as a compact bitmask over that layout. Leave values are filled lazily
// and cached, so each distinct leave is looked up at most once per turn. Build
// one per turn (HastyEquity::turn_leaves) and reuse it across the turn's moves.
class TurnLeaves {
 public:
  TurnLeaves(const Rack& rack, const LeaveValues& lv);

  // The move's leave as a bitmask: the full rack minus its played tiles. Each
  // played tile claims its letter's lowest still-available bit.
  uint8_t mask_for(const Move& move) const;

  double value(uint8_t mask);

  int point_value(uint8_t mask);

 private:
  void ensure(uint8_t mask);

  const LeaveValues& lv_;
  int size_;
  std::array<Tile, RACK_SIZE> tile_of_bit_{};
  std::array<uint8_t, 27> indices_{};
  uint8_t full_ = 0;
  std::array<float, 128> value_{};
  std::array<int16_t, 128> pv_{};
  std::bitset<128> computed_{};
};

// Computes HastyBot's static equity for a move, matching Macondo's four-
// calculator stack: leave value, opening adjustment, pre-endgame adjustment,
// and endgame adjustment.
//
// Load once at process startup with init(); thereafter equity() is safe to
// call concurrently from any number of threads.
class HastyEquity {
 public:
  static HastyEquity& instance();

  // Must be called exactly once before any call to equity(). Loads the leave
  // values file and the pre-endgame adjustment JSON. Throws on I/O failure.
  static void init(const std::string& klv2_path, const std::string& peg_json_path);

  // Lazy-initialization convenience for callers that just want HastyBot's
  // defaults: if init() has not already been called, load the default
  // leave-values and pre-endgame files for `lexicon` (see default_*_path()).
  // A no-op once loaded. Intended to be called during single-threaded setup;
  // not safe to race against concurrent equity() calls.
  static void ensure_initialized(const std::string& lexicon);

  // Default on-disk locations derived from Macondo's data layout: the
  // per-lexicon leave-values file (.../strategy/<lexicon>/leaves.klv2) and the
  // shared pre-endgame table (.../strategy/default/preendgame.json).
  static std::string default_leaves_path(const std::string& lexicon);
  static std::string default_peg_path();

  // Static HastyBot equity for `move` in the given position.
  //   bag_size : tiles remaining in the bag *before* the move
  //   my_rack  : the mover's rack *before* the move (used to derive the leave)
  //   opp_rack : opponent's rack (used only for the endgame adjustment)
  double equity(const Move& move, const Board& board, int bag_size, const Rack& opp_rack,
                const Rack& my_rack) const;

  // A per-turn leave cache for `my_rack`.
  TurnLeaves turn_leaves(const Rack& my_rack) const;

  // Equity of `move` reusing a per-turn TurnLeaves (the leave lookup is O(1) /
  // cached). Bit-identical to equity(); the leave is read from `leaves` instead
  // of reconstructed per call.
  double equity(const Move& move, const Board& board, int bag_size, const Rack& opp_rack,
                TurnLeaves& leaves) const;

  // Batched static-equity evaluation for a full legal-play list.
  //
  // Builds a per-turn leave table for `my_rack` once, then reads each move's
  // leave value in O(1) via Move::leave_mask() -- no sorting, hashing, or
  // per-move leave reconstruction.  Intended for both HastyBot and human UI
  // annotations to share one optimized implementation.
  std::vector<double> equities(const std::vector<Move>& moves, const Board& board, int bag_size,
                               const Rack& opp_rack, const Rack& my_rack) const;

  // Fill out[k] with the max leave value over every size-k sub-multiset of
  // `my_rack`, for k in [0, my_rack.size()] (out[k] = -inf for k > rack size).
  // A play that places e tiles leaves a size-(rack - e) rack, so pairing a
  // per-tile-count score bound with out[rack - e] gives a tight equity bound for
  // shadow-play pruning. Computed once per turn.
  void best_leaves_by_size(const Rack& my_rack, std::array<double, RACK_SIZE + 1>& out) const;

  // Max pre-endgame adjustment for a play that places `tiles_played` tiles with
  // `bag_size` tiles in the bag (0 outside the pre-endgame table's range).
  double peg_for_tiles(int tiles_played, int bag_size) const;

  // Raw leave equity for a specific leave (the tiles kept after a play). Used by
  // the WordMap best-first loop to bound a single subrack's plays before probing.
  double leave_value(const Rack& leave) const { return leave_values_.lookup(leave); }

  // The leave-value table, for callers that price many leaves at once via its
  // incremental cursor (cheaper than one hash per leave).
  const LeaveValues& leave_table() const { return leave_values_; }

 private:
  HastyEquity() = default;

  LeaveValues leave_values_;
  std::vector<double> peg_table_;  // indexed by bag_after_play + 7
  bool ready_ = false;
};

}  // namespace scribblez

#include "inlines/lexicon/hasty_equity.inl"
