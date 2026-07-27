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
// (bit i == the i-th sorted rack tile) over which a move's leave is a compact
// mask, and fills leave values lazily, so each distinct leave is looked up at
// most once per turn. Build one per turn and reuse it across the turn's moves.
class TurnLeaves {
 public:
  TurnLeaves(const Rack& rack, const LeaveValues& lv);

  // The full rack minus the move's played tiles, each of which claims its
  // letter's lowest still-available bit.
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

// HastyBot's static equity for a move, matching Macondo's four-calculator
// stack: leave value, opening adjustment, pre-endgame adjustment, and endgame
// adjustment. Load once with init(); equity() is then safe to call
// concurrently.
class HastyEquity {
 public:
  static HastyEquity& instance();

  // Exactly once before any equity() call. Throws on I/O failure.
  static void init(const std::string& klv2_path, const std::string& peg_json_path);

  // init() with `lexicon`'s default files, for callers that just want
  // HastyBot's defaults; a no-op once loaded. Call during single-threaded
  // setup: it must not race against concurrent equity() calls.
  static void ensure_initialized(const std::string& lexicon);

  // Derived from Macondo's data layout: the per-lexicon leave-values file
  // (.../strategy/<lexicon>/leaves.klv2) and the shared pre-endgame table
  // (.../strategy/default/preendgame.json).
  static std::string default_leaves_path(const std::string& lexicon);
  static std::string default_peg_path();

  //   bag_size : tiles in the bag *before* the move
  //   my_rack  : the mover's rack *before* the move, from which the leave comes
  //   opp_rack : read only by the endgame adjustment
  double equity(const Move& move, const Board& board, int bag_size, const Rack& opp_rack,
                const Rack& my_rack) const;

  TurnLeaves turn_leaves(const Rack& my_rack) const;

  // Bit-identical to equity(), reading the leave from `leaves` rather than
  // reconstructing it.
  double equity(const Move& move, const Board& board, int bag_size, const Rack& opp_rack,
                TurnLeaves& leaves) const;

  // Batched static-equity evaluation for a full legal-play list, sharing one
  // TurnLeaves across the moves so no leave is priced twice. The path both
  // HastyBot and the human UI's move annotations take.
  std::vector<double> equities(const std::vector<Move>& moves, const Board& board, int bag_size,
                               const Rack& opp_rack, const Rack& my_rack) const;

  // out[k] is the max leave value over every size-k sub-multiset of `my_rack`,
  // and -inf past the rack size. A play placing e tiles leaves a
  // size-(rack - e) rack, so pairing a per-tile-count score bound with
  // out[rack - e] gives a tight equity bound for shadow-play pruning.
  void best_leaves_by_size(const Rack& my_rack, std::array<double, RACK_SIZE + 1>& out) const;

  // 0 outside the pre-endgame table's range.
  double peg_for_tiles(int tiles_played, int bag_size) const;

  double leave_value(const Rack& leave) const { return leave_values_.lookup(leave); }

  // For callers pricing many leaves at once through its incremental cursor,
  // cheaper than one hash per leave.
  const LeaveValues& leave_table() const { return leave_values_; }

 private:
  HastyEquity() = default;

  LeaveValues leave_values_;
  std::vector<double> peg_table_;  // indexed by bag_after_play + 7
  bool ready_ = false;
};

}  // namespace scribblez

#include "inlines/lexicon/hasty_equity.inl"
