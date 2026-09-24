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

// Per-turn leave cache for one mover's rack. A leave is a 7-bit mask over the
// sorted rack (bit i = the i-th tile), and each distinct leave's value is
// looked up at most once, on first use. Build one per turn and share it across
// that turn's moves.
class TurnLeaves {
 public:
  TurnLeaves(const Rack& rack, const LeaveValues& lv);

  // The rack minus the move's played tiles. Duplicate letters clear their bits
  // lowest first, so equal leaves always get equal masks.
  uint8_t mask_for(const Move& move) const;

  double value(uint8_t mask);

  int point_value(uint8_t mask);

 private:
  void ensure(uint8_t mask);

  const LeaveValues& lv_;
  int size_;
  std::array<Tile, RACK_SIZE> tile_of_bit_{};
  std::array<uint8_t, TILE_KINDS> indices_{};
  uint8_t full_ = 0;
  std::array<float, 128> value_{};
  std::array<int16_t, 128> pv_{};
  std::bitset<128> computed_{};
};

// HastyBot's static equity for a move: score plus Macondo's four equity
// adjustments (leave value, opening, pre-endgame, endgame). A process-wide
// singleton; after init(), all const methods are safe to call concurrently.
class HastyEquity {
 public:
  static HastyEquity& instance();

  // Call once before any equity query. Throws if either file can't be read or
  // parsed. An empty `peg_json_path` opts out of the pre-endgame adjustment.
  static void init(const std::string& klv2_path, const std::string& peg_json_path);

  // init() with `lexicon`'s default files; a no-op once loaded. Call during
  // single-threaded setup, since it must not race with equity queries.
  static void ensure_initialized(const std::string& lexicon);

  // Paths into the Macondo checkout: the per-lexicon leave values
  // (strategy/<lexicon>/leaves.klv2) and the shared pre-endgame table
  // (strategy/default/preendgame.json).
  static std::string default_leaves_path(const std::string& lexicon);
  static std::string default_peg_path();

  // `bag_size` and `my_rack` are as of before the move; `opp_rack` matters only
  // to the endgame adjustment. For pricing many moves, prefer equities() or
  // the TurnLeaves overload.
  double equity(const Move& move, const Board& board, int bag_size, const Rack& opp_rack,
                const Rack& my_rack) const;

  TurnLeaves turn_leaves(const Rack& my_rack) const;

  // Bit-identical to the Rack overload, with the leave read from `leaves`.
  double equity(const Move& move, const Board& board, int bag_size, const Rack& opp_rack,
                TurnLeaves& leaves) const;

  // Equities for a list of moves, sharing one TurnLeaves across them.
  std::vector<double> equities(const std::vector<Move>& moves, const Board& board, int bag_size,
                               const Rack& opp_rack, const Rack& my_rack) const;

  // out[k] is the best leave value over all size-k sub-multisets of `my_rack`
  // (-1e18 for k beyond the rack size). A play placing e tiles keeps
  // rack - e, so out[rack - e] plus a score bound for e tiles bounds that
  // play's equity; MacondoBot's shadow pruning relies on this.
  void best_leaves_by_size(const Rack& my_rack, std::array<double, RACK_SIZE + 1>& out) const;

  // The pre-endgame adjustment for a play of `tiles_played` tiles; 0 outside
  // the table's range.
  double peg_for_tiles(int tiles_played, int bag_size) const;

  double leave_value(const Rack& leave) const { return leave_values_.lookup(leave); }

  // Exposed for LeaveValues' incremental cursor, which prices many leaves
  // more cheaply than one lookup() each.
  const LeaveValues& leave_table() const { return leave_values_; }

 private:
  HastyEquity() = default;

  LeaveValues leave_values_;
  std::vector<double> peg_table_;  // indexed by (bag size after the play) + 7
  bool ready_ = false;
};

}  // namespace scribblez

#include "inlines/lexicon/hasty_equity.inl"
