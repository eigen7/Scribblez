#pragma once

#include "scribblez/board.h"
#include "scribblez/leave_values.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <string>
#include <vector>

namespace scribblez {

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

  // Batched static-equity evaluation for a full legal-play list.
  //
  // Builds a per-turn leave table for `my_rack` once, then reads each move's
  // leave value in O(1) via Move::leave_mask() -- no sorting, hashing, or
  // per-move leave reconstruction.  Intended for both HastyBot and human UI
  // annotations to share one optimized implementation.
  std::vector<double> equities(const std::vector<Move>& moves, const Board& board, int bag_size,
                               const Rack& opp_rack, const Rack& my_rack) const;

 private:
  HastyEquity() = default;

  LeaveValues leave_values_;
  std::vector<double> peg_table_;  // indexed by bag_after_play + 7
  bool ready_ = false;
};

}  // namespace scribblez
