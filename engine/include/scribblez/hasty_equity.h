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

  // Static HastyBot equity for `move` in the given position.
  //   bag_size : tiles remaining in the bag *before* the move
  //   my_rack  : on-move player's rack (used to compute the leave)
  //   opp_rack : opponent's rack (used only for the endgame adjustment)
  double equity(const Move& move, const Board& board, int bag_size, const Rack& my_rack,
                const Rack& opp_rack) const;

 private:
  HastyEquity() = default;

  LeaveValues leave_values_;
  std::vector<double> peg_table_;  // indexed by bag_after_play + 7
  bool ready_ = false;
};

}  // namespace scribblez
