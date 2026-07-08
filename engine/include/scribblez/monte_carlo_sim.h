#pragma once

#include "scribblez/board.h"
#include "scribblez/dictionary.h"
#include "scribblez/rack.h"

#include <boost/json.hpp>

#include <array>
#include <map>
#include <string>

namespace scribblez {

// The analysis position parsed from a penultimate-bingo GCG: the board
// and cumulative scores after the final recorded move, plus the `start_player` --
// the seat that made that final move, whose POV the position evaluation model
// evaluates -- and its leave. The other seat (1 - start_player) bingoed on the
// penultimate move, so in a rollout it draws a clean full rack and, having moved
// earlier, plays first.
struct MonteCarloPosition {
  Board board;
  std::array<int, 2> scores{0, 0};
  int start_player = 0;  // seat that made the final move (the evaluated POV)
  Rack leave;            // start_player's leave = final rack_before minus the placed tiles
};

// Parse `gcg_text` into its post-move analysis position. Returns false (with *error
// set, if non-null) if the GCG has no turns or its final move is not a tile play.
bool parse_monte_carlo_position(const std::string& gcg_text, MonteCarloPosition* out,
                                std::string* error);

// A Monte-Carlo ground-truth result for one position, from `start_player`'s POV:
// W/L/D counts (summing to n) and the EXACT final-score-delta histogram (a delta is
// start_player_final - opponent_final).
struct MonteCarloResult {
  int start_player = 0;
  int n = 0;
  int wins = 0;
  int losses = 0;
  int draws = 0;
  std::map<int, int> delta_hist;  // score delta -> number of rollouts with that delta

  boost::json::object to_json() const;
};

// Play `n` HastyBot-vs-HastyBot rollouts from `pos` to a natural game end. Game g
// (for g in [1, n]) is seeded by g -- so the aggregate is fully deterministic and
// independent of how the games are spread across the `threads` workers.
MonteCarloResult run_monte_carlo(const MonteCarloPosition& pos, const Dictionary& dict, int n,
                                 int threads);

}  // namespace scribblez
