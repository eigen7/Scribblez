#pragma once

#include "game/board.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"

#include <boost/json.hpp>

#include <array>
#include <map>
#include <string>

namespace scribblez {

// The analysis position parsed from a penultimate-bingo GCG: the board and
// scores after the final recorded move, plus the seat that made it -- the POV
// the position evaluation model evaluates -- and its leave. The other seat
// bingoed on the penultimate move, so in a rollout it draws a clean full rack
// and, having moved earlier, plays first.
struct MonteCarloPosition {
  Board board;
  std::array<int, 2> scores{0, 0};
  int start_player = 0;  // seat that made the final move (the evaluated POV)
  Rack leave;            // start_player's leave = final rack_before minus the placed tiles
};

// False (with *error set, when non-null) if the GCG has no turns or its final
// move is not a tile play.
bool parse_monte_carlo_position(const std::string& gcg_text, MonteCarloPosition* out,
                                std::string* error);

// Per-square placement counts over the rollouts, from `start_player`'s POV, in
// board frame. Mirrors the position-evaluation model's four placement heads: in
// how many rollouts that seat's first move placed a tile on the square, and (the
// `*_win` planes) did so in a rollout it strictly won. "opp" is the seat to move
// first in the rollout; "self" is start_player. Each `*_win` plane is
// elementwise at most its `*_next` plane, and occupied board squares stay zero.
struct PlacementCounts {
  static constexpr int kCells = BOARD_SIZE * BOARD_SIZE;
  std::array<int, kCells> opp_next{};
  std::array<int, kCells> self_next{};
  std::array<int, kCells> opp_win{};
  std::array<int, kCells> self_win{};
};

// A Monte-Carlo ground-truth result for one position, from `start_player`'s
// POV, whose delta is start_player_final - opponent_final.
struct MonteCarloResult {
  int start_player = 0;
  int n = 0;
  int wins = 0;
  int losses = 0;
  int draws = 0;
  std::map<int, int> delta_hist;  // score delta -> number of rollouts with that delta
  PlacementCounts placement;

  boost::json::object to_json() const;
};

// Play `n` HastyBot-vs-HastyBot rollouts from `pos` to a natural game end. Game
// g is seeded by g, so the aggregate is deterministic and independent of how
// the games spread across the workers.
MonteCarloResult run_monte_carlo(const MonteCarloPosition& pos, const Dictionary& dict, int n,
                                 int threads);

}  // namespace scribblez
