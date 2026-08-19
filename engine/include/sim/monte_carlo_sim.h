#pragma once

#include "belief/rack_inference.h"
#include "data/gcg_post_move.h"
#include "game/board.h"
#include "lexicon/dictionary.h"

#include <boost/json.hpp>

#include <array>
#include <map>
#include <string>

namespace scribblez {

// The information condition a ground truth is computed under -- what a rollout
// knows about the opponent's leave (the tiles their last move retained; their
// replenishment is hidden either way). The position-evaluation model trained
// under a condition is measured against the truth of that same condition.
enum class LeaveCondition {
  // The leave is public: every rollout seats the opponent with it.
  kFaceUp,
  // The leave is inferred from their last move (belief::RackInferrer, the
  // Macondo rangefinder port) and sampled per rollout from the posterior; when
  // the move carries no information (a bingo, a pass, no recorded move) the
  // whole rack is a uniform draw from the unseen pool.
  kHidden,
};

// "face-up-leaves" / "hidden-leaves": the suffix of the results file the
// condition's ground truth is committed under.
const char* leave_condition_name(LeaveCondition condition);

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

// Play `n` rollouts from `pos` to a natural game end, EndgameHastyBot vs
// EndgameHastyBot at the self-play defaults (greedy static equity until the
// bag empties, then class-only endgame solves), the opponent's leave seated
// per `condition` (a face-up rollout is also played as the face-up variant,
// the information condition its training games are generated under). Game g
// is seeded by g, so the aggregate is deterministic and independent of how the
// games spread across the workers. `infer` parameterizes the hidden
// condition's leave inference.
MonteCarloResult run_monte_carlo(const ParsedGcgPostMove& pos, const Dictionary& dict, int n,
                                 int threads, LeaveCondition condition,
                                 const belief::RackInferrer::Params& infer = {});

}  // namespace scribblez
