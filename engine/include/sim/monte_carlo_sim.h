#pragma once

#include "belief/rack_inference.h"
#include "data/gcg_post_move.h"
#include "game/board.h"
#include "game/move.h"
#include "lexicon/dictionary.h"

#include <boost/json.hpp>

#include <array>
#include <map>
#include <string>

namespace scribblez {

// What a rollout knows about the opponent's leave: the tiles their last move
// kept. Their replenishment draw is hidden under both conditions. A model
// trained under one condition is measured against ground truth computed under
// the same one.
enum class LeaveCondition {
  // The leave is public: every rollout seats the opponent with it.
  kFaceUp,
  // Each rollout samples the leave from the posterior belief::RackInferrer
  // infers from the opponent's last move. When that move carries no
  // information (a bingo, a pass, or no recorded move), the whole rack is a
  // uniform draw from the unseen pool.
  kHidden,
};

// "face-up-leaves" / "hidden-leaves", the suffix of the condition's results file.
const char* leave_condition_name(LeaveCondition condition);

// Per-square placement counts over the rollouts, in board frame. These are the
// ground truth for the position-evaluation model's four placement heads.
// "opp" is the seat that moves first in the rollout; "self" is start_player.
// A `*_next` plane counts the rollouts in which that seat's first move covered
// the square; its `*_win` plane counts only the rollouts that seat strictly won.
// Occupied squares stay zero.
struct PlacementCounts {
  static constexpr int kCells = BOARD_SIZE * BOARD_SIZE;
  std::array<int, kCells> opp_next{};
  std::array<int, kCells> self_next{};
  std::array<int, kCells> opp_win{};
  std::array<int, kCells> self_win{};
};

// Fold one rollout's two first moves into `out`. A non-PLAY move covers nothing.
//
// `opp_first` is credited with the squares it places. `self_first` is credited
// with its footprint (training/footprint.h) decoded on `board`, the position
// before the opponent's move, as if the opponent had passed. The model never
// sees the opponent's move, and collapse_footprint_planes decodes its
// predictions the same way. Crediting literal squares instead would leave a
// residual the model cannot learn on every reply that plays through the
// opponent's new tiles.
void accumulate_rollout_placement(const Board& board, const Move& opp_first, bool opp_won,
                                  const Move& self_first, bool self_won, PlacementCounts& out);

// Monte-Carlo ground truth for one position, from start_player's point of view.
struct MonteCarloResult {
  int start_player = 0;
  int n = 0;
  int wins = 0;
  int losses = 0;
  int draws = 0;
  std::map<int, int> delta_hist;  // final score delta (own - opponent) -> rollout count
  PlacementCounts placement;

  boost::json::object to_json() const;
};

// Play `n` rollouts from `pos` to the end of the game, with EndgameHastyBot on
// both sides at its self-play defaults. The opponent's leave is seated per
// `condition`, and face-up rollouts play face-up-leaves rules, the rules their
// training games use. `infer` configures leave inference under kHidden.
//
// Rollout g is seeded by g, so the result is deterministic and independent of
// `threads`.
MonteCarloResult run_monte_carlo(const ParsedGcgPostMove& pos, const Dictionary& dict, int n,
                                 int threads, LeaveCondition condition,
                                 const belief::RackInferrer::Params& infer = {});

}  // namespace scribblez
