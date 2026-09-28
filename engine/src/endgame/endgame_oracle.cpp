#include "endgame/endgame_oracle.h"

#include "agent/agent.h"
#include "agent/hasty_bot.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <array>

namespace scribblez {

namespace {

// The final spread for the side to move if both sides play HastyBot's move to
// the end, under the solver's rules: going out collects twice the other rack,
// and two consecutive scoreless turns end the game with each side docked its
// own rack.
int32_t greedy_playout_spread(const EndgameState& state) {
  Board board = state.board;
  std::array<Rack, 2> racks = {state.my_rack, state.opp_rack};
  std::array<int, 2> scores = {state.my_score, state.opp_score};
  int scoreless = state.scoreless_turns > 0 ? 1 : 0;
  for (int on = 0;; on = 1 - on) {
    const MoveRequest req{board,         *state.dict, racks[on],
                          racks[1 - on], scores[on],  scores[1 - on],
                          /*bag_size=*/0};
    const Move m = hasty_best_move_wmp(req);
    for (int i = 0; i < m.num_glyphs(); ++i) racks[on].remove(m.glyph(i).rack_tile());
    if (m.type() == MoveType::PLAY) {
      board.apply(m);
      scores[on] += m.score();
      scoreless = 0;
      if (racks[on].empty()) {
        scores[on] += 2 * racks[1 - on].point_value();
        break;
      }
    } else if (++scoreless >= 2) {
      for (int p = 0; p < 2; ++p) scores[p] -= racks[p].point_value();
      break;
    }
  }
  return scores[0] - scores[1];
}

}  // namespace

EndgameVerdict SolverEndgameOracle::evaluate(const EndgameState& state, int effort,
                                             EndgameGoal goal) {
  const EndgameResult r = solver_.solve(
    state, {.budget = budget_, .plies = effort, .spread_matters = goal == EndgameGoal::kSpread});
  if (r.depth_completed < 1) return {.spread = greedy_playout_spread(state)};
  return {.cls = r.proven_class, .spread = r.value, .proven = r.proven};
}

}  // namespace scribblez
