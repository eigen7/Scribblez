#include "sim/monte_carlo_leaf_values.h"

#include "agent/endgame_agent.h"
#include "agent/hasty_bot.h"
#include "game/game.h"
#include "game/tile_counts.h"
#include "util/assert.h"
#include "util/work_pool.h"

#include <string>

namespace scribblez {

namespace {

// Player 0's final score minus player 1's after one playout from `leaf`.
int playout(const CpuLeaf& leaf, const Dictionary& dict, Agent& a0, Agent& a1, uint64_t seed) {
  TileCounts pool = leaf.board.unseen_tiles(leaf.leaves[0]);
  const bool seated = pool.remove(leaf.leaves[1].counts());
  DEBUG_ASSERT(seated, "the opponent's leave is not in the unseen pool");
  Game game(a0, a1, dict, seed);
  game.set_face_up_leaves(false);
  game.play_from(leaf.board, leaf.scores, leaf.leaves, pool, /*to_move=*/1);
  const GameLog log = game.log();
  return log.final_scores[0] - log.final_scores[1];
}

// The self-play generator's agent (py/scribblez/selfplay.py's hastybot-endgame
// at its defaults), so the playouts follow the policy the leaf's targets do.
EndgameAgent<HastyBot>::Params agent_params(size_t lane, const std::string& name) {
  EndgameAgent<HastyBot>::Params p;
  p.base.thread_id = int(lane);
  p.base.name = name;
  return p;
}

}  // namespace

std::vector<Wld> MonteCarloLeafValues::predict_wld(std::span<const CpuLeaf* const> leaves) {
  const size_t games = size_t(games_);
  std::vector<int> diffs(leaves.size() * games);
  util::parallel_lanes(diffs.size(), [&](util::LaneCursor& cursor) {
    EndgameAgent<HastyBot> a0(agent_params(cursor.lane(), "P0"));
    EndgameAgent<HastyBot> a1(agent_params(cursor.lane(), "P1"));
    for (size_t i; cursor.next(i);)
      diffs[i] = playout(*leaves[i / games], dict_, a0, a1, uint64_t(i % games + 1));
  });
  std::vector<Wld> out(leaves.size(), Wld{});
  for (size_t i = 0; i < diffs.size(); ++i) {
    const Wld w = terminal_wld(diffs[i]);
    for (size_t k = 0; k < w.size(); ++k) out[i / games][k] += w[k] / float(games);
  }
  return out;
}

}  // namespace scribblez
