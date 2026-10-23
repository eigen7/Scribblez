// sim/hasty_rollout_task's CPU reference: the leaves it records for the leaf
// eval model, checked against the racks each rollout drew and the moves it
// reports.

#include "agent/hasty_bot.h"
#include "game/board.h"
#include "game/game.h"
#include "game/game_log.h"
#include "game/move.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "sim/hasty_rollout_task.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <vector>

using namespace scribblez;

namespace {

bool have_data() {
  if (!std::ifstream(SCRIBBLEZ_DEFAULT_KWG).good()) return false;
  if (!std::ifstream(HastyEquity::default_leaves_path("NWL23")).good()) return false;
  HastyEquity::ensure_initialized("NWL23");
  return true;
}

const Dictionary& nwl23() {
  static const Dictionary dict = Dictionary::load_kwg(SCRIBBLEZ_DEFAULT_KWG);
  return dict;
}

// The position after `plies` plies of HastyBot self-play, from the side of
// the last mover.
HastyRolloutPosition position_after(const Dictionary& dict, uint64_t seed, int plies) {
  HastyBot p0({.thread_id = 0, .name = "p0"});
  HastyBot p1({.thread_id = 0, .name = "p1"});
  Game game(p0, p1, dict, seed);
  game.set_max_plies(plies);
  game.play();
  const GameLog log = game.log();
  const int us = log.records[log.num_records - 1].player;
  return make_rollout_position(game.board(), game.leave(us), game.score(us), game.score(1 - us),
                               seed);
}

// `rack` without the tiles `m` takes from it.
Rack after_move(Rack rack, const rollout_abi::PackedMove& m) {
  const Move move = unpack_move(m);
  for (int i = 0; i < move.num_glyphs(); ++i) EXPECT_TRUE(rack.remove(move.glyph(i).rack_tile()));
  return rack;
}

}  // namespace

TEST(HastyRolloutCpu, LeavesAreTheRacksAfterEachMoveBeforeItsDraw) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo leaves";
  const Dictionary& dict = nwl23();
  const HastyRolloutPosition pos = position_after(dict, /*seed=*/1, /*plies=*/10);
  std::vector<CpuLeaf> leaves(16);
  const std::vector<rollout_abi::RolloutResult> results =
    run_hasty_rollouts_cpu(pos, dict, /*n=*/32, /*threads=*/2, &leaves);
  for (uint32_t i = 0; i < leaves.size(); ++i) {
    const RolloutRacks racks = rollout_racks(pos, i);
    const rollout_abi::RolloutResult& r = results[i];
    const CpuLeaf& leaf = leaves[i];
    EXPECT_EQ(leaf.leaves[1], after_move(racks.opp, r.moves[0])) << "rollout " << i;
    EXPECT_EQ(leaf.leaves[0], after_move(racks.mine, r.moves[1])) << "rollout " << i;
    EXPECT_EQ(leaf.scores[0] - leaf.scores[1], r.diff) << "rollout " << i;
    Board board = pos.board;
    for (const rollout_abi::PackedMove& m : r.moves) {
      const Move move = unpack_move(m);
      if (move.type() == MoveType::PLAY) board.apply(move);
    }
    EXPECT_EQ(leaf.board.to_string(), board.to_string()) << "rollout " << i;
  }
}
