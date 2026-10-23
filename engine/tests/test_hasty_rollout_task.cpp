// sim/hasty_rollout_task's CPU reference and sim/rollout_values's CPU
// backend: the deal (known opponent leaves, refills, an exchange's returned
// tiles), the pairing of a group's starts, the leaves recorded for the leaf
// eval model, and the outcomes.

#include "agent/hasty_bot.h"
#include "game/board.h"
#include "game/game.h"
#include "game/game_log.h"
#include "game/move.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "sim/hasty_rollout_task.h"
#include "sim/rollout_values.h"
#include "sim/sim_runner.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
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

// The decision point after `plies` plies of HastyBot self-play.
struct Decision {
  Board board;
  Rack rack;
  int my_score = 0, opp_score = 0;
};

Decision decision_after(const Dictionary& dict, uint64_t seed, int plies) {
  HastyBot p0({.thread_id = 0, .name = "p0"});
  HastyBot p1({.thread_id = 0, .name = "p1"});
  Game game(p0, p1, dict, seed);
  game.set_max_plies(plies);
  game.play();
  const GameLog log = game.log();
  const int mover = 1 - log.records[log.num_records - 1].player;
  Decision d{game.board(), game.rack(mover), game.score(mover), game.score(1 - mover)};
  d.board.ensure_movegen_caches(dict);
  return d;
}

std::vector<Move> ranked_moves(const Decision& d, const Dictionary& dict) {
  const MoveRequest req{
    d.board, dict, d.rack, Rack{}, d.my_score, d.opp_score, d.board.pov_bag_size(d.rack.size())};
  return equity_top_k(req, std::numeric_limits<int>::max());
}

// A group of `moves` from `d`, `rollouts` each; rollout i seats `leaves[i]`.
RolloutBatch batch_of(const Decision& d, const std::vector<Move>& moves, int rollouts,
                      std::vector<Rack> leaves = {}) {
  RolloutBatch batch;
  batch.groups.push_back(make_rollout_group(d.board, d.rack, /*seed=*/7));
  batch.groups[0].opp_leaves = std::move(leaves);
  for (const Move& m : moves) {
    batch.starts.push_back(make_rollout_start(d.board, d.rack, d.my_score, d.opp_score, m, 0));
    batch.add_rollouts(uint32_t(batch.starts.size() - 1), 0, uint32_t(rollouts));
  }
  return batch;
}

// `rack` without the tiles `m` takes from it.
Rack after_move(Rack rack, const rollout_abi::PackedMove& m) {
  const Move move = unpack_move(m);
  for (int i = 0; i < move.num_glyphs(); ++i) EXPECT_TRUE(rack.remove(move.glyph(i).rack_tile()));
  return rack;
}

bool contains(const Rack& rack, const Rack& part) {
  for (int k = 0; k < TILE_KINDS; ++k)
    if (rack.count(Tile::of(k)) < part.count(Tile::of(k))) return false;
  return true;
}

}  // namespace

TEST(HastyRolloutCpu, LeavesAreTheRacksAfterEachMoveBeforeItsDraw) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo leaves";
  const Dictionary& dict = nwl23();
  const Decision d = decision_after(dict, /*seed=*/1, /*plies=*/10);
  const RolloutBatch batch = batch_of(d, {ranked_moves(d, dict).front()}, /*rollouts=*/32);
  std::vector<CpuLeaf> leaves(16);
  const std::vector<rollout_abi::RolloutResult> results =
    run_hasty_rollouts_cpu(batch, dict, /*threads=*/2, &leaves);
  for (size_t i = 0; i < leaves.size(); ++i) {
    const RolloutRacks racks = rollout_racks(batch, batch.tasks[i]);
    const rollout_abi::RolloutResult& r = results[i];
    const CpuLeaf& leaf = leaves[i];
    EXPECT_EQ(leaf.leaves[1], after_move(racks.opp, r.moves[0])) << "rollout " << i;
    EXPECT_EQ(leaf.leaves[0], after_move(racks.mine, r.moves[1])) << "rollout " << i;
    EXPECT_EQ(leaf.scores[0] - leaf.scores[1], r.diff) << "rollout " << i;
    Board board = batch.starts[0].board;
    for (const rollout_abi::PackedMove& m : r.moves) {
      const Move move = unpack_move(m);
      if (move.type() == MoveType::PLAY) board.apply(move);
    }
    EXPECT_EQ(leaf.board.to_string(), board.to_string()) << "rollout " << i;
  }
}

TEST(HastyRolloutCpu, AGroupsStartsDrawTheSameOpponentRack) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo leaves";
  const Dictionary& dict = nwl23();
  const Decision d = decision_after(dict, /*seed=*/2, /*plies=*/8);
  const std::vector<Move> ranked = ranked_moves(d, dict);
  const std::vector<Move> moves(ranked.begin(), ranked.begin() + 4);
  const RolloutBatch batch = batch_of(d, moves, /*rollouts=*/8);
  for (uint32_t i = 0; i < 8; ++i) {
    const Rack opp = rollout_racks(batch, {0, i}).opp;
    EXPECT_EQ(opp.size(), RACK_SIZE);
    for (uint32_t s = 1; s < moves.size(); ++s)
      EXPECT_EQ(rollout_racks(batch, {s, i}).opp, opp) << "start " << s << " rollout " << i;
  }
}

TEST(HastyRolloutCpu, AKnownLeaveIsSeatedAndTheRestDrawn) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo leaves";
  const Dictionary& dict = nwl23();
  const Decision d = decision_after(dict, /*seed=*/3, /*plies=*/8);
  const TileCounts pool = d.board.unseen_tiles(d.rack);
  Rack leave;
  for (int k = 0; k < TILE_KINDS && leave.size() < 3; ++k)
    if (pool.count(Tile::of(k)) > 0) leave.add(Tile::of(k));
  const RolloutBatch batch =
    batch_of(d, {ranked_moves(d, dict).front()}, /*rollouts=*/4, {leave, Rack{}, leave});
  for (uint32_t i = 0; i < 4; ++i) {
    const RolloutRacks racks = rollout_racks(batch, {0, i});
    EXPECT_EQ(racks.opp.size(), RACK_SIZE) << "rollout " << i;
    if (i == 0 || i == 2) EXPECT_TRUE(contains(racks.opp, leave)) << "rollout " << i;
  }
}

TEST(HastyRolloutCpu, AnExchangesTilesAreNotRedrawnAtOnce) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo leaves";
  const Dictionary& dict = nwl23();
  const Decision d = decision_after(dict, /*seed=*/4, /*plies=*/4);
  const std::vector<Move> ranked = ranked_moves(d, dict);
  const auto it = std::find_if(ranked.begin(), ranked.end(),
                               [](const Move& m) { return m.type() == MoveType::EXCHANGE; });
  ASSERT_NE(it, ranked.end());
  const RolloutBatch batch = batch_of(d, {*it}, /*rollouts=*/16);
  const Rack& returned = batch.starts[0].returned;
  ASSERT_EQ(returned.size(), it->num_glyphs());
  const TileCounts pool = batch.groups[0].pool;
  for (uint32_t i = 0; i < 16; ++i) {
    const RolloutRacks racks = rollout_racks(batch, {0, i});
    EXPECT_EQ(racks.mine.size(), RACK_SIZE);
    // Both racks come from the pool alone, which lacks the returned tiles
    // unless other copies are unseen.
    TileCounts drawn;
    for (const Rack* r : {&racks.opp, &racks.mine})
      for (int j = 0; j < r->size(); ++j) drawn.add(r->tiles()[size_t(j)]);
    for (int j = 0; j < batch.starts[0].leave.size(); ++j) {
      const Tile t = batch.starts[0].leave.tiles()[size_t(j)];
      drawn.add(t, -1);
    }
    for (int k = 0; k < TILE_KINDS; ++k)
      EXPECT_LE(drawn.count(Tile::of(k)), pool.count(Tile::of(k))) << "rollout " << i;
  }
}

TEST(HastyRolloutCpu, ATasksOutcomeDoesNotDependOnItsBatch) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo leaves";
  const Dictionary& dict = nwl23();
  const Decision d = decision_after(dict, /*seed=*/5, /*plies=*/10);
  const std::vector<Move> ranked = ranked_moves(d, dict);
  const RolloutBatch whole = batch_of(d, {ranked[0], ranked[1]}, /*rollouts=*/12);
  RolloutBatch part = whole;
  part.tasks = {{1, 9}, {0, 3}};
  const auto all = run_hasty_rollouts_cpu(whole, dict, /*threads=*/3);
  const auto some = run_hasty_rollouts_cpu(part, dict, /*threads=*/1);
  EXPECT_EQ(std::memcmp(&some[0], &all[12 + 9], sizeof(some[0])), 0);
  EXPECT_EQ(std::memcmp(&some[1], &all[3], sizeof(some[1])), 0);
}

TEST(RolloutValues, TheCpuBackendValuesLeavesAndFinishedGames) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo leaves";
  const Dictionary& dict = nwl23();
  const Decision d = decision_after(dict, /*seed=*/6, /*plies=*/10);
  const RolloutBatch batch = batch_of(d, {ranked_moves(d, dict).front()}, /*rollouts=*/8);
  CpuRolloutValues values(dict, /*threads=*/2, synthetic_leaf());
  const std::vector<RolloutOutcome> outcomes = values.run(batch);
  const std::vector<rollout_abi::RolloutResult> results =
    run_hasty_rollouts_cpu(batch, dict, /*threads=*/1);
  ASSERT_EQ(outcomes.size(), results.size());
  for (size_t k = 0; k < outcomes.size(); ++k) {
    EXPECT_EQ(outcomes[k].result.diff, results[k].diff);
    const double v = outcomes[k].win_value();
    EXPECT_GT(v, 0.0);
    EXPECT_LT(v, 1.0);
    EXPECT_EQ(v > 0.5, results[k].diff > 0) << "task " << k;
  }
  EXPECT_EQ(terminal_wld(3), (Wld{1, 0, 0}));
  EXPECT_EQ(terminal_wld(0), (Wld{0, 1, 0}));
  EXPECT_EQ(terminal_wld(-1), (Wld{0, 0, 1}));
}
