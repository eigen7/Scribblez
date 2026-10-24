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
#include <type_traits>
#include <utility>
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

// Every tile a task dealt (both racks, less our leave) came out of its
// group's pool: none is dealt twice, and no returned tile is redrawn.
void expect_dealt_from_pool(const RolloutBatch& batch, const rollout_abi::RolloutTask& task) {
  const RolloutRacks racks = rollout_racks(batch, task);
  const RolloutStart& start = batch.starts[task.position];
  const TileCounts& pool = batch.groups[start.group].pool;
  TileCounts dealt;
  for (const Rack* r : {&racks.opp, &racks.mine})
    for (int j = 0; j < r->size(); ++j) dealt.add(r->tiles()[size_t(j)]);
  for (int j = 0; j < start.leave.size(); ++j) dealt.add(start.leave.tiles()[size_t(j)], -1);
  for (int k = 0; k < TILE_KINDS; ++k)
    EXPECT_LE(dealt.count(Tile::of(k)), pool.count(Tile::of(k)))
      << "tile " << k << " rollout " << task.rollout;
}

// The first decision of games `seed`, `seed` + 1, ... with the bag holding
// at least a rack and the mover holding a tile none of which is unseen.
std::pair<Decision, Tile> decision_with_last_copy(const Dictionary& dict, uint64_t seed) {
  for (;; ++seed) {
    const Decision d = decision_after(dict, seed, /*plies=*/4);
    const TileCounts pool = d.board.unseen_tiles(d.rack);
    if (pool.size() < 2 * RACK_SIZE) continue;
    for (int j = 0; j < d.rack.size(); ++j) {
      const Tile t = d.rack.tiles()[size_t(j)];
      if (pool.count(t) == 0) return {d, t};
    }
  }
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
  // Tiles with a single unseen copy: seating one without taking it out of
  // the pool would let the draw deal it a second time.
  const TileCounts pool = d.board.unseen_tiles(d.rack);
  Rack leave;
  for (int k = 0; k < TILE_KINDS && leave.size() < 3; ++k)
    if (pool.count(Tile::of(k)) == 1) leave.add(Tile::of(k));
  ASSERT_FALSE(leave.empty());
  // Even rollouts below 48 seat it; odd ones, and those past the leaves,
  // draw the opponent's whole rack.
  std::vector<Rack> leaves(48);
  for (size_t i = 0; i < leaves.size(); i += 2) leaves[i] = leave;
  const RolloutBatch batch = batch_of(d, {ranked_moves(d, dict).front()}, /*rollouts=*/64, leaves);
  for (uint32_t i = 0; i < 64; ++i) {
    const RolloutRacks racks = rollout_racks(batch, {0, i});
    EXPECT_EQ(racks.opp.size(), RACK_SIZE) << "rollout " << i;
    if (i < leaves.size() && i % 2 == 0) {
      EXPECT_TRUE(contains(racks.opp, leave)) << "rollout " << i;
    }
    expect_dealt_from_pool(batch, {0, i});
  }
}

TEST(HastyRolloutCpu, AnExchangesTilesAreNotRedrawnAtOnce) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo leaves";
  const Dictionary& dict = nwl23();
  // Exchanging the last copy of a tile: the pool has none, so any deal that
  // holds it redrew a returned tile.
  const auto [d, last] = decision_with_last_copy(dict, /*seed=*/4);
  TileCounts tiles;
  tiles.add(last);
  const RolloutBatch batch = batch_of(d, {Move::exchange(tiles)}, /*rollouts=*/64);
  ASSERT_EQ(batch.starts[0].returned.size(), 1);
  for (uint32_t i = 0; i < 64; ++i) {
    EXPECT_EQ(rollout_racks(batch, {0, i}).mine.size(), RACK_SIZE) << "rollout " << i;
    expect_dealt_from_pool(batch, {0, i});
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
  static_assert(std::has_unique_object_representations_v<rollout_abi::RolloutResult>,
                "memcmp compares results field by field only without padding");
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

TEST(RolloutValues, LogitsBecomeWinDrawLossProbabilities) {
  const float even[3] = {0, 0, 0};
  const Wld w = wld_from_logits(even);
  EXPECT_FLOAT_EQ(w[0], 1.0f / 3);
  EXPECT_FLOAT_EQ(w[1], 1.0f / 3);
  const float huge[3] = {1000, 0, -1000};  // finite only if the max is subtracted
  EXPECT_EQ(wld_from_logits(huge), (Wld{1, 0, 0}));
  const float ordered[3] = {-1, 0, 2};
  const Wld o = wld_from_logits(ordered);
  EXPECT_LT(o[0], o[1]);
  EXPECT_LT(o[1], o[2]);
  EXPECT_FLOAT_EQ(o[0] + o[1] + o[2], 1.0f);
}

TEST(HastyRolloutCpu, AMoveThatGoesOutWithTheBagEmptyEndsTheGame) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo leaves";
  const Dictionary& dict = nwl23();
  const Decision d = decision_after(dict, /*seed=*/6, /*plies=*/18);
  ASSERT_LE(d.board.unseen_tiles(d.rack).size(), RACK_SIZE) << "the bag should be empty";
  const std::vector<Move> ranked = ranked_moves(d, dict);
  const auto out = std::ranges::find_if(ranked, [&](const Move& m) {
    return m.type() == MoveType::PLAY && m.num_glyphs() == d.rack.size();
  });
  ASSERT_NE(out, ranked.end()) << "no move plays out";
  const RolloutBatch batch = batch_of(d, {*out}, /*rollouts=*/4);
  const std::vector<RolloutOutcome> outcomes =
    CpuRolloutValues(dict, /*threads=*/1, synthetic_leaf()).run(batch);
  for (size_t i = 0; i < outcomes.size(); ++i) {
    const RolloutOutcome& o = outcomes[i];
    const Rack opp = rollout_racks(batch, batch.tasks[i]).opp;
    EXPECT_EQ(opp.size(), batch.groups[0].pool.size()) << "the opponent holds every unseen tile";
    const int bonus = 2 * opp.point_value();
    const int diff = d.my_score + out->score() + bonus - d.opp_score;
    EXPECT_EQ(o.result.game_over, 1) << "rollout " << i;
    EXPECT_EQ(o.result.diff, diff) << "rollout " << i;
    EXPECT_EQ(unpack_move(o.result.moves[0]).type(), MoveType::PASS) << "rollout " << i;
    EXPECT_EQ(o.wld, terminal_wld(diff)) << "rollout " << i;
  }
}
