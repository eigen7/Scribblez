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
#include "nwl23_equity.h"
#include "sim/hasty_rollout_task.h"
#include "sim/rollout_values.h"
#include "sim/sim_maps.h"
#include "sim/sim_runner.h"
#include "worker_budget.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

using namespace scribblez;

namespace {

bool have_data() {
  if (!std::ifstream(SCRIBBLEZ_DEFAULT_KWG).good()) return false;
  return scribblez::testing::install_nwl23_hasty_equity();
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

// The tiles a task's deal drew for us: our rack less our leave.
Rack our_refill(const RolloutBatch& batch, const rollout_abi::RolloutTask& task) {
  Rack drawn = rollout_racks(batch, task).mine;
  const Rack& leave = batch.starts[task.position].leave;
  for (int j = 0; j < leave.size(); ++j) EXPECT_TRUE(drawn.remove(leave.tiles()[size_t(j)]));
  return drawn;
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

constexpr uint64_t kBagSeed = 11;

// A bag of `n` tiles, tile i of kind i % kTileKinds, shuffled for `rollout`.
rollout_abi::Bag shuffled_bag(int n, uint32_t rollout) {
  rollout_abi::Bag bag{};
  for (int i = 0; i < n; ++i) bag.tiles[bag.back++] = uint8_t(i % rollout_abi::kTileKinds);
  bag.shuffle(kBagSeed, rollout);
  return bag;
}

std::vector<uint8_t> draw_n(rollout_abi::Bag& bag, int player, int n) {
  std::vector<uint8_t> out;
  for (int i = 0; i < n; ++i) out.push_back(bag.draw(player));
  return out;
}

std::vector<uint8_t> contents(const rollout_abi::Bag& bag) {
  return {bag.tiles + bag.front, bag.tiles + bag.back};
}

}  // namespace

// The opponent draws 2 then 3 tiles in one copy of a bag and 6 then 1 in the
// other, around a refill of ours: our tiles are the same, and the opponent's
// five are the first five of its seven.
TEST(RolloutBag, EachPlayersDrawsIgnoreTheOthers) {
  for (uint32_t rollout = 0; rollout < 16; ++rollout) {
    rollout_abi::Bag a = shuffled_bag(60, rollout);
    rollout_abi::Bag b = a;
    std::vector<uint8_t> a_opp = draw_n(a, 1, 2);
    const std::vector<uint8_t> a_us = draw_n(a, 0, 7);
    std::ranges::copy(draw_n(a, 1, 3), std::back_inserter(a_opp));
    std::vector<uint8_t> b_opp = draw_n(b, 1, 6);
    const std::vector<uint8_t> b_us = draw_n(b, 0, 7);
    std::ranges::copy(draw_n(b, 1, 1), std::back_inserter(b_opp));
    EXPECT_EQ(a_us, b_us) << "rollout " << rollout;
    EXPECT_TRUE(std::equal(a_opp.begin(), a_opp.end(), b_opp.begin())) << "rollout " << rollout;
  }
}

TEST(RolloutBag, APutBackChangesAtMostOneOfTheOtherPlayersTiles) {
  for (uint32_t rollout = 0; rollout < 16; ++rollout) {
    rollout_abi::Bag bag = shuffled_bag(60, rollout);
    rollout_abi::Bag untouched = bag;
    const std::vector<uint8_t> before = draw_n(untouched, 1, 20);
    for (uint8_t t : draw_n(bag, 0, 3)) bag.put_back(t, 0, kBagSeed, rollout);
    const std::vector<uint8_t> after = draw_n(bag, 1, 20);
    int changed = 0;
    for (size_t i = 0; i < before.size(); ++i) changed += before[i] != after[i];
    EXPECT_LE(changed, 3) << "rollout " << rollout;
  }
}

// Over 48000 rollouts of four distinct tiles: every shuffled order, and every
// order after we draw one and put back a fifth, about equally often (2000 per
// shuffled order and 500 per put-back order, standard deviations of about 44
// and 22, bounded here at four or more; the seed fixes the draws, so this
// cannot flake).
TEST(RolloutBag, ShufflesAndPutBacksLeaveEveryOrderEquallyLikely) {
  constexpr int kTrials = 48000;
  std::map<std::vector<uint8_t>, int> shuffled, put_back;
  for (uint32_t rollout = 0; rollout < kTrials; ++rollout) {
    rollout_abi::Bag bag = shuffled_bag(4, rollout);
    ++shuffled[contents(bag)];
    bag.draw(0);
    bag.put_back(4, 0, kBagSeed, rollout);
    ++put_back[contents(bag)];
  }
  ASSERT_EQ(shuffled.size(), 24u);
  for (const auto& [order, n] : shuffled) EXPECT_NEAR(n, kTrials / 24, 180);
  ASSERT_EQ(put_back.size(), 96u);
  for (const auto& [order, n] : put_back) EXPECT_NEAR(n, kTrials / 96, 100);
}

TEST(HastyRolloutCpu, LeavesAreTheRacksAfterEachMoveBeforeItsDraw) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo leaves";
  const Dictionary& dict = nwl23();
  const Decision d = decision_after(dict, /*seed=*/1, /*plies=*/10);
  const RolloutBatch batch = batch_of(d, {ranked_moves(d, dict).front()}, /*rollouts=*/32);
  std::vector<CpuLeaf> leaves(16);
  const std::vector<rollout_abi::RolloutResult> results =
    run_hasty_rollouts_cpu(batch, dict, &leaves);
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

// Rollout i's opponent rack is dealt first and our refill after it, from the
// other end: a start that plays more tiles draws more of the same sequence.
TEST(HastyRolloutCpu, AGroupsStartsDrawOurRefillFromOneSequence) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo leaves";
  const Dictionary& dict = nwl23();
  const Decision d = decision_after(dict, /*seed=*/2, /*plies=*/8);
  std::vector<Move> moves;
  for (const Move& m : ranked_moves(d, dict)) {
    const bool longer = moves.empty() || m.num_glyphs() > moves.back().num_glyphs();
    if (m.type() == MoveType::PLAY && longer) moves.push_back(m);
  }
  ASSERT_GE(moves.size(), 2u);
  const RolloutBatch batch = batch_of(d, moves, /*rollouts=*/8);
  for (uint32_t i = 0; i < 8; ++i) {
    for (uint32_t s = 1; s < moves.size(); ++s) {
      EXPECT_TRUE(contains(our_refill(batch, {s, i}), our_refill(batch, {s - 1, i})))
        << "start " << s << " rollout " << i;
    }
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
  const auto all = with_worker_budget(3, [&] { return run_hasty_rollouts_cpu(whole, dict); });
  const auto some = with_worker_budget(1, [&] { return run_hasty_rollouts_cpu(part, dict); });
  EXPECT_EQ(std::memcmp(&some[0], &all[12 + 9], sizeof(some[0])), 0);
  EXPECT_EQ(std::memcmp(&some[1], &all[3], sizeof(some[1])), 0);
}

TEST(RolloutValues, TheCpuBackendValuesLeavesAndFinishedGames) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo leaves";
  const Dictionary& dict = nwl23();
  const Decision d = decision_after(dict, /*seed=*/6, /*plies=*/10);
  const RolloutBatch batch = batch_of(d, {ranked_moves(d, dict).front()}, /*rollouts=*/8);
  CpuRolloutValues values(dict, synthetic_leaf());
  const BatchValues valued = values.run(batch);
  EXPECT_TRUE(valued.maps.empty()) << "the batch does not sum its maps";
  const std::vector<RolloutOutcome>& outcomes = valued.outcomes;
  const std::vector<rollout_abi::RolloutResult> results = run_hasty_rollouts_cpu(batch, dict);
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
  const Decision d = decision_after(dict, /*seed=*/6, /*plies=*/21);
  ASSERT_LE(d.board.unseen_tiles(d.rack).size(), RACK_SIZE) << "the bag should be empty";
  const std::vector<Move> ranked = ranked_moves(d, dict);
  const auto out = std::ranges::find_if(ranked, [&](const Move& m) {
    return m.type() == MoveType::PLAY && m.num_glyphs() == d.rack.size();
  });
  ASSERT_NE(out, ranked.end()) << "no move plays out";
  RolloutBatch batch = batch_of(d, {*out}, /*rollouts=*/4);
  batch.sum_maps = true;
  const BatchValues values = CpuRolloutValues(dict, synthetic_leaf()).run(batch);
  const std::vector<RolloutOutcome>& outcomes = values.outcomes;
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
  // The maps: we went out in every rollout, collecting the bonus, and no
  // square's fate is left open.
  ASSERT_EQ(values.maps.size(), 1u);
  const rollout_abi::SimMaps& m = values.maps[0];
  EXPECT_EQ(m.sims, 4.0f);
  EXPECT_EQ(m.went_out_sum[0], 4.0f);
  EXPECT_EQ(m.went_out_sum[1], 0.0f);
  EXPECT_EQ(m.went_out_sum[2], 0.0f);
  float bonus_sum = 0;
  for (size_t i = 0; i < outcomes.size(); ++i)
    bonus_sum += float(2 * rollout_racks(batch, batch.tasks[i]).opp.point_value());
  EXPECT_EQ(m.end_points_sum, bonus_sum);
  const Board& after = batch.starts[0].board;
  for (int sq = 0; sq < rollout_abi::kSquares; ++sq) {
    const bool empty = after.at(sq / BOARD_SIZE, sq % BOARD_SIZE).is_empty();
    float settled = 0;
    for (int c = 0; c < rollout_abi::kFateClasses; ++c) settled += m.class_sum[c][sq];
    EXPECT_EQ(settled, empty ? 4.0f : 0.0f) << "square " << sq;
    EXPECT_EQ(m.class_sum[rollout_abi::kFateStaysEmpty][sq], empty ? 4.0f : 0.0f)
      << "square " << sq;
  }
}

namespace {

// The fate class and signed points `m` by `mover` gives the squares it fills.
std::pair<int8_t, int16_t> expected_fate(const Move& m, int mover) {
  return {rollout_abi::placer_fate(mover, m.num_glyphs() == RACK_SIZE),
          int16_t(mover == 0 ? m.score() : -m.score())};
}

}  // namespace

// Mid-game, with tiles in the bag, a rollout is its two plies: the squares
// they fill are labelled by their mover and points, the start's tiles carry
// no label, every other square and the end of the game stay open.
TEST(HastyRolloutCpu, FatesLabelTheSquaresThePliesFillAndLeaveTheRestOpen) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo leaves";
  const Dictionary& dict = nwl23();
  const Decision d = decision_after(dict, /*seed=*/1, /*plies=*/10);
  const RolloutBatch batch = batch_of(d, {ranked_moves(d, dict).front()}, /*rollouts=*/16);
  std::vector<rollout_abi::RolloutFates> fates;
  const auto results = run_hasty_rollouts_cpu(batch, dict, nullptr, &fates);
  ASSERT_EQ(fates.size(), results.size());
  const Board& start = batch.starts[0].board;
  for (size_t k = 0; k < results.size(); ++k) {
    ASSERT_EQ(results[k].game_over, 0) << "rollout " << k;
    const rollout_abi::RolloutFates& f = fates[k];
    EXPECT_EQ(f.went_out, rollout_abi::kWentOutOpen);
    EXPECT_EQ(f.end_points, 0);
    std::array<std::pair<int8_t, int16_t>, rollout_abi::kSquares> expected;
    for (int sq = 0; sq < rollout_abi::kSquares; ++sq) {
      const bool empty = start.at(sq / BOARD_SIZE, sq % BOARD_SIZE).is_empty();
      expected[sq] = {empty ? rollout_abi::kFateOpen : rollout_abi::kFateFilledAtStart, 0};
    }
    for (int ply = 0; ply < 2; ++ply) {
      const Move m = unpack_move(results[k].moves[ply]);
      const int mover = ply == 0 ? 1 : 0;
      visit_placed_squares(
        m, [&](int r, int c) { expected[r * BOARD_SIZE + c] = expected_fate(m, mover); });
    }
    for (int sq = 0; sq < rollout_abi::kSquares; ++sq) {
      EXPECT_EQ(f.fate[sq], expected[sq].first) << "rollout " << k << " square " << sq;
      EXPECT_EQ(f.points[sq], expected[sq].second) << "rollout " << k << " square " << sq;
    }
  }
}

// The maps sum each start's own rollouts, in ascending task order, however
// the batch interleaves the starts' tasks.
TEST(SimMaps, SumEachStartsRolloutsInTaskOrder) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo leaves";
  const Dictionary& dict = nwl23();
  const Decision d = decision_after(dict, /*seed=*/3, /*plies=*/8);
  const std::vector<Move> ranked = ranked_moves(d, dict);
  RolloutBatch batch = batch_of(d, {ranked[0], ranked[1]}, /*rollouts=*/0);
  batch.sum_maps = true;
  for (uint32_t i = 0; i < 6; ++i) {
    batch.add_rollouts(i % 2, i, 1);
    if (i % 3 == 0) batch.add_rollouts(1, 10 + i, 1);
  }
  const TasksByStart by_start = tasks_by_start(batch);
  ASSERT_EQ(by_start.offsets, (std::vector<uint32_t>{0, 3, 8}));
  ASSERT_EQ(by_start.tasks, (std::vector<uint32_t>{0, 3, 6, 1, 2, 4, 5, 7}));
  std::vector<rollout_abi::RolloutFates> fates;
  run_hasty_rollouts_cpu(batch, dict, nullptr, &fates);
  const std::vector<rollout_abi::SimMaps> maps = sum_sim_maps(fates, by_start);
  ASSERT_EQ(maps.size(), 2u);
  const BatchValues values = CpuRolloutValues(dict, synthetic_leaf()).run(batch);
  ASSERT_EQ(values.maps.size(), 2u);
  for (size_t s = 0; s < 2; ++s)
    EXPECT_EQ(sim_maps_difference(values.maps[s], maps[s]), "") << "start " << s;
  for (size_t s = 0; s < 2; ++s) {
    const rollout_abi::SimMaps& m = maps[s];
    const float n = float(by_start.offsets[s + 1] - by_start.offsets[s]);
    EXPECT_EQ(m.sims, n);
    for (int sq = 0; sq < rollout_abi::kSquares; ++sq) {
      float settled = 0, points = 0, points_sq = 0, got = 0;
      for (int c = 0; c < rollout_abi::kFateClasses; ++c) got += m.class_sum[c][sq];
      for (uint32_t i = by_start.offsets[s]; i < by_start.offsets[s + 1]; ++i) {
        const rollout_abi::RolloutFates& f = fates[by_start.tasks[i]];
        if (f.fate[sq] < 0 || f.fate[sq] >= rollout_abi::kFateClasses) continue;
        settled += 1;
        points += float(f.points[sq]);
        points_sq += float(f.points[sq] * f.points[sq]);
      }
      EXPECT_EQ(got, settled) << "start " << s << " square " << sq;
      EXPECT_EQ(m.points_sum[sq], points) << "start " << s << " square " << sq;
      EXPECT_EQ(m.points_sq_sum[sq], points_sq) << "start " << s << " square " << sq;
    }
  }
}

namespace {

// A batch of `rollouts` of a play that draws the bag's last tiles without
// going out: the opponent then moves with the bag empty, so every rollout
// plays to the end. Has no tasks if no early game reaches such a play.
RolloutBatch bag_emptying_batch(const Dictionary& dict, int rollouts) {
  for (uint64_t seed = 1; seed < 64; ++seed) {
    const Decision d = decision_after(dict, seed, /*plies=*/20);
    const int bag = d.board.pov_bag_size(d.rack.size());
    if (bag < 1 || bag >= d.rack.size()) continue;
    const std::vector<Move> ranked = ranked_moves(d, dict);
    const auto it = std::ranges::find_if(ranked, [&](const Move& m) {
      return m.type() == MoveType::PLAY && m.num_glyphs() >= bag && m.num_glyphs() < d.rack.size();
    });
    if (it != ranked.end()) return batch_of(d, {*it}, rollouts);
  }
  return {};
}

}  // namespace

TEST(HastyRolloutCpu, ARolloutWhoseBagEmptiesIsPlayedToTheEnd) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo leaves";
  const Dictionary& dict = nwl23();
  const RolloutBatch batch = bag_emptying_batch(dict, /*rollouts=*/8);
  ASSERT_FALSE(batch.tasks.empty()) << "no game reached a play that empties the bag";
  const std::vector<RolloutOutcome> outcomes =
    CpuRolloutValues(dict, synthetic_leaf()).run(batch).outcomes;
  for (size_t i = 0; i < outcomes.size(); ++i) {
    EXPECT_EQ(outcomes[i].result.game_over, 1) << "rollout " << i;
    EXPECT_EQ(outcomes[i].wld, terminal_wld(outcomes[i].result.diff)) << "rollout " << i;
  }
}

// A finished rollout's end of game, against its final racks: whoever went
// out holds nothing and gains twice the other's rack, signed from player 0's
// side; in a stalemate each loses its own rack's value.
TEST(HastyRolloutCpu, FatesRecordWhoWentOutAndTheEndPoints) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo leaves";
  const Dictionary& dict = nwl23();
  const RolloutBatch batch = bag_emptying_batch(dict, /*rollouts=*/32);
  ASSERT_FALSE(batch.tasks.empty()) << "no game reached a play that empties the bag";
  std::vector<CpuLeaf> leaves(batch.tasks.size());
  std::vector<rollout_abi::RolloutFates> fates;
  run_hasty_rollouts_cpu(batch, dict, &leaves, &fates);
  std::array<int, rollout_abi::kWentOutClasses> seen{};
  for (size_t k = 0; k < fates.size(); ++k) {
    const rollout_abi::RolloutFates& f = fates[k];
    ASSERT_GE(f.went_out, 0) << "rollout " << k;
    ++seen[size_t(f.went_out)];
    const int v0 = leaves[k].leaves[0].point_value(), v1 = leaves[k].leaves[1].point_value();
    if (f.went_out == 2) {
      EXPECT_EQ(f.end_points, v1 - v0) << "rollout " << k;
      continue;
    }
    const int out = f.went_out;
    EXPECT_TRUE(leaves[k].leaves[size_t(out)].empty()) << "rollout " << k;
    const int bonus = 2 * (out == 0 ? v1 : v0);
    EXPECT_EQ(f.end_points, out == 0 ? bonus : -bonus) << "rollout " << k;
  }
  EXPECT_GT(seen[1], 0) << "no rollout's opponent went out";
}
