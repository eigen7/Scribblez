// LeafBot on the CPU reference with the synthetic leaf (the suites hide the
// GPU): it plays whole games, deterministically for a seed; plays the
// candidate the leaf values best, whatever its static equity; values each at
// the position after it, over opponent leaves of the length the opponent
// kept, empty before the opponent's first move; asks the leave inference
// model at each modelled turn; and values a candidate that empties the bag
// like any other. Skips without the NWL23 lexicon and HastyBot's leave
// values.

#include "agent/hasty_bot.h"
#include "agent/leaf_bot.h"
#include "game/game.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/leave_value_arrays.h"
#include "lexicon/leave_values.h"
#include "nwl23_equity.h"
#include "sim/monte_carlo_leaf_values.h"
#include "sim/rollout_values.h"
#include "sim/sim_runner.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <mutex>
#include <span>
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

std::shared_ptr<const LeaveValueArrays> leave_table() {
  static const auto table = std::make_shared<const LeaveValueArrays>(
    LeaveValues::load(HastyEquity::default_leaves_path("NWL23")));
  return table;
}

LeafBot::Params small_params(uint64_t seed) {
  return {.thread_id = 0,
          .name = "LeafBot",
          .leaf_model = "synthetic",
          .candidates = 4,
          .opp_leaves = 3,
          .seed = seed};
}

// The CPU reference's synthetic leaf, recording every leaf it is asked about.
class RecordingValues : public RolloutValues {
 public:
  RecordingValues() : inner_(nwl23(), synthetic_leaf()) {}

  BatchValues run(const RolloutBatch& batch) override { return inner_.run(batch); }
  std::vector<LeafPrediction> predict_leaves(std::span<const CpuLeaf* const> leaves) override;

  std::vector<CpuLeaf> seen;

 private:
  CpuRolloutValues inner_;
  std::mutex mutex_;
};

std::vector<LeafPrediction> RecordingValues::predict_leaves(
  std::span<const CpuLeaf* const> leaves) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const CpuLeaf* leaf : leaves) seen.push_back(*leaf);
  }
  return inner_.predict_leaves(leaves);
}

// A leaf that wins every position whose board is `favoured` and loses the
// rest.
class FavouringValues : public RolloutValues {
 public:
  explicit FavouringValues(Board favoured) : favoured_(std::move(favoured)) {}

  BatchValues run(const RolloutBatch&) override { return {}; }
  std::vector<LeafPrediction> predict_leaves(std::span<const CpuLeaf* const> leaves) override;

 private:
  Board favoured_;
};

bool same_tiles(const Board& a, const Board& b) {
  for (int r = 0; r < BOARD_SIZE; ++r)
    for (int c = 0; c < BOARD_SIZE; ++c)
      if (!(a.at(r, c) == b.at(r, c))) return false;
  return true;
}

std::vector<LeafPrediction> FavouringValues::predict_leaves(
  std::span<const CpuLeaf* const> leaves) {
  std::vector<LeafPrediction> out(leaves.size());
  for (size_t i = 0; i < leaves.size(); ++i)
    out[i].wld = same_tiles(leaves[i]->board, favoured_) ? Wld{1, 0, 0} : Wld{0, 0, 1};
  return out;
}

// A leaf that loses every position whose opponent holds the Q and wins the
// rest, from the side of the player who moved.
class QHolderLoses : public RolloutValues {
 public:
  BatchValues run(const RolloutBatch&) override { return {}; }
  std::vector<LeafPrediction> predict_leaves(std::span<const CpuLeaf* const> leaves) override;
};

std::vector<LeafPrediction> QHolderLoses::predict_leaves(std::span<const CpuLeaf* const> leaves) {
  std::vector<LeafPrediction> out(leaves.size());
  for (size_t i = 0; i < leaves.size(); ++i) {
    const bool q = leaves[i]->leaves[1].count(Tile::from_char('Q')) > 0;
    out[i].wld = q ? Wld{0, 0, 1} : Wld{1, 0, 0};
  }
  return out;
}

// A noisy leaf that loses everywhere, recording each call's size.
class NoisyRecorder : public LeafValues {
 public:
  std::vector<Wld> predict_wld(std::span<const CpuLeaf* const> leaves) override;
  bool noisy() const override { return true; }

  std::vector<size_t> sizes;
};

std::vector<Wld> NoisyRecorder::predict_wld(std::span<const CpuLeaf* const> leaves) {
  sizes.push_back(leaves.size());
  return std::vector<Wld>(leaves.size(), Wld{0, 0, 1});
}

// A leaf that loses where the opponent holds the Q and wins elsewhere, as
// QHolderLoses, recording each call's size.
class CallRecorder : public LeafValues {
 public:
  std::vector<Wld> predict_wld(std::span<const CpuLeaf* const> leaves) override;

  std::vector<size_t> sizes;
};

std::vector<Wld> CallRecorder::predict_wld(std::span<const CpuLeaf* const> leaves) {
  sizes.push_back(leaves.size());
  std::vector<Wld> out;
  out.reserve(leaves.size());
  for (const CpuLeaf* leaf : leaves)
    out.push_back(leaf->leaves[1].count(Tile::from_char('Q')) > 0 ? Wld{0, 0, 1} : Wld{1, 0, 0});
  return out;
}

LeafBot::Backend backend(std::shared_ptr<LeafValues> values, belief::LeaveParamsFn predict = {}) {
  return {.values = std::move(values), .predict = std::move(predict), .table = leave_table()};
}

// A game of `bot` against HastyBot from a fixed deal, `bot` moving first iff
// `first`; returns its moves.
std::vector<Move> play(Agent& bot, bool first = true) {
  HastyBot hasty({.thread_id = 0, .name = "HastyBot"});
  const std::unique_ptr<Game> g = first ? std::make_unique<Game>(bot, hasty, nwl23(), /*seed=*/1)
                                        : std::make_unique<Game>(hasty, bot, nwl23(), /*seed=*/1);
  g->play();
  const GameLogStorage log = g->extract_log();
  std::vector<Move> moves;
  for (const TurnRecord& t : log.turns) moves.push_back(t.move);
  return moves;
}

}  // namespace

TEST(LeafBot, PlaysWholeGamesTheSameForASeed) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  LeafBot a(small_params(3),
            backend(std::make_shared<CpuRolloutValues>(nwl23(), synthetic_leaf())));
  LeafBot b(small_params(3),
            backend(std::make_shared<CpuRolloutValues>(nwl23(), synthetic_leaf())));
  const std::vector<Move> first = play(a);
  EXPECT_GT(first.size(), 10u);
  EXPECT_EQ(play(b), first);
}

// Static equity only chooses the candidates; the leaf chooses among them.
TEST(LeafBot, PlaysTheCandidateTheLeafFavours) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const Board board;
  const Rack rack = Rack::from_string("AEINRST");
  const Rack hidden;
  const MoveRequest req{board, nwl23(), rack, hidden, 0, 0, 86};
  const std::vector<Move> ranked = equity_top_k(req, 4);
  ASSERT_EQ(ranked.size(), 4u);
  Board favoured = board;
  favoured.apply(ranked[3]);
  LeafBot bot(small_params(1), backend(std::make_shared<FavouringValues>(favoured)));
  EXPECT_EQ(bot.make_move(req).move, ranked[3]);
}

TEST(LeafBot, ValuesEachCandidateAtThePositionAfterIt) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const Board board;
  const Rack rack = Rack::from_string("AEINRST");
  const Rack hidden;
  const MoveRequest req{board, nwl23(), rack, hidden, 30, 12, 86};
  const std::vector<Move> moves = equity_top_k(req, 2);
  const std::vector<WeightedLeave> opp = {{Rack::from_string("QU"), 0.25},
                                          {Rack::from_string("EE"), 0.75}};
  auto values = std::make_shared<RecordingValues>();
  LeafBot bot(small_params(1), backend(values));
  EXPECT_EQ(bot.values(req, moves, opp).size(), moves.size());
  ASSERT_EQ(values->seen.size(), 4u);
  for (size_t i = 0; i < 4; ++i) {
    const Move& m = moves[i / 2];
    const CpuLeaf& leaf = values->seen[i];
    Rack leave = rack;
    for (int g = 0; g < m.num_glyphs(); ++g) leave.remove(m.glyph(g).rack_tile());
    EXPECT_EQ(leaf.leaves[0], leave);
    EXPECT_EQ(leaf.leaves[1], opp[i % 2].leave);
    EXPECT_EQ(leaf.scores[0], 30 + m.score());
    EXPECT_EQ(leaf.scores[1], 12);
    EXPECT_EQ(leaf.board.num_tiles(), m.num_glyphs());
  }
}

// Moving first, the opponent has kept nothing yet, as the leaf's training rows
// have it; after its first move, the leaves are its rack less what it played.
TEST(LeafBot, DrawsTheOpponentsLeavesAtTheLengthItKept) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  auto first = std::make_shared<RecordingValues>();
  LeafBot a(small_params(1), backend(first));
  const std::vector<Move> moves = play(a, /*first=*/true);
  ASSERT_FALSE(first->seen.empty());
  EXPECT_EQ(first->seen.front().leaves[1].size(), 0);

  auto second = std::make_shared<RecordingValues>();
  LeafBot b(small_params(1), backend(second));
  const std::vector<Move> replies = play(b, /*first=*/false);
  ASSERT_FALSE(second->seen.empty());
  // HastyBot's opening move kept RACK_SIZE minus the tiles it played.
  EXPECT_EQ(second->seen.front().leaves[1].size(), RACK_SIZE - replies.front().num_glyphs());
}

TEST(LeafBot, AsksTheModelForTheOpponentsLeaveAtEachModelledTurn) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  int calls = 0;
  belief::LeaveParamsFn predict = [&](const float*, int count, belief::LeaveParams* out) {
    ++calls;
    ASSERT_EQ(count, 1);
    out[0] = belief::LeaveParams{};
  };
  LeafBot bot(small_params(4),
              backend(std::make_shared<CpuRolloutValues>(nwl23(), synthetic_leaf()), predict));
  play(bot);
  EXPECT_GE(calls, 5);  // every turn but the first, a bingo's or a pass's, and the late ones
}

TEST(LeafBot, ItsDecisionReportsEachCandidatesValue) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const Board board;
  const Rack rack = Rack::from_string("AEINRST");
  const Rack hidden;
  const MoveRequest req{board, nwl23(), rack, hidden, 0, 0, 86};
  const std::vector<Move> ranked = equity_top_k(req, 4);
  Board favoured = board;
  favoured.apply(ranked[2]);
  LeafBot bot(small_params(1), backend(std::make_shared<FavouringValues>(favoured)));
  const MoveDecision d = bot.make_move(req);
  ASSERT_EQ(d.values.size(), ranked.size());
  for (size_t i = 0; i < ranked.size(); ++i) {
    EXPECT_EQ(d.values[i].move, ranked[i]);
    EXPECT_DOUBLE_EQ(d.values[i].value, i == 2 ? 1.0 : 0.0);
  }
  EXPECT_EQ(d.move, ranked[2]);
}

// A candidate that empties the bag is valued like any other.
TEST(LeafBot, ValuesTheCandidatesThatEmptyTheBag) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const Board board;
  const Rack rack = Rack::from_string("AEINRST");
  const Rack hidden;
  const MoveRequest req{board, nwl23(), rack, hidden, 0, 0, 3};
  const std::vector<Move> ranked = equity_top_k(req, 4);
  ASSERT_TRUE(std::ranges::any_of(ranked, [](const Move& m) { return m.num_glyphs() >= 3; }));
  Board favoured = board;
  favoured.apply(ranked[1]);
  LeafBot bot(small_params(1), backend(std::make_shared<FavouringValues>(favoured)));
  const MoveDecision d = bot.make_move(req);
  ASSERT_EQ(d.values.size(), ranked.size());
  EXPECT_EQ(d.move, ranked[1]);
}

// A value is the leaf's weighted mean over the leaves: with a leaf that wins
// only where the opponent holds the Q, the leave weights are the value.
TEST(LeafBot, AValueIsTheLeafsMeanOverTheLeavesWeighted) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const Board board;
  const Rack rack = Rack::from_string("AEINRST");
  const Rack hidden;
  const MoveRequest req{board, nwl23(), rack, hidden, 0, 0, 86};
  const std::vector<Move> moves = equity_top_k(req, 1);
  LeafBot bot(small_params(1), backend(std::make_shared<QHolderLoses>()));
  const std::vector<double> v =
    bot.values(req, moves, {{Rack::from_string("QU"), 0.25}, {Rack::from_string("EE"), 0.75}});
  ASSERT_EQ(v.size(), 1u);
  EXPECT_DOUBLE_EQ(v[0], 0.75);
}

// The leaf is asked whole candidates at a time, no more than kLeavesPerCall
// leaves a call, and each candidate's value is as one call would have it.
TEST(LeafBot, AsksTheLeafAFewCandidatesAtATime) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const Board board;
  const Rack rack = Rack::from_string("AEINRST");
  const Rack hidden;
  const MoveRequest req{board, nwl23(), rack, hidden, 0, 0, 86};
  const std::vector<Move> moves = equity_top_k(req, 5);
  ASSERT_EQ(moves.size(), 5u);
  // Two candidates' leaves fit in a call, a quarter of them holding the Q.
  const size_t n = LeafBot::kLeavesPerCall / 2 - 1;
  std::vector<WeightedLeave> opp;
  for (size_t i = 0; i < n; ++i)
    opp.push_back({Rack::from_string(i % 4 == 0 ? "QU" : "EE"), 1.0 / double(n)});
  auto recorder = std::make_shared<CallRecorder>();
  LeafBot bot(small_params(1), backend(recorder));
  const std::vector<double> v = bot.values(req, moves, opp);
  EXPECT_EQ(recorder->sizes, (std::vector<size_t>{2 * n, 2 * n, n}));
  const double expected = double(n - (n + 3) / 4) / double(n);
  for (double x : v) EXPECT_NEAR(x, expected, 1e-12);
}

TEST(LeafBot, RepeatedDrawsAreValuedOnceWeightedByTheirCount) {
  const std::vector<WeightedLeave> w =
    weighted_leaves({Rack::from_string("AB"), Rack::from_string("CD"), Rack::from_string("BA"),
                     Rack::from_string("AB")});
  ASSERT_EQ(w.size(), 2u);
  EXPECT_EQ(w[0].leave, Rack::from_string("AB"));
  EXPECT_DOUBLE_EQ(w[0].weight, 0.75);
  EXPECT_DOUBLE_EQ(w[1].weight, 0.25);
}

TEST(LeafBot, EqualWeightsKeepEveryDraw) {
  const std::vector<WeightedLeave> w =
    equal_weights({Rack::from_string("AB"), Rack::from_string("CD"), Rack::from_string("AB")});
  ASSERT_EQ(w.size(), 3u);
  EXPECT_EQ(w[2].leave, Rack::from_string("AB"));
  for (const WeightedLeave& x : w) EXPECT_DOUBLE_EQ(x.weight, 1.0 / 3.0);
}

// Under a noisy leaf, every draw is valued, repeats included: each call holds
// the candidates times the draws, even where the opponent's leave is known to
// be empty (LeafBot's first turn here), which is played out once per draw.
TEST(LeafBot, ANoisyLeafValuesEveryDraw) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  auto noisy = std::make_shared<NoisyRecorder>();
  LeafBot::Params p = small_params(1);
  p.opp_leaves = 64;  // far more draws than one-tile leaves the pool can hold
  LeafBot bot(p, backend(noisy));
  play(bot);
  ASSERT_FALSE(noisy->sizes.empty());
  EXPECT_TRUE(std::ranges::any_of(noisy->sizes, [](size_t n) { return n >= 64; }));
  EXPECT_EQ(noisy->sizes.front(), size_t(p.candidates) * 64);  // before the opponent's first move
  for (size_t n : noisy->sizes) EXPECT_EQ(n % 64, 0u) << n;
}

TEST(LeafBot, TheBestValueIsTheEarliestOnTies) {
  const Move m = Move::pass();
  EXPECT_EQ(best_value({{m, 0.2}, {m, 0.5}, {m, 0.5}, {m, 0.1}}), 1u);
  EXPECT_EQ(best_value({{m, 0.7}}), 0u);
}

// The monte-carlo leaf (sim/monte_carlo_leaf_values.h) plays each leaf out
// once: its values are the same on every call and follow a lead no game can
// overturn.
TEST(MonteCarloLeafValues, PlaysEachLeafOutToItsOutcome) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const Board board;
  const Rack rack = Rack::from_string("AEINRST");
  const Rack hidden;
  const MoveRequest req{board, nwl23(), rack, hidden, 0, 0, 86};
  const Move opening = equity_top_k(req, 1).front();
  const RolloutStart s = make_rollout_start(board, rack, 0, 0, opening, 0);
  const CpuLeaf ahead{s.board, {s.leave, Rack{}}, {1000, 0}};
  const CpuLeaf behind{s.board, {s.leave, Rack{}}, {0, 1000}};
  const std::vector<const CpuLeaf*> leaves{&ahead, &behind};
  MonteCarloLeafValues values(nwl23());
  EXPECT_TRUE(values.noisy());
  const std::vector<Wld> first = values.predict_wld(leaves);
  EXPECT_EQ(values.predict_wld(leaves), first);
  EXPECT_EQ(first[0], (Wld{1.0f, 0.0f, 0.0f}));
  EXPECT_EQ(first[1], (Wld{0.0f, 0.0f, 1.0f}));
}
