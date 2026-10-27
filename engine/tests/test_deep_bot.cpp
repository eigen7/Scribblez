// DeepBot under both reply models, on the CPU reference with the synthetic
// leaf (the suites hide the GPU): it plays whole games, deterministically for a
// seed; asks the leave inference model for Alice's belief from her own view
// and, in Bob's subtrees, for Bob's with her private past hidden; Bob replies
// from his own equity top N; and --subtree-dump writes a line per turn. Skips
// without the NWL23 lexicon and HastyBot's leave values.

#include "agent/deep_bot.h"
#include "agent/hasty_bot.h"
#include "belief/leave_params.h"
#include "encoding/leave_inference_input_encoder.h"
#include "game/game.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/leave_value_arrays.h"
#include "lexicon/leave_values.h"
#include "sim/rollout_values.h"
#include "temp_dir.h"

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
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

DeepBot::Params small_params(uint64_t seed, const std::string& reply = "subtree") {
  DeepBot::Params p{.thread_id = 0,
                    .name = "DeepBot",
                    .leaf_model = "synthetic",
                    .candidates = 2,
                    .late_candidates = 3,
                    .reply = reply,
                    .rollouts = 8,
                    .bob_racks = 3,
                    .sims_per_rack = 2,
                    .seed = seed};
  p.subtree = {.alice_racks = 3, .candidates = 2, .late_candidates = 2};
  return p;
}

using Encoder = LeaveInferenceInputEncoder;
constexpr size_t kHiddenFlag = Encoder::kSpatialFloats + Encoder::kCurrentScalars - 1;

DeepBot::Backend cpu_backend(belief::LeaveParamsFn predict = {},
                             std::shared_ptr<const LeaveValueArrays> table = {}) {
  return {std::make_unique<CpuRolloutValues>(nwl23(), synthetic_leaf()), std::move(predict),
          std::move(table)};
}

// A game of `deepbot` against HastyBot from a fixed deal; returns its moves.
std::vector<Move> play(DeepBot& deepbot) {
  HastyBot hasty({.thread_id = 0, .name = "HastyBot"});
  Game g(deepbot, hasty, nwl23(), /*seed=*/5);
  g.play();
  const GameLogStorage log = g.extract_log();
  std::vector<Move> moves;
  for (const TurnRecord& t : log.turns) moves.push_back(t.move);
  return moves;
}

}  // namespace

TEST(DeepBot, PlaysWholeGamesTheSameForASeed) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  DeepBot a(small_params(3), cpu_backend());
  DeepBot b(small_params(3), cpu_backend());
  const std::vector<Move> first = play(a);
  EXPECT_GT(first.size(), 10u);
  EXPECT_EQ(play(b), first);
  // The empty bag's turns fall back to HastyBot's move: the game ends.
}

TEST(DeepBot, AsksTheModelForAlicesBeliefAtEachModelledTurn) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  auto table = std::make_shared<const LeaveValueArrays>(
    LeaveValues::load(HastyEquity::default_leaves_path("NWL23")));
  int calls = 0;
  belief::LeaveParamsFn predict = [&](const float* rows, int count, belief::LeaveParams* out) {
    ++calls;
    ASSERT_EQ(count, 1);
    // Alice's own view of her position: her private history is not hidden.
    EXPECT_EQ(rows[kHiddenFlag], 0.0f);
    out[0] = belief::LeaveParams{};
  };
  DeepBot bot(small_params(4, "hasty"), cpu_backend(predict, table));
  play(bot);
  EXPECT_GE(calls, 5);  // every turn but the first, a bingo's or a pass's, and the endgame
}

// In Bob's subtrees the model is asked once per of Alice's candidates, for
// every rack she sampled for Bob at once, and each row hides Alice's private
// past: it is Alice's model of Bob's view.
TEST(DeepBot, AsksTheModelForBobsBeliefWithAlicesPastHidden) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  auto table = std::make_shared<const LeaveValueArrays>(
    LeaveValues::load(HastyEquity::default_leaves_path("NWL23")));
  int bob_rows = 0;
  belief::LeaveParamsFn predict = [&](const float* rows, int count, belief::LeaveParams* out) {
    for (int k = 0; k < count; ++k) {
      bob_rows += rows[size_t(k) * Encoder::kInputFloats + kHiddenFlag] == 1.0f;
      out[k] = belief::LeaveParams{};
    }
  };
  DeepBot bot(small_params(4), cpu_backend(predict, table));
  play(bot);
  EXPECT_GT(bob_rows, 10);
}

// Bob's reply at each rack is one of his own static-equity top N, chosen by
// his sims; the dump reports it per turn as one JSON line.
TEST(DeepBot, TheSubtreeDumpWritesALinePerSearchedTurn) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const auto dir = scribblez::testing::make_temp_dir("scribblez_test_subtree_dump");
  DeepBot::Params p = small_params(6);
  p.subtree_dump = (dir / "dump.jsonl").string();
  DeepBot bot(p, cpu_backend());
  play(bot);
  std::ifstream in(p.subtree_dump);
  int lines = 0;
  for (std::string line; std::getline(in, line); ++lines) {
    const boost::json::object turn = boost::json::parse(line).as_object();
    for (const boost::json::value& v : turn.at("candidates").as_array()) {
      const boost::json::object& c = v.as_object();
      const double value = c.at("subtree_value").as_double();
      EXPECT_TRUE(value >= 0.0 && value <= 1.0);
      if (!c.at("searched").as_bool()) continue;
      const boost::json::object& r = c.at("replies").as_object();
      const boost::json::array& ranks = r.at("rank_hist").as_array();
      EXPECT_LE(ranks.size(), size_t(p.subtree.late_candidates));
      int64_t racks = 0;
      for (const boost::json::value& n : ranks) racks += n.as_int64();
      EXPECT_EQ(racks, int64_t(p.bob_racks));
    }
  }
  EXPECT_GT(lines, 5);
  std::filesystem::remove_all(dir);
}
