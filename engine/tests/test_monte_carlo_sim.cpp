// Monte-Carlo ground truth under the two leave conditions
// (sim/monte_carlo_sim.h). Gated on the lexicon mount, like the other
// HastyBot-driven tests.

#include "data/gcg_post_move.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "sim/monte_carlo_sim.h"
#include "util/io.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>

namespace scribblez {
namespace {

namespace fs = std::filesystem;

class MonteCarloSimTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string kwg = SCRIBBLEZ_DEFAULT_KWG;
    const std::string leaves = HastyEquity::default_leaves_path("NWL23");
    if (!fs::exists(kwg) || !fs::exists(leaves)) GTEST_SKIP() << "no NWL23 kwg/leaves";
    dict_ = std::make_unique<Dictionary>(Dictionary::load_kwg(kwg));
    HastyEquity::ensure_initialized("NWL23");
  }

  // The fixture position: Hasty_2 bingoed INCASED, Hasty_1 then played GAVE --
  // the final mover is the POV, and the opponent kept nothing.
  static std::string postbingo_text() {
    const fs::path p = fs::path(SCRIBBLEZ_TEST_DATA_DIR) / "postbingo-gave.gcg";
    std::string text = util::read_file(p.string());
    EXPECT_FALSE(text.empty()) << "cannot read " << p;
    return text;
  }

  // The same game with its last two moves dropped: Hasty_2 just played .O from
  // ACEINOS keeping ACEINS, then Hasty_1 played .OJI. A large known leave.
  static std::string two_moves_earlier() {
    std::string text = postbingo_text();
    for (int i = 0; i < 2; ++i) {
      const size_t cut = text.find_last_of('\n', text.size() - 2);
      text.erase(cut + 1);
    }
    return text;
  }

  ParsedGcgPostMove parse(const std::string& text) {
    ParsedGcgPostMove pos;
    std::string error;
    EXPECT_TRUE(read_gcg_post_move(text, &pos, &error)) << error;
    return pos;
  }

  MonteCarloResult run(const ParsedGcgPostMove& pos, LeaveCondition condition, int n = 40) {
    return run_monte_carlo(pos, *dict_, n, /*threads=*/4, condition);
  }

  std::unique_ptr<Dictionary> dict_;
};

void expect_same(const MonteCarloResult& a, const MonteCarloResult& b) {
  EXPECT_EQ(a.n, b.n);
  EXPECT_EQ(a.wins, b.wins);
  EXPECT_EQ(a.losses, b.losses);
  EXPECT_EQ(a.draws, b.draws);
  EXPECT_EQ(a.delta_hist, b.delta_hist);
  EXPECT_EQ(a.placement.opp_next, b.placement.opp_next);
}

// After a bingo nothing is known face-up and there is nothing to infer, so
// the two conditions are the same rollouts -- what lets the tool write one
// truth under both names for such a position.
TEST_F(MonteCarloSimTest, ConditionsCoincideAfterABingo) {
  const ParsedGcgPostMove pos = parse(postbingo_text());
  ASSERT_TRUE(pos.opp_leave.empty());
  expect_same(run(pos, LeaveCondition::kHidden), run(pos, LeaveCondition::kFaceUp));
}

// The rollouts are seeded per game, so a truth is a pure function of the
// position, the condition, and the seeds -- not of the thread split.
TEST_F(MonteCarloSimTest, DeterministicAcrossThreadCounts) {
  const ParsedGcgPostMove pos = parse(two_moves_earlier());
  ASSERT_EQ(pos.opp_leave.to_string(), "ACEINS");
  for (const LeaveCondition c : {LeaveCondition::kHidden, LeaveCondition::kFaceUp}) {
    expect_same(run_monte_carlo(pos, *dict_, 40, 1, c), run_monte_carlo(pos, *dict_, 40, 7, c));
  }
}

// With a known leave the conditions see different opponents: face-up seats
// ACEINS every rollout; hidden draws from the posterior its last move induces.
TEST_F(MonteCarloSimTest, AKnownLeaveSeparatesTheConditions) {
  const ParsedGcgPostMove pos = parse(two_moves_earlier());
  ASSERT_TRUE(pos.opp_observation.has_value());
  const MonteCarloResult hidden = run(pos, LeaveCondition::kHidden, 200);
  const MonteCarloResult face_up = run(pos, LeaveCondition::kFaceUp, 200);
  EXPECT_EQ(hidden.n, 200);
  EXPECT_EQ(face_up.n, 200);
  EXPECT_NE(hidden.delta_hist, face_up.delta_hist);
}

}  // namespace
}  // namespace scribblez
