// Monte-Carlo ground truth (sim/monte_carlo_sim.h): the per-square placement
// planes, and the rollouts under the two leave conditions. The rollout tests
// need the NWL23 lexicon and leaves on the data mount and skip without them.

#include "data/gcg_post_move.h"
#include "game/board.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "sim/monte_carlo_sim.h"
#include "util/io.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace scribblez {
namespace {

namespace fs = std::filesystem;

Glyph G(int letter_index) { return Glyph::of(Tile::of(letter_index)); }

int cell(int r, int c) { return r * BOARD_SIZE + c; }

// The self planes credit a reply's footprint decoded on the pre-reply board,
// not its literal squares. The opponent plays a lone tile at (7,8); self then
// plays down column 8 through it, placing (5,8), (6,8) and (8,8). On the
// pre-reply board (7,8) is still empty, so the footprint (anchor (5,8),
// vertical, 3 tiles) decodes to (5,8), (6,8), (7,8): the third tile lands on
// the opponent's square rather than the one past it, matching how the model's
// placement heads decode a footprint.
TEST(PlacementPlanes, SelfReplyIsProjectedOntoThePreReplyBoard) {
  Board board;
  board.set(7, 7, G(4));  // the position's one tile
  Glyph one[1] = {G(0)};
  const Move opp = Move::play(true, 7, uint16_t(1u << 8), 0, one, 1);  // (7,8)
  Glyph three[3] = {G(1), G(2), G(3)};
  const Move self = Move::play(false, 8, uint16_t((1u << 5) | (1u << 6) | (1u << 8)), 0, three, 3);

  PlacementCounts out;
  accumulate_rollout_placement(board, opp, /*opp_won=*/false, self, /*self_won=*/true, out);

  std::array<int, PlacementCounts::kCells> opp_next{}, self_next{};
  opp_next[cell(7, 8)] = 1;
  self_next[cell(5, 8)] = 1;
  self_next[cell(6, 8)] = 1;
  self_next[cell(7, 8)] = 1;  // projected: not (8,8)
  EXPECT_EQ(out.opp_next, opp_next);
  EXPECT_EQ(out.self_next, self_next);
  EXPECT_EQ(out.opp_win, (std::array<int, PlacementCounts::kCells>{}));
  EXPECT_EQ(out.self_win, self_next);
}

// A pass, an exchange, or a rollout that ended before a seat moved covers
// nothing on either plane.
TEST(PlacementPlanes, NonPlaysCoverNothing) {
  Board board;
  PlacementCounts out;
  accumulate_rollout_placement(board, Move::pass(), true, Move::pass(), false, out);
  EXPECT_EQ(out.opp_next, (std::array<int, PlacementCounts::kCells>{}));
  EXPECT_EQ(out.self_next, (std::array<int, PlacementCounts::kCells>{}));
}

class MonteCarloSimTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string kwg = SCRIBBLEZ_DEFAULT_KWG;
    const std::string leaves = HastyEquity::default_leaves_path("NWL23");
    if (!fs::exists(kwg) || !fs::exists(leaves)) GTEST_SKIP() << "no NWL23 kwg/leaves";
    dict_ = std::make_unique<Dictionary>(Dictionary::load_kwg(kwg));
    HastyEquity::ensure_initialized("NWL23");
  }

  // Hasty_2 bingoed INCASED, then Hasty_1 (the POV) played GAVE; the opponent
  // kept nothing.
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

// After a bingo the opponent kept nothing, so there is nothing to show face-up
// or to infer and the two conditions must produce identical rollouts.
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
