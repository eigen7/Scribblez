// DeepBot with the hasty reply model, on the CPU reference with the synthetic
// leaf (the suites hide the GPU): it plays whole games, deterministically for a
// seed, and asks the leave inference model for Alice's belief at every turn
// the model covers, from her own view. Skips without the NWL23
// lexicon and HastyBot's leave values.

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

#include <gtest/gtest.h>

#include <fstream>
#include <memory>
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

DeepBot::Params small_params(uint64_t seed) {
  return {.thread_id = 0,
          .name = "DeepBot",
          .leaf_model = "synthetic",
          .candidates = 3,
          .late_candidates = 4,
          .rollouts = 8,
          .seed = seed};
}

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
    using Encoder = LeaveInferenceInputEncoder;
    EXPECT_EQ(rows[Encoder::kSpatialFloats + Encoder::kCurrentScalars - 1], 0.0f);
    out[0] = belief::LeaveParams{};
  };
  DeepBot bot(small_params(4), cpu_backend(predict, table));
  play(bot);
  EXPECT_GE(calls, 5);  // every turn but the first, a bingo's or a pass's, and the endgame
}
