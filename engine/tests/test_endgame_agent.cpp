// EndgameHastyBotAgent: HastyBot while the bag holds tiles, the endgame solver
// once it is empty. Compiled into test_endgame alongside the solver suite.
// Every case needs the Macondo-bundled NWL23 leave table and skips without it.

#include "agent/endgame_hasty_bot.h"
#include "agent/macondo_bot.h"
#include "endgame/endgame_solver.h"
#include "endgame_positions.h"
#include "game/board.h"
#include "game/game.h"
#include "game/move.h"
#include "game/movegen.h"
#include "game/rack.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace scribblez;

namespace {

// Loads the NWL23 equity tables; false when the leaves file is absent.
bool ensure_equity() {
  const std::string leaves = HastyEquity::default_leaves_path("NWL23");
  if (!std::ifstream(leaves).good()) return false;
  HastyEquity::ensure_initialized("NWL23");
  return true;
}

// A canonical key for a play (placed squares and glyphs, plus score), for
// checking a move's membership in a generated list.
std::string move_key(const Move& m) {
  if (m.type() != MoveType::PLAY) return "PASS";
  struct Placement {
    int r, c, code;
  };
  std::vector<Placement> tiles;
  const bool horiz = m.horizontal();
  uint16_t mask = m.square_mask();
  int gi = 0;
  for (int pos = 0; mask; ++pos, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    const int r = horiz ? m.start() : pos;
    const int c = horiz ? pos : m.start();
    tiles.push_back({r, c, m.glyph(gi++).code()});
  }
  std::sort(tiles.begin(), tiles.end(), [](const Placement& a, const Placement& b) {
    if (a.r != b.r) return a.r < b.r;
    return a.c < b.c;
  });
  std::string k;
  char buf[32];
  for (const auto& t : tiles) {
    std::snprintf(buf, sizeof(buf), "%d,%d,%d;", t.r, t.c, t.code);
    k += buf;
  }
  std::snprintf(buf, sizeof(buf), "|%d", m.score());
  k += buf;
  return k;
}

std::set<std::string> key_set(const std::vector<Move>& ms) {
  std::set<std::string> s;
  for (const auto& m : ms) s.insert(move_key(m));
  return s;
}

// Recomputes a finished game's scores from its turn records and checks them
// against the log: score deltas sum to the last cumulative scores, and the
// end-of-game rack adjustment for the end reason yields final_scores.
void check_log_consistency(const GameLog& log) {
  ASSERT_GT(log.num_records, 0);
  std::array<int, 2> running = log.initial_scores;
  for (int i = 0; i < log.num_records; ++i)
    running[log.records[i].player] += log.records[i].score_delta;
  const TurnRecord& last = log.records[log.num_records - 1];
  ASSERT_EQ(running[0], last.cumulative_scores[0]);
  ASSERT_EQ(running[1], last.cumulative_scores[1]);

  const std::string reason = log.end_reason ? log.end_reason : "";
  ASSERT_TRUE(reason == "out" || reason == "stalemate" || reason == "max_turns") << reason;

  std::array<int, 2> expected = last.cumulative_scores;
  if (reason == "out") {
    const int o = last.player;  // the player who emptied their rack moved last
    expected[o] += 2 * log.final_racks[1 - o].point_value();
  } else if (reason == "stalemate") {
    for (int p = 0; p < 2; ++p) expected[p] -= log.final_racks[p].point_value();
  }
  ASSERT_EQ(log.final_scores[0], expected[0]);
  ASSERT_EQ(log.final_scores[1], expected[1]);
}

constexpr uint64_t kSolveBudget = 1ull << 20;
constexpr int kSolvePlies = 24;

// Forwards to an inner agent and counts how often the game loop prompts it,
// which makes projection fast-tracking observable from outside.
class PromptCountingAgent : public Agent {
 public:
  PromptCountingAgent(Agent& inner) : Agent(inner.thread_id(), inner.name()), inner_(inner) {}
  MoveDecision make_move(const MoveRequest& req) override {
    ++prompts;
    return inner_.make_move(req);
  }
  void observe_move(const Move& move) override { inner_.observe_move(move); }
  void begin_game(const BeginGameRequest& req) override { inner_.begin_game(req); }
  int prompts = 0;

 private:
  Agent& inner_;
};

EndgameHastyBotAgent::Params endgame_params(uint64_t budget, int plies) {
  EndgameHastyBotAgent::Params p;
  p.hasty = HastyBotAgent::Params{.thread_id = 0, .name = "EndgameHastyBot"};
  p.solver.budget = budget;
  p.solver.plies = plies;
  return p;
}

}  // namespace

// While the bag holds tiles the agent plays exactly like a plain HastyBot.
TEST(EndgameAgent, PreEndgameDelegatesToHasty) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  Dictionary d = tiny_dict();

  std::mt19937 rng(0x1234ABCDu);
  EndgameHastyBotAgent eg(endgame_params(50000, 25));
  HastyBotAgent hasty({.thread_id = 0, .name = "HastyBot"});

  int checked = 0;
  for (int i = 0; i < 30; ++i) {
    Board b;
    const int setup = std::uniform_int_distribution<int>(1, 4)(rng);
    for (int k = 0; k < setup; ++k) {
      const Rack seed = random_rack(rng);
      const std::vector<Move> plays = MoveGenerator(b, d).generate(seed);
      if (plays.empty()) break;
      b.apply(plays[std::uniform_int_distribution<size_t>(0, plays.size() - 1)(rng)]);
    }
    const Rack my = random_rack(rng);
    const Rack opp = random_rack(rng);
    const MoveRequest req{b, d, my, opp, 0, 0, /*bag_size=*/50};
    EXPECT_EQ(eg.make_move(req).move, hasty.make_move(req).move) << "position " << i;
    ++checked;
  }
  ASSERT_GT(checked, 0);
}

// On an endgame where the solver's move differs from HastyBot's greedy move,
// the agent plays the solver's move, or the greedy move with budget 0.
TEST(EndgameAgent, EndgameTakeoverVsGreedy) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  Dictionary d = tiny_dict();
  std::mt19937 rng(0xA5A5F00Du);

  EndgameSolver ref;
  bool found = false;
  for (int i = 0; i < 1200 && !found; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/3);
    const MoveRequest req = endgame_request(p, d);
    const Move greedy = hasty_best_move_wmp(req);
    // Same configuration as the agent's (spread_matters off), so the moves must
    // match exactly.
    ref.clear();
    const EndgameResult r = ref.solve(
      {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, /*scoreless_turns=*/0},
      {kSolveBudget, kSolvePlies, false});
    if (r.best == greedy) continue;

    EndgameHastyBotAgent solving(endgame_params(kSolveBudget, kSolvePlies));
    EXPECT_EQ(solving.make_move(req).move, r.best);
    EXPECT_NE(solving.make_move(req).move, greedy);

    EndgameHastyBotAgent disabled(endgame_params(/*nodes=*/0, kSolvePlies));
    EXPECT_EQ(disabled.make_move(req).move, greedy);

    found = true;
  }
  ASSERT_TRUE(found) << "no solver-beats-greedy endgame found in the scan";
}

// A budget too small for the solver's first iteration (one node against a
// multi-move root) declines the solve, and the agent plays HastyBot's move: a
// partial search never replaces the greedy policy with a noisier one.
TEST(EndgameAgent, ShallowSolveFallsBackToHasty) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  Dictionary d = tiny_dict();
  std::mt19937 rng(0xFA11BACCu);

  EndgameHastyBotAgent tiny(endgame_params(/*nodes=*/1, kSolvePlies));
  HastyBotAgent hasty({.thread_id = 0, .name = "HastyBot"});
  int checked = 0;
  for (int i = 0; i < 30; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/3);
    // A pass-only root fits any budget, so it would not exercise the decline.
    if (MoveGenerator(p.board, d).generate(p.my_rack).empty()) continue;
    const MoveRequest req = endgame_request(p, d);
    EXPECT_EQ(tiny.make_move(req).move, hasty.make_move(req).move) << "position " << i;
    ++checked;
  }
  ASSERT_GT(checked, 0);
}

TEST(EndgameAgent, EndgameMoveIsLegal) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  Dictionary d = tiny_dict();
  std::mt19937 rng(0xBADF00D1u);

  EndgameHastyBotAgent agent(endgame_params(20000, 20));
  int checked = 0;
  for (int i = 0; i < 60; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/3);
    const MoveRequest req = endgame_request(p, d);
    const Move m = agent.make_move(req).move;
    if (m.type() == MoveType::PASS) {
      ++checked;
      continue;
    }
    const std::set<std::string> legal = key_set(MoveGenerator(p.board, d).generate(p.my_rack));
    EXPECT_TRUE(legal.count(move_key(m)) > 0) << "position " << i;
    ++checked;
  }
  ASSERT_GT(checked, 0);
}

// Full EndgameHastyBot-vs-HastyBot games terminate normally and log consistent
// scores.
TEST(EndgameAgent, FullGameTinyDict) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  Dictionary d = tiny_dict();
  for (uint64_t seed = 1; seed <= 8; ++seed) {
    EndgameHastyBotAgent a0(endgame_params(20000, 25));
    HastyBotAgent a1({.thread_id = 0, .name = "HastyBot"});
    Game g(a0, a1, d, seed);
    g.play();
    check_log_consistency(g.log());
  }
}

TEST(EndgameAgent, FullGameRealLexicon) {
  const std::string kwg = SCRIBBLEZ_DEFAULT_KWG;
  if (!std::ifstream(kwg).good() || !ensure_equity()) GTEST_SKIP() << "no NWL23 kwg/leaves";
  Dictionary d = Dictionary::load_kwg(kwg);
  for (uint64_t seed = 1; seed <= 6; ++seed) {
    EndgameHastyBotAgent a0(endgame_params(50000, 25));
    HastyBotAgent a1({.thread_id = 0, .name = "HastyBot"});
    Game g(a0, a1, d, seed);
    g.play();
    check_log_consistency(g.log());
  }
}

// from_spec accepts both the endgame- and the HastyBot options, and rejects bad
// ones.
TEST(EndgameAgent, FromSpecParsing) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";

  EXPECT_NE(EndgameHastyBotAgent::from_spec(
              {"--endgame-budget=1234", "--endgame-plies=7", "--temperature=0"}, 0, "X"),
            nullptr);
  EXPECT_NE(
    EndgameHastyBotAgent::from_spec({"--top-k=5", "--temperature=1.5", "--seed=42"}, 0, "Y"),
    nullptr);

  // A parsed --endgame-budget=0 must disable the solver.
  Dictionary d = tiny_dict();
  std::mt19937 rng(0x0FF5E7u);
  const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/3);
  const MoveRequest req = endgame_request(p, d);
  auto disabled = EndgameHastyBotAgent::from_spec({"--endgame-budget=0"}, 0, "Z");
  EXPECT_EQ(disabled->make_move(req).move, hasty_best_move_wmp(req));

  EXPECT_THROW(EndgameHastyBotAgent::from_spec({"--endgame-plies=notanint"}, 0, "B"),
               std::runtime_error);
  EXPECT_THROW(EndgameHastyBotAgent::from_spec({"--bogus-option=1"}, 0, "C"), std::runtime_error);
}

TEST(EndgameAgent, SpreadMattersFromSpec) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";

  EXPECT_NE(
    EndgameHastyBotAgent::from_spec({"--endgame-spread-matters=1", "--endgame-budget=777"}, 0, "W"),
    nullptr);
  EXPECT_NE(EndgameHastyBotAgent::from_spec({"--endgame-spread-matters=0"}, 0, "X"), nullptr);
  EXPECT_NE(EndgameHastyBotAgent::from_spec({"--endgame-budget=777"}, 0, "V"), nullptr);
  EXPECT_THROW(EndgameHastyBotAgent::from_spec({"--endgame-spread-matters=maybe"}, 0, "B"),
               std::runtime_error);
}

// With spread_matters off (the solve stops at the win/draw/loss proof), checked
// against a reference solve of the same position:
//   * a proof certificate becomes the decision's projection, even for a proven
//     loss: the result is settled, so the game loop can stop spending compute;
//   * a proven loss without a certificate falls back to HastyBot's move, which
//     shapes the final spread better than an arbitrary losing move.
TEST(EndgameAgent, FirstWinProjectsCertificates) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  Dictionary d = tiny_dict();
  std::mt19937 rng(0x105510FFu);

  EndgameSolver ref;
  HastyBotAgent hasty({.thread_id = 0, .name = "HastyBot"});
  EndgameHastyBotAgent::Params wp = endgame_params(kSolveBudget, kSolvePlies);
  wp.solver.spread_matters = false;
  EndgameHastyBotAgent wld(wp);

  int projected = 0, loss_fallbacks = 0, checked = 0;
  for (int i = 0; i < 200; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/3);
    ref.clear();
    const EndgameResult r = ref.solve(
      {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, /*scoreless_turns=*/0},
      {kSolveBudget, kSolvePlies, false});
    if (r.depth_completed < 1) continue;
    ++checked;

    // begin_game clears the agent's transposition table. A warm table would
    // make its capped solve diverge from the freshly cleared reference solve.
    wld.begin_game({});
    const MoveRequest req = endgame_request(p, d);
    const MoveDecision decision = wld.make_move(req);
    if (r.proven_class == -1 && r.continuation.empty()) {
      EXPECT_EQ(decision.move, hasty.make_move(req).move) << "position " << i;
      EXPECT_TRUE(decision.projected_remaining_moves.empty()) << "position " << i;
      ++loss_fallbacks;
    } else {
      EXPECT_EQ(decision.move, r.best) << "position " << i;
      EXPECT_EQ(decision.projected_remaining_moves.size(), r.continuation.size())
        << "position " << i;
      if (!decision.projected_remaining_moves.empty()) ++projected;
    }
  }
  ASSERT_GT(checked, 50);
  ASSERT_GT(projected, 0) << "no solve produced a projected certificate";
  std::cout << "  first-win decisions: " << projected << " projected, " << loss_fallbacks
            << " loss fallbacks, " << checked << " checked\n";
}

// In real-lexicon self-play, a game loop that respects projections fast-tracks
// proven endgames to their end, so agents are prompted less, and every game
// still ends naturally. Plays the same seeds both ways and compares prompts.
TEST(EndgameAgent, FastTrackReducesPrompts) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  const char* path = SCRIBBLEZ_DEFAULT_KWG;
  if (!std::ifstream(path).good()) GTEST_SKIP() << "no lexicon at " << path;
  Dictionary d = Dictionary::load_kwg(path);

  int prompts_with = 0, prompts_without = 0, fast_tracked = 0;
  for (int mode = 0; mode < 2; ++mode) {
    const bool respect = mode == 0;
    EndgameHastyBotAgent::Params params = endgame_params(1600, 25);
    params.solver.spread_matters = false;
    EndgameHastyBotAgent inner0(params), inner1(params);
    PromptCountingAgent a0(inner0), a1(inner1);
    // These games are the whole cost of the test. Ten is plenty: most games
    // reach a proven endgame, so the prompt gap is many times the
    // seed-to-seed variation.
    for (int i = 0; i < 10; ++i) {
      Game g(a0, a1, d, /*seed=*/9000 + i);
      g.set_respect_projections(respect);
      g.play();
      const GameLogStorage log = g.extract_log();
      ASSERT_NE(log.end_reason, "max_turns") << "seed " << 9000 + i;
    }
    (respect ? prompts_with : prompts_without) = a0.prompts + a1.prompts;
  }
  fast_tracked = prompts_without - prompts_with;
  std::cout << "  prompts: " << prompts_with << " with fast-track vs " << prompts_without
            << " without (" << fast_tracked << " skipped)\n";
  EXPECT_LT(prompts_with, prompts_without);
}
