// GoogleTest suite for EndgameHastyBotAgent: pre-endgame delegation to plain
// HastyBot, endgame takeover by the solver (and its disablement at
// endgame_nodes=0), legality of returned endgame moves, full-game integration,
// and --type=hastybot-endgame from_spec parsing.

#include "agent/endgame_hasty_bot.h"
#include "agent/macondo_bot.h"
#include "endgame/endgame_solver.h"
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

Dictionary tiny_dict() {
  return Dictionary::build_from_words({"CAT", "CATS", "AT",     "AS",     "BAT", "BATS", "HE",
                                       "TO",  "ON",   "NO",     "IT",     "IS",  "OAT",  "OATS",
                                       "HAT", "HATS", "RAT",    "RATS",   "DOG", "GOD",  "GO",
                                       "OD",  "DO",   "AERIES", "PARTIED"});
}

// Load HastyBot's default equity tables for NWL23 (leaves + pre-endgame), which
// the agent's greedy path needs. Returns false (so the caller skips) when the
// Macondo-bundled leaves file is absent; ensure_initialized is idempotent.
bool ensure_equity() {
  const std::string leaves = HastyEquity::default_leaves_path("NWL23");
  if (!std::ifstream(leaves).good()) return false;
  HastyEquity::ensure_initialized("NWL23");
  return true;
}

// A canonical key for a play (placed squares/glyphs + score), so a returned move
// can be checked for membership in a generated list ignoring enumeration order.
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

Rack random_rack(std::mt19937& rng) {
  Rack r;
  std::uniform_int_distribution<int> pick(0, 26);  // 26 -> blank
  for (int i = 0; i < RACK_SIZE; ++i) {
    const int v = pick(rng);
    r.add(v == 26 ? BLANK : Tile::of(v));
  }
  return r;
}

// A small bag-empty endgame: a board seeded with a few random plays, two short
// racks drawn from a curated tiny-dict letter set (so plays exist but branching
// stays low), and small random scores.
struct EndgamePos {
  Board board;
  Rack my_rack;
  Rack opp_rack;
  int my_score;
  int opp_score;
};

EndgamePos random_endgame(std::mt19937& rng, const Dictionary& d, int rack_tiles) {
  static const char kLetters[] = "ATSOCHEBDGRINO";
  EndgamePos p;
  const int setup = std::uniform_int_distribution<int>(1, 3)(rng);
  for (int k = 0; k < setup; ++k) {
    const Rack seed = random_rack(rng);
    const std::vector<Move> plays = MoveGenerator(p.board, d).generate(seed);
    if (plays.empty()) break;
    p.board.apply(plays[std::uniform_int_distribution<size_t>(0, plays.size() - 1)(rng)]);
  }
  std::uniform_int_distribution<int> letter(0, static_cast<int>(sizeof(kLetters) - 2));
  for (int i = 0; i < rack_tiles; ++i) {
    p.my_rack.add(Tile::from_char(kLetters[letter(rng)]));
    p.opp_rack.add(Tile::from_char(kLetters[letter(rng)]));
  }
  p.my_score = std::uniform_int_distribution<int>(0, 40)(rng);
  p.opp_score = std::uniform_int_distribution<int>(0, 40)(rng);
  return p;
}

MoveRequest endgame_request(const EndgamePos& p, const Dictionary& d) {
  return MoveRequest{p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, /*bag_size=*/0};
}

// Reconstruct a completed game's end-of-game bookkeeping from its turn records
// and assert it matches the logged final scores: the per-player score-delta sum
// equals the last recorded cumulative score, and the endgame adjustment implied
// by the end reason reproduces final_scores.
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

EndgameHastyBotAgent::Params endgame_params(uint64_t nodes, int plies) {
  EndgameHastyBotAgent::Params p;
  p.hasty = HastyBotAgent::Params{.thread_id = 0, .name = "EndgameHastyBot"};
  p.endgame_nodes = nodes;
  p.endgame_plies = plies;
  return p;
}

}  // namespace

// While the bag holds tiles the agent must play exactly like a plain HastyBot:
// for a mid-game request (bag_size > 0), both return the identical move.
TEST(EndgameAgent, PreEndgameDelegatesToHasty) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  Dictionary d = tiny_dict();

  std::mt19937 rng(0x1234ABCDu);
  EndgameHastyBotAgent eg(endgame_params(50000, 25));
  HastyBotAgent hasty({.thread_id = 0, .name = "HastyBot"});

  int checked = 0;
  for (int i = 0; i < 30; ++i) {
    // A board with a few tiles down, but plenty of tiles still in the bag.
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
    EXPECT_EQ(eg.make_move(req), hasty.make_move(req)) << "position " << i;
    ++checked;
  }
  ASSERT_GT(checked, 0);
}

// On an endgame where the exact solver's move differs from HastyBot's greedy
// move, the agent plays the solver's move; with the solver disabled
// (endgame_nodes = 0) it plays the greedy move instead.
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
    ref.clear();
    const EndgameResult r = ref.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score,
                                      /*scoreless_turns=*/0, kSolveBudget, kSolvePlies);
    if (r.best == greedy) continue;

    // Solver takes over: the agent (with a matching budget/plies) plays the
    // solver's move, not the greedy one.
    EndgameHastyBotAgent solving(endgame_params(kSolveBudget, kSolvePlies));
    EXPECT_EQ(solving.make_move(req), r.best);
    EXPECT_NE(solving.make_move(req), greedy);

    // Solver disabled: the agent falls back to the greedy move.
    EndgameHastyBotAgent disabled(endgame_params(/*nodes=*/0, kSolvePlies));
    EXPECT_EQ(disabled.make_move(req), greedy);

    found = true;
  }
  ASSERT_TRUE(found) << "no solver-beats-greedy endgame found in the scan";
}

// Every endgame move the agent returns is legal: a pass, or one of the moves the
// generator produces for its rack.
TEST(EndgameAgent, EndgameMoveIsLegal) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  Dictionary d = tiny_dict();
  std::mt19937 rng(0xBADF00D1u);

  EndgameHastyBotAgent agent(endgame_params(20000, 20));
  int checked = 0;
  for (int i = 0; i < 60; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/3);
    const MoveRequest req = endgame_request(p, d);
    const Move m = agent.make_move(req);
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

// A full EndgameHastyBot-vs-HastyBot game on the tiny dictionary terminates
// normally and logs internally consistent scores.
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

// The same integration against the real NWL23 lexicon (gated on the KWG).
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

// --type=hastybot-endgame from_spec: valid specs (endgame + hasty options)
// produce agents, and bad options throw.
TEST(EndgameAgent, FromSpecParsing) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";

  EXPECT_NE(EndgameHastyBotAgent::from_spec(
              {"--endgame-nodes=1234", "--endgame-plies=7", "--temperature=0"}, 0, "X"),
            nullptr);
  EXPECT_NE(
    EndgameHastyBotAgent::from_spec({"--top-k=5", "--temperature=1.5", "--seed=42"}, 0, "Y"),
    nullptr);

  // A parsed --endgame-nodes=0 disables the solver: the agent plays the greedy
  // move on a bag-empty request.
  Dictionary d = tiny_dict();
  std::mt19937 rng(0x0FF5E7u);
  const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/3);
  const MoveRequest req = endgame_request(p, d);
  auto disabled = EndgameHastyBotAgent::from_spec({"--endgame-nodes=0"}, 0, "Z");
  EXPECT_EQ(disabled->make_move(req), hasty_best_move_wmp(req));

  EXPECT_THROW(EndgameHastyBotAgent::from_spec({"--endgame-plies=notanint"}, 0, "B"),
               std::runtime_error);
  EXPECT_THROW(EndgameHastyBotAgent::from_spec({"--bogus-option=1"}, 0, "C"), std::runtime_error);
}
