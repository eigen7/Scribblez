// BestBot, the port of Macondo's simming bot, and its parts: the running
// statistic, the win-percentage table, the stopping rule and the simmer. The
// cases that need the NWL23 lexicon or Macondo's data files skip without them.

#include "agent/agent.h"
#include "agent/best_bot.h"
#include "agent/endgame_agent.h"
#include "agent/hasty_bot.h"
#include "game/board.h"
#include "game/game.h"
#include "game/move.h"
#include "game/movegen.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "sim/macondo_autostopper.h"
#include "sim/macondo_simmer.h"
#include "sim/sim_runner.h"
#include "sim/win_pct_table.h"
#include "util/running_stat.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace scribblez;

namespace {

constexpr char kWinPct[] = "/workspace/mount/macondo/data/strategy/default/winpct.csv";

// Loads the NWL23 equity tables; false when the leaves file is absent.
bool ensure_equity() {
  if (!std::ifstream(HastyEquity::default_leaves_path("NWL23")).good()) return false;
  HastyEquity::ensure_initialized("NWL23");
  return true;
}

bool have_macondo_data() {
  return std::ifstream(SCRIBBLEZ_DEFAULT_KWG).good() && std::ifstream(kWinPct).good() &&
         ensure_equity();
}

const Dictionary& nwl23() {
  static const Dictionary dict = Dictionary::load_kwg(SCRIBBLEZ_DEFAULT_KWG);
  return dict;
}

util::RunningStat stat_of(const std::vector<double>& xs) {
  util::RunningStat s;
  for (double x : xs) s.push(x);
  return s;
}

// `n` samples alternating mean - spread and mean + spread.
util::RunningStat alternating(double mean, double spread, int n) {
  util::RunningStat s;
  for (int i = 0; i < n; ++i) s.push(i % 2 == 0 ? mean - spread : mean + spread);
  return s;
}

SimmedPlay simmed(const util::RunningStat& win_prob, const util::RunningStat& equity) {
  SimmedPlay p;
  p.win_prob = win_prob;
  p.equity = equity;
  return p;
}

// A decision point: the board, the mover's rack, and the scores from the
// mover's point of view.
struct Position {
  Board board;
  Rack rack;
  int my_score = 0;
  int opp_score = 0;
  int bag_size = 0;

  MoveRequest request(const Dictionary& dict) const {
    static const Rack kHidden;
    return MoveRequest{board, dict, rack, kHidden, my_score, opp_score, bag_size};
  }
};

// The position after `plies` HastyBot-vs-HastyBot moves of game `seed`, or
// nullopt if the game did not stop there (it ended, or the bag was already
// empty).
std::optional<Position> hasty_position(const Dictionary& dict, uint64_t seed, int plies) {
  HastyBot a0({.thread_id = 0, .name = "A"}), a1({.thread_id = 0, .name = "B"});
  Game g(a0, a1, dict, seed);
  g.set_max_plies(plies);
  g.play();
  if (!g.truncated()) return std::nullopt;
  const int mover = plies % 2;
  return Position{g.board(), g.rack(mover), g.score(mover), g.score(1 - mover), g.bag_size()};
}

// The first position of game `seed` whose bag size satisfies `want`.
std::optional<Position> find_position(const Dictionary& dict, uint64_t seed,
                                      const std::function<bool(int)>& want) {
  for (int plies = 1; plies < 40; ++plies) {
    const std::optional<Position> p = hasty_position(dict, seed, plies);
    if (p && want(p->bag_size)) return p;
  }
  return std::nullopt;
}

void expect_same_results(const MacondoSimmer::Result& a, const MacondoSimmer::Result& b) {
  EXPECT_EQ(a.iterations, b.iterations);
  ASSERT_EQ(a.plays.size(), b.plays.size());
  for (size_t i = 0; i < a.plays.size(); ++i) {
    EXPECT_EQ(a.plays[i].move, b.plays[i].move) << "rank " << i;
    EXPECT_EQ(a.plays[i].ignored, b.plays[i].ignored) << "rank " << i;
    EXPECT_EQ(a.plays[i].win_prob.count(), b.plays[i].win_prob.count()) << "rank " << i;
    EXPECT_EQ(a.plays[i].win_prob.mean(), b.plays[i].win_prob.mean()) << "rank " << i;
    EXPECT_EQ(a.plays[i].equity.mean(), b.plays[i].equity.mean()) << "rank " << i;
  }
}

}  // namespace

// Macondo's own stats test vectors (stats/stats_test.go).
TEST(RunningStat, MatchesMacondoTestVectors) {
  const util::RunningStat a = stat_of({10, 12, 23, 23, 16, 23, 21, 16});
  EXPECT_NEAR(a.mean(), 18, 1e-6);
  EXPECT_NEAR(std::sqrt(a.variance()), 5.2372293656638, 1e-6);
  const util::RunningStat b = stat_of({14, 35, 71, 124, 10, 24, 55, 33, 87, 19});
  EXPECT_NEAR(b.mean(), 47.2, 1e-6);
  EXPECT_NEAR(std::sqrt(b.variance()), 36.937785531891, 1e-6);
  const util::RunningStat one = stat_of({1});
  EXPECT_EQ(one.mean(), 1);
  EXPECT_EQ(one.variance(), 0);
  EXPECT_EQ(util::RunningStat().mean(), 0);
}

TEST(WinPctTable, ReadsMacondoTable) {
  if (!std::ifstream(kWinPct).good()) GTEST_SKIP() << "no Macondo winpct.csv";
  const WinPctTable& t = WinPctTable::macondo_default();
  // Cells read straight from the CSV: the side to move leads at a tied score.
  EXPECT_FLOAT_EQ(t.win_prob(0, 79), 0.586229f);
  EXPECT_FLOAT_EQ(t.win_prob(1, 50), 0.627713f);
  EXPECT_FLOAT_EQ(t.win_prob(-1, 93), 0.556054f);
  EXPECT_GT(t.win_prob(300, 50), 0.99);
  EXPECT_LT(t.win_prob(-300, 50), 0.01);
  for (int s = -299; s <= 300; ++s) EXPECT_GE(t.win_prob(s, 50), t.win_prob(s - 1, 50)) << s;
  // Out-of-range arguments read the edge of the table.
  EXPECT_EQ(t.win_prob(1000, 50), t.win_prob(300, 50));
  EXPECT_EQ(t.win_prob(-1000, 50), t.win_prob(-300, 50));
  EXPECT_EQ(t.win_prob(40, 200), t.win_prob(40, 93));
}

TEST(MacondoAutoStopper, StopsPastTheIterationCap) {
  Board board;
  const MacondoAutoStopper stopper(board);
  std::vector<SimmedPlay> plays(3, simmed(alternating(0.5, 0.2, 200), alternating(0, 10, 200)));
  EXPECT_FALSE(stopper.should_stop(128, plays, /*plies=*/5));
  // The cap is 2000 + 625 per ply, tested at check granularity.
  EXPECT_FALSE(stopper.should_stop(5125, plays, 5));
  EXPECT_TRUE(stopper.should_stop(5126, plays, 5));
}

TEST(MacondoAutoStopper, PrunesOnlyConfidentlyTrailingPlays) {
  Board board;
  const MacondoAutoStopper stopper(board);
  const util::RunningStat eq = alternating(0, 10, 200);
  std::vector<SimmedPlay> plays = {simmed(alternating(0.30, 0.1, 200), eq),
                                   simmed(alternating(0.70, 0.1, 200), eq),
                                   simmed(alternating(0.69, 0.1, 200), eq)};
  EXPECT_FALSE(stopper.should_stop(256, plays, 2));
  EXPECT_TRUE(plays[0].ignored);
  EXPECT_FALSE(plays[1].ignored);
  EXPECT_FALSE(plays[2].ignored);
}

TEST(MacondoAutoStopper, StopsWhenOnePlayRemains) {
  Board board;
  const MacondoAutoStopper stopper(board);
  const util::RunningStat eq = alternating(0, 10, 200);
  std::vector<SimmedPlay> plays = {simmed(alternating(0.30, 0.1, 200), eq),
                                   simmed(alternating(0.70, 0.1, 200), eq)};
  EXPECT_TRUE(stopper.should_stop(256, plays, 2));
  EXPECT_TRUE(plays[0].ignored);
}

TEST(MacondoAutoStopper, WaitsForEnoughSamplesBeforePruning) {
  Board board;
  const MacondoAutoStopper stopper(board);
  const util::RunningStat eq = alternating(0, 10, 100);
  std::vector<SimmedPlay> plays = {simmed(alternating(0.30, 0.1, 100), eq),
                                   simmed(alternating(0.70, 0.1, 100), eq)};
  EXPECT_FALSE(stopper.should_stop(128, plays, 2));
  EXPECT_FALSE(plays[0].ignored);
}

// When every play is a near-certain win, win probability cannot separate them
// and equity ranks them instead.
TEST(MacondoAutoStopper, TiebreaksCertainWinsByEquity) {
  Board board;
  const MacondoAutoStopper stopper(board);
  const util::RunningStat certain = alternating(1.0, 0.0, 200);
  std::vector<SimmedPlay> plays = {simmed(certain, alternating(10, 5, 200)),
                                   simmed(certain, alternating(40, 5, 200)),
                                   simmed(certain, alternating(39, 5, 200))};
  EXPECT_FALSE(stopper.should_stop(256, plays, 2));
  EXPECT_TRUE(plays[0].ignored);
  EXPECT_FALSE(plays[1].ignored);
  EXPECT_FALSE(plays[2].ignored);
}

// Past 750 iterations, a play that lays the leader's tiles along the leader's
// word extent in another order is pruned even when the statistics cannot
// separate the two.
TEST(MacondoAutoStopper, PrunesMateriallySimilarPlaysLate) {
  const Dictionary dict = Dictionary::build_from_words({"ATE", "EAT", "ETA", "TEA"});
  const Board board;
  const std::vector<Move> plays = MoveGenerator(board, dict).generate(Rack::from_string("AET"));
  std::optional<std::pair<Move, Move>> pair;
  for (const Move& a : plays)
    for (const Move& b : plays)
      if (!pair && !(a == b) && a.horizontal() && b.horizontal() && a.start() == b.start() &&
          a.square_mask() == b.square_mask())
        pair = {a, b};
  ASSERT_TRUE(pair);

  const MacondoAutoStopper stopper(board);
  const util::RunningStat win = alternating(0.5, 0.2, 200), eq = alternating(0, 10, 200);
  std::vector<SimmedPlay> simmed_plays(2, simmed(win, eq));
  simmed_plays[0].move = pair->first;
  simmed_plays[1].move = pair->second;
  EXPECT_FALSE(stopper.should_stop(640, simmed_plays, 2));
  EXPECT_FALSE(simmed_plays[1].ignored);
  EXPECT_TRUE(stopper.should_stop(768, simmed_plays, 2));
  EXPECT_TRUE(simmed_plays[1].ignored);
}

TEST(MacondoSimmer, ResultsDoNotDependOnThreadCount) {
  if (!have_macondo_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo data";
  const std::optional<Position> pos = hasty_position(nwl23(), 7, 6);
  ASSERT_TRUE(pos);
  const MoveRequest req = pos->request(nwl23());
  const std::vector<Move> candidates = equity_top_k(req, 6);
  MacondoSimmer::Params params{.plies = 2, .threads = 1, .max_iterations = 300};
  const MacondoSimmer::Result one =
    MacondoSimmer::simulate(nwl23(), sim_position_from(req), candidates, 0, params, 99);
  params.threads = 4;
  const MacondoSimmer::Result four =
    MacondoSimmer::simulate(nwl23(), sim_position_from(req), candidates, 0, params, 99);
  expect_same_results(one, four);
  EXPECT_EQ(one.iterations, 300u);
}

TEST(MacondoSimmer, RanksPlaysAndRespectsTheIterationCap) {
  if (!have_macondo_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo data";
  const std::optional<Position> pos = hasty_position(nwl23(), 11, 4);
  ASSERT_TRUE(pos);
  const MoveRequest req = pos->request(nwl23());
  const std::vector<Move> candidates = equity_top_k(req, 5);
  const MacondoSimmer::Result r = MacondoSimmer::simulate(
    nwl23(), sim_position_from(req), candidates, 0, {.plies = 3, .max_iterations = 100}, 5);
  EXPECT_EQ(r.iterations, 100u);
  ASSERT_EQ(r.plays.size(), candidates.size());
  for (size_t i = 0; i < r.plays.size(); ++i) {
    // No stopping check falls before the cap, so nothing was pruned.
    EXPECT_FALSE(r.plays[i].ignored);
    EXPECT_EQ(r.plays[i].win_prob.count(), 100);
    EXPECT_GE(r.plays[i].win_prob.mean(), 0.0);
    EXPECT_LE(r.plays[i].win_prob.mean(), 1.0);
    if (i > 0) {
      EXPECT_LE(r.plays[i].win_prob.mean(), r.plays[i - 1].win_prob.mean() + 1e-9);
    }
  }
}

// With 8 or fewer tiles unseen (one or none in the bag), the bot plays
// HastyBot's move.
TEST(BestBot, PlaysHastyBotsMoveInThePreEndgame) {
  if (!have_macondo_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo data";
  int checked = 0;
  for (uint64_t seed = 1; seed <= 4; ++seed) {
    const std::optional<Position> pos =
      find_position(nwl23(), seed, [](int bag) { return bag == 1; });
    if (!pos) continue;
    BestBot bot({.name = "B", .max_iterations = 1, .seed = 1});
    const MoveRequest req = pos->request(nwl23());
    EXPECT_EQ(bot.make_move(req).move, hasty_best_move_wmp(req)) << "seed " << seed;
    ++checked;
  }
  EXPECT_GT(checked, 0);
}

// The bot hands MacondoSimmer Macondo's phase parameters and its own seed: a
// replay of the sim picks the same move.
TEST(BestBot, SimsTopMovesWithMacondosPhaseParameters) {
  if (!have_macondo_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo data";
  struct Phase {
    std::function<bool(int)> bag;
    int candidates;
    std::function<int(const Position&)> plies;
  };
  const std::vector<Phase> phases = {
    {[](int bag) { return bag > 7; }, 40, [](const Position&) { return 5; }},
    {[](int bag) { return bag >= 2 && bag <= 7; }, 80,
     [](const Position& p) { return p.bag_size + RACK_SIZE; }},
  };
  for (const Phase& phase : phases) {
    const std::optional<Position> pos = find_position(nwl23(), 3, phase.bag);
    ASSERT_TRUE(pos);
    const MoveRequest req = pos->request(nwl23());
    BestBot bot({.name = "B", .max_iterations = 8, .seed = 77});
    bot.begin_game({});
    const Move played = bot.make_move(req).move;
    const MacondoSimmer::Result replay = MacondoSimmer::simulate(
      nwl23(), sim_position_from(req), equity_top_k(req, phase.candidates), 0,
      {.plies = phase.plies(*pos), .max_iterations = 8}, bot.sim_seed(0));
    EXPECT_EQ(played, replay.plays.front().move) << "bag " << pos->bag_size;
  }
}

TEST(BestBot, FromSpecParsing) {
  if (!have_macondo_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo data";
  EXPECT_NE(BestBot::from_spec({"--min-sim-plies=3", "--max-iterations=500", "--seed=4"}, 0, "B"),
            nullptr);
  EXPECT_NE(EndgameAgent<BestBot>::from_spec({"--sim-threads=2", "--endgame-budget=5000"}, 0, "E"),
            nullptr);
  EXPECT_THROW(BestBot::from_spec({"--bogus=1"}, 0, "B"), std::runtime_error);
  EXPECT_THROW(BestBot::from_spec({"--min-sim-plies=0"}, 0, "B"), std::runtime_error);
  EXPECT_THROW(BestBot::from_spec({"--endgame-budget=5"}, 0, "B"), std::runtime_error);
}

// Whole games against HastyBot finish normally, through every phase: sims,
// pre-endgame, and the solved endgame.
TEST(BestBot, PlaysFullGames) {
  if (!have_macondo_data()) GTEST_SKIP() << "no NWL23 kwg / Macondo data";
  for (uint64_t seed = 1; seed <= 1; ++seed) {
    EndgameAgent<BestBot>::Params params;
    params.base = {.name = "BestBot", .max_iterations = 8, .seed = seed};
    params.solver.budget = 20000;
    EndgameAgent<BestBot> best(params);
    HastyBot hasty({.thread_id = 0, .name = "HastyBot"});
    Game g(best, hasty, nwl23(), seed);
    g.play();
    const GameLog log = g.log();
    ASSERT_GT(log.num_records, 0);
    EXPECT_TRUE(std::string(log.end_reason) == "out" || std::string(log.end_reason) == "stalemate")
      << log.end_reason;
  }
}
