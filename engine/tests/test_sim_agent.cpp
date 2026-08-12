// Unit tests for SimAgent, the Monte-Carlo simming baseline.
//
//  * it plays the candidate its own rollouts rank best -- checked by replaying
//    the decision through SimRunner directly, which pins the position and the
//    seed it hands the simulator.
//  * it sims against the opponent's public leave, checked on a position where
//    doing so demonstrably changes which candidate wins.
//  * both selection objectives run and choose a candidate that was simmed.
//  * two agents on one seed agree, turn after turn.
//  * a bag-empty turn with solving disabled falls back to static equity rather
//    than simulating without a bag to draw from.
//  * an unusable rollout count or candidate cap is rejected -- in every build,
//    since these are the bounds a Release build once let through.

#include "agent/agent.h"
#include "agent/sim_agent.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "sim/sim_runner.h"
#include "synthetic_equity.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using namespace scribblez;

namespace {

Rack rack_from(const std::string& s) {
  Rack r;
  for (char c : s) r.add(c == '?' ? BLANK : Tile::from_char(c));
  return r;
}

Dictionary opening_dict() {
  return Dictionary::build_from_words(
    {"AE",   "AR",    "AT",    "ARC",    "ARCS",   "ARE",   "ART",  "ARTS",  "ATE",
     "CAR",  "CARE",  "CARES", "CARET",  "CARETS", "CARS",  "CART", "CARTS", "CAT",
     "CATS", "CATER", "CRATE", "CRATES", "EAR",    "EARS",  "EAT",  "EATS",  "ERA",
     "ETA",  "RACE",  "RACES", "RAT",    "RATE",   "RATES", "RATS", "SCAR",  "SCARE",
     "SET",  "TARE",  "TEA",   "TEAR",   "TEARS",  "TRACE"});
}

class SimAgentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tmp_ = std::filesystem::temp_directory_path() / "scribblez_test_sim_agent_XXXXXX";
    std::filesystem::create_directories(tmp_);
    scribblez::testing::install_synthetic_hasty_equity(tmp_);
  }
  void TearDown() override { std::filesystem::remove_all(tmp_); }

  SimAgent::Params params() const {
    SimAgent::Params p;
    p.name = "S";
    p.dict = &dict_;
    p.top_k = 4;
    p.sim.rollouts = 8;
    p.sim.threads = 1;
    p.seed = 12345;
    p.endgame.budget = 0;  // the endgame is not what these tests are about
    return p;
  }

  // An opening turn with a non-empty opponent leave, so the position handed to
  // the simulator carries every field the agent is responsible for filling.
  MoveRequest request() const {
    return MoveRequest{board_,          dict_,    my_rack_, opp_leave_, /*my_score=*/13,
                       /*opp_score=*/7, bag_size_};
  }

  Dictionary dict_ = opening_dict();
  Board board_;
  Rack my_rack_ = rack_from("CARTES");
  Rack opp_leave_ = rack_from("AE");
  int bag_size_ = 86;
  std::filesystem::path tmp_;
};

// The agent's own ranking rule, so the reproduction below breaks ties the way
// the agent does rather than the way a test author guessed.
int best_of(const std::vector<SimObservation>& obs, SimObjective objective) {
  auto value = [&](const SimObservation& o) {
    if (o.n == 0) return 0.0;
    return objective == SimObjective::kWinRate
             ? (o.wins + 0.5 * o.draws) / static_cast<double>(o.n)
             : static_cast<double>(o.delta_sum) / static_cast<double>(o.n);
  };
  int best = 0;
  for (size_t i = 1; i < obs.size(); ++i)
    if (value(obs[i]) > value(obs[best])) best = static_cast<int>(i);
  return best;
}

}  // namespace

TEST_F(SimAgentTest, PlaysTheCandidateItsOwnRolloutsRankBest) {
  for (SimObjective objective : {SimObjective::kWinRate, SimObjective::kSpread}) {
    SimAgent::Params p = params();
    p.objective = objective;
    SimAgent agent(p);
    agent.begin_game({});
    const Move played = agent.make_move(request()).move;

    // Replay the same decision independently: the same candidate set, the same
    // position, the same seed. A disagreement means the agent handed the
    // simulator something other than the turn it was asked about.
    const std::vector<Move> candidates = equity_top_k(request(), p.top_k);
    ASSERT_GT(candidates.size(), 1u);
    SimPosition pos;
    pos.board = board_;
    pos.mover = 0;
    pos.scores = {13, 7};
    pos.rack = my_rack_;
    pos.opp_leave = opp_leave_;
    const std::vector<SimObservation> obs =
      SimRunner(dict_, p.sim).run(pos, candidates, agent.sim_seed(0));

    EXPECT_TRUE(played == candidates[static_cast<size_t>(best_of(obs, objective))])
      << "objective=" << static_cast<int>(objective);
  }
}

TEST_F(SimAgentTest, SimulatesAgainstTheOpponentsPublicLeave) {
  // A leave changes a decision only where it changes which candidate the
  // rollouts favour, which no particular position is guaranteed to do. So
  // search for one that does -- comparing the same rollouts run with and
  // without the leave -- and only then check that the agent went the way the
  // leave-aware run did. Finding no such position is itself a failure: the
  // check would be vacuous.
  const SimAgent::Params p = params();
  const std::vector<std::string> leaves = {"AE", "SS", "AAAA", "CARTES", "EEEEEE"};

  bool discriminated = false;
  for (const std::string& text : leaves) {
    const Rack leave = rack_from(text);
    const MoveRequest req{board_, dict_, my_rack_, leave, 13, 7, bag_size_};
    const std::vector<Move> candidates = equity_top_k(req, p.top_k);
    ASSERT_GT(candidates.size(), 1u);

    SimPosition pos;
    pos.board = board_;
    pos.mover = 0;
    pos.scores = {13, 7};
    pos.rack = my_rack_;
    SimPosition blind = pos;
    pos.opp_leave = leave;

    const SimRunner runner(dict_, p.sim);
    const int with = best_of(runner.run(pos, candidates, SimAgent(p).sim_seed(0)), p.objective);
    const int without =
      best_of(runner.run(blind, candidates, SimAgent(p).sim_seed(0)), p.objective);
    if (with == without) continue;

    discriminated = true;
    SimAgent agent(p);
    agent.begin_game({});
    EXPECT_TRUE(agent.make_move(req).move == candidates[static_cast<size_t>(with)])
      << "leave=" << text << ": the agent simmed as though the leave were hidden";
    break;
  }
  ASSERT_TRUE(discriminated) << "no leave changed the decision; the check proved nothing";
}

TEST_F(SimAgentTest, PicksFromTheCandidatesItSimmed) {
  SimAgent agent(params());
  agent.begin_game({});
  const Move played = agent.make_move(request()).move;

  const std::vector<Move> candidates = equity_top_k(request(), params().top_k);
  bool found = false;
  for (const Move& c : candidates) found = found || (c == played);
  EXPECT_TRUE(found) << "played a move outside its own candidate set";
}

TEST_F(SimAgentTest, OneSeedGivesOneGame) {
  SimAgent a(params()), b(params());
  a.begin_game({});
  b.begin_game({});
  // Step both through the same few turns: the seed advances with the ply
  // count, so agreeing once is weaker than agreeing as the game moves on.
  for (int turn = 0; turn < 3; ++turn) {
    const Move ma = a.make_move(request()).move;
    const Move mb = b.make_move(request()).move;
    ASSERT_TRUE(ma == mb) << "turn " << turn;
    a.observe_move(ma);
    b.observe_move(mb);
  }
}

TEST_F(SimAgentTest, AnEmptyBagFallsBackToStaticEquity) {
  SimAgent agent(params());  // endgame budget 0, so the solver declines
  agent.begin_game({});
  MoveRequest req{board_,          dict_,         my_rack_, opp_leave_, /*my_score=*/13,
                  /*opp_score=*/7, /*bag_size=*/0};
  const Move played = agent.make_move(req).move;
  EXPECT_TRUE(played == equity_top_k(req, params().top_k).front());
}

TEST_F(SimAgentTest, AnUnusableRolloutCountIsRejected) {
  // SimRunner once only ASSERTED its bound, so a Release build accepted
  // --rollouts=0 and ran none: every observation mean became 0/0, and the NaN
  // comparisons made the agent play its first candidate every turn without a
  // word of complaint. The bound throws now, from the constructor.
  const auto build = [&](int rollouts) {
    SimAgent::Params p = params();
    p.sim.rollouts = rollouts;
    return SimAgent(p);
  };
  EXPECT_THROW(build(0), std::runtime_error);
  EXPECT_THROW(build(SimRunner::kMaxRollouts + 1), std::runtime_error);
  EXPECT_NO_THROW(build(1));
  EXPECT_NO_THROW(build(SimRunner::kMaxRollouts));
}

TEST_F(SimAgentTest, AnUnusableCandidateCapIsRejected) {
  // The same rule one level down, where every simming caller gets its cap
  // checked whatever it validated for itself: k == 0 silently returns no
  // candidates at all, and k < 0 walks partial_sort's middle iterator before
  // the range's start.
  EXPECT_THROW(equity_top_k(request(), 0), std::runtime_error);
  EXPECT_THROW(equity_top_k(request(), -1), std::runtime_error);
  EXPECT_EQ(equity_top_k(request(), 1).size(), 1u);
}
