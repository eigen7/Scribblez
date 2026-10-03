// SimAgent, the Monte-Carlo simming baseline. The central check replays the
// agent's decision through SimRunner directly: agreement pins the position,
// candidates and seed the agent hands the simulator. Runs off a synthetic leave
// table, so it needs no data mount.

#include "agent/agent.h"
#include "agent/sim_agent.h"
#include "encoding/input_encoder.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "nn/eval_service.h"
#include "sim/sim_runner.h"
#include "synthetic_equity.h"
#include "temp_dir.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace scribblez;

namespace {

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
    tmp_ = scribblez::testing::make_temp_dir("scribblez_test_sim_agent");
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

  // The request's position as the simulator takes it.
  SimPosition position() const {
    SimPosition pos;
    pos.board = board_;
    pos.mover = 0;
    pos.scores = {13, 7};
    pos.rack = my_rack_;
    pos.opp_leave = opp_leave_;
    return pos;
  }

  Dictionary dict_ = opening_dict();
  Board board_;
  Rack my_rack_ = Rack::from_string("CARTES");
  Rack opp_leave_ = Rack::from_string("AE");
  int bag_size_ = 86;
  std::filesystem::path tmp_;
};

// A deterministic leaf evaluator for the truncation test. Its outputs are a
// function of the row's score-diff input, so different horizon states get
// different values and a replay reproduces them.
class LeafStub : public nn::PositionEvalService {
 public:
  bool opp_leave_input() const override { return false; }
  int spatial_planes() const override { return scribblez::spatial_planes(); }
  int scalar_floats() const override { return scribblez::scalar_floats({nullptr}); }
  void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) override {
    const InputEncodingSpec spec{nullptr};
    const size_t row_floats = input_floats(spec);
    const size_t sd_off = spatial_floats() + scalar_block_offset(spec, ScalarBlockId::kScoreDiff);
    for (int i = 0; i < batch.count; ++i) {
      const float sd = batch.rows[size_t(i) * row_floats + sd_off];
      const float w = 0.5f + 0.4f * std::tanh(sd);
      float* wld = head_out[0] + size_t(i) * nn::WldOutput::kRowElems;
      wld[0] = w;
      wld[1] = 0.1f;
      wld[2] = 0.9f - w;
      float* out_sd = head_out[1] + size_t(i) * nn::ScoreDiffOutput::kRowElems;
      out_sd[0] = sd * kScoreDiffInputScale;
      out_sd[1] = 5.0f;
    }
  }
};

}  // namespace

TEST_F(SimAgentTest, PlaysTheCandidateItsOwnRolloutsRankBest) {
  for (SimObjective objective : {SimObjective::kWinRate, SimObjective::kSpread}) {
    SimAgent::Params p = params();
    p.objective = objective;
    SimAgent agent(p);
    agent.begin_game({});
    const Move played = agent.make_move(request()).move;

    // A disagreement means the agent handed the simulator something other than
    // the turn it was asked about.
    const std::vector<Move> candidates = equity_top_k(request(), p.top_k);
    ASSERT_GT(candidates.size(), 1u);
    SimPosition pos;
    pos.board = board_;
    pos.mover = 0;
    pos.scores = {13, 7};
    pos.rack = my_rack_;
    pos.opp_leave = opp_leave_;
    const std::vector<RolloutStats> obs =
      SimRunner(dict_, p.sim).run(pos, candidates, agent.sim_seed(0));

    EXPECT_TRUE(played == candidates[size_t(best_observation_index(obs, objective))])
      << "objective=" << int(objective);
  }
}

TEST_F(SimAgentTest, SimulatesAgainstTheOpponentsPublicLeave) {
  // A leave matters only where it changes which candidate the rollouts favour,
  // which no single position guarantees. So search for a leave that flips the
  // winner between leave-aware and leave-blind runs, then check the agent
  // followed the leave-aware one. Finding none fails the test, since the check
  // would otherwise be vacuous.
  const SimAgent::Params p = params();
  const std::vector<std::string> leaves = {"AE", "SS", "AAAA", "CARTES", "EEEEEE"};

  bool discriminated = false;
  for (const std::string& text : leaves) {
    const Rack leave = Rack::from_string(text);
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
    const int with =
      best_observation_index(runner.run(pos, candidates, SimAgent(p).sim_seed(0)), p.objective);
    const int without =
      best_observation_index(runner.run(blind, candidates, SimAgent(p).sim_seed(0)), p.objective);
    if (with == without) continue;

    discriminated = true;
    SimAgent agent(p);
    agent.begin_game({});
    EXPECT_TRUE(agent.make_move(req).move == candidates[size_t(with)])
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
  // The seed advances with the ply count, so agreeing on several turns is a
  // stronger check than agreeing on one.
  for (int turn = 0; turn < 3; ++turn) {
    const Move ma = a.make_move(request()).move;
    const Move mb = b.make_move(request()).move;
    ASSERT_TRUE(ma == mb) << "turn " << turn;
    a.observe_move(ma);
    b.observe_move(mb);
  }
}

TEST_F(SimAgentTest, AnEmptyBagFallsBackToStaticEquity) {
  // With endgame budget 0 the solver declines, and there is no bag to sim from.
  SimAgent agent(params());
  agent.begin_game({});
  MoveRequest req{board_,          dict_,         my_rack_, opp_leave_, /*my_score=*/13,
                  /*opp_score=*/7, /*bag_size=*/0};
  const Move played = agent.make_move(req).move;
  EXPECT_TRUE(played == equity_top_k(req, params().top_k).front());
}

// With a horizon and a leaf service, the decision reproduces through a SimRunner
// configured the same way: the agent passes both to the simulator unchanged.
TEST_F(SimAgentTest, TruncatedRolloutsReproduceThroughSimRunner) {
  SimAgent::Params p = params();
  p.sim_horizon = 4;
  SimAgent agent(p, std::make_shared<LeafStub>());
  agent.begin_game({});
  const Move played = agent.make_move(request()).move;

  const std::vector<Move> candidates = equity_top_k(request(), p.top_k);
  ASSERT_GT(candidates.size(), 1u);
  SimPosition pos;
  pos.board = board_;
  pos.mover = 0;
  pos.scores = {13, 7};
  pos.rack = my_rack_;
  pos.opp_leave = opp_leave_;
  LeafStub replay_leaf;
  SimRunner::Params sp = p.sim;
  sp.horizon_plies = p.sim_horizon;
  sp.leaf_service = &replay_leaf;
  const std::vector<RolloutStats> obs =
    SimRunner(dict_, sp).run(pos, candidates, agent.sim_seed(0));
  EXPECT_TRUE(played == candidates[size_t(best_observation_index(obs, p.objective))]);
}

// A horizon without a leaf service, or a leaf service without a horizon, is
// rejected at construction.
TEST_F(SimAgentTest, TruncationPairingIsValidated) {
  SimAgent::Params p = params();
  p.sim_horizon = 4;
  EXPECT_THROW(SimAgent agent(p), std::runtime_error);
  p.sim_horizon = 0;
  EXPECT_THROW(SimAgent agent(p, std::make_unique<LeafStub>()), std::runtime_error);
}

TEST_F(SimAgentTest, AnUnusableRolloutCountIsRejected) {
  // The bound must throw in every build type, not just assert in Debug. With
  // zero rollouts every observation mean is 0/0, and the NaN comparisons make
  // the agent play its first candidate every turn without complaint.
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
  // Checked in equity_top_k itself, so every simming caller is covered. k == 0
  // would silently return no candidates, and k < 0 would move partial_sort's
  // middle iterator before the start of the range.
  EXPECT_THROW(equity_top_k(request(), 0), std::runtime_error);
  EXPECT_THROW(equity_top_k(request(), -1), std::runtime_error);
  EXPECT_EQ(equity_top_k(request(), 1).size(), 1u);
}

// A rollout's trace is that rollout step by step: the opponent replies first,
// from the rack the Rollout records, and a truncated rollout stops after the
// horizon's plies. Asking for traces changes no outcome.
TEST_F(SimAgentTest, RolloutTracesMatchTheirRollouts) {
  const std::vector<Move> candidates = equity_top_k(request(), 3);
  ASSERT_GT(candidates.size(), 1u);
  LeafStub leaf;
  for (const int horizon : {0, SimRunner::kMinHorizonPlies}) {
    SimRunner::Params sp;
    sp.rollouts = 6;
    sp.horizon_plies = horizon;
    sp.leaf_service = horizon > 0 ? &leaf : nullptr;
    const SimRunner runner(dict_, sp);
    std::vector<RolloutTrace> traces;
    const std::vector<Rollout> traced =
      runner.run_rollouts(position(), candidates, 7, sp.rollouts, &traces);
    const std::vector<Rollout> plain = runner.run_rollouts(position(), candidates, 7);
    ASSERT_EQ(traces.size(), traced.size());
    for (size_t k = 0; k < traced.size(); ++k) {
      const Rollout& r = traced[k];
      const RolloutTrace& t = traces[k];
      EXPECT_EQ(r.p_win, plain[k].p_win);
      EXPECT_EQ(r.delta, plain[k].delta);
      ASSERT_GE(t.turns.size(), 2u);
      EXPECT_EQ(t.turns[0].player, 1);
      EXPECT_TRUE(t.turns[0].rack_before == r.opp_rack);
      EXPECT_TRUE(t.turns[0].move == r.opp_reply);
      EXPECT_TRUE(t.turns[1].move == r.self_next);
      EXPECT_EQ(t.truncated, horizon > 0);
      if (horizon > 0) EXPECT_EQ(int(t.turns.size()), horizon);
    }
  }
}

TEST_F(SimAgentTest, RolloutTracesDoNotDependOnTheThreadCount) {
  const std::vector<Move> candidates = equity_top_k(request(), 3);
  LeafStub leaf;
  auto traces_with = [&](int threads) {
    SimRunner::Params sp;
    sp.rollouts = 8;
    sp.threads = threads;
    sp.horizon_plies = SimRunner::kMinHorizonPlies;
    sp.leaf_service = &leaf;
    std::vector<RolloutTrace> traces;
    SimRunner(dict_, sp).run_rollouts(position(), candidates, 11, sp.rollouts, &traces);
    return traces;
  };
  const std::vector<RolloutTrace> one = traces_with(1);
  const std::vector<RolloutTrace> many = traces_with(3);
  ASSERT_EQ(one.size(), many.size());
  for (size_t k = 0; k < one.size(); ++k) {
    ASSERT_EQ(one[k].turns.size(), many[k].turns.size());
    EXPECT_EQ(one[k].truncated, many[k].truncated);
    for (size_t i = 0; i < one[k].turns.size(); ++i) {
      const TurnRecord& a = one[k].turns[i];
      const TurnRecord& b = many[k].turns[i];
      EXPECT_EQ(a.player, b.player);
      EXPECT_TRUE(a.rack_before == b.rack_before);
      EXPECT_EQ(a.bag_size_before, b.bag_size_before);
      EXPECT_TRUE(a.move == b.move);
      EXPECT_EQ(a.score_delta, b.score_delta);
      EXPECT_TRUE(a.drawn == b.drawn);
    }
  }
}
