// Unit tests for NeuralSimAgent, the position-evaluation-top-K agent, with the
// model replaced by a scripted EvalService stub -- no ONNX, no TensorRT, no
// GPU.
//
//  * the sim set is the scripted model's top K, not static equity's: the agent
//    plays the rollouts' favourite among them, checked by replaying the
//    decision through SimRunner with the agent's own seed.
//  * the shortlist caps what the model evaluates, and shortlist 0 evaluates
//    every candidate -- exchanges included, which the model may promote.
//  * drop-best-prob=1 excludes the model's top-ranked candidate from the sim
//    set; 0 keeps it.
//  * two agents on one seed agree.
//  * a bag-empty turn with solving disabled falls back to static equity
//    without consulting the model.

#include "agent/agent.h"
#include "agent/neural_sim_agent.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "selfplay/sim_runner.h"
#include "sim_agent_fixture.h"
#include "stub_eval_service.h"
#include "synthetic_equity.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace scribblez;
using scribblez::testing::CountingStubEvalService;
using scribblez::testing::model_rank;
using scribblez::testing::opening_dict;
using scribblez::testing::rack_from;
using scribblez::testing::script_favouring;
using scribblez::testing::shortlist_candidates;
using scribblez::testing::StubEvalService;

namespace {

class NeuralSimAgentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tmp_ = std::filesystem::temp_directory_path() / "scribblez_test_neural_sim_agent_XXXXXX";
    std::filesystem::create_directories(tmp_);
    scribblez::testing::install_synthetic_hasty_equity(tmp_);
  }
  void TearDown() override { std::filesystem::remove_all(tmp_); }

  NeuralSimAgent::Params params() const {
    NeuralSimAgent::Params p;
    p.name = "NS";
    p.dict = &dict_;
    p.shortlist = 0;  // every legal candidate, so tests can script them all
    p.sim_top_k = 2;
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

}  // namespace

TEST_F(NeuralSimAgentTest, SimsTheModelsTopKAndPlaysTheRolloutsFavourite) {
  const NeuralSimAgent::Params p = params();
  const std::vector<Move> candidates = shortlist_candidates(request(), p.shortlist);
  ASSERT_GT(candidates.size(), 4u);

  // The model favours the equity ranking's 3rd and 4th candidates; equity's
  // own favourites score low, so the sim set differs from SimAgent's.
  const std::vector<nn::Eval> scripted = script_favouring(candidates.size(), {2, 3});

  auto stub = std::make_unique<StubEvalService>();
  stub->scripted = scripted;
  NeuralSimAgent agent(p, std::move(stub), /*max_batch=*/1024);
  agent.begin_game({});
  const Move played = agent.make_move(request()).move;

  // Replay the same decision independently: the model's top-2 by scripted
  // value, simmed from the same position with the agent's own seed.
  const std::vector<int> rank = model_rank(scripted, p.rank_objective);
  const std::vector<Move> simmed = {candidates[static_cast<size_t>(rank[0])],
                                    candidates[static_cast<size_t>(rank[1])]};
  SimPosition pos;
  pos.board = board_;
  pos.mover = 0;
  pos.scores = {13, 7};
  pos.rack = my_rack_;
  pos.opp_leave = opp_leave_;
  const std::vector<SimObservation> obs =
    SimRunner(dict_, p.sim).run(pos, simmed, agent.sim_seed(0));

  EXPECT_TRUE(played == simmed[static_cast<size_t>(best_observation_index(obs, p.sim_objective))]);
}

TEST_F(NeuralSimAgentTest, ShortlistCapsWhatTheModelEvaluates) {
  NeuralSimAgent::Params p = params();
  p.shortlist = 3;
  p.sim_top_k = 1;  // the model's favourite plays without any rollouts
  const std::vector<Move> candidates = shortlist_candidates(request(), p.shortlist);
  ASSERT_EQ(candidates.size(), 3u);

  auto stub = std::make_unique<CountingStubEvalService>();
  CountingStubEvalService* sp = stub.get();
  sp->scripted = script_favouring(candidates.size(), {1});
  NeuralSimAgent agent(p, std::move(stub), /*max_batch=*/1024);
  agent.begin_game({});

  const Move played = agent.make_move(request()).move;
  EXPECT_EQ(sp->total_rows, 3);  // exactly the shortlist reached the model
  EXPECT_TRUE(played == candidates[1]);
}

TEST_F(NeuralSimAgentTest, TheModelCanPromoteAnExchange) {
  NeuralSimAgent::Params p = params();
  p.sim_top_k = 1;
  const std::vector<Move> candidates = shortlist_candidates(request(), p.shortlist);

  // With shortlist 0 the candidate space includes every exchange; find one
  // (static equity buries them all far below the plays on this rack).
  int exchange_idx = -1;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].type() == MoveType::EXCHANGE) {
      exchange_idx = static_cast<int>(i);
      break;
    }
  }
  ASSERT_GE(exchange_idx, 0) << "no exchange candidate; the check would be vacuous";

  auto stub = std::make_unique<StubEvalService>();
  stub->scripted = script_favouring(candidates.size(), {exchange_idx});
  NeuralSimAgent agent(p, std::move(stub), /*max_batch=*/1024);
  agent.begin_game({});
  EXPECT_TRUE(agent.make_move(request()).move == candidates[static_cast<size_t>(exchange_idx)]);
}

TEST_F(NeuralSimAgentTest, DropBestProbExcludesTheModelsFavourite) {
  NeuralSimAgent::Params p = params();
  p.sim_top_k = 1;  // sim set of one: the drop decides the move outright
  const std::vector<Move> candidates = shortlist_candidates(request(), p.shortlist);
  const std::vector<nn::Eval> scripted = script_favouring(candidates.size(), {2, 3});

  for (const double prob : {0.0, 1.0}) {
    auto stub = std::make_unique<StubEvalService>();
    stub->scripted = scripted;
    p.drop_best_prob = prob;
    NeuralSimAgent agent(p, std::move(stub), /*max_batch=*/1024);
    agent.begin_game({});
    EXPECT_EQ(agent.drop_best(0), prob == 1.0);
    // Undropped, the model's favourite (equity rank 2) plays; dropped, the
    // runner-up (equity rank 3) does.
    const Move expected = candidates[prob == 1.0 ? 3u : 2u];
    EXPECT_TRUE(agent.make_move(request()).move == expected) << "drop_best_prob=" << prob;
  }
}

TEST_F(NeuralSimAgentTest, OneSeedGivesOneDecision) {
  const NeuralSimAgent::Params p = params();
  const std::vector<Move> candidates = shortlist_candidates(request(), p.shortlist);
  const std::vector<nn::Eval> scripted = script_favouring(candidates.size(), {1, 4});

  Move moves[2];
  for (int i = 0; i < 2; ++i) {
    auto stub = std::make_unique<StubEvalService>();
    stub->scripted = scripted;
    NeuralSimAgent agent(p, std::move(stub), /*max_batch=*/1024);
    agent.begin_game({});
    moves[i] = agent.make_move(request()).move;
  }
  EXPECT_TRUE(moves[0] == moves[1]);
}

TEST_F(NeuralSimAgentTest, ASoleCandidatePlaysWithoutModelOrRollouts) {
  // A rack with no legal play and a bag too small for exchanges leaves a lone
  // PASS candidate. drop_best_prob=1 makes this the sharpest edge: were the
  // sole candidate ranked and dropped, the sim set would be empty -- the
  // size-1 early return must fire before either can happen.
  NeuralSimAgent::Params p = params();
  p.drop_best_prob = 1.0;
  auto stub = std::make_unique<CountingStubEvalService>();
  CountingStubEvalService* sp = stub.get();
  NeuralSimAgent agent(p, std::move(stub), /*max_batch=*/1024);
  agent.begin_game({});

  const Rack unplayable = rack_from("QQQQQQ");  // no dict word uses Q
  MoveRequest req{board_,          dict_,         unplayable, opp_leave_, /*my_score=*/13,
                  /*opp_score=*/7, /*bag_size=*/3};  // < RACK_SIZE: exchanges illegal
  const Move played = agent.make_move(req).move;
  EXPECT_EQ(played.type(), MoveType::PASS);
  EXPECT_EQ(sp->total_rows, 0);  // the model was never consulted
}

TEST_F(NeuralSimAgentTest, SimTopKLargerThanTheCandidateSetIsCapped) {
  // sim_top_k above the candidate count (here also shaved by a certain drop)
  // must clamp to what exists: the drop excludes the model's favourite, the
  // remaining two candidates sim, and one of them plays.
  NeuralSimAgent::Params p = params();
  p.shortlist = 3;
  p.sim_top_k = 10;
  p.drop_best_prob = 1.0;
  const std::vector<Move> candidates = shortlist_candidates(request(), p.shortlist);
  ASSERT_EQ(candidates.size(), 3u);

  auto stub = std::make_unique<CountingStubEvalService>();
  CountingStubEvalService* sp = stub.get();
  sp->scripted = script_favouring(candidates.size(), {1});
  NeuralSimAgent agent(p, std::move(stub), /*max_batch=*/1024);
  agent.begin_game({});

  const Move played = agent.make_move(request()).move;
  EXPECT_EQ(sp->total_rows, 3);
  // The favourite (equity rank 1) was dropped; the survivors are ranks 0 and 2.
  EXPECT_TRUE(played == candidates[0] || played == candidates[2]);
  EXPECT_FALSE(played == candidates[1]);
}

TEST_F(NeuralSimAgentTest, AnEmptyBagFallsBackToStaticEquity) {
  auto stub = std::make_unique<CountingStubEvalService>();
  CountingStubEvalService* sp = stub.get();
  NeuralSimAgent agent(params(), std::move(stub),
                       /*max_batch=*/1024);  // endgame budget 0: solver declines
  agent.begin_game({});
  MoveRequest req{board_,          dict_,         my_rack_, opp_leave_, /*my_score=*/13,
                  /*opp_score=*/7, /*bag_size=*/0};
  const Move played = agent.make_move(req).move;
  EXPECT_TRUE(played == equity_top_k(req, 1).front());
  EXPECT_EQ(sp->total_rows, 0);  // the model was never consulted
}

TEST_F(NeuralSimAgentTest, AnUnusableRolloutCountIsRejected) {
  // As for SimAgent: the bound SimRunner only asserts, enforced here so a
  // Release build cannot quietly stop simulating (see sim_runner.h).
  const auto build = [&](int rollouts) {
    NeuralSimAgent::Params p = params();
    p.sim.rollouts = rollouts;
    return NeuralSimAgent(p, std::make_unique<StubEvalService>(), /*max_batch=*/1024);
  };
  EXPECT_THROW(build(0), std::runtime_error);
  EXPECT_THROW(build(SimRunner::kMaxRollouts + 1), std::runtime_error);
  EXPECT_NO_THROW(build(1));
  EXPECT_NO_THROW(build(SimRunner::kMaxRollouts));
}
