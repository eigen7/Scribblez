// MsetSimAgent, which sims the move-set evaluation model's top K candidates,
// with the model replaced by a scripted stub (no GPU). Besides the selection
// checks it shares with the NeuralSimAgent suite, it pins the agent's side of
// the model contract: one service call carries the whole candidate set, each
// candidate's features are what move_set::encode_move produces, and the board
// row matches the one the training decoder reconstructs. Chunking a set larger
// than the engine is the service's job and is tested in
// test_mset_inference_parity.

#include "agent/agent.h"
#include "agent/mset_sim_agent.h"
#include "agent_parity_fixture.h"
#include "encoding/input_encoder.h"
#include "game/board.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "game_fixture.h"
#include "lexicon/dictionary.h"
#include "sim/sim_runner.h"
#include "sim_agent_fixture.h"
#include "stub_eval_service.h"
#include "stub_move_set_eval_service.h"
#include "synthetic_equity.h"
#include "training/move_set_encoder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace scribblez;
using scribblez::testing::check_pre_move_row_matches_decoder;
using scribblez::testing::expect_move_features_match;
using scribblez::testing::make_play_full;
using scribblez::testing::model_rank;
using scribblez::testing::opening_dict;
using scribblez::testing::rack_from;
using scribblez::testing::script_favouring;
using scribblez::testing::shortlist_candidates;
using scribblez::testing::StubMoveSetEvalService;

namespace {

// The stub declares the base input layout.
const int kInputFloats = input_floats(InputEncodingSpec{nullptr});

class MsetSimAgentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tmp_ = std::filesystem::temp_directory_path() / "scribblez_test_mset_sim_agent_XXXXXX";
    std::filesystem::create_directories(tmp_);
    scribblez::testing::install_synthetic_hasty_equity(tmp_);
  }
  void TearDown() override { std::filesystem::remove_all(tmp_); }

  MsetSimAgent::Params params() const {
    MsetSimAgent::Params p;
    p.name = "MS";
    p.dict = &dict_;
    p.sim_top_k = 2;
    p.sim.rollouts = 8;
    p.sim.threads = 1;
    p.seed = 12345;
    p.endgame.budget = 0;  // the endgame is not what these tests are about
    return p;
  }

  // An opening turn with a non-empty opponent leave, so the position handed to
  // the simulator has every field the agent is responsible for filling.
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

TEST_F(MsetSimAgentTest, OutOfRangeScalarParamsAreRejected) {
  // Every guard in validate(), at its boundaries. Zero rollouts is the
  // dangerous one: without the guard the agent would play the model's top
  // candidate every turn without complaint.
  const auto build = [&](int shortlist, int sim_top_k, int rollouts) {
    MsetSimAgent::Params p = params();
    p.shortlist = shortlist;
    p.sim_top_k = sim_top_k;
    p.sim.rollouts = rollouts;
    return MsetSimAgent(p, std::make_unique<StubMoveSetEvalService>());
  };

  EXPECT_THROW(build(-1, 2, 8), std::runtime_error);  // --shortlist
  EXPECT_THROW(build(0, 0, 8), std::runtime_error);   // --sim-top-k
  EXPECT_THROW(build(0, 2, 0), std::runtime_error);   // --rollouts, lower
  EXPECT_THROW(build(0, 2, SimRunner::kMaxRollouts + 1), std::runtime_error);

  // The accepted boundaries, so the bounds cannot tighten unnoticed.
  EXPECT_NO_THROW(build(0, 1, 1));
  EXPECT_NO_THROW(build(0, 1, SimRunner::kMaxRollouts));
}

// The agent must forward sim_horizon to its SimRunner, which validates it at
// construction: it needs a leaf service and must meet the minimum. An agent
// that dropped sim_horizon would run terminal rollouts and throw in neither
// case.
TEST_F(MsetSimAgentTest, TruncationHorizonIsWiredToTheRunner) {
  using scribblez::testing::StubEvalService;
  MsetSimAgent::Params p = params();
  p.sim_horizon = SimRunner::kMinHorizonPlies;  // no leaf service
  EXPECT_THROW(MsetSimAgent(p, std::make_unique<StubMoveSetEvalService>()), std::runtime_error);
  p.sim_horizon = SimRunner::kMinHorizonPlies - 1;  // below the minimum, even with a leaf
  EXPECT_THROW(MsetSimAgent(p, std::make_unique<StubMoveSetEvalService>(),
                            std::make_shared<StubEvalService>()),
               std::runtime_error);
  p.sim_horizon = SimRunner::kMinHorizonPlies;
  EXPECT_NO_THROW(MsetSimAgent(p, std::make_unique<StubMoveSetEvalService>(),
                               std::make_shared<StubEvalService>()));
}

TEST_F(MsetSimAgentTest, SimsTheModelsTopKAndPlaysTheRolloutsFavourite) {
  const MsetSimAgent::Params p = params();
  const std::vector<Move> candidates = shortlist_candidates(request(), p.shortlist);
  ASSERT_GT(candidates.size(), 4u);

  // The model favours equity's 3rd and 4th candidates, so its sim set differs
  // from the one SimAgent would pick.
  const auto scripted = script_favouring(candidates.size(), {2, 3});

  auto stub = std::make_unique<StubMoveSetEvalService>();
  stub->scripted = scripted;
  MsetSimAgent agent(p, std::move(stub));
  agent.begin_game({});
  const Move played = agent.make_move(request()).move;

  const std::vector<int> rank = model_rank(scripted, p.rank_objective);
  const std::vector<Move> simmed = {candidates[size_t(rank[0])], candidates[size_t(rank[1])]};
  SimPosition pos;
  pos.board = board_;
  pos.mover = 0;
  pos.scores = {13, 7};
  pos.rack = my_rack_;
  pos.opp_leave = opp_leave_;
  const std::vector<SimObservation> obs =
    SimRunner(dict_, p.sim).run(pos, simmed, agent.sim_seed(0));

  EXPECT_TRUE(played == simmed[size_t(best_observation_index(obs, p.sim_objective))]);
}

TEST_F(MsetSimAgentTest, ShortlistCapsWhatTheModelScores) {
  MsetSimAgent::Params p = params();
  p.shortlist = 3;
  p.sim_top_k = 1;  // the model's favourite plays without any rollouts
  const std::vector<Move> candidates = shortlist_candidates(request(), p.shortlist);
  ASSERT_EQ(candidates.size(), 3u);

  auto stub = std::make_unique<StubMoveSetEvalService>();
  StubMoveSetEvalService* sp = stub.get();
  sp->scripted = script_favouring(candidates.size(), {1});
  MsetSimAgent agent(p, std::move(stub));
  agent.begin_game({});

  const Move played = agent.make_move(request()).move;
  EXPECT_EQ(sp->total_moves, 3);  // exactly the shortlist reached the model
  EXPECT_TRUE(played == candidates[1]);
}

TEST_F(MsetSimAgentTest, TheModelCanPromoteAnExchange) {
  MsetSimAgent::Params p = params();
  p.sim_top_k = 1;
  const std::vector<Move> candidates = shortlist_candidates(request(), p.shortlist);

  // The default shortlist includes every exchange, which static equity ranks
  // far below the plays on this rack.
  int exchange_idx = -1;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].type() == MoveType::EXCHANGE) {
      exchange_idx = i;
      break;
    }
  }
  ASSERT_GE(exchange_idx, 0) << "no exchange candidate; the check would be vacuous";

  auto stub = std::make_unique<StubMoveSetEvalService>();
  stub->scripted = script_favouring(candidates.size(), {exchange_idx});
  MsetSimAgent agent(p, std::move(stub));
  agent.begin_game({});
  EXPECT_TRUE(agent.make_move(request()).move == candidates[size_t(exchange_idx)]);
}

TEST_F(MsetSimAgentTest, OneSeedGivesOneDecision) {
  const MsetSimAgent::Params p = params();
  const std::vector<Move> candidates = shortlist_candidates(request(), p.shortlist);
  const auto scripted = script_favouring(candidates.size(), {1, 4});

  Move moves[2];
  for (int i = 0; i < 2; ++i) {
    auto stub = std::make_unique<StubMoveSetEvalService>();
    stub->scripted = scripted;
    MsetSimAgent agent(p, std::move(stub));
    agent.begin_game({});
    moves[i] = agent.make_move(request()).move;
  }
  EXPECT_TRUE(moves[0] == moves[1]);
}

TEST_F(MsetSimAgentTest, ASoleCandidatePlaysWithoutModelOrRollouts) {
  // No legal play and a bag too small to exchange leave PASS as the only
  // candidate.
  auto stub = std::make_unique<StubMoveSetEvalService>();
  StubMoveSetEvalService* sp = stub.get();
  MsetSimAgent agent(params(), std::move(stub));
  agent.begin_game({});

  const Rack unplayable = rack_from("QQQQQQ");  // no dict word uses Q
  MoveRequest req{board_,          dict_,         unplayable, opp_leave_, /*my_score=*/13,
                  /*opp_score=*/7, /*bag_size=*/3};  // < RACK_SIZE: exchanges illegal
  const Move played = agent.make_move(req).move;
  EXPECT_EQ(played.type(), MoveType::PASS);
  EXPECT_EQ(sp->calls, 0);  // the model was never consulted
}

TEST_F(MsetSimAgentTest, SimTopKLargerThanTheCandidateSetIsCapped) {
  // sim_top_k above the candidate count clamps to what exists.
  MsetSimAgent::Params p = params();
  p.shortlist = 3;
  p.sim_top_k = 10;
  const std::vector<Move> candidates = shortlist_candidates(request(), p.shortlist);
  ASSERT_EQ(candidates.size(), 3u);

  auto stub = std::make_unique<StubMoveSetEvalService>();
  StubMoveSetEvalService* sp = stub.get();
  sp->scripted = script_favouring(candidates.size(), {1});
  MsetSimAgent agent(p, std::move(stub));
  agent.begin_game({});

  const Move played = agent.make_move(request()).move;
  EXPECT_EQ(sp->total_moves, 3);
  EXPECT_TRUE(
    std::any_of(candidates.begin(), candidates.end(), [&](const Move& m) { return m == played; }));
}

TEST_F(MsetSimAgentTest, AnEmptyBagFallsBackToStaticEquity) {
  auto stub = std::make_unique<StubMoveSetEvalService>();
  StubMoveSetEvalService* sp = stub.get();
  // With endgame budget 0 the solver declines, and there is no bag to sim from.
  MsetSimAgent agent(params(), std::move(stub));
  agent.begin_game({});
  MoveRequest req{board_,          dict_,         my_rack_, opp_leave_, /*my_score=*/13,
                  /*opp_score=*/7, /*bag_size=*/0};
  const Move played = agent.make_move(req).move;
  EXPECT_TRUE(played == equity_top_k(req, 1).front());
  EXPECT_EQ(sp->calls, 0);  // the model was never consulted
}

TEST_F(MsetSimAgentTest, TheRolloutSeedFollowsTheAdvancingPly) {
  // The other rollout tests decide at ply 0, where seeding from the current ply
  // and a hardcoded ply 0 look the same. Here two moves are observed first, and
  // the position is chosen so the two seeds pick different candidates.
  const MsetSimAgent::Params p = params();
  const Move opening =
    make_play_full(7, 7, /*horizontal=*/true, 0b111, 10,
                   {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                    Glyph::of(Tile::from_char('T'))});
  const Move reply =
    make_play_full(6, 7, /*horizontal=*/false, 0b11, 5,
                   {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('T'))});

  Board board;
  board.apply(opening);
  board.apply(reply);
  const MoveRequest req{board,
                        dict_,
                        my_rack_,
                        opp_leave_,
                        /*my_score=*/5,
                        /*opp_score=*/10,
                        /*bag_size=*/72};
  const std::vector<Move> candidates = shortlist_candidates(req, p.shortlist);
  ASSERT_GT(candidates.size(), 4u);
  const auto scripted = script_favouring(candidates.size(), {2, 3});

  auto stub = std::make_unique<StubMoveSetEvalService>();
  stub->scripted = scripted;
  MsetSimAgent agent(p, std::move(stub));
  agent.begin_game({});
  agent.observe_move(opening);
  agent.observe_move(reply);
  const Move played = agent.make_move(req).move;

  const std::vector<int> rank = model_rank(scripted, p.rank_objective);
  const std::vector<Move> simmed = {candidates[size_t(rank[0])], candidates[size_t(rank[1])]};
  const SimRunner runner(dict_, p.sim);
  const SimPosition pos = sim_position_from(req);
  const Move at_ply_2 = simmed[size_t(
    best_observation_index(runner.run(pos, simmed, agent.sim_seed(2)), p.sim_objective))];
  const Move at_ply_0 = simmed[size_t(
    best_observation_index(runner.run(pos, simmed, agent.sim_seed(0)), p.sim_objective))];

  ASSERT_FALSE(at_ply_2 == at_ply_0) << "the two plies agree here; the test proves nothing";
  EXPECT_TRUE(played == at_ply_2);
}

TEST_F(MsetSimAgentTest, TheWholeCandidateSetGoesToTheModelInOnePass) {
  // The opponent opened for 10, so the move features see a non-zero pre-move
  // score differential.
  const Move opening =
    make_play_full(7, 7, /*horizontal=*/true, 0b111, 10,
                   {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                    Glyph::of(Tile::from_char('T'))});
  const int pre_diff = -10;

  Board board;
  board.apply(opening);
  const MsetSimAgent::Params p = params();
  const MoveRequest req{board,
                        dict_,
                        my_rack_,
                        opp_leave_,
                        /*my_score=*/0,
                        /*opp_score=*/10,
                        /*bag_size=*/86};
  const std::vector<Move> candidates = shortlist_candidates(req, p.shortlist);

  auto stub = std::make_unique<StubMoveSetEvalService>();
  StubMoveSetEvalService* sp = stub.get();
  sp->scripted = script_favouring(candidates.size(), {0});
  MsetSimAgent agent(p, std::move(stub));
  agent.begin_game({});
  agent.observe_move(opening);
  agent.make_move(req);

  // Amortizing one board encode over the whole set is why this model exists.
  ASSERT_EQ(sp->calls, 1);
  ASSERT_EQ(sp->last_moves.count, int(candidates.size()));

  // The board row is the position's pre-move row, which
  // ThePreMoveRowMatchesTheTrainingDecoder ties to the training row.
  std::vector<float> expected_row(size_t(kInputFloats), 0.0f);
  agent.encode_board_row(req, expected_row.data());
  EXPECT_EQ(sp->last_board_row, expected_row);

  // Each candidate's features must be what encode_move, the encoder the
  // training rows go through, makes of it at this differential.
  expect_move_features_match(sp->last_moves, candidates, pre_diff);
}

TEST(MsetSimAgent, ThePreMoveRowMatchesTheTrainingDecoder) {
  check_pre_move_row_matches_decoder<MsetSimAgent, StubMoveSetEvalService>({0, 0});
}

TEST(MsetSimAgent, AHandicapReachesTheModelRow) {
  // The training replay seeds its scores from the handicap the .slog records.
  // An agent that started every game at 0-0 would feed the model a score
  // differential wrong by the head start all game, with nothing else in the row
  // to reveal it.
  check_pre_move_row_matches_decoder<MsetSimAgent, StubMoveSetEvalService>({50, 0});
  check_pre_move_row_matches_decoder<MsetSimAgent, StubMoveSetEvalService>({0, 37});
}
