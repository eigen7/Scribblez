// The evidence loop (agent/evidence_loop.h) and UltimateBotAgent, with the
// move proposal model replaced by a scripted stub (no GPU). The loop sims the
// anchor (the highest-scoring candidate) first, whatever the model says; each
// later sim is the model's highest-gain unsimmed candidate, conditioned on the
// sims so far, until the budget or the gain threshold stops it. The agent then
// plays the rollouts' favourite among the simmed set.
//
// The stub dictates each conditioned pass's gains and records the evidence it
// was shown, so the tests check the sequence of picks directly. The model
// contract checks (one encode per turn, move features, board row) mirror the
// MsetSimAgent suite's.

#include "agent/agent.h"
#include "agent/evidence_loop.h"
#include "agent/ultimate_bot_agent.h"
#include "data/binary_log.h"
#include "data/block_decoder.h"
#include "data/data_loader.h"  // kLabelFloats
#include "encoding/input_encoder.h"
#include "game/board.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "game_fixture.h"
#include "lexicon/dictionary.h"
#include "nn/model_specs.h"
#include "sim/sim_runner.h"
#include "sim_agent_fixture.h"
#include "stub_eval_service.h"
#include "stub_move_proposal_service.h"
#include "synthetic_equity.h"
#include "training/evidence_trajectory_select.h"
#include "training/move_set_encoder.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace scribblez;
using scribblez::agent::EvidenceSet;
using scribblez::testing::build_slog;
using scribblez::testing::make_play_full;
using scribblez::testing::opening_dict;
using scribblez::testing::rack_from;
using scribblez::testing::StubMoveProposalService;

namespace {

// The stub declares the base input layout.
const int kInputFloats = input_floats(InputEncodingSpec{nullptr});

// Gains over `n` candidates, descending along `favoured`, low elsewhere.
std::vector<float> gains_favouring(size_t n, const std::vector<int>& favoured) {
  std::vector<float> g(n, 0.1f);
  float v = 1.0f;
  for (int idx : favoured) {
    g[size_t(idx)] = v;
    v -= 0.2f;
  }
  return g;
}

// `count` candidate indices other than `anchor`, lowest first.
std::vector<int> picks_avoiding(int anchor, int n, int count) {
  std::vector<int> out;
  for (int i = 0; i < n && int(out.size()) < count; ++i)
    if (i != anchor) out.push_back(i);
  return out;
}

bool same_observation(const SimObservation& a, const SimObservation& b) {
  return std::memcmp(&a, &b, sizeof(SimObservation)) == 0;
}

class UltimateBotAgentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tmp_ = std::filesystem::temp_directory_path() / "scribblez_test_ultimate_bot_XXXXXX";
    std::filesystem::create_directories(tmp_);
    scribblez::testing::install_synthetic_hasty_equity(tmp_);
  }
  void TearDown() override { std::filesystem::remove_all(tmp_); }

  UltimateBotAgent::Params params() const {
    UltimateBotAgent::Params p;
    p.name = "UB";
    p.dict = &dict_;
    p.max_sims = 3;
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

  // The agent's candidate space: every legal move, in static-equity order.
  static std::vector<Move> candidates(const MoveRequest& req) {
    return equity_top_k(req, std::numeric_limits<int>::max());
  }

  Dictionary dict_ = opening_dict();
  Board board_;
  Rack my_rack_ = rack_from("CARTES");
  Rack opp_leave_ = rack_from("AE");
  int bag_size_ = 86;
  std::filesystem::path tmp_;
};

}  // namespace

TEST_F(UltimateBotAgentTest, OutOfRangeScalarParamsAreRejected) {
  const auto build = [&](int max_sims, float threshold, int rollouts) {
    UltimateBotAgent::Params p = params();
    p.max_sims = max_sims;
    p.gain_threshold = threshold;
    p.sim.rollouts = rollouts;
    return UltimateBotAgent(p, std::make_unique<StubMoveProposalService>());
  };
  EXPECT_THROW(build(0, 0.0f, 8), std::runtime_error);                     // --max-sims, lower
  EXPECT_THROW(build(nn::kMaxEvidence + 1, 0.0f, 8), std::runtime_error);  // past the width
  EXPECT_THROW(build(3, -0.1f, 8), std::runtime_error);                    // --gain-threshold
  EXPECT_THROW(build(3, 0.0f, 0), std::runtime_error);                     // --rollouts, lower
  EXPECT_THROW(build(3, 0.0f, SimRunner::kMaxRollouts + 1), std::runtime_error);

  // The accepted boundaries, so the bounds cannot tighten unnoticed.
  EXPECT_NO_THROW(build(1, 0.0f, 1));
  EXPECT_NO_THROW(build(nn::kMaxEvidence, 0.0f, SimRunner::kMaxRollouts));
}

// The agent must forward sim_horizon to its SimRunner, which validates it at
// construction: it needs a leaf service and must meet the minimum.
TEST_F(UltimateBotAgentTest, TruncationHorizonIsWiredToTheRunner) {
  using scribblez::testing::StubEvalService;
  UltimateBotAgent::Params p = params();
  p.sim_horizon = SimRunner::kMinHorizonPlies;  // no leaf service
  EXPECT_THROW(UltimateBotAgent(p, std::make_unique<StubMoveProposalService>()),
               std::runtime_error);
  p.sim_horizon = SimRunner::kMinHorizonPlies - 1;  // below the minimum, even with a leaf
  EXPECT_THROW(UltimateBotAgent(p, std::make_unique<StubMoveProposalService>(),
                                std::make_shared<StubEvalService>()),
               std::runtime_error);
  p.sim_horizon = SimRunner::kMinHorizonPlies;
  EXPECT_NO_THROW(UltimateBotAgent(p, std::make_unique<StubMoveProposalService>(),
                                   std::make_shared<StubEvalService>()));
}

TEST_F(UltimateBotAgentTest, TheAnchorIsSimmedFirstAndLaterSimsFollowTheConditionedGain) {
  const UltimateBotAgent::Params p = params();  // max_sims 3
  const std::vector<Move> cands = candidates(request());
  const int n = int(cands.size());
  ASSERT_GT(n, 4);
  const int anchor = int(evidence::anchor_index(cands));
  const std::vector<int> picks = picks_avoiding(anchor, n, 2);

  // The passes favour picks[0], then picks[1]; the anchor's gain is never the
  // highest, yet it is simmed first.
  auto stub = std::make_unique<StubMoveProposalService>();
  StubMoveProposalService* sp = stub.get();
  sp->scripted_gains = {gains_favouring(n, {picks[0]}), gains_favouring(n, {picks[1]})};
  UltimateBotAgent agent(p, std::move(stub));
  agent.begin_game({});
  const Move played = agent.make_move(request()).move;

  EXPECT_EQ(sp->encode_calls, 1);
  // One conditioned pass before each sim after the anchor.
  ASSERT_EQ(sp->condition_calls, 2);
  EXPECT_EQ(sp->seen_evidence[0], std::vector<int>{anchor});
  EXPECT_EQ(sp->seen_evidence[1], (std::vector<int>{anchor, picks[0]}));

  // Replay the three sims as one batch with the agent's seed.
  const std::vector<Move> simmed = {cands[size_t(anchor)], cands[size_t(picks[0])],
                                    cands[size_t(picks[1])]};
  const std::vector<SimObservation> obs =
    SimRunner(dict_, p.sim).run(sim_position_from(request()), simmed, agent.sim_seed(0));
  EXPECT_TRUE(played == simmed[size_t(best_observation_index(obs, SimObjective::kWinRate))]);
}

TEST_F(UltimateBotAgentTest, ASimmedCandidateIsNeverRepicked) {
  const UltimateBotAgent::Params p = params();
  const std::vector<Move> cands = candidates(request());
  const int n = int(cands.size());
  const int anchor = int(evidence::anchor_index(cands));
  const int runner_up = picks_avoiding(anchor, n, 1)[0];

  // Every pass rates the already-simmed anchor highest, so the second sim must
  // be the runner-up.
  auto stub = std::make_unique<StubMoveProposalService>();
  StubMoveProposalService* sp = stub.get();
  sp->scripted_gains = {gains_favouring(n, {anchor, runner_up}),
                        gains_favouring(n, {anchor, runner_up})};
  UltimateBotAgent agent(p, std::move(stub));
  agent.begin_game({});
  agent.make_move(request());
  ASSERT_EQ(sp->condition_calls, 2);
  EXPECT_EQ(sp->seen_evidence[1], (std::vector<int>{anchor, runner_up}));
}

TEST_F(UltimateBotAgentTest, EqualGainsGoToTheEquityPreferredCandidate) {
  const UltimateBotAgent::Params p = params();
  const std::vector<Move> cands = candidates(request());
  const int anchor = int(evidence::anchor_index(cands));

  // With the stub's default all-zero gains, the tie goes to the lowest unsimmed
  // index, i.e. the best by static equity.
  auto stub = std::make_unique<StubMoveProposalService>();
  StubMoveProposalService* sp = stub.get();
  UltimateBotAgent agent(p, std::move(stub));
  agent.begin_game({});
  agent.make_move(request());
  ASSERT_EQ(sp->condition_calls, 2);
  EXPECT_EQ(sp->seen_evidence[1][1], anchor == 0 ? 1 : 0);
}

TEST_F(UltimateBotAgentTest, TheGainThresholdStopsTheLoop) {
  const std::vector<Move> cands = candidates(request());
  const int n = int(cands.size());
  const int anchor = int(evidence::anchor_index(cands));

  // Every gain is below the threshold, so the anchor is the only sim and plays.
  UltimateBotAgent::Params p = params();
  p.gain_threshold = 0.5f;
  auto stub = std::make_unique<StubMoveProposalService>();
  StubMoveProposalService* sp = stub.get();
  sp->scripted_gains = {std::vector<float>(size_t(n), 0.1f)};
  UltimateBotAgent agent(p, std::move(stub));
  agent.begin_game({});
  EXPECT_TRUE(agent.make_move(request()).move == cands[size_t(anchor)]);
  EXPECT_EQ(sp->condition_calls, 1);

  // At the default threshold of 0 only the budget stops the loop.
  p.gain_threshold = 0.0f;
  auto stub2 = std::make_unique<StubMoveProposalService>();
  StubMoveProposalService* sp2 = stub2.get();
  sp2->scripted_gains = {std::vector<float>(size_t(n), 0.1f), std::vector<float>(size_t(n), 0.1f)};
  UltimateBotAgent agent2(p, std::move(stub2));
  agent2.begin_game({});
  agent2.make_move(request());
  EXPECT_EQ(sp2->condition_calls, p.max_sims - 1);

  // A gain equal to the threshold continues the loop: the bound is inclusive.
  p.gain_threshold = 0.5f;
  auto stub3 = std::make_unique<StubMoveProposalService>();
  StubMoveProposalService* sp3 = stub3.get();
  sp3->scripted_gains = {std::vector<float>(size_t(n), 0.5f), std::vector<float>(size_t(n), 0.1f)};
  UltimateBotAgent agent3(p, std::move(stub3));
  agent3.begin_game({});
  agent3.make_move(request());
  EXPECT_EQ(sp3->condition_calls, 2);
}

// A non-finite gain is a broken model output. Left unchecked, NaN's false
// comparisons would corrupt the argmax and slip past the threshold.
TEST_F(UltimateBotAgentTest, ANonFiniteGainIsAHardError) {
  const std::vector<Move> cands = candidates(request());
  const int n = int(cands.size());
  ASSERT_GE(n, 3);
  UltimateBotAgent::Params p = params();
  p.gain_threshold = 0.5f;
  auto stub = std::make_unique<StubMoveProposalService>();
  std::vector<float> gains(size_t(n), 0.9f);
  // On an unsimmed candidate, since the anchor's gain is never read.
  gains[(evidence::anchor_index(cands) + 1) % size_t(n)] = std::numeric_limits<float>::quiet_NaN();
  stub->scripted_gains = {gains};
  UltimateBotAgent agent(p, std::move(stub));
  agent.begin_game({});
  EXPECT_THROW(agent.make_move(request()), std::runtime_error);
}

TEST_F(UltimateBotAgentTest, ABudgetPastTheCandidateCountSimsThemAll) {
  // No legal play in this dictionary, so the candidates are the rack's 35
  // distinct exchanges, few enough to exhaust. The bag must be able to supply
  // the rack, since the sims draw from the bag less these tiles.
  const Rack rack = rack_from("VVWWXQ");
  const MoveRequest req{board_,          dict_,    rack, opp_leave_, /*my_score=*/13,
                        /*opp_score=*/7, bag_size_};
  const std::vector<Move> cands = candidates(req);
  ASSERT_GT(cands.size(), 1u);
  ASSERT_LT(cands.size(), size_t(nn::kMaxEvidence));
  for (const Move& m : cands) ASSERT_EQ(m.type(), MoveType::EXCHANGE);

  UltimateBotAgent::Params p = params();
  p.max_sims = nn::kMaxEvidence;
  auto stub = std::make_unique<StubMoveProposalService>();
  StubMoveProposalService* sp = stub.get();
  UltimateBotAgent agent(p, std::move(stub));
  agent.begin_game({});
  agent.make_move(req);
  EXPECT_EQ(sp->condition_calls, int(cands.size()) - 1);
  EXPECT_EQ(sp->seen_evidence.back().size(), cands.size() - 1);
}

TEST_F(UltimateBotAgentTest, ABudgetOfOnePlaysTheAnchorUnsimmed) {
  const std::vector<Move> cands = candidates(request());
  const int anchor = int(evidence::anchor_index(cands));
  UltimateBotAgent::Params p = params();
  p.max_sims = 1;
  auto stub = std::make_unique<StubMoveProposalService>();
  StubMoveProposalService* sp = stub.get();
  UltimateBotAgent agent(p, std::move(stub));
  agent.begin_game({});
  EXPECT_TRUE(agent.make_move(request()).move == cands[size_t(anchor)]);
  EXPECT_EQ(sp->encode_calls, 0);
  EXPECT_EQ(sp->condition_calls, 0);
  for (const Move& m : cands) EXPECT_LE(m.score(), cands[size_t(anchor)].score());
}

TEST_F(UltimateBotAgentTest, OneSeedGivesOneDecision) {
  const UltimateBotAgent::Params p = params();
  const std::vector<Move> cands = candidates(request());
  const int n = int(cands.size());
  const int anchor = int(evidence::anchor_index(cands));
  const std::vector<int> picks = picks_avoiding(anchor, n, 2);

  Move moves[2];
  for (int i = 0; i < 2; ++i) {
    auto stub = std::make_unique<StubMoveProposalService>();
    stub->scripted_gains = {gains_favouring(n, {picks[1]}), gains_favouring(n, {picks[0]})};
    UltimateBotAgent agent(p, std::move(stub));
    agent.begin_game({});
    moves[i] = agent.make_move(request()).move;
  }
  EXPECT_TRUE(moves[0] == moves[1]);
}

TEST_F(UltimateBotAgentTest, ASoleCandidatePlaysWithoutModelOrRollouts) {
  // No legal play and a bag too small to exchange leave PASS as the only
  // candidate.
  auto stub = std::make_unique<StubMoveProposalService>();
  StubMoveProposalService* sp = stub.get();
  UltimateBotAgent agent(params(), std::move(stub));
  agent.begin_game({});

  const Rack unplayable = rack_from("QQQQQQ");  // no dict word uses Q
  MoveRequest req{board_,          dict_,         unplayable, opp_leave_, /*my_score=*/13,
                  /*opp_score=*/7, /*bag_size=*/3};  // < RACK_SIZE: exchanges illegal
  const Move played = agent.make_move(req).move;
  EXPECT_EQ(played.type(), MoveType::PASS);
  EXPECT_EQ(sp->encode_calls, 0);  // the model was never consulted
}

TEST_F(UltimateBotAgentTest, AnEmptyBagFallsBackToStaticEquity) {
  auto stub = std::make_unique<StubMoveProposalService>();
  StubMoveProposalService* sp = stub.get();
  // With endgame budget 0 the solver declines, and there is no bag to sim from.
  UltimateBotAgent agent(params(), std::move(stub));
  agent.begin_game({});
  MoveRequest req{board_,          dict_,         my_rack_, opp_leave_, /*my_score=*/13,
                  /*opp_score=*/7, /*bag_size=*/0};
  const Move played = agent.make_move(req).move;
  EXPECT_TRUE(played == equity_top_k(req, 1).front());
  EXPECT_EQ(sp->encode_calls, 0);  // the model was never consulted
}

TEST_F(UltimateBotAgentTest, TheRolloutSeedFollowsTheAdvancingPly) {
  // The other rollout tests decide at ply 0, where seeding from the current ply
  // and a hardcoded ply 0 look the same. Here two moves are observed first, and
  // the sim pair is chosen so the two seeds pick different candidates.
  UltimateBotAgent::Params p = params();
  p.max_sims = 2;
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
  const std::vector<Move> cands = candidates(req);
  const int n = int(cands.size());
  ASSERT_GT(n, 4);
  const int anchor = int(evidence::anchor_index(cands));

  const SimRunner runner(dict_, p.sim);
  const SimPosition pos = sim_position_from(req);
  int partner = -1;
  Move at_ply_2;
  for (int i = 0; i < n && partner < 0; ++i) {
    if (i == anchor) continue;
    const std::vector<Move> simmed = {cands[size_t(anchor)], cands[size_t(i)]};
    // sim_seed depends only on the params, so a probe agent stands in for the
    // one under test.
    UltimateBotAgent probe(p, std::make_unique<StubMoveProposalService>());
    const Move p2 = simmed[size_t(
      best_observation_index(runner.run(pos, simmed, probe.sim_seed(2)), SimObjective::kWinRate))];
    const Move p0 = simmed[size_t(
      best_observation_index(runner.run(pos, simmed, probe.sim_seed(0)), SimObjective::kWinRate))];
    if (!(p2 == p0)) {
      partner = i;
      at_ply_2 = p2;
    }
  }
  ASSERT_GE(partner, 0) << "every pair agrees across plies here; the test proves nothing";

  auto stub = std::make_unique<StubMoveProposalService>();
  stub->scripted_gains = {gains_favouring(n, {partner})};
  UltimateBotAgent agent(p, std::move(stub));
  agent.begin_game({});
  agent.observe_move(opening);
  agent.observe_move(reply);
  EXPECT_TRUE(agent.make_move(req).move == at_ply_2);
}

TEST_F(UltimateBotAgentTest, TheWholeCandidateSetGoesToTheModelInOnePass) {
  // The opponent opened for 10, so the move features see a non-zero pre-move
  // score differential.
  const Move opening =
    make_play_full(7, 7, /*horizontal=*/true, 0b111, 10,
                   {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                    Glyph::of(Tile::from_char('T'))});
  const int pre_diff = -10;

  Board board;
  board.apply(opening);
  const UltimateBotAgent::Params p = params();
  const MoveRequest req{board,
                        dict_,
                        my_rack_,
                        opp_leave_,
                        /*my_score=*/0,
                        /*opp_score=*/10,
                        /*bag_size=*/86};
  const std::vector<Move> cands = candidates(req);

  auto stub = std::make_unique<StubMoveProposalService>();
  StubMoveProposalService* sp = stub.get();
  UltimateBotAgent agent(p, std::move(stub));
  agent.begin_game({});
  agent.observe_move(opening);
  agent.make_move(req);

  // Amortizing one board encode over the whole set is why this model exists.
  ASSERT_EQ(sp->encode_calls, 1);
  ASSERT_EQ(sp->last_moves.count, int(cands.size()));

  // The board row is the position's pre-move row, which
  // ThePreMoveRowMatchesTheTrainingDecoder ties to the training row.
  std::vector<float> expected_row(size_t(kInputFloats), 0.0f);
  agent.encode_board_row(req, expected_row.data());
  EXPECT_EQ(sp->last_board_row, expected_row);

  // Each candidate's features must be what encode_move, the encoder the
  // training rows go through, makes of it at this differential.
  for (size_t i = 0; i < cands.size(); ++i) {
    int32_t letters[move_set::kMoveMaxPlaced];
    uint8_t blanks[move_set::kMoveMaxPlaced];
    int32_t squares[move_set::kMoveMaxPlaced];
    uint8_t tile_mask[move_set::kMoveMaxPlaced];
    float scalars[move_set::kMoveScalars];
    move_set::encode_move(cands[i], pre_diff, letters, blanks, squares, tile_mask, scalars);

    const size_t tile_base = i * move_set::kMoveMaxPlaced;
    for (int t = 0; t < move_set::kMoveMaxPlaced; ++t) {
      EXPECT_EQ(sp->last_moves.letters[tile_base + t], letters[t]) << "move " << i << " tile " << t;
      EXPECT_EQ(sp->last_moves.blanks[tile_base + t], blanks[t]) << "move " << i << " tile " << t;
      EXPECT_EQ(sp->last_moves.squares[tile_base + t], squares[t]) << "move " << i << " tile " << t;
      EXPECT_EQ(sp->last_moves.tile_mask[tile_base + t], tile_mask[t])
        << "move " << i << " tile " << t;
    }
    for (int s = 0; s < move_set::kMoveScalars; ++s) {
      EXPECT_EQ(sp->last_moves.scalars[i * move_set::kMoveScalars + s], scalars[s])
        << "move " << i << " scalar " << s;
    }
  }
}

// The loop sims one candidate at a time, yet each observation must equal, bit
// for bit, that candidate's result in one batched SimRunner::run: the final
// pick compares sims under common random numbers. (This holds for terminal
// rollouts; under truncation the leaf batches differ and equality is only up to
// a tolerance.)
//
// Also prints the per-sim setup cost of running one at a time, bounded above
// by a one-candidate, one-rollout run, against the rollout time of a
// 400-rollout sim.
TEST_F(UltimateBotAgentTest, OneAtATimeSimsEqualOneBatchedRun) {
  const std::vector<Move> cands = candidates(request());
  const int n = int(cands.size());
  const int anchor = int(evidence::anchor_index(cands));
  const std::vector<int> picks = picks_avoiding(anchor, n, 3);
  const SimRunner::Params sim{64, 1};
  const SimRunner runner(dict_, sim);
  const SimPosition pos = sim_position_from(request());
  const uint64_t seed = 777;

  StubMoveProposalService stub;
  stub.scripted_gains = {gains_favouring(n, {picks[0]}), gains_favouring(n, {picks[1]}),
                         gains_favouring(n, {picks[2]})};
  std::vector<float> board_row(size_t(kInputFloats), 0.0f);
  move_set::MoveFeatureArrays features;
  features.encode(cands.data(), n, 0);
  stub.encode(board_row.data(), features);
  agent::SimRunnerCandidateSimmer simmer(runner, pos, seed);
  agent::ArgmaxGainPolicy policy(0.0f);

  const EvidenceSet evidence = agent::run_evidence_loop(cands, stub, simmer, policy, 4);
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<SimObservation> batched = runner.run(pos, evidence.moves, seed);
  const auto t1 = std::chrono::steady_clock::now();

  ASSERT_EQ(evidence.size(), 4);
  EXPECT_EQ(evidence.scored_indices, (std::vector<int>{anchor, picks[0], picks[1], picks[2]}));
  for (int j = 0; j < 4; ++j) {
    EXPECT_TRUE(same_observation(evidence.observations[size_t(j)], batched[size_t(j)]))
      << "sim " << j;
  }

  const SimRunner one(dict_, SimRunner::Params{1, 1});
  const auto t2 = std::chrono::steady_clock::now();
  one.run(pos, {evidence.moves.front()}, seed);
  const auto t3 = std::chrono::steady_clock::now();
  const double per_rollout_ms =
    std::chrono::duration<double, std::milli>(t1 - t0).count() / (4.0 * sim.rollouts);
  const double setup_bound_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
  std::cout << "  per-sim setup <= " << setup_bound_ms << " ms against " << 400 * per_rollout_ms
            << " ms of rollouts at 400/sim (<= " << 100.0 * setup_bound_ms / (400 * per_rollout_ms)
            << "%)\n";
}

namespace {

// The pre-move row the agent encodes must equal, float for float, the row the
// training BlockDecoder reconstructs by replay for the same position. Uses a
// three-turn game sampled at turn 2 (player 0's), so both players have a prior
// move and the last-move placement planes are exercised.
void check_pre_move_row_matches_decoder(std::array<int, 2> initial_scores) {
  const Move move0 =
    make_play_full(7, 7, /*horizontal=*/true, 0b111, 10,
                   {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                    Glyph::of(Tile::from_char('T'))});
  const Move move1 =
    make_play_full(0, 0, /*horizontal=*/true, 0b1, 5, {Glyph::of(Tile::from_char('S'))});
  const Move move2 =
    make_play_full(2, 2, /*horizontal=*/true, 0b11, 8,
                   {Glyph::of(Tile::from_char('D')), Glyph::of(Tile::from_char('O'))});
  const uint32_t sampled_turn = 2;

  // Player 0's rack at turn 2 replays to DONERST: CATERST, plays CAT, draws
  // DON. Player 1 holds the S it plays on turn 1.
  binlog::InitialRacks ir{};
  ir.p0 = rack_from("CATERST");
  ir.p1 = rack_from("SAINTED");

  binlog::TurnBlob t0{};
  t0.move = move0;
  t0.drawn = rack_from("DON");
  binlog::TurnBlob t1{};
  t1.move = move1;
  binlog::TurnBlob t2{};
  t2.move = move2;

  const std::vector<char> buf = build_slog(ir, {t0, t1, t2}, sampled_turn, initial_scores);

  // Training path: the pre-move row, untransposed. Both paths use the same
  // dictionary for the cross-check planes.
  Dictionary dict = opening_dict();
  binlog::BlockDecoder dec(InputEncodingSpec{&dict});
  const uint8_t flips[1] = {0};
  std::vector<float> dec_row(size_t(kInputFloats + kLabelFloats), 0.0f);
  dec.decode(buf.data(), "test.slog", /*local_start=*/0, /*n_rows=*/1, flips, /*post_move=*/false,
             /*output_row_start=*/0, dec_row.data());

  UltimateBotAgent::Params p;
  p.name = "UB";
  p.dict = &dict;
  UltimateBotAgent agent(p, std::make_unique<StubMoveProposalService>());
  agent.begin_game({initial_scores});
  agent.observe_move(move0);
  agent.observe_move(move1);

  // The agent takes board and scores from its own replay of observed moves;
  // only the rack comes from the request, so the rest of it is arbitrary.
  const Rack my_rack = rack_from("DONERST");
  const Rack no_leave;
  const Board board;
  const MoveRequest req{board,          dict, my_rack, no_leave, /*my_score=*/10, /*opp_score=*/5,
                        /*bag_size=*/50};
  std::vector<float> agent_row(size_t(kInputFloats), 0.0f);
  agent.encode_board_row(req, agent_row.data());

  bool any_nonzero = false;
  for (int i = 0; i < kInputFloats; ++i) {
    ASSERT_EQ(agent_row[size_t(i)], dec_row[size_t(i)]) << "input float " << i;
    any_nonzero = any_nonzero || agent_row[size_t(i)] != 0.0f;
  }
  ASSERT_TRUE(any_nonzero);  // an all-zero match would prove nothing
}

}  // namespace

TEST(UltimateBotAgent, ThePreMoveRowMatchesTheTrainingDecoder) {
  check_pre_move_row_matches_decoder({0, 0});
}

TEST(UltimateBotAgent, AHandicapReachesTheModelRow) {
  // The training replay seeds its scores from the handicap the .slog records.
  // An agent that started every game at 0-0 would feed the model a score
  // differential wrong by the head start all game, with nothing else in the row
  // to reveal it.
  check_pre_move_row_matches_decoder({50, 0});
  check_pre_move_row_matches_decoder({0, 37});
}
