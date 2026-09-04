// Unit tests for the evidence loop (agent/evidence_loop.h) and UltimateBotAgent,
// with the move proposal model replaced by a scripted MoveProposalService stub
// -- no ONNX, no TensorRT, no GPU.
//
//  * the anchor -- the highest-raw-score candidate -- is simmed first whatever
//    the model says, and every later sim is the scripted proves-best argmax
//    over the UNSIMMED candidates, conditioned on exactly the sims so far.
//  * the budget (--max-sims) and the gain threshold both stop the loop; a
//    budget of one plays the anchor with no sim and no model pass; a budget
//    past the candidate count sims them all.
//  * the loop's one-at-a-time sims equal, bit for bit, one batched
//    SimRunner::run over the same candidates -- the common-random-numbers
//    pairing the final pick and the stopping rule ride on.
//  * the agent plays the rollouts' favourite among the simmed set, checked by
//    replaying the decision through SimRunner with the agent's own seed.
//  * two agents on one seed agree; the seed follows the advancing ply.
//  * a bag-empty turn, and a sole candidate, are answered without the model.
//  * ONE encode() carries the whole candidate set, with the move features
//    move_set::encode_move makes of each candidate at the turn's differential,
//    and the pre-move board row is byte-identical to the training decoder's.

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

// The base input layout the stub declares.
const int kInputFloats = input_floats(InputEncodingSpec{nullptr});

// A gain vector over `n` candidates favouring `favoured` in that order
// (descending), everyone else at a low constant.
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
  // the simulator carries every field the agent is responsible for filling.
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

  // The accepted boundaries, so the bounds cannot silently tighten.
  EXPECT_NO_THROW(build(1, 0.0f, 1));
  EXPECT_NO_THROW(build(nn::kMaxEvidence, 0.0f, SimRunner::kMaxRollouts));
}

// The agent forwards sim_horizon into its SimRunner: the runner validates the
// horizon (pairing against the injected leaf, and the lower bound) at
// construction, so a bad horizon is rejected.
TEST_F(UltimateBotAgentTest, TruncationHorizonIsWiredToTheRunner) {
  using scribblez::testing::StubEvalService;
  UltimateBotAgent::Params p = params();
  p.sim_horizon = SimRunner::kMinHorizonPlies;  // a horizon with no leaf service
  EXPECT_THROW(UltimateBotAgent(p, std::make_unique<StubMoveProposalService>()),
               std::runtime_error);
  p.sim_horizon = SimRunner::kMinHorizonPlies - 1;  // below the minimum, even with a leaf
  EXPECT_THROW(UltimateBotAgent(p, std::make_unique<StubMoveProposalService>(),
                                std::make_shared<StubEvalService>()),
               std::runtime_error);
  p.sim_horizon = SimRunner::kMinHorizonPlies;  // a valid horizon with a leaf
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

  // The first conditioned pass favours picks[0], the second picks[1]. Neither
  // is the anchor, and the anchor's gain is never the highest -- yet it is
  // simmed first.
  auto stub = std::make_unique<StubMoveProposalService>();
  StubMoveProposalService* sp = stub.get();
  sp->scripted_gains = {gains_favouring(n, {picks[0]}), gains_favouring(n, {picks[1]})};
  UltimateBotAgent agent(p, std::move(stub));
  agent.begin_game({});
  const Move played = agent.make_move(request()).move;

  EXPECT_EQ(sp->encode_calls, 1);
  // sims - 1 conditioned passes: none before the anchor, one before each later
  // sim, none after the budget is spent.
  ASSERT_EQ(sp->condition_calls, 2);
  EXPECT_EQ(sp->seen_evidence[0], std::vector<int>{anchor});
  EXPECT_EQ(sp->seen_evidence[1], (std::vector<int>{anchor, picks[0]}));

  // Replay the same decision independently: the three simmed candidates,
  // batched from the same position with the agent's own seed, the rollouts'
  // favourite by win rate.
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

  // Every conditioned pass rates the (already simmed) anchor highest; the
  // argmax over the UNSIMMED candidates is the runner-up.
  auto stub = std::make_unique<StubMoveProposalService>();
  StubMoveProposalService* sp = stub.get();
  sp->scripted_gains = {gains_favouring(n, {anchor, runner_up}),
                        gains_favouring(n, {anchor, runner_up})};
  UltimateBotAgent agent(p, std::move(stub));
  agent.begin_game({});
  agent.make_move(request());
  ASSERT_EQ(sp->condition_calls, 2);
  EXPECT_EQ(sp->seen_evidence[1], (std::vector<int>{anchor, runner_up}));
  // And the second pass, with both taken, picked neither: three distinct sims.
  // (The third sim is not in any evidence set the stub saw; the agent's own
  // final pick covers it in the replay test above.)
}

TEST_F(UltimateBotAgentTest, EqualGainsGoToTheEquityPreferredCandidate) {
  const UltimateBotAgent::Params p = params();
  const std::vector<Move> cands = candidates(request());
  const int anchor = int(evidence::anchor_index(cands));

  // An all-zero gain vector (the stub's default): the tie falls to the lowest
  // unsimmed index -- the static-equity order the candidates arrive in.
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

  // Every predicted gain sits below the threshold: the anchor is the turn's
  // only sim, and plays.
  UltimateBotAgent::Params p = params();
  p.gain_threshold = 0.5f;
  auto stub = std::make_unique<StubMoveProposalService>();
  StubMoveProposalService* sp = stub.get();
  sp->scripted_gains = {std::vector<float>(size_t(n), 0.1f)};
  UltimateBotAgent agent(p, std::move(stub));
  agent.begin_game({});
  EXPECT_TRUE(agent.make_move(request()).move == cands[size_t(anchor)]);
  EXPECT_EQ(sp->condition_calls, 1);

  // At the default threshold of 0 the same gains never stop the loop: the
  // budget does.
  p.gain_threshold = 0.0f;
  auto stub2 = std::make_unique<StubMoveProposalService>();
  StubMoveProposalService* sp2 = stub2.get();
  sp2->scripted_gains = {std::vector<float>(size_t(n), 0.1f), std::vector<float>(size_t(n), 0.1f)};
  UltimateBotAgent agent2(p, std::move(stub2));
  agent2.begin_game({});
  agent2.make_move(request());
  EXPECT_EQ(sp2->condition_calls, p.max_sims - 1);

  // A gain exactly at the threshold "reaches" it (the flag's words): the loop
  // continues, so the bound is inclusive and cannot silently tighten.
  p.gain_threshold = 0.5f;
  auto stub3 = std::make_unique<StubMoveProposalService>();
  StubMoveProposalService* sp3 = stub3.get();
  sp3->scripted_gains = {std::vector<float>(size_t(n), 0.5f), std::vector<float>(size_t(n), 0.1f)};
  UltimateBotAgent agent3(p, std::move(stub3));
  agent3.begin_game({});
  agent3.make_move(request());
  EXPECT_EQ(sp3->condition_calls, 2);
}

// A non-finite gain is a broken model output, not a candidate: it would win
// every argmax (NaN compares false) and defeat the threshold.
TEST_F(UltimateBotAgentTest, ANonFiniteGainIsAHardError) {
  const std::vector<Move> cands = candidates(request());
  const int n = int(cands.size());
  ASSERT_GE(n, 3);
  UltimateBotAgent::Params p = params();
  p.gain_threshold = 0.5f;
  auto stub = std::make_unique<StubMoveProposalService>();
  std::vector<float> gains(size_t(n), 0.9f);
  // On an unsimmed candidate: the anchor's own gain is never read.
  gains[(evidence::anchor_index(cands) + 1) % size_t(n)] = std::numeric_limits<float>::quiet_NaN();
  stub->scripted_gains = {gains};
  UltimateBotAgent agent(p, std::move(stub));
  agent.begin_game({});
  EXPECT_THROW(agent.make_move(request()), std::runtime_error);
}

TEST_F(UltimateBotAgentTest, ABudgetPastTheCandidateCountSimsThemAll) {
  // A legal rack (the sims draw from the bag less these tiles, so it must be
  // one the bag can supply) with no legal play in this dictionary: the
  // candidate set is its 35 distinct exchanges -- small enough to exhaust.
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
  // One conditioned pass before each sim after the anchor, then nothing left
  // to pick from.
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
  EXPECT_EQ(sp->encode_calls, 0);  // the model was never consulted...
  EXPECT_EQ(sp->condition_calls, 0);
  // ...and the anchor is the highest-scoring candidate, by the rule no model
  // can be wrong about.
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
  // A rack with no legal play and a bag too small for exchanges leaves a lone
  // PASS candidate, which the size-1 early return plays outright.
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
  UltimateBotAgent agent(params(), std::move(stub));  // endgame budget 0: solver declines
  agent.begin_game({});
  MoveRequest req{board_,          dict_,         my_rack_, opp_leave_, /*my_score=*/13,
                  /*opp_score=*/7, /*bag_size=*/0};
  const Move played = agent.make_move(req).move;
  EXPECT_TRUE(played == equity_top_k(req, 1).front());
  EXPECT_EQ(sp->encode_calls, 0);  // the model was never consulted
}

TEST_F(UltimateBotAgentTest, TheRolloutSeedFollowsTheAdvancingPly) {
  // Every other rollout test decides at ply 0, where sim_seed(ply_) and a
  // hardcoded sim_seed(0) are indistinguishable. Here two moves have been
  // observed first, so a decision seeded off a stale ply would draw different
  // rollouts and, on a scripted pair chosen to split, play the other candidate.
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

  // A partner for the anchor on which the two plies disagree -- searched for,
  // since the test proves nothing on a pair where they agree.
  const SimRunner runner(dict_, p.sim);
  const SimPosition pos = sim_position_from(req);
  int partner = -1;
  Move at_ply_2;
  for (int i = 0; i < n && partner < 0; ++i) {
    if (i == anchor) continue;
    const std::vector<Move> simmed = {cands[size_t(anchor)], cands[size_t(i)]};
    // sim_seed is a function of the params' seed alone, so any agent over
    // these params answers for the one under test.
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
  // Second turn of the game: the opponent has opened for 10, so the pre-move
  // differential the move features resolve is a signed non-zero number rather
  // than the game-start 0.
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

  // And that one board row is the position's own pre-move row (which the
  // decoder cross-check separately proves is the training row).
  std::vector<float> expected_row(size_t(kInputFloats), 0.0f);
  agent.encode_board_row(req, expected_row.data());
  EXPECT_EQ(sp->last_board_row, expected_row);

  // Each candidate's features are what the shared encoder makes of it at this
  // turn's pre-move differential -- the same encode_move the training rows go
  // through, reached here by the agent instead of the FFI.
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

// The loop's sims, taken one candidate at a time, pair with each other exactly:
// each observation is bit-identical to the same candidate's in one batched
// SimRunner::run over the whole simmed set (terminal rollouts; under
// truncation the leaf batches differ and the equality is a tolerance). Also
// prints an upper bound on what running one-at-a-time costs per sim -- the
// thread spawn, agent construction, and candidate setup each run() pays,
// bounded above by a whole one-rollout run -- against the rollouts of a
// deployment-sized sim, the number docs/roadmap.md item 6's plan accepted as
// negligible.
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

  // A one-candidate, one-rollout run is setup plus one rollout: an upper bound
  // on the per-sim setup cost, against a 400-rollout sim's rollouts scaled
  // from the batched run's per-rollout time.
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

// The row cross-check, run at a chosen head-start handicap: a three-turn game
// whose sampled turn (2) is player 0's, so both players already have a prior
// move -- which exercises the last-self / last-opp placement-plane features.
// The agent scores that turn's candidates against the position's PRE-move row,
// which is the row MsetDataset reconstructs by replay for the very same
// position, so the two must agree float for float.
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

  // Initial racks and post-turn draws chosen so the replay reconstructs player
  // 0's pre-move rack at turn 2 as DONERST: starting CATERST, play CAT (leave
  // ERST), draw DON. Player 1 holds an S to play on turn 1.
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

  // Training path: decode the PRE-move sampled row (no symmetry transpose). The same
  // dictionary drives both paths' cross-check planes.
  Dictionary dict = opening_dict();
  binlog::BlockDecoder dec(InputEncodingSpec{&dict});
  const uint8_t flips[1] = {0};
  std::vector<float> dec_row(size_t(kInputFloats + kLabelFloats), 0.0f);
  dec.decode(buf.data(), "test.slog", /*local_start=*/0, /*n_rows=*/1, flips, /*post_move=*/false,
             /*output_row_start=*/0, dec_row.data());

  // Inference path: the agent starts from the same handicap, observes turns
  // 0..1, then encodes its own decision point at turn 2.
  UltimateBotAgent::Params p;
  p.name = "UB";
  p.dict = &dict;
  UltimateBotAgent agent(p, std::make_unique<StubMoveProposalService>());
  agent.begin_game({initial_scores});
  agent.observe_move(move0);
  agent.observe_move(move1);

  // The board and scores of the encoded row come from the agent's own mirrored
  // replay, not from the request -- only the rack does, so the rest of the
  // request is whatever a turn would carry.
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
  ASSERT_TRUE(any_nonzero);  // guard against a vacuous all-zero match
}

}  // namespace

TEST(UltimateBotAgent, ThePreMoveRowMatchesTheTrainingDecoder) {
  check_pre_move_row_matches_decoder({0, 0});
}

TEST(UltimateBotAgent, AHandicapReachesTheModelRow) {
  // A head start is a score-differential feature like any other, and the
  // training replay seeds its encoder from the handicap the .slog records. An
  // agent that began every game at 0-0 would feed the model a differential
  // wrong by the head start for the whole game -- silently, since nothing
  // about the row's shape changes. The two rows must still agree.
  check_pre_move_row_matches_decoder({50, 0});
  check_pre_move_row_matches_decoder({0, 37});
}
