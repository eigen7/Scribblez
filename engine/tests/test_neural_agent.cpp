// NeuralAgent, which scores candidate plays with the position-evaluation model,
// here replaced by a scripted stub (no GPU). Covers candidate selection in
// top-K and all-moves modes, chunked evaluation, temperature sampling, the
// endgame hand-off to the solver, and encode_candidate() parity: the agent must
// feed the model exactly the row the training BlockDecoder produces for the
// same position.

#include "agent/agent.h"
#include "agent/neural_agent.h"
#include "data/binary_log.h"
#include "data/block_decoder.h"
#include "data/data_loader.h"
#include "encoding/game_state_encoder.h"
#include "encoding/input_encoder.h"
#include "endgame/endgame_solver.h"
#include "endgame_positions.h"
#include "game/board.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "game_fixture.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "nn/eval_service.h"
#include "sim/sim_runner.h"
#include "stub_eval_service.h"
#include "synthetic_equity.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace scribblez;
using scribblez::testing::build_slog;
using scribblez::testing::make_play_full;

static Rack rack_from(const std::string& s) {
  Rack r;
  for (char c : s) {
    if (c == '?')
      r.add(BLANK);
    else
      r.add(Tile::from_char(c));
  }
  return r;
}

// A modest word list with enough overlapping racks to yield many opening plays
// of differing scores -- the candidate sets the selection tests rank and prune.
static Dictionary medium_dict() {
  return Dictionary::build_from_words(
    {"AA",     "AB",      "AD",     "AE",     "AG",    "AH",     "AI",      "AL",      "AN",
     "AR",     "AS",      "AT",     "AW",     "AX",    "AY",     "BA",      "BE",      "BI",
     "BO",     "BY",      "CAB",    "CAR",    "CARS",  "CART",   "CARTS",   "CAT",     "CATS",
     "CARE",   "CARES",   "CARET",  "CARETS", "CASTE", "CASTER", "CASTERS", "DOG",     "DOGS",
     "DOT",    "DOTS",    "EAR",    "EARS",   "EAT",   "EATS",   "RAT",     "RATE",    "RATES",
     "RATS",   "STARE",   "STARED", "TARE",   "TARES", "TEAR",   "TEARS",   "REACT",   "REACTS",
     "TRACE",  "TRACES",  "CRATE",  "CRATES", "CATER", "CATERS", "RECAST",  "RECASTS", "TASTE",
     "TASTER", "TASTERS", "SET",    "TEA",    "ATE",   "ETA",    "ACE",     "ACES",    "ACRE",
     "ACRES",  "RACE",    "RACES",  "SCAR",   "SCARE", "ARC",    "ARCS",    "ARE",     "ERA"});
}

// Same type, orientation, anchor, squares and glyphs; ignores score.
static bool same_move(const Move& a, const Move& b) {
  if (a.type() != b.type()) return false;
  if (a.horizontal() != b.horizontal() || a.start() != b.start() ||
      a.square_mask() != b.square_mask() || a.num_glyphs() != b.num_glyphs())
    return false;
  for (int i = 0; i < a.num_glyphs(); ++i)
    if (a.glyph(i).code() != b.glyph(i).code()) return false;
  return true;
}

// An opening position for `rack`, so a test can script the model against the
// candidates the agent will evaluate.
struct OpeningPosition {
  Board board;
  Dictionary dict = medium_dict();
  Rack my_rack;
  Rack opp;
  int bag_size = 50;

  explicit OpeningPosition(const std::string& rack) : my_rack(rack_from(rack)) {}

  MoveRequest request() const {
    return MoveRequest{board, dict, my_rack, opp, /*my_score=*/0, /*opp_score=*/0, bag_size};
  }

  // The candidates an agent with `top_k` evaluates, in its evaluation order.
  std::vector<Move> candidates(int top_k) const {
    return equity_top_k(request(), top_k == 0 ? std::numeric_limits<int>::max() : top_k);
  }
};

static int count_exchanges(const std::vector<Move>& moves) {
  return std::count_if(moves.begin(), moves.end(),
                       [](const Move& m) { return m.type() == MoveType::EXCHANGE; });
}

// Installs a synthetic leave table for the tests that rank by equity.
class NeuralAgentEquityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tmp_ = std::filesystem::temp_directory_path() / "scribblez_test_neural_agent_XXXXXX";
    std::filesystem::create_directories(tmp_);
    scribblez::testing::install_synthetic_hasty_equity(tmp_);
  }
  void TearDown() override { std::filesystem::remove_all(tmp_); }

  std::filesystem::path tmp_;
};

using scribblez::testing::CountingStubEvalService;
using scribblez::testing::ScriptedEval;
using scribblez::testing::StubEvalService;

// The base input layout, which the stubs declare.
static const int kInputFloats = input_floats(InputEncodingSpec{nullptr});
static const int kRowFloats = kInputFloats + kLabelFloats;

static ScriptedEval eval_with(float score_diff_mean, float win_prob) {
  return {{win_prob, 0.0f, 0.0f}, {score_diff_mean, 0.0f}};
}

static ScriptedEval sd(float score_diff_mean) { return eval_with(score_diff_mean, 0.0f); }

TEST_F(NeuralAgentEquityTest, TopKSelectionUsesObjective) {
  // The stub's rows follow `order`: highest equity first.
  OpeningPosition pos("CARETS");
  ASSERT_GE(pos.candidates(0).size(), 3u);  // more than top_k, so the filter drops moves
  const int top_k = 2;
  const std::vector<Move> order = pos.candidates(top_k);
  ASSERT_EQ(int(order.size()), top_k);
  const MoveRequest req = pos.request();

  // The model prefers equity's second choice, overriding the equity argmax.
  {
    auto stub = std::make_shared<StubEvalService>();
    StubEvalService* sp = stub.get();
    NeuralAgent agent({.thread_id = 0,
                       .name = "stub",
                       .dict = &pos.dict,
                       .top_k = top_k,
                       .objective = EvalObjective::kScoreDiff},
                      std::move(stub));
    agent.begin_game({});
    sp->scripted = {sd(1.0f), sd(9.0f)};
    ASSERT_TRUE(same_move(agent.make_move(req).move, order[1]));
  }

  // The model agrees with equity.
  {
    auto stub = std::make_shared<StubEvalService>();
    StubEvalService* sp = stub.get();
    NeuralAgent agent({.thread_id = 0,
                       .name = "stub",
                       .dict = &pos.dict,
                       .top_k = top_k,
                       .objective = EvalObjective::kScoreDiff},
                      std::move(stub));
    agent.begin_game({});
    sp->scripted = {sd(9.0f), sd(1.0f)};
    ASSERT_TRUE(same_move(agent.make_move(req).move, order[0]));
  }

  // The win-prob objective ignores score_diff_mean.
  {
    auto stub = std::make_shared<StubEvalService>();
    StubEvalService* sp = stub.get();
    NeuralAgent agent({.thread_id = 0,
                       .name = "stub-wp",
                       .dict = &pos.dict,
                       .top_k = top_k,
                       .objective = EvalObjective::kWinProb},
                      std::move(stub));
    agent.begin_game({});
    sp->scripted = {eval_with(9.0f, 0.1f), eval_with(1.0f, 0.9f)};
    ASSERT_TRUE(same_move(agent.make_move(req).move, order[1]));
  }
}

TEST_F(NeuralAgentEquityTest, TopKExcludesLowEquityMoves) {
  // Moves outside the top_k by equity never reach the model.
  OpeningPosition pos("CARETS");
  ASSERT_GE(pos.candidates(0).size(), 3u);
  const int top_k = 2;
  const std::vector<Move> order = pos.candidates(top_k);
  const MoveRequest req = pos.request();

  auto stub = std::make_shared<CountingStubEvalService>();
  CountingStubEvalService* sp = stub.get();
  sp->scripted = {sd(1.0f), sd(5.0f)};
  NeuralAgent agent({.thread_id = 0,
                     .name = "topk",
                     .dict = &pos.dict,
                     .top_k = top_k,
                     .objective = EvalObjective::kScoreDiff},
                    std::move(stub));
  agent.begin_game({});

  Move got = agent.make_move(req).move;
  ASSERT_TRUE(same_move(got, order[1]));
  ASSERT_EQ(sp->total_rows, top_k);
}

TEST_F(NeuralAgentEquityTest, AllMovesEvaluated) {
  // With top_k 0 every legal play and exchange is evaluated, best equity
  // first. The model prefers the lowest-equity move, which a top-K agent would
  // never see.
  OpeningPosition pos("CARETS");
  const std::vector<Move> cands = pos.candidates(0);
  const int n = cands.size();
  ASSERT_GE(n, 3);
  ASSERT_EQ(n, int(generate_legal_plays(pos.request()).size() +
                   generate_legal_exchanges(pos.request()).size()));
  const int lo = n - 1;
  const MoveRequest req = pos.request();

  auto stub = std::make_shared<CountingStubEvalService>();
  CountingStubEvalService* sp = stub.get();
  sp->scripted.assign(size_t(n), sd(0.0f));
  sp->scripted[size_t(lo)] = sd(9.0f);
  NeuralAgent agent({.thread_id = 0,
                     .name = "full",
                     .dict = &pos.dict,
                     .top_k = 0,
                     .objective = EvalObjective::kScoreDiff},
                    std::move(stub));
  agent.begin_game({});

  Move got = agent.make_move(req).move;
  ASSERT_TRUE(same_move(got, cands[lo]));
  ASSERT_EQ(sp->total_rows, n);
}

TEST_F(NeuralAgentEquityTest, ChunkedEvaluation) {
  // With a batch limit of 2 the agent scores the moves across several
  // evaluate() calls and still picks the global best.
  OpeningPosition pos("CARETS");
  const std::vector<Move> cands = pos.candidates(0);
  const int n = cands.size();
  ASSERT_GE(n, 3);           // at least two chunks
  const int target = n - 1;  // in the final chunk
  const MoveRequest req = pos.request();

  auto stub = std::make_shared<CountingStubEvalService>();
  CountingStubEvalService* sp = stub.get();
  sp->scripted.assign(size_t(n), sd(0.0f));
  sp->scripted[size_t(target)] = sd(9.0f);
  NeuralAgent agent({.thread_id = 0,
                     .name = "chunk",
                     .dict = &pos.dict,
                     .top_k = 0,
                     .objective = EvalObjective::kScoreDiff},
                    std::move(stub), /*max_batch=*/2);
  agent.begin_game({});

  Move got = agent.make_move(req).move;
  ASSERT_TRUE(same_move(got, cands[target]));
  ASSERT_EQ(sp->total_rows, n);
  ASSERT_LE(sp->max_chunk, 2);
  ASSERT_EQ(sp->calls, (n + 1) / 2);
}

// The rack minus the play's tiles.
static Rack leave_after(const Rack& rack, const Move& mv) {
  Rack leave = rack;
  for (int i = 0; i < mv.num_glyphs(); ++i) leave.remove(mv.glyph(i).rack_tile());
  return leave;
}

TEST(NeuralAgent, EncodeCandidateMatchesReplay) {
  // The same history goes to the agent and to an independent reference encoder.
  Move move_a = make_play_full(7, 7, /*horizontal=*/true, 0b111, 10,
                               {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                                Glyph::of(Tile::from_char('T'))});
  Move move_b =
    make_play_full(9, 7, /*horizontal=*/true, 0b1, 5, {Glyph::of(Tile::from_char('S'))});

  // encode_candidate never calls the model.
  Dictionary dict = medium_dict();
  NeuralAgent agent({.thread_id = 0,
                     .name = "stub",
                     .dict = &dict,
                     .top_k = 4,
                     .objective = EvalObjective::kScoreDiff},
                    std::make_shared<StubEvalService>());
  agent.begin_game({});
  agent.observe_move(move_a);
  agent.observe_move(move_b);

  GameStateEncoder ref{InputEncodingSpec{&dict}};
  ref.apply_move(move_a);
  ref.apply_move(move_b);
  const int my_seat = ref.active_player();

  Move candidate =
    make_play_full(0, 0, /*horizontal=*/true, 0b11, 8,
                   {Glyph::of(Tile::from_char('D')), Glyph::of(Tile::from_char('O'))});
  Rack my_rack = rack_from("DONERST");

  std::vector<float> agent_row(kInputFloats);
  agent.encode_candidate(candidate, my_rack, my_seat, Rack{}, agent_row.data());

  std::vector<float> ref_row(kInputFloats);
  ref.board().ensure_movegen_caches(dict);
  GameStateEncoder post = ref;
  post.apply_move(candidate);
  post.encode_input(my_seat, leave_after(my_rack, candidate), ref_row.data());

  for (size_t i = 0; i < agent_row.size(); ++i)
    ASSERT_EQ(agent_row[i], ref_row[i]) << "input float " << i;
}

// The row encode_candidate() produces for a move must equal, float for float,
// the post-move row the training BlockDecoder reconstructs for it. Uses a
// three-turn game sampled at turn 2 (player 0's), so both players have a prior
// move and the last-move placement planes are exercised.
static void check_candidate_row_matches_decoder(std::array<int, 2> initial_scores) {
  Move move0 = make_play_full(7, 7, /*horizontal=*/true, 0b111, 10,
                              {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                               Glyph::of(Tile::from_char('T'))});
  Move move1 = make_play_full(0, 0, /*horizontal=*/true, 0b1, 5, {Glyph::of(Tile::from_char('S'))});
  Move move2 = make_play_full(2, 2, /*horizontal=*/true, 0b11, 8,
                              {Glyph::of(Tile::from_char('D')), Glyph::of(Tile::from_char('O'))});
  const uint32_t sampled_turn = 2;
  const int mover = int(sampled_turn % 2);  // turn k is played by k % 2

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

  std::vector<char> buf = build_slog(ir, {t0, t1, t2}, sampled_turn, initial_scores);

  // Training path: the post-move row, untransposed. Both paths use the same
  // dictionary for the cross-check planes.
  Dictionary dict = medium_dict();
  binlog::BlockDecoder dec(InputEncodingSpec{&dict});
  const uint8_t flips[1] = {0};
  std::vector<float> dec_row(kRowFloats, 0.0f);
  dec.decode(buf.data(), "test.slog", /*local_start=*/0, /*n_rows=*/1, flips, /*post_move=*/true,
             /*output_row_start=*/0, dec_row.data());

  NeuralAgent agent({.thread_id = 0,
                     .name = "stub",
                     .dict = &dict,
                     .top_k = 4,
                     .objective = EvalObjective::kScoreDiff},
                    std::make_shared<StubEvalService>());
  agent.begin_game({initial_scores});
  agent.observe_move(move0);
  agent.observe_move(move1);

  std::vector<float> agent_row(kInputFloats, 0.0f);
  agent.encode_candidate(move2, rack_from("DONERST"), mover, Rack{}, agent_row.data());

  bool any_nonzero = false;
  for (int i = 0; i < kInputFloats; ++i) {
    ASSERT_EQ(agent_row[i], dec_row[i]) << "input float " << i;
    any_nonzero = any_nonzero || agent_row[i] != 0.0f;
  }
  ASSERT_TRUE(any_nonzero);  // an all-zero match would prove nothing
}

TEST(NeuralAgent, EncodeCandidateMatchesTrainingDecoder) {
  check_candidate_row_matches_decoder({0, 0});
}

TEST(NeuralAgent, AHandicapReachesTheModelRow) {
  // The training replay seeds its scores from the handicap the .slog records.
  // An agent that started every game at 0-0 would feed the model a score
  // differential wrong by the head start all game, with nothing else in the row
  // to reveal it. Also covers NeuralSimAgent, which encodes through the same
  // CandidateEvaluator.
  check_candidate_row_matches_decoder({50, 0});
  check_candidate_row_matches_decoder({0, 37});
}

TEST_F(NeuralAgentEquityTest, TemperatureSamplingSpreads) {
  // The stub rates order[0] above order[1].
  OpeningPosition pos("CARETS");
  ASSERT_GE(pos.candidates(0).size(), 3u);
  const int top_k = 2;
  const std::vector<Move> order = pos.candidates(top_k);
  const MoveRequest req = pos.request();

  // Temperature 0 always plays the model's favourite.
  {
    auto stub = std::make_shared<StubEvalService>();
    StubEvalService* gp = stub.get();
    NeuralAgent greedy({.thread_id = 0,
                        .name = "greedy",
                        .dict = &pos.dict,
                        .top_k = top_k,
                        .objective = EvalObjective::kScoreDiff,
                        .temperature = 0.0},
                       std::move(stub));
    greedy.begin_game({});
    gp->scripted = {sd(2.0f), sd(0.0f)};
    for (int i = 0; i < 50; ++i) ASSERT_TRUE(same_move(greedy.make_move(req).move, order[0]));
  }

  // A high temperature samples both, with the favourite still more often.
  {
    auto stub = std::make_shared<StubEvalService>();
    StubEvalService* sp = stub.get();
    NeuralAgent sampler({.thread_id = 0,
                         .name = "sampler",
                         .dict = &pos.dict,
                         .top_k = top_k,
                         .objective = EvalObjective::kScoreDiff,
                         .temperature = 5.0,
                         .seed = 12345},
                        std::move(stub));
    sampler.begin_game({});
    sp->scripted = {sd(2.0f), sd(0.0f)};
    int high = 0, low = 0;
    for (int i = 0; i < 400; ++i) {
      const Move got = sampler.make_move(req).move;
      if (same_move(got, order[0]))
        ++high;
      else if (same_move(got, order[1]))
        ++low;
    }
    ASSERT_GT(high, 0);
    ASSERT_GT(low, 0);
    ASSERT_GT(high, low);
  }
}

// --- Endgame ----------------------------------------------------------------

static constexpr uint64_t kSolveBudget = 1ull << 20;
static constexpr int kSolvePlies = 24;

static EndgameSolver::Params solver_params(uint64_t budget) {
  EndgameSolver::Params p;
  p.budget = budget;
  p.plies = kSolvePlies;
  return p;
}

// The agent's fallback when the solver declines a bag-empty turn: the
// static-equity argmax, with ties broken as the agent's equity_top_k does.
static Move greedy_equity_move(const MoveRequest& req) { return equity_top_k(req, 1).front(); }

TEST_F(NeuralAgentEquityTest, EndgameGoesToTheSolver) {
  // Only a position where the solver and static equity disagree tells the two
  // policies apart, so scan random endgames for one. Neither configuration may
  // call the model.
  Dictionary d = tiny_dict();
  std::mt19937 rng(0xE9DA3E01u);
  EndgameSolver ref;

  bool found = false;
  for (int i = 0; i < 1200 && !found; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/3);
    const MoveRequest req = endgame_request(p, d);
    const std::vector<Move> plays = generate_legal_plays(req);
    if (plays.empty()) continue;
    const Move greedy = greedy_equity_move(req);

    ref.clear();
    const EndgameResult r = ref.solve(
      {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, /*scoreless_turns=*/0},
      solver_params(kSolveBudget));
    // Skip outcomes where the agent plays its own move: no completed iteration,
    // or a proven loss with no certificate (its move is arbitrary).
    if (r.depth_completed < 1) continue;
    if (r.proven_class == -1 && r.continuation.empty()) continue;
    if (r.best == greedy) continue;

    // The solver's move, with its certificate as the projection.
    auto solving_stub = std::make_shared<CountingStubEvalService>();
    CountingStubEvalService* solving_sp = solving_stub.get();
    NeuralAgent solving({.thread_id = 0,
                         .name = "solving",
                         .dict = &d,
                         .top_k = 0,
                         .objective = EvalObjective::kScoreDiff,
                         .endgame = solver_params(kSolveBudget)},
                        std::move(solving_stub));
    // begin_game() clears the transposition table, matching the freshly
    // cleared reference solve.
    solving.begin_game({});
    const MoveDecision decision = solving.make_move(req);
    EXPECT_EQ(decision.move, r.best);
    EXPECT_EQ(decision.projected_remaining_moves.size(), r.continuation.size());
    EXPECT_EQ(solving_sp->calls, 0);

    // With solving disabled, the static-equity move.
    auto disabled_stub = std::make_shared<CountingStubEvalService>();
    CountingStubEvalService* disabled_sp = disabled_stub.get();
    NeuralAgent disabled({.thread_id = 0,
                          .name = "disabled",
                          .dict = &d,
                          .top_k = 0,
                          .objective = EvalObjective::kScoreDiff,
                          .endgame = solver_params(/*budget=*/0)},
                         std::move(disabled_stub));
    disabled.begin_game({});
    const MoveDecision fallback = disabled.make_move(req);
    EXPECT_EQ(fallback.move, greedy);
    EXPECT_TRUE(fallback.projected_remaining_moves.empty());
    EXPECT_EQ(disabled_sp->calls, 0);

    found = true;
  }
  ASSERT_TRUE(found) << "no solver-beats-greedy endgame found in the scan";
}

// --- Shared service (nn::PositionEvalService::create() path) ----------------

TEST_F(NeuralAgentEquityTest, AgentsShareOneService) {
  // PositionEvalService::create() hands every thread's agent the same
  // service. Two agents sharing one stub must each drive it correctly.
  OpeningPosition pos("CARETS");
  ASSERT_GE(pos.candidates(0).size(), 3u);
  const int top_k = 2;
  const std::vector<Move> order = pos.candidates(top_k);
  ASSERT_EQ(int(order.size()), top_k);
  const MoveRequest req = pos.request();

  auto shared = std::make_shared<StubEvalService>();
  const NeuralAgent::Params base{.thread_id = 0,
                                 .name = "shared",
                                 .dict = &pos.dict,
                                 .top_k = top_k,
                                 .objective = EvalObjective::kScoreDiff};
  NeuralAgent a(base, shared, /*max_batch=*/top_k);
  NeuralAgent b(base, shared, /*max_batch=*/top_k);
  EXPECT_EQ(shared.use_count(), 3);  // the test plus both agents
  a.begin_game({});
  b.begin_game({});
  shared->scripted = {sd(1.0f), sd(9.0f)};  // second-ranked candidate wins
  EXPECT_TRUE(same_move(a.make_move(req).move, order[1]));
  shared->scripted = {sd(9.0f), sd(1.0f)};  // top-ranked candidate wins
  EXPECT_TRUE(same_move(b.make_move(req).move, order[0]));
}

TEST(NeuralNetSharingKey, EqualityDistinguishesEngineDeterminingFields) {
  // create() shares one loaded engine among equal NeuralNetParams, so every
  // field that changes the built engine or its buffers must break equality.
  // copy_aux and fast_build matter for correctness, not just speed: aux host
  // buffers exist only under copy_aux, and a fast_build plan is cached
  // separately and optimized differently.
  using Params = nn::NeuralNetParams<nn::PositionEvaluationSpec>;
  Params a;
  a.onnx_path = "model.onnx";

  EXPECT_EQ(a, Params(a));  // identical params share

  Params b = a;
  b.onnx_path = "other.onnx";
  EXPECT_NE(a, b);
  b = a;
  b.precision = nn::Precision::kFP32;
  EXPECT_NE(a, b);
  b = a;
  b.max_rows = a.max_rows + 1;
  EXPECT_NE(a, b);
  b = a;
  b.cuda_device_id = a.cuda_device_id + 1;
  EXPECT_NE(a, b);
  b = a;
  b.copy_aux = !a.copy_aux;
  EXPECT_NE(a, b);
  b = a;
  b.fast_build = !a.fast_build;
  EXPECT_NE(a, b);
}
