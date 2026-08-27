// Unit tests for MsetSimAgent, the move-set-evaluation agent, with the model
// replaced by a scripted MoveSetEvalService stub -- no ONNX, no TensorRT, no
// GPU.
//
//  * the sim set is the scripted model's top K, not static equity's: the agent
//    plays the rollouts' favourite among them, checked by replaying the
//    decision through SimRunner with the agent's own seed.
//  * the default shortlist scores every candidate -- exchanges included, which
//    the model may promote -- and a shortlist caps what reaches the model.
//  * two agents on one seed agree.
//  * a bag-empty turn, and a sole candidate, are answered without the model.
//  * ONE service call carries the whole candidate set, and the move features in
//    it are exactly what move_set::encode_move makes of each candidate at the
//    turn's score differential. (Chunking a set past the engine's ceiling is
//    the service's job, not the agent's, and is tested where it lives.)
//  * the pre-move board row the agent feeds the model is byte-identical to the
//    row the training BlockDecoder reconstructs for the same position -- the
//    drift the model itself could never reveal.

#include "agent/agent.h"
#include "agent/mset_sim_agent.h"
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
using scribblez::testing::build_slog;
using scribblez::testing::make_play_full;
using scribblez::testing::model_rank;
using scribblez::testing::opening_dict;
using scribblez::testing::rack_from;
using scribblez::testing::script_favouring;
using scribblez::testing::shortlist_candidates;
using scribblez::testing::StubMoveSetEvalService;

namespace {

// The base input layout the stub declares.
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

TEST_F(MsetSimAgentTest, OutOfRangeScalarParamsAreRejected) {
  // Every guard in validate(), including its boundaries. --rollouts is the one
  // whose absence was silent rather than loud: SimRunner only asserts its
  // bound, so a Release build ran 0 rollouts and played the model's rank-0
  // candidate every turn (see validate()'s comment).
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

  // The accepted boundaries, so the bounds cannot silently tighten.
  EXPECT_NO_THROW(build(0, 1, 1));
  EXPECT_NO_THROW(build(0, 1, SimRunner::kMaxRollouts));
}

// The agent forwards sim_horizon into its SimRunner: the runner validates the
// horizon (pairing against the injected leaf, and the lower bound) at
// construction, so a bad horizon is rejected. If the agent silently dropped
// sim_horizon it would run terminal rollouts and none of these would throw.
TEST_F(MsetSimAgentTest, TruncationHorizonIsWiredToTheRunner) {
  using scribblez::testing::StubEvalService;
  MsetSimAgent::Params p = params();
  p.sim_horizon = SimRunner::kMinHorizonPlies;  // a horizon with no leaf service
  EXPECT_THROW(MsetSimAgent(p, std::make_unique<StubMoveSetEvalService>()), std::runtime_error);
  p.sim_horizon = SimRunner::kMinHorizonPlies - 1;  // below the minimum, even with a leaf
  EXPECT_THROW(MsetSimAgent(p, std::make_unique<StubMoveSetEvalService>(),
                            std::make_unique<StubEvalService>()),
               std::runtime_error);
  p.sim_horizon = SimRunner::kMinHorizonPlies;  // a valid horizon with a leaf
  EXPECT_NO_THROW(MsetSimAgent(p, std::make_unique<StubMoveSetEvalService>(),
                               std::make_unique<StubEvalService>()));
}

TEST_F(MsetSimAgentTest, SimsTheModelsTopKAndPlaysTheRolloutsFavourite) {
  const MsetSimAgent::Params p = params();
  const std::vector<Move> candidates = shortlist_candidates(request(), p.shortlist);
  ASSERT_GT(candidates.size(), 4u);

  // The model favours the equity ranking's 3rd and 4th candidates; equity's
  // own favourites score low, so the sim set differs from SimAgent's.
  const auto scripted = script_favouring(candidates.size(), {2, 3});

  auto stub = std::make_unique<StubMoveSetEvalService>();
  stub->scripted = scripted;
  MsetSimAgent agent(p, std::move(stub));
  agent.begin_game({});
  const Move played = agent.make_move(request()).move;

  // Replay the same decision independently: the model's top-2 by scripted
  // value, simmed from the same position with the agent's own seed.
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

  // The default shortlist is every legal move, so the candidate space includes
  // every exchange; find one (static equity buries them all far below the plays
  // on this rack).
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
  // A rack with no legal play and a bag too small for exchanges leaves a lone
  // PASS candidate, which the size-1 early return plays outright.
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
  // sim_top_k above the candidate count must clamp to what exists: all three
  // candidates sim, and one of them plays.
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
  MsetSimAgent agent(params(), std::move(stub));  // endgame budget 0: solver declines
  agent.begin_game({});
  MoveRequest req{board_,          dict_,         my_rack_, opp_leave_, /*my_score=*/13,
                  /*opp_score=*/7, /*bag_size=*/0};
  const Move played = agent.make_move(req).move;
  EXPECT_TRUE(played == equity_top_k(req, 1).front());
  EXPECT_EQ(sp->calls, 0);  // the model was never consulted
}

TEST_F(MsetSimAgentTest, TheRolloutSeedFollowsTheAdvancingPly) {
  // Every other rollout test decides at ply 0, where sim_seed(ply_) and a
  // hardcoded sim_seed(0) are indistinguishable. Here two moves have been
  // observed first, so a decision seeded off a stale ply would draw different
  // rollouts and, on this scripted pair, play the other candidate.
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

  // Replay at ply 2 -- the ply the agent is actually at -- and separately at
  // ply 0, the value a stale-seed regression would use. The decision must
  // follow the former; the latter must be a different decision, or this test
  // could not tell them apart.
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

  // And that one board row is the position's own pre-move row (which the
  // decoder cross-check separately proves is the training row).
  std::vector<float> expected_row(size_t(kInputFloats), 0.0f);
  agent.encode_board_row(req, expected_row.data());
  EXPECT_EQ(sp->last_board_row, expected_row);

  // Each candidate's features are what the shared encoder makes of it at this
  // turn's pre-move differential -- the same encode_move the training rows go
  // through, reached here by the agent instead of the FFI.
  for (size_t i = 0; i < candidates.size(); ++i) {
    int32_t letters[move_set::kMoveMaxPlaced];
    uint8_t blanks[move_set::kMoveMaxPlaced];
    int32_t squares[move_set::kMoveMaxPlaced];
    uint8_t tile_mask[move_set::kMoveMaxPlaced];
    float scalars[move_set::kMoveScalars];
    move_set::encode_move(candidates[i], pre_diff, letters, blanks, squares, tile_mask, scalars);

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

  // Training path: decode the PRE-move sampled row (no symmetry flip). The same
  // dictionary drives both paths' cross-check planes.
  Dictionary dict = opening_dict();
  binlog::BlockDecoder dec(InputEncodingSpec{&dict});
  const uint8_t flips[1] = {0};
  std::vector<float> dec_row(size_t(kInputFloats + kLabelFloats), 0.0f);
  dec.decode(buf.data(), "test.slog", /*local_start=*/0, /*n_rows=*/1, flips, /*post_move=*/false,
             /*output_row_start=*/0, dec_row.data());

  // Inference path: the agent starts from the same handicap, observes turns
  // 0..1, then encodes its own decision point at turn 2.
  MsetSimAgent::Params p;
  p.name = "MS";
  p.dict = &dict;
  MsetSimAgent agent(p, std::make_unique<StubMoveSetEvalService>());
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

TEST(MsetSimAgent, ThePreMoveRowMatchesTheTrainingDecoder) {
  check_pre_move_row_matches_decoder({0, 0});
}

TEST(MsetSimAgent, AHandicapReachesTheModelRow) {
  // A head start is a score-differential feature like any other, and the
  // training replay seeds its encoder from the handicap the .slog records. An
  // agent that began every game at 0-0 would feed the model a differential
  // wrong by the head start for the whole game -- silently, since nothing
  // about the row's shape changes. The two rows must still agree.
  check_pre_move_row_matches_decoder({50, 0});
  check_pre_move_row_matches_decoder({0, 37});
}
