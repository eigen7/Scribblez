// PolicyBot with a stub policy-set model: it feeds the model, at every turn of
// a played game, exactly the inputs the training pipeline builds for that
// position (the pool labeler's candidates and belief, the dataset replay's
// row and candidate features); it plays the highest logit, the better static
// equity on ties; and it leaves the bag-empty turns to HastyBot's move. Reads
// HastyBot's NWL23 leave values and skips without the Macondo checkout.

#include "agent/agent.h"
#include "agent/hasty_bot.h"
#include "agent/policy_bot.h"
#include "belief/leave_params.h"
#include "belief/slog_leave_beliefs.h"
#include "data/binary_log.h"
#include "data/block_decoder.h"
#include "data/forgone_play_log.h"
#include "encoding/input_encoder.h"
#include "game/board.h"
#include "game/game.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/leave_value_arrays.h"
#include "lexicon/leave_values.h"
#include "nn/eval_service.h"
#include "sim/pool_jobs.h"
#include "sim/sim_runner.h"
#include "sim/slog_position_simmer.h"
#include "temp_dir.h"
#include "training/policy_set_inputs.h"
#include "util/exception.h"
#include "util/io.h"
#include "util/progress.h"

#include <gtest/gtest.h>

#include <array>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace scribblez;

namespace {

const std::string kLeaves = HastyEquity::default_leaves_path("NWL23");
constexpr int kBlankPrune = 40;

Dictionary small_dict() {
  return Dictionary::build_from_words(
    {"AT",   "AS",   "AN",    "AR",    "AE",    "AX",   "EX",   "OX",   "IT",    "IS",
     "IN",   "ON",   "NO",    "TO",    "OR",    "RE",   "ARE",  "ART",  "ARTS",  "CAT",
     "CATS", "EAT",  "EATS",  "ERA",   "ETA",   "OAT",  "OATS", "RAT",  "RATS",  "RATE",
     "SET",  "SIT",  "SIN",   "TAN",   "TEA",   "TEN",  "TIN",  "TOE",  "TON",   "NET",
     "NOT",  "ROT",  "SAT",   "SEA",   "TAR",   "TIE",  "TIES", "STAR", "STARE", "TEAR",
     "NOTE", "TONE", "TONES", "IRATE", "RAISE", "ARISE"});
}

// One scored position, as the stub saw it.
struct ScoredPosition {
  std::vector<float> board_row;
  std::array<float, policy_set::kOppRackFloats> opp_rack;
  policy_set::CandidateInputs candidates;
};

// A scripted policy-set model that keeps its inputs. Its logits favour
// candidate `favourite` (the first when the pool is shorter), or follow
// `logits` when set; it declares the hidden-leaves layout, or the face-up one
// with `face_up`.
class StubPolicySetService : public nn::PolicySetEvalService {
 public:
  int favourite = 1;
  bool face_up = false;
  std::vector<float> logits;
  std::vector<ScoredPosition> calls;

  bool opp_leave_input() const override { return face_up; }
  int spatial_planes() const override { return scribblez::spatial_planes(); }
  int scalar_floats() const override { return scribblez::scalar_floats({nullptr, face_up}); }

 protected:
  void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) override;
};

void StubPolicySetService::do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) {
  const int n = batch.candidates->count();
  ScoredPosition& seen = calls.emplace_back();
  seen.board_row.assign(batch.board_row, batch.board_row + input_floats(InputEncodingSpec{}));
  std::copy(batch.opp_rack, batch.opp_rack + policy_set::kOppRackFloats, seen.opp_rack.begin());
  seen.candidates = *batch.candidates;
  for (int i = 0; i < n; ++i) {
    head_out[0][i] = logits.empty() ? float(i == std::min(favourite, n - 1)) : logits[size_t(i)];
  }
}

// A plays-the-top-score opponent, so the games are real but cheap.
class TopScoreAgent : public Agent {
 public:
  TopScoreAgent() : Agent(1, "TopScore") {}
  MoveDecision make_move(const MoveRequest& req) override;
};

MoveDecision TopScoreAgent::make_move(const MoveRequest& req) {
  const std::vector<Move> plays = generate_legal_plays(req);
  if (plays.empty()) return Move::pass();
  const Move* best = &plays[0];
  for (const Move& m : plays)
    if (m.score() > best->score()) best = &m;
  return *best;
}

// A belief that differs by tile, so a wrong summary shows.
void tilted_belief(const float* /*rows*/, int count, belief::LeaveParams* out) {
  for (int i = 0; i < count; ++i) {
    out[i] = belief::LeaveParams{};
    out[i].theta[4] = 1.5f;  // E
  }
}

class PolicyBotTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!std::filesystem::exists(kLeaves)) return;
    table_ = std::make_shared<const LeaveValueArrays>(LeaveValues::load(kLeaves));
  }
  void SetUp() override {
    if (!table_) GTEST_SKIP() << "no HastyBot leave values at " << kLeaves;
    // Other suites in this binary install a synthetic table.
    HastyEquity::init(kLeaves, HastyEquity::default_peg_path());
    tmp_ = scribblez::testing::make_temp_dir("scribblez_policy_bot");
  }
  void TearDown() override {
    if (!tmp_.empty()) std::filesystem::remove_all(tmp_);
  }

  PolicyBot make_bot(belief::LeaveParamsFn predict = {}) {
    auto stub = std::make_shared<StubPolicySetService>();
    stub_ = stub.get();
    std::shared_ptr<const LeaveValueArrays> table = predict ? table_ : nullptr;
    return PolicyBot({.thread_id = 0, .name = "PolicyBot", .dict = &dict_},
                     {.policy = stub,
                      .blank_prune_points = kBlankPrune,
                      .predict = std::move(predict),
                      .table = std::move(table)});
  }

  // A game of `bot` (seat 0) against TopScoreAgent, as a .slog with its
  // .forgone sidecar; returns its path.
  std::string write_game(PolicyBot& bot);

  static std::shared_ptr<const LeaveValueArrays> table_;
  Dictionary dict_ = small_dict();
  std::filesystem::path tmp_;
  StubPolicySetService* stub_ = nullptr;
};
std::shared_ptr<const LeaveValueArrays> PolicyBotTest::table_;

std::string PolicyBotTest::write_game(PolicyBot& bot) {
  TopScoreAgent opp;
  {
    binlog::BinaryLogWriter writer(tmp_.string(), /*games_per_file=*/1);
    Game g(bot, opp, dict_, /*seed=*/11);
    g.play();
    writer.append(g.extract_log());
  }
  std::string path;
  for (const auto& e : std::filesystem::directory_iterator(tmp_))
    if (e.path().extension() == ".slog") path = e.path().string();
  write_forgone_sidecar(util::read_file_bytes(path), dict_, forgone_path_of(path));
  return path;
}

// The position's inputs as training builds them: the pool labeler's
// candidates and belief summary, and the dataset replay's row and candidate
// inputs.
struct TrainingInputs {
  std::vector<Move> moves;
  ScoredPosition scored;
};

TrainingInputs training_inputs(const std::vector<char>& slog, const std::string& path,
                               const Dictionary& dict, const belief::SlogLeaveBeliefs& beliefs,
                               uint32_t turn) {
  SlogSimConfig config;
  config.selector = pool_selector(kBlankPrune);
  util::ProgressMeter meter(1, "positions");
  const SimmedPosition selected =
    select_slog_candidates(slog, dict, config, {{0, turn}}, &meter).front();
  TrainingInputs out;
  out.moves = selected.candidates.moves;
  const PoolJob job =
    make_pool_job(selected, beliefs.belief({0, turn}).get(), dict, /*rollouts=*/1);
  out.scored.opp_rack = policy_set::encode_opp_rack(job.record.opp_rack);
  const InputEncodingSpec spec{&dict};
  binlog::BlockDecoder decoder(spec);
  std::vector<float> row(
    size_t(binlog::BlockDecoder::row_floats_for(binlog::DecodeTask::kPositionEval, spec)));
  policy_set::replay_candidate_inputs(decoder, dict, slog.data(), path, 0, turn,
                                      /*transpose=*/false, out.moves, row.data(), 0,
                                      &out.scored.candidates);
  row.resize(size_t(input_floats(spec)));
  out.scored.board_row = std::move(row);
  return out;
}

void expect_same_inputs(const ScoredPosition& got, const ScoredPosition& want, uint32_t turn) {
  EXPECT_EQ(got.board_row, want.board_row) << "turn " << turn;
  EXPECT_EQ(got.opp_rack, want.opp_rack) << "turn " << turn;
  const move_set::MoveFeatureArrays& g = got.candidates.features;
  const move_set::MoveFeatureArrays& w = want.candidates.features;
  ASSERT_EQ(g.count, w.count) << "turn " << turn;
  EXPECT_EQ(g.letters, w.letters) << "turn " << turn;
  EXPECT_EQ(g.blanks, w.blanks) << "turn " << turn;
  EXPECT_EQ(g.squares, w.squares) << "turn " << turn;
  EXPECT_EQ(g.tile_mask, w.tile_mask) << "turn " << turn;
  EXPECT_EQ(g.scalars, w.scalars) << "turn " << turn;
  EXPECT_EQ(g.cross_cells, w.cross_cells) << "turn " << turn;
  EXPECT_EQ(g.cross_letters, w.cross_letters) << "turn " << turn;
  EXPECT_EQ(got.candidates.equity, want.candidates.equity) << "turn " << turn;
}

}  // namespace

// Every turn PolicyBot scored, in a game it played with inferred beliefs, fed
// the model the inputs training builds at that position, and it played the
// pool candidate the stub favoured.
TEST_F(PolicyBotTest, FeedsTheModelTheTrainingInputsOfEachPosition) {
  PolicyBot bot = make_bot(tilted_belief);
  const std::string path = write_game(bot);
  const std::vector<char> slog = util::read_file_bytes(path);
  const belief::SlogLeaveBeliefs beliefs(slog, path, dict_, *table_, tilted_belief);
  std::vector<TurnRecord> scratch;
  const GameLog g = binlog::make_game_view(slog.data(), 0, scratch, nullptr);
  const auto* meta =
    reinterpret_cast<const binlog::GameMetadata*>(slog.data() + sizeof(binlog::FileHeader));

  size_t call = 0;
  int modelled = 0;
  for (uint32_t turn = 0; turn < uint32_t(g.num_records); turn += 2) {
    if (turn >= meta->eligible_end) break;  // the bag is empty: HastyBot's move
    const TrainingInputs want = training_inputs(slog, path, dict_, beliefs, turn);
    if (want.moves.size() == 1) continue;  // nothing to score
    ASSERT_LT(call, stub_->calls.size()) << "turn " << turn << " was not scored";
    expect_same_inputs(stub_->calls[call++], want.scored, turn);
    EXPECT_EQ(g.records[turn].move, want.moves[1]) << "turn " << turn;
    modelled += want.scored.opp_rack[TILE_KINDS] == 1.0f;
  }
  EXPECT_EQ(call, stub_->calls.size());
  EXPECT_GE(call, 3u);
  EXPECT_GE(modelled, 2);  // the belief came from the model, not only uniform
}

TEST_F(PolicyBotTest, PlaysTheHighestLogitTheBetterEquityOnTies) {
  PolicyBot bot = make_bot();
  bot.begin_game({});
  const Board board;
  const Rack rack = Rack::from_string("AERSTIN");
  const Rack hidden;
  const MoveRequest req{board, dict_, rack, hidden, 0, 0, /*bag_size=*/86};
  const std::vector<Move> pool =
    policy_set::pool_candidates(equity_top_k(req, std::numeric_limits<int>::max()), kBlankPrune)
      .moves;
  ASSERT_GT(pool.size(), 3u);
  stub_->logits.assign(pool.size(), 0.0f);
  stub_->logits[2] = stub_->logits[3] = 1.0f;
  EXPECT_EQ(bot.make_move(req).move, pool[2]);
}

TEST_F(PolicyBotTest, LeavesTheBagEmptyTurnsToHastyBot) {
  PolicyBot bot = make_bot();
  bot.begin_game({});
  const Board board;
  const Rack rack = Rack::from_string("AERSTIN");
  const Rack opp = Rack::from_string("EOT");
  const MoveRequest req{board, dict_, rack, opp, 0, 0, /*bag_size=*/0};
  EXPECT_EQ(bot.make_move(req).move, hasty_best_move_wmp(req));
  EXPECT_TRUE(stub_->calls.empty());
}

TEST_F(PolicyBotTest, RefusesAFaceUpOpponentRack) {
  PolicyBot bot = make_bot();
  bot.begin_game({});
  const Board board;
  const Rack rack = Rack::from_string("AERSTIN");
  const Rack opp = Rack::from_string("EO");
  const MoveRequest req{board, dict_, rack, opp, 0, 0, /*bag_size=*/86};
  EXPECT_THROW(bot.make_move(req), std::exception);
}

TEST_F(PolicyBotTest, RefusesANonFiniteLogit) {
  PolicyBot bot = make_bot();
  bot.begin_game({});
  const Board board;
  const Rack rack = Rack::from_string("AERSTIN");
  const Rack hidden;
  const MoveRequest req{board, dict_, rack, hidden, 0, 0, /*bag_size=*/86};
  const size_t pool =
    policy_set::pool_candidates(equity_top_k(req, std::numeric_limits<int>::max()), kBlankPrune)
      .moves.size();
  stub_->logits.assign(pool, 0.0f);
  stub_->logits[1] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(bot.make_move(req), util::Exception);
}

TEST_F(PolicyBotTest, RefusesAFaceUpLeavesModel) {
  auto stub = std::make_shared<StubPolicySetService>();
  stub->face_up = true;
  PolicyBot::Backend backend;
  backend.policy = stub;
  backend.blank_prune_points = kBlankPrune;
  PolicyBot::Params params;
  params.dict = &dict_;
  EXPECT_THROW(PolicyBot(params, std::move(backend)), util::CleanException);
}

namespace {

// The factory's metadata checks, on what it reads before building an engine:
// a PolicyBot naming a model and a leave inference export, and models trained
// on uniform beliefs and on the beliefs of tag "li" generation 7.
PolicyBot::Params named_params() {
  PolicyBot::Params p;
  p.service.model = "policy.onnx";
  p.leave_inference = "leave.onnx";
  return p;
}

nn::ModelIdentity leave_model(const std::string& tag, int generation) {
  nn::ModelIdentity id;
  id.tag = tag;
  id.generation = generation;
  return id;
}

PolicyBot::TrainedOn trained_on(const nn::ModelIdentity& leave) {
  PolicyBot::TrainedOn t;
  t.blank_prune_points = kBlankPrune;
  t.leave_model = leave;
  return t;
}

}  // namespace

TEST(PolicyBotTrainedOn, AcceptsTheLeaveModelItTrainedOn) {
  const nn::ModelIdentity leave = leave_model("li", 7);
  EXPECT_NO_THROW(PolicyBot::check_trained_on(named_params(), trained_on(leave), &leave));
  EXPECT_NO_THROW(PolicyBot::check_trained_on(named_params(), trained_on({}), nullptr));
}

TEST(PolicyBotTrainedOn, RefusesAnotherLeaveModelGeneration) {
  const PolicyBot::TrainedOn trained = trained_on(leave_model("li", 7));
  const nn::ModelIdentity other_generation = leave_model("li", 8);
  const nn::ModelIdentity other_tag = leave_model("other", 7);
  EXPECT_THROW(PolicyBot::check_trained_on(named_params(), trained, &other_generation),
               util::CleanException);
  EXPECT_THROW(PolicyBot::check_trained_on(named_params(), trained, &other_tag),
               util::CleanException);
}

TEST(PolicyBotTrainedOn, RefusesAMissingOrUnwantedLeaveModel) {
  const nn::ModelIdentity leave = leave_model("li", 7);
  EXPECT_THROW(PolicyBot::check_trained_on(named_params(), trained_on(leave), nullptr),
               util::CleanException);
  EXPECT_THROW(PolicyBot::check_trained_on(named_params(), trained_on({}), &leave),
               util::CleanException);
}

TEST(PolicyBotTrainedOn, RefusesAnExportWithoutABlankPrune) {
  EXPECT_THROW(PolicyBot::check_trained_on(named_params(), PolicyBot::TrainedOn{}, nullptr),
               util::CleanException);
}
