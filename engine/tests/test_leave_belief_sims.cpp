// Sims under a leave belief: SimRunner seats rollout i's opponent leave from
// the belief, the same for every candidate (sim/sim_runner.h); and
// SlogLeaveBeliefs builds a position's belief from its training row, only
// where the leave is modelled (belief/slog_leave_beliefs.h). Reads HastyBot's
// NWL23 leave values and skips without the Macondo checkout.

#include "agent/agent.h"
#include "belief/leave_params.h"
#include "belief/leave_sampler.h"
#include "belief/slog_leave_beliefs.h"
#include "data/binary_log.h"
#include "data/forgone_play_log.h"
#include "game/board.h"
#include "game/game.h"
#include "game/rack.h"
#include "game/tile_counts.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/leave_value_arrays.h"
#include "lexicon/leave_values.h"
#include "sim/sim_runner.h"
#include "sim/slog_position_simmer.h"
#include "temp_dir.h"
#include "training/leave_inference_task.h"
#include "util/exception.h"
#include "util/io.h"
#include "util/progress.h"

#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace scribblez;
using belief::LeaveParams;
using belief::LeaveQuery;
using belief::LeaveSampler;

namespace {

const std::string kLeaves = HastyEquity::default_leaves_path("NWL23");
constexpr Tile kQ = Tile::of('Q' - 'A');

Dictionary small_dict() {
  return Dictionary::build_from_words(
    {"AT",    "AS",    "AN",   "AR",   "AE",    "AX",    "EX",    "OX",    "IT",    "IS",
     "IN",    "ON",    "NO",   "TO",   "OR",    "RE",    "ARE",   "ART",   "ARTS",  "CAT",
     "CATS",  "EAT",   "EATS", "ERA",  "ETA",   "OAT",   "OATS",  "RAT",   "RATS",  "RATE",
     "SET",   "SIT",   "SIN",  "TAN",  "TEA",   "TEN",   "TIN",   "TOE",   "TON",   "NET",
     "NOT",   "ROT",   "SAT",  "SEA",  "TAR",   "TIE",   "TIES",  "STAR",  "STARE", "TEAR",
     "RATES", "STONE", "NOTE", "TONE", "TONES", "IRATE", "RAISE", "ARISE", "SATIRE"});
}

// A plays-the-top-score agent, so the games are real but need no equity.
class TopScoreAgent : public Agent {
 public:
  TopScoreAgent(int tid, std::string name) : Agent(tid, std::move(name)) {}
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

class LeaveBeliefSimsTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!std::filesystem::exists(kLeaves)) return;
    table_ = std::make_unique<LeaveValueArrays>(LeaveValues::load(kLeaves));
  }
  void SetUp() override {
    if (!table_) GTEST_SKIP() << "no HastyBot leave values at " << kLeaves;
    // Other suites in this binary install a synthetic table.
    HastyEquity::init(kLeaves, HastyEquity::default_peg_path());
    tmp_ = scribblez::testing::make_temp_dir("scribblez_leave_belief_sims");
  }
  void TearDown() override {
    if (!tmp_.empty()) std::filesystem::remove_all(tmp_);
  }

  // One played game as a .slog with its .forgone sidecar; returns its path.
  std::string write_game() const;

  static std::unique_ptr<LeaveValueArrays> table_;
  Dictionary dict_ = small_dict();
  std::filesystem::path tmp_;
};
std::unique_ptr<LeaveValueArrays> LeaveBeliefSimsTest::table_;

std::string LeaveBeliefSimsTest::write_game() const {
  {
    binlog::BinaryLogWriter writer(tmp_.string(), /*games_per_file=*/1);
    TopScoreAgent a0(0, "A0"), a1(1, "A1");
    Game g(a0, a1, dict_, /*seed=*/11);
    g.play();
    writer.append(g.extract_log());
  }
  std::string path;
  for (const auto& e : std::filesystem::directory_iterator(tmp_))
    if (e.path().extension() == ".slog") path = e.path().string();
  write_forgone_sidecar(util::read_file_bytes(path), dict_, forgone_path_of(path), 1);
  return path;
}

// A first-turn position whose opponent leave belief all but forces a Q.
SimPosition q_belief_position(const LeaveValueArrays& table) {
  SimPosition pos;
  pos.rack = Rack::from_string("AERSTIN");
  LeaveParams params{};
  params.theta[kQ] = 40.0f;
  pos.opp_leave_belief =
    std::make_shared<const LeaveSampler>(params, 2, pos.board.unseen_tiles(pos.rack), table);
  return pos;
}

TEST_F(LeaveBeliefSimsTest, RolloutsSeatPairedLeavesFromTheBelief) {
  const SimPosition pos = q_belief_position(*table_);
  const MoveRequest req{pos.board, dict_, pos.rack, Rack{}, 0, 0, 86};
  const std::vector<Move> candidates = equity_top_k(req, 3);
  ASSERT_GT(candidates.size(), 1u);
  const SimRunner runner(dict_, {.rollouts = 12});
  const std::vector<Rollout> rollouts = runner.run_rollouts(pos, candidates, /*base_seed=*/5);

  std::set<std::string> racks;
  for (int i = 0; i < runner.rollouts(); ++i) {
    const Rack& opp = rollouts[size_t(i)].opp_rack;
    EXPECT_EQ(opp.size(), RACK_SIZE);
    EXPECT_EQ(opp.count(kQ), 1) << opp.to_string();
    for (size_t c = 1; c < candidates.size(); ++c)
      EXPECT_TRUE(rollouts[c * runner.rollouts() + i].opp_rack == opp) << "rollout " << i;
    racks.insert(opp.to_string());
  }
  EXPECT_GT(racks.size(), 1u) << "every rollout drew the same rack";
}

TEST_F(LeaveBeliefSimsTest, RefusesAKnownLeaveWithABelief) {
  SimPosition pos = q_belief_position(*table_);
  pos.opp_leave = Rack::from_string("E");
  const SimRunner runner(dict_, {.rollouts = 2});
  EXPECT_THROW(runner.run_rollouts(pos, {Move::pass()}, 0), util::Exception);
}

// The simmer attaches the config's belief to the position it sims: the
// callback runs once, for the position asked about, and the rollouts seat its
// leave. The belief is the game's own at its first modelled turn, tilted
// toward a tile the leave pool holds.
TEST_F(LeaveBeliefSimsTest, SimmerSeatsTheConfiguredBelief) {
  const std::string path = write_game();
  const std::vector<char> slog = util::read_file_bytes(path);
  const auto* meta =
    reinterpret_cast<const binlog::GameMetadata*>(slog.data() + sizeof(binlog::FileHeader));
  Tile favoured = kQ;
  LeaveParams params{};
  const belief::LeaveParamsFn stub = [&](const float*, int, LeaveParams* out) { out[0] = params; };
  const belief::SlogLeaveBeliefs beliefs(slog, path, dict_, *table_, stub);
  std::optional<binlog::GamePositionIndex> at;
  for (uint32_t t = meta->eligible_begin; t < meta->eligible_end && !at; ++t) {
    const LeaveQuery q = beliefs.query({0, t});
    if (!q.modelled()) continue;
    for (int k = 0; k < TILE_KINDS - 1; ++k)
      if (q.leave_pool.count(Tile::of(k)) > 0) favoured = Tile::of(k);
    at = binlog::GamePositionIndex{0, t};
  }
  ASSERT_TRUE(at.has_value());
  params.theta[favoured] = 40.0f;

  std::atomic<int> calls = 0;
  SlogSimConfig config;
  config.runner.rollouts = 4;
  config.output = SimOutput::kRollouts;
  config.opp_leave_belief = [&](const binlog::GamePositionIndex& w) {
    EXPECT_TRUE(w == *at);
    ++calls;
    return beliefs.belief(w);
  };
  util::ProgressMeter meter(1, "positions");
  const std::vector<SimmedPosition> out = sim_slog_positions(slog, dict_, config, {*at}, &meter);
  ASSERT_EQ(out.size(), 1u);
  ASSERT_FALSE(out[0].rollouts.empty());
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(out[0].position.opp_leave_belief, nullptr)
    << "the simmer must drop the belief from the result";
  for (const Rollout& r : out[0].rollouts[0]) EXPECT_GE(r.opp_rack.count(favoured), 1);
}

// Every eligible turn's query reads the observer-visible target fields, and a
// belief exists exactly where the leave is modelled, from the row's input.
TEST_F(LeaveBeliefSimsTest, BeliefsComeFromTheTrainingRows) {
  const std::string path = write_game();
  const std::vector<char> slog = util::read_file_bytes(path);
  std::vector<float> seen;
  belief::LeaveParamsFn stub = [&](const float* rows, int count, LeaveParams* out) {
    ASSERT_EQ(count, 1);
    seen.assign(rows, rows + LeaveInferenceTask::kInputFloats);
    out[0] = LeaveParams{};
  };
  const belief::SlogLeaveBeliefs beliefs(slog, path, dict_, *table_, stub);
  const auto* meta =
    reinterpret_cast<const binlog::GameMetadata*>(slog.data() + sizeof(binlog::FileHeader));
  int modelled = 0;
  for (uint32_t t = meta->eligible_begin; t < meta->eligible_end; ++t) {
    const LeaveQuery q = beliefs.query({0, t});
    if (t == 0) {
      EXPECT_EQ(q.cause, LeaveRowCause::kNoOpponentMove);
    }
    EXPECT_GT(q.bag, 0);
    EXPECT_GE(q.leave_pool.size(), q.length);
    seen.clear();
    const std::shared_ptr<const LeaveSampler> b = beliefs.belief({0, t});
    ASSERT_EQ(b != nullptr, q.modelled()) << "turn " << t;
    if (!b) continue;
    ++modelled;
    EXPECT_EQ(b->length(), q.length);
    EXPECT_EQ(seen, q.input);
  }
  EXPECT_GT(modelled, 0) << "the game has no modelled turn; the check proved nothing";
}

// With the bag empty the opponent holds every unseen tile, whatever they
// kept, so the leave is not modelled.
TEST(LeaveQuery, NotModelledWithTheBagEmpty) {
  LeaveQuery q;
  q.length = 3;
  q.bag = 1;
  EXPECT_TRUE(q.modelled());
  q.bag = 0;
  EXPECT_FALSE(q.modelled());
  q.bag = 1;
  q.cause = LeaveRowCause::kPass;
  EXPECT_FALSE(q.modelled());
}

}  // namespace
