// The .belief sidecar (data/belief_log.h): one entry per turn of the leave
// inference rows' region, each the query's cause and length, the model's
// parameters and the belief's summaries within the stored precision; the
// held-out draws are DeepBot's (belief/leave_draws.h); and a sidecar that does
// not belong to its .slog or leave table is refused. The model is a stub.
// Reads HastyBot's NWL23 leave values and skips without the Macondo checkout.

#include "agent/agent.h"
#include "belief/leave_draws.h"
#include "belief/leave_params.h"
#include "belief/leave_query.h"
#include "belief/slog_leave_beliefs.h"
#include "data/belief_log.h"
#include "data/binary_log.h"
#include "data/forgone_play_log.h"
#include "game/game.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/leave_value_arrays.h"
#include "lexicon/leave_values.h"
#include "temp_dir.h"
#include "util/exception.h"
#include "util/io.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace scribblez;
using belief::LeaveParams;
using belief::LeaveQuery;

namespace {

const std::string kLeaves = HastyEquity::default_leaves_path("NWL23");

Dictionary small_dict() {
  return Dictionary::build_from_words(
    {"AT",    "AS",    "AN",   "AR",   "AE",    "AX",    "EX",    "OX",    "IT",    "IS",
     "IN",    "ON",    "NO",   "TO",   "OR",    "RE",    "ARE",   "ART",   "ARTS",  "CAT",
     "CATS",  "EAT",   "EATS", "ERA",  "ETA",   "OAT",   "OATS",  "RAT",   "RATS",  "RATE",
     "SET",   "SIT",   "SIN",  "TAN",  "TEA",   "TEN",   "TIN",   "TOE",   "TON",   "NET",
     "NOT",   "ROT",   "SAT",  "SEA",  "TAR",   "TIE",   "TIES",  "STAR",  "STARE", "TEAR",
     "RATES", "STONE", "NOTE", "TONE", "TONES", "IRATE", "RAISE", "ARISE", "SATIRE"});
}

// Plays the top-scoring move, or passes at its `pass_at`-th move, so the
// games are real but need no equity.
class TopScoreAgent : public Agent {
 public:
  TopScoreAgent(int tid, std::string name, int pass_at = -1)
      : Agent(tid, std::move(name)), pass_at_(pass_at) {}
  MoveDecision make_move(const MoveRequest& req) override;

 private:
  int pass_at_;
  int moves_ = 0;
};

MoveDecision TopScoreAgent::make_move(const MoveRequest& req) {
  if (moves_++ == pass_at_) return Move::pass();
  const std::vector<Move> plays = generate_legal_plays(req);
  if (plays.empty()) return Move::pass();
  const Move* best = &plays[0];
  for (const Move& m : plays)
    if (m.score() > best->score()) best = &m;
  return *best;
}

// The stub model: a tilt toward A and against E, the same for every row.
LeaveParams stub_params() {
  LeaveParams p{};
  p.theta[0] = 1.5f;
  p.theta[4] = -0.75f;
  p.a = 0.01f;
  return p;
}

const belief::LeaveParamsFn kStub = [](const float*, int count, LeaveParams* out) {
  for (int i = 0; i < count; ++i) out[i] = stub_params();
};

class BeliefLogTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!std::filesystem::exists(kLeaves)) return;
    table_ = std::make_unique<LeaveValueArrays>(LeaveValues::load(kLeaves));
  }
  void SetUp() override {
    if (!table_) GTEST_SKIP() << "no HastyBot leave values at " << kLeaves;
    HastyEquity::init(kLeaves, HastyEquity::default_peg_path());
    tmp_ = scribblez::testing::make_temp_dir("scribblez_belief_log");
  }
  void TearDown() override {
    if (!tmp_.empty()) std::filesystem::remove_all(tmp_);
  }

  // One game as a .slog with its .forgone sidecar under `dir`; returns its
  // path. The second player passes at its `pass_at`-th move.
  std::string write_game(const std::string& dir, uint64_t seed, int pass_at = -1) const;
  // The sidecar of the .slog at `path`, with `draws` stored leaves.
  void write_sidecar(const std::string& path, int draws) const;

  static std::unique_ptr<LeaveValueArrays> table_;
  Dictionary dict_ = small_dict();
  std::filesystem::path tmp_;
};

std::unique_ptr<LeaveValueArrays> BeliefLogTest::table_;

std::string BeliefLogTest::write_game(const std::string& dir, uint64_t seed, int pass_at) const {
  const std::filesystem::path out = tmp_ / dir;
  std::filesystem::create_directories(out);
  {
    TopScoreAgent a0(0, "A0"), a1(1, "A1", pass_at);
    binlog::BinaryLogWriter writer(out.string(), /*games_per_file=*/1);
    Game g(a0, a1, dict_, seed);
    g.play();
    writer.append(g.extract_log());
  }
  std::string path;
  for (const auto& e : std::filesystem::directory_iterator(out))
    if (e.path().extension() == ".slog") path = e.path().string();
  write_forgone_sidecar(util::read_file_bytes(path), dict_, forgone_path_of(path));
  return path;
}

void BeliefLogTest::write_sidecar(const std::string& path, int draws) const {
  const BeliefModel model{kStub, {"0123456789abcdef", "stub", 7, 2}};
  write_belief_sidecar(util::read_file_bytes(path), path, dict_, *table_, model, draws,
                       belief_path_of(path));
}

const binlog::GameMetadata& first_game(const std::vector<char>& slog) {
  return *reinterpret_cast<const binlog::GameMetadata*>(slog.data() + sizeof(binlog::FileHeader));
}

// float16 rounding, relative.
constexpr double kHalf = 1.0 / 1024;

TEST_F(BeliefLogTest, EachTurnOfTheRegionHoldsItsBeliefsSummaries) {
  const std::string path = write_game("plain", 11);
  write_sidecar(path, /*draws=*/0);
  const std::vector<char> slog = util::read_file_bytes(path);
  const BeliefSidecar sidecar(belief_path_of(path), slog.data(), table_->content_hash());
  EXPECT_EQ(sidecar.identity().tag, "stub");
  EXPECT_EQ(sidecar.identity().generation, 7);
  EXPECT_EQ(sidecar.draws(), 0);

  const binlog::GameMetadata& meta = first_game(slog);
  const std::span<const BeliefEntry> entries = sidecar.game(0);
  ASSERT_EQ(int(entries.size()), meta.eligible_end - meta.eligible_begin);
  const belief::SlogLeaveBeliefs beliefs(slog, path, dict_, *table_, kStub);
  int modelled = 0;
  for (size_t k = 0; k < entries.size(); ++k) {
    const LeaveQuery q = beliefs.query({0, uint32_t(meta.eligible_begin + int(k))});
    const LeaveBelief b = LeaveBelief::unpack(entries[k]);
    ASSERT_EQ(b.cause, q.cause) << k;
    ASSERT_EQ(b.length, q.length) << k;
    if (!q.modelled()) {
      // An empty leave's summaries are exact; a pass's are the next test's.
      if (q.length == 0) {
        for (double e : b.expected) EXPECT_EQ(e, 0) << k;
      }
      continue;
    }
    ++modelled;
    EXPECT_NEAR(b.params.theta[0], 1.5, 1.5 * kHalf);
    EXPECT_NEAR(b.params.theta[4], -0.75, 0.75 * kHalf);
    const belief::LeaveMoments m = belief::belief_of(q, stub_params(), *table_)->moments();
    EXPECT_NEAR(b.value_mean, m.value_mean, std::abs(m.value_mean) * kHalf + 1e-3) << k;
    EXPECT_NEAR(b.value_sd, m.value_sd, m.value_sd * kHalf + 1e-3) << k;
    for (int t = 0; t < TILE_KINDS; ++t)
      EXPECT_NEAR(b.expected[size_t(t)], m.expected[size_t(t)], q.length / 510.0 + 1e-9) << k;
  }
  EXPECT_GT(modelled, 0);
}

// After a pass the opponent kept seven tiles, a leave the table does not
// hold: its expected counts are seven times each kind's share of the pool.
TEST_F(BeliefLogTest, APassKeepsSevenTilesAtThePoolsShares) {
  const std::string path = write_game("pass", 11, /*pass_at=*/1);
  write_sidecar(path, /*draws=*/0);
  const std::vector<char> slog = util::read_file_bytes(path);
  const BeliefSidecar sidecar(belief_path_of(path), slog.data(), table_->content_hash());
  const binlog::GameMetadata& meta = first_game(slog);
  const belief::SlogLeaveBeliefs beliefs(slog, path, dict_, *table_, kStub);
  bool seen = false;
  for (size_t k = 0; k < sidecar.game(0).size(); ++k) {
    const LeaveQuery q = beliefs.query({0, uint32_t(meta.eligible_begin + int(k))});
    if (q.cause != LeaveRowCause::kPass) continue;
    seen = true;
    const LeaveBelief b = LeaveBelief::unpack(sidecar.game(0)[k]);
    EXPECT_EQ(b.length, RACK_SIZE);
    EXPECT_EQ(b.value_mean, 0);
    for (int t = 0; t < TILE_KINDS; ++t) {
      const double share = double(q.leave_pool.count(Tile::of(t))) / q.leave_pool.size();
      EXPECT_NEAR(b.expected[size_t(t)], RACK_SIZE * share, RACK_SIZE / 510.0 + 1e-9) << t;
    }
  }
  EXPECT_TRUE(seen);
}

TEST_F(BeliefLogTest, TheHeldOutDrawsAreDeepBots) {
  const std::string path = write_game("draws", 11);
  write_sidecar(path, /*draws=*/16);
  const std::vector<char> slog = util::read_file_bytes(path);
  const BeliefSidecar sidecar(belief_path_of(path), slog.data(), table_->content_hash());
  ASSERT_EQ(sidecar.draws(), 16);
  const binlog::GameMetadata& meta = first_game(slog);
  const belief::SlogLeaveBeliefs beliefs(slog, path, dict_, *table_, kStub);
  for (size_t k = 0; k < sidecar.game(0).size(); ++k) {
    const int turn = meta.eligible_begin + int(k);
    const LeaveQuery q = beliefs.query({0, uint32_t(turn)});
    std::shared_ptr<const belief::LeaveSampler> sampler;
    if (belief::draws_read_belief(q))
      sampler = belief::belief_of(q, q.modelled() ? stub_params() : LeaveParams{}, *table_);
    const std::vector<Rack> want =
      belief::opp_leave_draws(q, sampler.get(), 16, belief_draw_seed(0, turn));
    for (int i = 0; i < 16; ++i) EXPECT_EQ(sidecar.draw(0, k, i), want[size_t(i)]) << k << " " << i;
  }
}

TEST_F(BeliefLogTest, RefusesASidecarThatIsNotItsSlogs) {
  const std::string path = write_game("a", 11);
  write_sidecar(path, /*draws=*/0);
  const std::vector<char> slog = util::read_file_bytes(path);
  EXPECT_THROW(BeliefSidecar(belief_path_of(path), slog.data(), "other values"), util::Exception);

  const std::string other = write_game("b", 12);
  const std::vector<char> other_slog = util::read_file_bytes(other);
  EXPECT_THROW(BeliefSidecar(belief_path_of(path), other_slog.data(), table_->content_hash()),
               util::Exception);

  const std::filesystem::path sidecar = belief_path_of(path);
  std::filesystem::resize_file(sidecar, std::filesystem::file_size(sidecar) - 1);
  EXPECT_THROW(BeliefSidecar(belief_path_of(path), slog.data(), table_->content_hash()),
               util::Exception);
}

}  // namespace
