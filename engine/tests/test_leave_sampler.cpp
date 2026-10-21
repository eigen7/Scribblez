// LeaveSampler (belief/leave_sampler.h): untilted, its probabilities are the
// pool's hypergeometric prior; tilted, its quantiles follow its own
// probabilities; it never yields a leave the pool cannot supply; its moments
// are its rows'. Also summarize_opp_rack (belief/opp_rack_summary.h), the rack
// its leaves and a refill make. The tilt's
// exact numbers are checked against the trainer's PyTorch pass in
// py/tests/test_leave_inference_export.py. Reads HastyBot's NWL23 leave
// values and skips without the Macondo checkout.

#include "belief/leave_params.h"
#include "belief/leave_prior.h"
#include "belief/leave_sampler.h"
#include "belief/opp_rack_summary.h"
#include "game/rack.h"
#include "game/tile_counts.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/leave_value_arrays.h"
#include "lexicon/leave_values.h"
#include "util/exception.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <vector>

using namespace scribblez;
using belief::LeaveParams;
using belief::LeaveSampler;

namespace {

const std::string kLeaves = HastyEquity::default_leaves_path("NWL23");

class LeaveSamplerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!std::filesystem::exists(kLeaves)) return;
    table_ = std::make_unique<LeaveValueArrays>(LeaveValues::load(kLeaves));
  }
  void SetUp() override {
    if (!table_) GTEST_SKIP() << "no HastyBot leave values at " << kLeaves;
  }
  static std::unique_ptr<LeaveValueArrays> table_;
};
std::unique_ptr<LeaveValueArrays> LeaveSamplerTest::table_;

// A mid-game pool: the full bag less a typical board's and rack's tiles, so
// some kinds are short and some (Q, Z, the blanks) gone.
TileCounts mid_game_pool() {
  TileCounts pool = TileCounts::full_distribution();
  pool.remove(TileCounts::from_string("QZ??EEEEEAAAAIIIOOONNRRTTSSLDGX"));
  return pool;
}

LeaveParams tilted() {
  LeaveParams p{};
  for (int t = 0; t < TILE_KINDS; ++t) p.theta[t] = 0.1f * float((t * 7) % 11 - 5);
  p.a = 0.08f;
  p.b = 0.002f;
  return p;
}

TEST_F(LeaveSamplerTest, UntiltedIsTheHypergeometricPrior) {
  const TileCounts pool = mid_game_pool();
  const LeaveSampler sampler(LeaveParams{}, 3, pool, *table_);
  for (size_t r = 0; r < table_->rows(3); ++r) {
    const double expected = belief::log_hypergeometric_prior(sampler.leave(r), pool);
    if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(sampler.log_prob(r))) << sampler.leave(r).to_string();
    } else {
      EXPECT_NEAR(sampler.log_prob(r), expected, 1e-9) << sampler.leave(r).to_string();
    }
  }
}

// On an even grid of N quantiles, each row is drawn within one of N times
// its probability.
TEST_F(LeaveSamplerTest, QuantilesFollowTheProbabilities) {
  const LeaveSampler sampler(tilted(), 2, mid_game_pool(), *table_);
  constexpr int kGrid = 20000;
  std::map<size_t, int> drawn;
  for (int i = 0; i < kGrid; ++i) ++drawn[sampler.sample((i + 0.5) / kGrid)];
  double total = 0;
  for (size_t r = 0; r < table_->rows(2); ++r) {
    const double p = std::exp(sampler.log_prob(r));
    total += p;
    EXPECT_NEAR(double(drawn[r]) / kGrid, p, 1.0 / kGrid) << sampler.leave(r).to_string();
  }
  EXPECT_NEAR(total, 1.0, 1e-9);
}

TEST_F(LeaveSamplerTest, NeverYieldsWhatThePoolCannotSupply) {
  const TileCounts pool = mid_game_pool();
  const LeaveSampler sampler(tilted(), 4, pool, *table_);
  for (int i = 0; i < 1000; ++i) {
    const Rack leave = sampler.leave(sampler.sample(i / 1000.0));
    TileCounts left = pool;
    EXPECT_TRUE(left.remove(leave.counts())) << leave.to_string();
  }
  for (size_t r = 0; r < table_->rows(4); ++r) {
    if (sampler.leave(r).count(Tile::from_char('Q')) > 0) {
      EXPECT_TRUE(std::isinf(sampler.log_prob(r)));
    }
  }
}

TEST_F(LeaveSamplerTest, RefusesQueriesWithNoLeaves) {
  const TileCounts pool = mid_game_pool();
  EXPECT_THROW(LeaveSampler(LeaveParams{}, 0, pool, *table_), util::Exception);
  EXPECT_THROW(LeaveSampler(LeaveParams{}, kLeaveArraysMaxLength + 1, pool, *table_),
               util::Exception);
  EXPECT_THROW(LeaveSampler(LeaveParams{}, 3, TileCounts::from_string("AB"), *table_),
               util::Exception);
}

TEST_F(LeaveSamplerTest, MomentsAreTheRowsMoments) {
  const LeaveSampler sampler(tilted(), 4, mid_game_pool(), *table_);
  std::array<double, TILE_KINDS> presence{};
  double mean = 0, sq = 0;
  for (size_t r = 0; r < table_->rows(4); ++r) {
    const double p = std::exp(sampler.log_prob(r));
    if (p == 0) continue;
    const Rack leave = sampler.leave(r);
    for (int t = 0; t < TILE_KINDS; ++t)
      if (leave.count(Tile::of(t)) > 0) presence[t] += p;
    const double v = table_->values(4)[r];
    mean += p * v;
    sq += p * v * v;
  }
  const belief::LeaveMoments m = sampler.moments();
  for (int t = 0; t < TILE_KINDS; ++t) EXPECT_NEAR(m.presence[t], presence[t], 1e-9) << t;
  EXPECT_NEAR(m.value_mean, mean, 1e-9);
  EXPECT_NEAR(m.value_sd, std::sqrt(sq - mean * mean), 1e-9);
  EXPECT_EQ(m.presence[Tile::from_char('Q').index()], 0);
}

// With no leave, the rack is a uniform full draw: it lacks a kind with
// probability C(unseen - copies, 7) / C(unseen, 7).
TEST(OppRackSummary, AnUnmodelledRackIsAUniformDraw) {
  const TileCounts unseen = TileCounts::from_string("AABBCDEFGH");  // 10 tiles
  const belief::OppRackSummary s = belief::summarize_opp_rack(unseen, nullptr);
  EXPECT_FALSE(s.modelled);
  EXPECT_NEAR(s.presence[Tile::from_char('A').index()], 1 - 8.0 / 120, 1e-6);   // C(8,7)/C(10,7)
  EXPECT_NEAR(s.presence[Tile::from_char('C').index()], 1 - 36.0 / 120, 1e-6);  // C(9,7)/C(10,7)
  EXPECT_EQ(s.presence[Tile::from_char('Z').index()], 0);
  EXPECT_EQ(s.leave_value_mean, 0);
}

// A modelled rack's presence is that of the believed leave plus a uniform
// refill, checked against drawing both.
TEST_F(LeaveSamplerTest, AModelledRackIsTheLeavePlusARefill) {
  TileCounts unseen = mid_game_pool();
  unseen.remove(TileCounts::from_string("AEEINRSTTULLOO"));  // the bag gets short
  const LeaveSampler sampler(tilted(), 3, unseen, *table_);
  const belief::OppRackSummary s = belief::summarize_opp_rack(unseen, &sampler);
  EXPECT_TRUE(s.modelled);
  EXPECT_NEAR(s.leave_value_mean, sampler.moments().value_mean, 1e-4);

  std::mt19937_64 rng(7);
  constexpr int kDraws = 40000;
  std::array<int, TILE_KINDS> held{};
  for (int i = 0; i < kDraws; ++i) {
    const Rack leave = sampler.leave(sampler.sample((i + 0.5) / kDraws));
    TileCounts rest = unseen;
    rest.remove(leave.counts());
    std::vector<int> tiles;
    for (int t = 0; t < TILE_KINDS; ++t) tiles.insert(tiles.end(), rest.count(Tile::of(t)), t);
    std::shuffle(tiles.begin(), tiles.end(), rng);
    std::array<bool, TILE_KINDS> in{};
    for (int j = 0; j < leave.size(); ++j) in[leave.tiles()[j].index()] = true;
    for (int j = 0; j < RACK_SIZE - leave.size(); ++j) in[tiles[j]] = true;
    for (int t = 0; t < TILE_KINDS; ++t) held[t] += in[t];
  }
  for (int t = 0; t < TILE_KINDS; ++t)
    EXPECT_NEAR(s.presence[t], double(held[t]) / kDraws, 0.01) << t;
}

}  // namespace
