// Unit tests for Bayesian rack inference (docs/roadmap.md, track B).
//
//  * the hypergeometric prior: enumeration agrees with the count, priors
//    normalize, and drawing from the pool reproduces them empirically -- the
//    property the sampling path relies on to skip the prior term.
//  * the observations that carry nothing to infer: a bingo, a pass, an empty
//    bag.
//  * posterior shape: normalized, ordered, every leave the right size, and
//    reproducible across runs.
//  * that inference actually informs -- over a spread of racks, the leave the
//    opponent really kept ends up weighted well above its prior. This is the
//    unit-test-scale version of the offline ground-truth readout that sets the
//    temperature (docs/roadmap.md, B2).

#include "agent/agent.h"
#include "belief/leave_prior.h"
#include "belief/move_likelihood.h"
#include "belief/rack_inference.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "game/tile_counts.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "synthetic_equity.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace scribblez;
using namespace scribblez::belief;

namespace {

Rack rack_from(const std::string& s) {
  Rack r;
  for (char c : s) r.add(c == '?' ? BLANK : Tile::from_char(c));
  return r;
}

TileCounts pool_from(const std::string& s) {
  TileCounts c;
  for (char ch : s) c.add(ch == '?' ? BLANK : Tile::from_char(ch));
  return c;
}

// Words spanning a common letter set at six and seven letters, so that adding
// one tile to a rack can unlock a bingo. That is what makes hypotheses
// distinguishable: a leave the opponent demonstrably did not have would have
// led them to a different, better play.
Dictionary bingo_capable_dict() {
  return Dictionary::build_from_words(
    {"AE",    "AR",     "AS",      "AT",    "ARC",    "ARCS",   "ARE",     "ART",   "ARTS",
     "ATE",   "CAR",    "CARE",    "CARES", "CARET",  "CARETS", "CARS",    "CART",  "CARTS",
     "CASTE", "CASTER", "CASTERS", "CAT",   "CATS",   "CATER",  "CATERS",  "CRATE", "CRATES",
     "EAR",   "EARS",   "EAT",     "EATS",  "ERA",    "ETA",    "RACE",    "RACES", "RAT",
     "RATE",  "RATES",  "RATS",    "REACT", "REACTS", "RECAST", "RECASTS", "SCAR",  "SCARE",
     "SET",   "STARE",  "TARE",    "TARES", "TASTE",  "TASTER", "TASTERS", "TEA",   "TEAR",
     "TEARS", "TRACE",  "TRACES"});
}

// A fixture over an opening position: an empty board, a pool the opponent's
// rack was drawn from, and the synthetic leave table the equity model reads.
class RackInferenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tmp_ = std::filesystem::temp_directory_path() / "scribblez_test_rack_inference_XXXXXX";
    std::filesystem::create_directories(tmp_);
    scribblez::testing::install_synthetic_hasty_equity(tmp_);
  }
  void TearDown() override { std::filesystem::remove_all(tmp_); }

  // Roomy enough that exchanges are legal and several leaves are plausible,
  // small enough that a one- or two-tile leave space enumerates.
  TileCounts pool() const { return pool_from("AAAABBCCEEEERRSSTTT?"); }

  int bag_size() const { return pool().size() - RACK_SIZE; }

  // The move HastyBot would make from `rack` here -- the equity argmax, which
  // is exactly the process the likelihood model assumes.
  Move best_move(const Rack& rack) const {
    const Rack no_opp;
    const MoveRequest req{board_, dict_, rack, no_opp, 0, 0, bag_size()};
    std::vector<Move> moves = generate_legal_plays(req);
    const std::vector<Move> exchanges = generate_legal_exchanges(req);
    moves.insert(moves.end(), exchanges.begin(), exchanges.end());
    EXPECT_FALSE(moves.empty());
    const std::vector<double> eq =
      HastyEquity::instance().equities(moves, board_, bag_size(), no_opp, rack);
    return moves[size_t(std::max_element(eq.begin(), eq.end()) - eq.begin())];
  }

  OppMoveObservation observation_of(const Move& move) const {
    return OppMoveObservation{board_, move, pool()};
  }

  // A rack whose best play here is CRATE, keeping BB -- a two-tile leave
  // space, small enough to enumerate and awkward enough that a different
  // leave would have meant a different play.
  static Rack partial_play_rack() { return rack_from("BBCARTE"); }

  std::filesystem::path tmp_;
  Board board_;
  Dictionary dict_ = bingo_capable_dict();
};

double weight_of(const RackPosterior& p, const Rack& leave) {
  for (int i = 0; i < p.size(); ++i)
    if (p.entry(i).leave == leave) return p.entry(i).weight;
  return 0.0;
}

}  // namespace

TEST(LeavePrior, EnumerationAgreesWithTheCount) {
  const TileCounts pool = pool_from("AABBCDE");
  for (int k = 1; k <= 4; ++k)
    EXPECT_EQ(int64_t(enumerate_leaves(pool, k).size()), count_multisets(pool, k, 1'000'000))
      << "k=" << k;
}

TEST(LeavePrior, PriorsNormalize) {
  const TileCounts pool = pool_from("AABBCDE");
  for (int k = 1; k <= 4; ++k) {
    double total = 0.0;
    for (const ScoredLeave& h : enumerate_leaves(pool, k)) total += std::exp(h.log_weight);
    EXPECT_NEAR(total, 1.0, 1e-9) << "k=" << k;
  }
}

TEST(LeavePrior, SingleTilePriorIsTheTileFraction) {
  const TileCounts pool = pool_from("AABBCDE");
  const std::vector<ScoredLeave> hyps = enumerate_leaves(pool, 1);
  ASSERT_EQ(hyps.size(), 5u);  // A, B, C, D, E
  for (const ScoredLeave& h : hyps) {
    const Tile t = h.leave.tiles()[0];
    EXPECT_NEAR(std::exp(h.log_weight), double(pool.count(t)) / pool.size(), 1e-12);
  }
}

TEST(LeavePrior, UndrawableLeaveHasZeroProbability) {
  const TileCounts pool = pool_from("AABBCDE");
  EXPECT_FALSE(std::isfinite(log_hypergeometric_prior(rack_from("AAA"), pool)));
  EXPECT_FALSE(std::isfinite(log_hypergeometric_prior(rack_from("Z"), pool)));
}

TEST(LeavePrior, CountingStopsOnceTheCapIsExceeded) {
  const TileCounts pool = pool_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
  EXPECT_EQ(count_multisets(pool, 5, 10), 11);  // cap + 1 means "more than cap"
  EXPECT_EQ(count_multisets(pool, 1, 100), 26);
}

TEST(LeavePrior, DrawingReproducesThePrior) {
  const TileCounts pool = pool_from("AABBC");
  const std::vector<ScoredLeave> hyps = enumerate_leaves(pool, 2);
  std::mt19937_64 rng(7);
  std::vector<int> hits(hyps.size(), 0);
  constexpr int kDraws = 200000;
  for (int i = 0; i < kDraws; ++i) {
    const Rack drawn = draw_leave(pool, 2, rng);
    for (size_t j = 0; j < hyps.size(); ++j)
      if (hyps[j].leave == drawn) ++hits[j];
  }
  for (size_t j = 0; j < hyps.size(); ++j)
    EXPECT_NEAR(double(hits[j]) / kDraws, std::exp(hyps[j].log_weight), 0.01)
      << "leave=" << hyps[j].leave.to_string();
}

TEST_F(RackInferenceTest, ABingoLeavesNothingToInfer) {
  const RackInferrer inferrer(dict_, {});
  // CASTERS uses the whole rack, so nothing was kept.
  const Move bingo = best_move(rack_from("CASTERS"));
  ASSERT_EQ(bingo.num_glyphs(), RACK_SIZE);
  EXPECT_TRUE(inferrer.infer(observation_of(bingo), 1).empty());
}

TEST_F(RackInferenceTest, APassRevealsNothing) {
  const RackInferrer inferrer(dict_, {});
  EXPECT_TRUE(inferrer.infer(observation_of(Move::pass()), 1).empty());
}

TEST_F(RackInferenceTest, AnEmptyBagSkipsInference) {
  const RackInferrer inferrer(dict_, {});
  OppMoveObservation obs = observation_of(best_move(partial_play_rack()));
  obs.pool = pool_from("ABCDEFG");  // exactly one rack: the bag is gone
  EXPECT_TRUE(inferrer.infer(obs, 1).empty());
}

TEST_F(RackInferenceTest, PosteriorIsNormalizedOrderedAndTheRightShape) {
  const RackInferrer inferrer(dict_, {});
  const Rack truth = partial_play_rack();
  const Move played = best_move(truth);
  const int kept = RACK_SIZE - played.num_glyphs();
  ASSERT_GT(kept, 0);

  const RackPosterior posterior = inferrer.infer(observation_of(played), 1);
  ASSERT_FALSE(posterior.empty());

  double total = 0.0;
  for (int i = 0; i < posterior.size(); ++i) {
    EXPECT_EQ(posterior.entry(i).leave.size(), kept);
    EXPECT_GT(posterior.entry(i).weight, 0.0);
    if (i > 0) {
      EXPECT_TRUE(posterior.entry(i - 1).leave < posterior.entry(i).leave);
    }
    total += posterior.entry(i).weight;
  }
  EXPECT_NEAR(total, 1.0, 1e-9);
}

TEST_F(RackInferenceTest, SmallLeaveSpacesAreEnumeratedExactly) {
  const Move played = best_move(partial_play_rack());
  ASSERT_LT(RACK_SIZE - played.num_glyphs(), 3);  // a small space, so exhaustive

  EXPECT_TRUE(RackInferrer(dict_, {}).infer(observation_of(played), 1).exhaustive());

  RackInferrer::Params sampled;
  sampled.max_enumerated = 1;  // force the space over the threshold
  sampled.samples = 200;
  EXPECT_FALSE(RackInferrer(dict_, sampled).infer(observation_of(played), 1).exhaustive());
}

TEST_F(RackInferenceTest, SamplingIsReproducibleFromItsSeed) {
  RackInferrer::Params params;
  params.max_enumerated = 1;
  params.samples = 300;
  const RackInferrer inferrer(dict_, params);
  const OppMoveObservation obs = observation_of(best_move(partial_play_rack()));

  const RackPosterior a = inferrer.infer(obs, 99);
  const RackPosterior b = inferrer.infer(obs, 99);
  ASSERT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); ++i) {
    EXPECT_TRUE(a.entry(i).leave == b.entry(i).leave);
    EXPECT_DOUBLE_EQ(a.entry(i).weight, b.entry(i).weight);
  }
}

TEST_F(RackInferenceTest, SamplingWeightsMatchEnumerationOnASmallSpace) {
  const OppMoveObservation obs = observation_of(best_move(partial_play_rack()));
  const RackPosterior exact = RackInferrer(dict_, {}).infer(obs, 1);

  RackInferrer::Params params;
  params.max_enumerated = 1;
  params.samples = 12000;
  const RackPosterior sampled = RackInferrer(dict_, params).infer(obs, 5);

  // Importance sampling is unbiased for the same posterior, so with enough
  // draws the two paths agree -- which is what says the sampling path really
  // does pick the prior up from its proposal.
  ASSERT_FALSE(exact.empty());
  for (int i = 0; i < exact.size(); ++i)
    EXPECT_NEAR(weight_of(sampled, exact.entry(i).leave), exact.entry(i).weight, 0.03)
      << "leave=" << exact.entry(i).leave.to_string();
}

TEST_F(RackInferenceTest, AnExchangeInfersWhatWasKept) {
  const RackInferrer inferrer(dict_, {});
  // A rack of awkward consonants: the equity argmax here is an exchange.
  const Rack truth = rack_from("BBCTTRC");
  const Move played = best_move(truth);
  ASSERT_EQ(played.type(), MoveType::EXCHANGE);

  const RackPosterior posterior = inferrer.infer(observation_of(played), 1);
  ASSERT_FALSE(posterior.empty());
  for (int i = 0; i < posterior.size(); ++i)
    EXPECT_EQ(posterior.entry(i).leave.size(), RACK_SIZE - played.num_glyphs());
}

TEST_F(RackInferenceTest, TheTrueLeaveGainsWeightOverItsPrior) {
  RackInferrer::Params params;
  params.temperature = 3.0;  // pinned, so retuning the default cannot move this bar
  const RackInferrer inferrer(dict_, params);
  const std::vector<std::string> racks = {"BBCARTE", "AACERST", "ACEERST", "ABCERST",
                                          "AAERSTT", "AAABEST", "RSTTEAB", "BBTTCAE"};

  int gained = 0;
  double total_log_ratio = 0.0;
  int scored = 0;
  for (const std::string& s : racks) {
    const Rack truth = rack_from(s);
    const Move played = best_move(truth);
    if (played.num_glyphs() == RACK_SIZE) continue;  // a bingo tells us nothing

    Rack kept = truth;
    for (int i = 0; i < played.num_glyphs(); ++i) kept.remove(played.glyph(i).rack_tile());

    TileCounts remaining = pool();
    for (int i = 0; i < played.num_glyphs(); ++i) remaining.remove(played.glyph(i).rack_tile());

    const double prior = std::exp(log_hypergeometric_prior(kept, remaining));
    const double posterior = weight_of(inferrer.infer(observation_of(played), 1), kept);
    ASSERT_GT(prior, 0.0) << s;
    ASSERT_GT(posterior, 0.0) << s << " eliminated the leave it actually held";

    if (posterior > prior) ++gained;
    total_log_ratio += std::log2(posterior / prior);
    ++scored;
  }

  ASSERT_GT(scored, 4);
  EXPECT_GT(gained * 2, scored) << "inference should raise the true leave more often than not";
  EXPECT_GT(total_log_ratio / scored, 0.4) << "mean information gain, in bits, over the prior";
}

TEST_F(RackInferenceTest, ALowTemperatureKeepsEveryHypothesis) {
  // A sharp temperature divides equity gaps into the hundreds, which is where
  // a likelihood held as a plain probability rounds to zero and takes its
  // hypothesis out of the posterior altogether. Support must not depend on the
  // temperature: a hypothesis the evidence merely disfavors should rank last,
  // not disappear.
  const OppMoveObservation obs = observation_of(best_move(partial_play_rack()));
  RackInferrer::Params warm;
  warm.temperature = 3.0;
  const int support = RackInferrer(dict_, warm).infer(obs, 1).size();
  ASSERT_GT(support, 1);

  for (double sharp : {0.5, 0.1, 0.01, 0.001}) {
    RackInferrer::Params params;
    params.temperature = sharp;
    EXPECT_EQ(RackInferrer(dict_, params).infer(obs, 1).size(), support) << "temperature=" << sharp;
  }
}
