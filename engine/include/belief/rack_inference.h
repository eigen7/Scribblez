#pragma once

// Bayesian rack inference (docs/roadmap.md, track B): what the opponent's last
// move says about the tiles they kept.
//
// posterior(leave) is proportional to prior(leave) * P(their move | leave) --
// a multivariate hypergeometric prior over draws from the pool unseen to us
// ([leave_prior.h](leave_prior.h)) times a model of how they choose moves
// ([move_likelihood.h](move_likelihood.h)). The result is the distribution a
// simulation samples opponent racks from instead of drawing them uniformly,
// which is only right when the opponent's rack is genuinely fresh.
//
// Ported from Macondo's rangefinder package, the machinery behind its
// SIMMING_INFER_BOT.

#include "belief/move_likelihood.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile_counts.h"

#include <cstdint>
#include <vector>

namespace scribblez {
class Dictionary;
}

namespace scribblez::belief {

// The opponent's last action and the state it was taken from, all as we saw
// it. `pool` is what was unseen to us at that moment -- their whole rack plus
// the bag -- which is the population their kept tiles were drawn from.
struct OppMoveObservation {
  Board board_before;
  Move move;
  TileCounts pool;
};

// A distribution over the tiles the opponent kept. An empty posterior means
// the observation carried no information to condition on, so a consumer should
// fall back to drawing uniformly from the unseen pool: a bingo keeps nothing,
// a pass reveals nothing, and once the bag is empty there is nothing hidden
// that this could infer.
class RackPosterior {
 public:
  struct Entry {
    Rack leave;
    double weight;
  };

  RackPosterior() = default;
  RackPosterior(std::vector<Entry> entries, bool exhaustive);

  bool empty() const { return entries_.empty(); }
  int size() const { return entries_.size(); }
  const Entry& entry(int i) const { return entries_[i]; }

  // The leave that `u`, a uniform variate in [0, 1), selects. Taking the
  // variate rather than an RNG keeps sampling a pure function of the caller's
  // seed, which is what lets a simulation hand every candidate the same
  // opponent rack on a given rollout (the common-random-numbers scheme in
  // sim_runner.h).
  const Rack& sample(double u) const;

  // True when every possible leave was enumerated and scored rather than the
  // space being sampled. An exhaustive posterior has full support, so a
  // consumer has no coverage gap to hedge against; a sampled one may simply
  // have missed the truth.
  bool exhaustive() const { return exhaustive_; }

 private:
  std::vector<Entry> entries_;  // ordered by leave; weights sum to 1
  bool exhaustive_ = false;
};

class RackInferrer {
 public:
  struct Params {
    // Equity points. As it approaches zero the likelihood hardens into "the
    // move they played was the best one for this rack". Set by the offline
    // ground-truth sweep (docs/roadmap.md, B2).
    double temperature = 3.0;

    // The leave-space size at or below which every leave is enumerated and
    // scored exactly; above it the space is sampled instead. A static-equity
    // hypothesis costs one move generation -- around 0.1 ms, against the ~100
    // ms Macondo pays to mini-simulate one -- which is what puts this some
    // three orders of magnitude above its threshold of 750.
    int64_t max_enumerated = 4000;

    // Hypotheses drawn when the space is too large to enumerate. Together
    // with max_enumerated this bounds one inference at a few hundred
    // milliseconds whichever path it takes.
    int samples = 2000;
  };

  RackInferrer(const Dictionary& dict, const Params& params);

  // `seed` drives the sampling path only; an enumerated posterior is exact and
  // ignores it.
  RackPosterior infer(const OppMoveObservation& obs, uint64_t seed) const;

 private:
  const Dictionary& dict_;
  Params params_;
};

}  // namespace scribblez::belief
