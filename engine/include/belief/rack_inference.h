#pragma once

// Bayesian rack inference (docs/roadmap.md, track B): what the opponent's last
// move says about the tiles they kept.
//
// posterior(leave) is proportional to prior(leave) * P(their move | leave): a
// hypergeometric prior over draws from the pool unseen to us (leave_prior.h)
// times a model of how they choose moves (move_likelihood.h). A simulation
// samples opponent racks from this posterior instead of drawing them
// uniformly, which is only right when the opponent's rack is entirely fresh.
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

// The opponent's last action and the state it was taken from, as we saw it.
// `pool` is what was unseen to us at that moment: their whole rack plus the
// bag.
struct OppMoveObservation {
  Board board_before;
  Move move;
  TileCounts pool;
};

// A distribution over the tiles the opponent kept. Empty means there was
// nothing to condition on (a bingo keeps nothing, a pass reveals nothing, and
// with an empty bag nothing is hidden), so a consumer should draw uniformly
// from the unseen pool.
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

  // The leave that `u`, a uniform variate in [0, 1), selects. Taking a variate
  // rather than an RNG lets a simulation hand every candidate the same
  // opponent rack on a given rollout (common random numbers, sim_runner.h).
  const Rack& sample(double u) const;

  // Whether every possible leave was enumerated rather than sampled. A sampled
  // posterior may lack support on the true leave; an exhaustive one cannot.
  bool exhaustive() const { return exhaustive_; }

 private:
  std::vector<Entry> entries_;  // ordered by leave; weights sum to 1
  bool exhaustive_ = false;
};

class RackInferrer {
 public:
  struct Params {
    // The likelihood's softmax temperature, in equity points. Set by the
    // offline ground-truth sweep (docs/roadmap.md, B2).
    double temperature = 3.0;

    // The leave-space size at or below which every leave is enumerated and
    // scored exactly; above it the space is sampled. Scoring a hypothesis costs
    // one move generation (~0.1 ms), far cheaper than Macondo's ~100 ms
    // mini-sim, which is why this sits well above Macondo's threshold of 750.
    int64_t max_enumerated = 4000;

    // Hypotheses drawn when the space is too large to enumerate. With
    // max_enumerated, this bounds one inference at a few hundred milliseconds.
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
