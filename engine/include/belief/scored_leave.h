#pragma once

// The currency of rack inference (docs/roadmap.md, track B): a candidate leave
// and an unnormalized log weight.
//
// Every stage of the inference trades in this one type, because every stage is
// measuring the same leaves on the same scale -- a prior on its own, a
// likelihood on its own, or the product that forms the posterior numerator.
// Keeping the stages in log space is what makes them composable: combining two
// weights is an addition that cannot overflow, where multiplying probabilities
// underflows to zero and silently erases a hypothesis before anything has
// established how small "small" is for the position at hand.
//
// Weights are normalized only at the very end, when the whole set is in hand
// and can be shifted by its own maximum. That result is a RackPosterior
// (rack_inference.h), whose entries carry true probabilities rather than these
// weights -- the one place the distinction is worth two types.

#include "game/rack.h"

namespace scribblez::belief {

struct ScoredLeave {
  Rack leave;
  double log_weight;
};

}  // namespace scribblez::belief
