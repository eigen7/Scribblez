#pragma once

// A candidate leave and an unnormalized log weight: the one type every stage
// of rack inference (docs/roadmap.md, track B) trades in, whether the weight
// is a prior, a likelihood, or their product.
//
// Log space makes stages composable: combining weights is an addition that
// cannot overflow, whereas multiplying probabilities can underflow to zero and
// silently erase a hypothesis. Weights are normalized only at the end, into a
// RackPosterior (rack_inference.h), whose entries are true probabilities.

#include "game/rack.h"

namespace scribblez::belief {

struct ScoredLeave {
  Rack leave;
  double log_weight;
};

}  // namespace scribblez::belief
