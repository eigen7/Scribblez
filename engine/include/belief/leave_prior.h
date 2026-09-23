#pragma once

// The prior half of Bayesian rack inference (docs/roadmap.md, track B): what
// the opponent's kept tiles look like before their move tells us anything.
// Whatever they kept is a draw from the pool of tiles unseen to us, so the
// prior over leaves is multivariate hypergeometric.

#include "belief/scored_leave.h"
#include "game/rack.h"
#include "game/tile_counts.h"

#include <cstdint>
#include <random>
#include <vector>

namespace scribblez::belief {

// The number of distinct size-k multisets drawable from `pool`, or cap + 1 if
// it exceeds `cap`. The count grows exponentially in k, and callers only ask
// which side of a threshold it falls on, so counting stops at the cap.
int64_t count_multisets(const TileCounts& pool, int k, int64_t cap);

// Every distinct size-k multiset drawable from `pool`, with its exact log prior.
std::vector<ScoredLeave> enumerate_leaves(const TileCounts& pool, int k);

// log P(a size-k draw from `pool` comes out exactly `leave`), and -infinity
// when `leave` cannot be drawn from `pool` at all.
double log_hypergeometric_prior(const Rack& leave, const TileCounts& pool);

// k tiles drawn from `pool` without replacement: a sample from the
// distribution log_hypergeometric_prior scores, so draws need no prior weight.
Rack draw_leave(const TileCounts& pool, int k, std::mt19937_64& rng);

}  // namespace scribblez::belief
