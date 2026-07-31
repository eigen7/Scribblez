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

// The number of distinct size-k multisets drawable from `pool`, counted
// without materializing them and abandoned as soon as the count exceeds `cap`
// (which then returns cap + 1). The count grows exponentially in k, and the
// only question a caller asks of it is which side of a threshold it falls on.
int64_t count_multisets(const TileCounts& pool, int k, int64_t cap);

// Every distinct size-k multiset drawable from `pool`, weighted by its exact
// prior: exponentiated, the weights sum to 1.
std::vector<ScoredLeave> enumerate_leaves(const TileCounts& pool, int k);

// log P(a size-k draw from `pool` comes out exactly `leave`), and -infinity
// when `leave` cannot be drawn from `pool` at all.
double log_hypergeometric_prior(const Rack& leave, const TileCounts& pool);

// k tiles drawn from `pool` without replacement -- a sample from the very
// distribution log_hypergeometric_prior scores, which is what lets the sampling
// path treat the prior as already applied.
Rack draw_leave(const TileCounts& pool, int k, std::mt19937_64& rng);

}  // namespace scribblez::belief
