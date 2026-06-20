#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace scribblez {

// Numerically-stable softmax sampler over the first `k` entries of `values`.
// Returns an index in [0, k) drawn with probability proportional to
// exp((values[i] - max) / temperature), shifting by the max for stability.
// `scratch` is a caller-owned weight buffer reused across calls to avoid
// per-call allocation; it is grown to at least k entries as needed.
//
// Preconditions: k >= 1 and temperature > 0. Shared by every agent that turns a
// vector of candidate scores (static equity, or a value head's output) into an
// exploratory move choice.
inline int softmax_sample(const std::vector<double>& values, int k, double temperature,
                          std::mt19937_64& rng, std::vector<double>& scratch) {
  if (static_cast<int>(scratch.size()) < k) scratch.resize(static_cast<size_t>(k));

  double max_v = values[0];
  for (int j = 1; j < k; ++j) max_v = std::max(max_v, values[j]);

  double sum = 0.0;
  for (int j = 0; j < k; ++j) {
    scratch[j] = std::exp((values[j] - max_v) / temperature);
    sum += scratch[j];
  }

  double r = std::uniform_real_distribution<double>(0.0, sum)(rng);
  double acc = 0.0;
  for (int j = 0; j < k; ++j) {
    acc += scratch[j];
    if (r <= acc) return j;
  }
  return k - 1;
}

}  // namespace scribblez
