#include "util/math.h"

#include <algorithm>
#include <cmath>

namespace util {

int SoftmaxSampler::sample(const std::vector<double>& values, int k, double temperature,
                           std::mt19937_64& rng) {
  if (static_cast<int>(weights_.size()) < k) weights_.resize(static_cast<size_t>(k));

  double max_v = values[0];
  for (int j = 1; j < k; ++j) max_v = std::max(max_v, values[j]);

  double sum = 0.0;
  for (int j = 0; j < k; ++j) {
    weights_[j] = std::exp((values[j] - max_v) / temperature);
    sum += weights_[j];
  }

  double r = std::uniform_real_distribution<double>(0.0, sum)(rng);
  double acc = 0.0;
  for (int j = 0; j < k; ++j) {
    acc += weights_[j];
    if (r <= acc) return j;
  }
  return k - 1;
}

}  // namespace util
