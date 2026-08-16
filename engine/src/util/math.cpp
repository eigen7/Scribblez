#include "util/math.h"

#include <Eigen/Core>

namespace scribblez::util {

int SoftmaxSampler::sample(const std::vector<double>& values, int k, double temperature,
                           std::mt19937_64& rng) {
  if (int(weights_.size()) < k) weights_.resize(size_t(k));

  // Numerically stable softmax weights: subtract the max before exponentiating.
  // Both arrays are mapped zero-copy; `weights_` holds the result for the
  // cumulative-sum draw below.
  Eigen::Map<const Eigen::ArrayXd> v(values.data(), k);
  Eigen::Map<Eigen::ArrayXd> w(weights_.data(), k);
  w = ((v - v.maxCoeff()) / temperature).exp();

  // Inverse-CDF draw: walk the running weight sum and return the first index
  // whose cumulative weight reaches the sampled threshold.
  double r = std::uniform_real_distribution<double>(0.0, w.sum())(rng);
  double acc = 0.0;
  for (int j = 0; j < k; ++j) {
    acc += weights_[j];
    if (r <= acc) return j;
  }
  return k - 1;
}

}  // namespace scribblez::util
