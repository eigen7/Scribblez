#include "util/math.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace scribblez::util {

void softmax(const float* logits, int n, float* out) {
  const float m = *std::max_element(logits, logits + n);
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    out[i] = std::exp(logits[i] - m);
    sum += out[i];
  }
  for (int i = 0; i < n; ++i) out[i] = float(out[i] / sum);
}

int SoftmaxSampler::sample(const std::vector<double>& values, int k, double temperature,
                           std::mt19937_64& rng) {
  if (int(weights_.size()) < k) weights_.resize(size_t(k));

  // Subtract the max before exponentiating so the largest weight is 1.
  Eigen::Map<const Eigen::ArrayXd> v(values.data(), k);
  Eigen::Map<Eigen::ArrayXd> w(weights_.data(), k);
  w = ((v - v.maxCoeff()) / temperature).exp();

  // Inverse-CDF draw. The trailing return covers rounding in the running sum.
  double r = std::uniform_real_distribution<double>(0.0, w.sum())(rng);
  double acc = 0.0;
  for (int j = 0; j < k; ++j) {
    acc += weights_[j];
    if (r <= acc) return j;
  }
  return k - 1;
}

}  // namespace scribblez::util
