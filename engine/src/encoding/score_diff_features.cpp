#include "encoding/score_diff_features.h"

#include <cmath>

namespace scribblez {
namespace {

// Bump centers are u = -1, ..., +1 in equal steps, so the spacing is also the
// Gaussian width: adjacent bumps cross near half height, giving a partition
// smooth enough to interpolate and peaked enough to localize.
inline constexpr float kBasisSpacing = 2.0f / float(kScoreDiffBasisFloats - 1);

}  // namespace

void encode_score_diff_features(int score_diff, float* out) {
  const float d = float(score_diff);
  const float u = d / (std::fabs(d) + kScoreDiffBasisSoftening);
  out[0] = d / kScoreDiffInputScale;
  for (int i = 0; i < kScoreDiffBasisFloats; ++i) {
    const float z = (u - (-1.0f + float(i) * kBasisSpacing)) / kBasisSpacing;
    out[1 + i] = std::exp(-0.5f * z * z);
  }
}

}  // namespace scribblez
