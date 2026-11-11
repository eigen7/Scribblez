#include "sim/map_residuals.h"

#include <cmath>

namespace scribblez {

namespace {

using rollout_abi::kLeafPointsScale;

// The sims' mean and unbiased per-rollout variance of a quantity with sum
// `s` and sum of squares `s2` over `n` rollouts (0 for one rollout).
struct Moments {
  double mean, var;
};

Moments moments(double s, double s2, double n) {
  const double mean = s / n;
  if (n < 2) return {mean, 0.0};
  return {mean, std::max(0.0, (s2 - n * mean * mean) / (n - 1))};
}

// Adds a mean-and-variance residual: the sims' moments and the leaf's
// summed variance `leaf_var` against the prior's (`mean`, `var`), scaled
// from the heads' units to points.
void add_moment_residual(const Moments& sims, float leaf_var, float mean, float var, double n,
                         double& r2, double& noise, double& sims_var, double& sims_leaf_var,
                         double& prior_var) {
  const double d = sims.mean - double(mean) * kLeafPointsScale;
  r2 += d * d;
  noise += sims.var / n;
  sims_var += sims.var;
  sims_leaf_var += double(leaf_var) / n;
  prior_var += double(var) * kLeafPointsScale * kLeafPointsScale;
}

// Adds a class residual: the sims' shares `sum[c * stride] / n` against the
// softmax of `logits[c * stride]`, their spread from `sq_sum`.
void add_class_residual(const float* sum, const float* sq_sum, const float* logits, int classes,
                        int stride, double n, double& r2, double& noise, double* shift) {
  float prior[rollout_abi::kFateClasses] = {};
  class_probabilities(logits, classes, stride, prior);
  for (int c = 0; c < classes; ++c) {
    const Moments share = moments(sum[c * stride], sq_sum[c * stride], n);
    const double d = share.mean - double(prior[c]);
    r2 += d * d;
    noise += share.var / n;
    shift[c] += d;
  }
}

}  // namespace

void class_probabilities(const float* logits, int classes, int stride, float* probs) {
  float top = logits[0];
  for (int c = 1; c < classes; ++c) top = std::max(top, logits[c * stride]);
  float total = 0;
  for (int c = 0; c < classes; ++c) {
    probs[c] = std::exp(logits[c * stride] - top);
    total += probs[c];
  }
  for (int c = 0; c < classes; ++c) probs[c] /= total;
}

MapResidual map_residual(const rollout_abi::SimMaps& maps, const LeafPrediction& prior,
                         const Board& start, double sim_value) {
  MapResidual r;
  r.sims = int(std::lround(maps.sims));
  const double n = maps.sims;
  for (int sq = 0; sq < kBoardCells; ++sq) {
    if (!start.at(sq / BOARD_SIZE, sq % BOARD_SIZE).is_empty()) continue;
    ++r.squares;
    add_moment_residual(moments(maps.points_sum[sq], maps.points_sq_sum[sq], n),
                        maps.points_leaf_var[sq], prior.square_points[size_t(sq)],
                        prior.square_points[size_t(kBoardCells + sq)], n, r.points_r2,
                        r.points_noise, r.points_var, r.points_leaf_var, r.points_prior_var);
    add_class_residual(&maps.class_sum[0][sq], &maps.class_sq_sum[0][sq],
                       prior.ownership.data() + sq, rollout_abi::kFateClasses, kBoardCells, n,
                       r.ownership_r2, r.ownership_noise, r.ownership_shift);
  }
  add_moment_residual(moments(maps.end_points_sum, maps.end_points_sq_sum, n),
                      maps.end_points_leaf_var, prior.end_points[0], prior.end_points[1], n,
                      r.end_r2, r.end_noise, r.end_var, r.end_leaf_var, r.end_prior_var);
  add_class_residual(maps.went_out_sum, maps.went_out_sq_sum, prior.going_out.data(),
                     rollout_abi::kWentOutClasses, 1, n, r.going_out_r2, r.going_out_noise,
                     r.going_out_shift);
  r.value_r = sim_value - win_value(prior.wld);
  return r;
}

}  // namespace scribblez
