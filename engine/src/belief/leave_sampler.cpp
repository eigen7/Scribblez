#include "belief/leave_sampler.h"

#include "util/exception.h"
#include "util/math.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scribblez::belief {

namespace {

constexpr double kInfeasible = -std::numeric_limits<double>::infinity();
constexpr uint64_t kLeaveQuantileSalt = 0x1EAFE5EEDull;

// The per-code table a row sums over its codes: copy j of a tile kind the
// pool holds n of adds log((n - j) / (j + 1)), so a leave's sum is its log
// prior Σ_t log C(n_t, k_t) up to the pool's normalizer, which cancels in
// log Z. The kind's first copy also adds its theta.
std::array<double, LeaveValueArrays::kCodes> code_scores(const LeaveParams& params,
                                                         const TileCounts& pool) {
  std::array<double, LeaveValueArrays::kCodes> scores;
  for (int t = 0; t < TILE_KINDS; ++t) {
    const Tile tile = Tile::of(t);
    const int n = pool.count(tile);
    for (int j = 0; j < LeaveValueArrays::kCopiesPerTile; ++j) {
      const double step = j < n ? std::log(double(n - j)) - std::log(double(j + 1)) : kInfeasible;
      scores[LeaveValueArrays::code(tile, j)] = j == 0 ? step + params.theta[t] : step;
    }
  }
  return scores;
}

}  // namespace

LeaveSampler::LeaveSampler(const LeaveParams& params, int length, const TileCounts& leave_pool,
                           const LeaveValueArrays& table)
    : length_(length),
      values_(table.values(length)),
      a_(params.a),
      b_(params.b),
      scores_(code_scores(params, leave_pool)),
      cdf_(values_.size()) {
  if (leave_pool.size() < length) {
    throw util::Exception("a pool of {} tiles holds no leave of length {}", leave_pool.size(),
                          length);
  }
  for (int slot = 0; slot < length; ++slot) codes_[slot] = table.codes(length, slot);
  accumulate();
}

double LeaveSampler::log_weight(size_t row) const {
  double w = 0;
  for (int slot = 0; slot < length_; ++slot) w += scores_[codes_[slot][row]];
  const double v = values_[row];
  return w + a_ * v - b_ * v * v;
}

// Two passes: the log weights and their max, then the running sum of the
// weights shifted by the max, so no exp() overflows.
void LeaveSampler::accumulate() {
  max_log_weight_ = kInfeasible;
  for (size_t r = 0; r < cdf_.size(); ++r) {
    cdf_[r] = log_weight(r);
    max_log_weight_ = std::max(max_log_weight_, cdf_[r]);
  }
  double total = 0;
  for (double& c : cdf_) {
    total += std::exp(c - max_log_weight_);
    c = total;
  }
  log_z_ = max_log_weight_ + std::log(total);
}

size_t LeaveSampler::sample(double u) const {
  // The first row whose running sum passes u's share of the total. A row the
  // pool cannot supply adds nothing to the sum, so it is never the first.
  const auto it = std::upper_bound(cdf_.begin(), cdf_.end(), u * cdf_.back());
  return std::min(size_t(it - cdf_.begin()), cdf_.size() - 1);
}

double LeaveSampler::log_prob(size_t row) const { return log_weight(row) - log_z_; }

Rack LeaveSampler::leave(size_t row) const {
  Rack rack;
  for (int slot = 0; slot < length_; ++slot) {
    rack.add(Tile::of(codes_[slot][row] / LeaveValueArrays::kCopiesPerTile));
  }
  return rack;
}

LeaveMoments LeaveSampler::moments() const {
  LeaveMoments m{};
  double value_sq = 0;
  const double total = cdf_.back();
  for (size_t r = 0; r < cdf_.size(); ++r) {
    const double p = (cdf_[r] - (r > 0 ? cdf_[r - 1] : 0.0)) / total;
    if (p == 0) continue;
    // A leave holds a kind iff it holds that kind's first copy.
    for (int slot = 0; slot < length_; ++slot) {
      const uint8_t code = codes_[slot][r];
      if (code % LeaveValueArrays::kCopiesPerTile == 0)
        m.presence[code / LeaveValueArrays::kCopiesPerTile] += p;
    }
    m.value_mean += p * values_[r];
    value_sq += p * values_[r] * values_[r];
  }
  m.value_sd = std::sqrt(std::max(0.0, value_sq - m.value_mean * m.value_mean));
  return m;
}

double rollout_leave_quantile(uint64_t seed) {
  return double(util::splitmix64(seed ^ kLeaveQuantileSalt) >> 11) * 0x1.0p-53;
}

}  // namespace scribblez::belief
