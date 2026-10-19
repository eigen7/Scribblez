#include "sim/pool_labels.h"

#include "sim/rollout_report.h"
#include "util/exception.h"

#include <algorithm>
#include <cmath>

namespace scribblez {

namespace {

double mean_win(const std::vector<Rollout>& rs) {
  double sum = 0;
  for (const Rollout& r : rs) sum += r.p_win + 0.5 * r.p_draw;
  return rs.empty() ? 0 : sum / double(rs.size());
}

size_t leader_of(const std::vector<std::vector<Rollout>>& rollouts) {
  size_t most = 0;
  for (const auto& rs : rollouts) most = std::max(most, rs.size());
  size_t leader = 0;
  double best = -1;
  for (size_t c = 0; c < rollouts.size(); ++c) {
    if (rollouts[c].size() != most) continue;
    const double m = mean_win(rollouts[c]);
    if (m > best) {
      best = m;
      leader = c;
    }
  }
  return leader;
}

// P(a beats the leader), from a's mean paired difference and its standard
// error, kept off 1 so the odds stay finite. A zero error (identical outcomes
// on every paired rollout) decides by the sign alone.
double beats_leader(double diff, double se) {
  constexpr double kMax = 1 - 1e-9;
  if (se <= 0) return diff > 0 ? kMax : diff == 0 ? 0.5 : 0.0;
  return std::min(0.5 * std::erfc(-diff / (se * std::sqrt(2.0))), kMax);
}

}  // namespace

std::vector<PoolLabel> pool_labels(const std::vector<std::vector<Rollout>>& rollouts) {
  std::vector<PoolLabel> out(rollouts.size());
  if (rollouts.empty()) return out;
  const size_t leader = leader_of(rollouts);
  std::vector<double> odds(rollouts.size(), 0.0);
  double total = 1;  // the leader's own odds
  for (size_t c = 0; c < rollouts.size(); ++c) {
    const auto& rs = rollouts[c];
    if (rs.empty()) throw util::Exception("pool_labels: candidate {} has no rollouts", c);
    out[c].win_prob = mean_win(rs);
    out[c].rollouts = int(rs.size());
    if (c == leader) continue;
    const size_t n = rs.size();
    const PairedWinDiff d = paired_win_diff(rs, rollouts[leader]);
    const double mean = d.sum / double(n);
    out[c].paired_se = std::sqrt(std::max(d.sq_sum / double(n) - mean * mean, 0.0) / double(n));
    const double p = beats_leader(mean, out[c].paired_se);
    odds[c] = p / (1 - p);
    total += odds[c];
  }
  odds[leader] = 1;
  for (size_t c = 0; c < rollouts.size(); ++c) out[c].q = odds[c] / total;
  return out;
}

}  // namespace scribblez
