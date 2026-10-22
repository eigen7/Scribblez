#include "sim/pool_labels.h"

#include "sim/rollout_report.h"
#include "util/exception.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace scribblez {

namespace {

using CandidateRollouts = std::vector<std::vector<Rollout>>;

// Wins over the first n rollouts, a draw counting half.
double prefix_wins(const std::vector<Rollout>& rs, size_t n) {
  double sum = 0;
  for (size_t i = 0; i < n; ++i) sum += rs[i].win_value();
  return sum;
}

// The leader once every surviving candidate had run n rollouts, as racing
// picked it: among the candidates that reached n, the most wins over those n,
// the first on ties.
size_t leader_at(const CandidateRollouts& rollouts, size_t n) {
  size_t leader = 0;
  double best = -1;
  for (size_t c = 0; c < rollouts.size(); ++c) {
    if (rollouts[c].size() < n) continue;
    const double w = prefix_wins(rollouts[c], n);
    if (w > best) {
      best = w;
      leader = c;
    }
  }
  return leader;
}

// Each candidate's comparator: the final leader for a candidate that ran
// every rollout, else the leader at the checkpoint that stopped it.
std::vector<size_t> comparators(const CandidateRollouts& rollouts) {
  std::map<size_t, size_t> leader_by_length;
  std::vector<size_t> out;
  for (const auto& rs : rollouts) {
    const auto [it, added] = leader_by_length.try_emplace(rs.size(), 0);
    if (added) it->second = leader_at(rollouts, rs.size());
    out.push_back(it->second);
  }
  return out;
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
  for (size_t c = 0; c < rollouts.size(); ++c) {
    if (rollouts[c].empty()) throw util::Exception("pool_labels: candidate {} has no rollouts", c);
  }
  const std::vector<size_t> against = comparators(rollouts);
  size_t most = 0;
  for (const auto& rs : rollouts) most = std::max(most, rs.size());
  const size_t leader = leader_at(rollouts, most);
  std::vector<double> odds(rollouts.size(), 0.0);
  double total = 1;  // the leader's own odds
  for (size_t c = 0; c < rollouts.size(); ++c) {
    const auto& rs = rollouts[c];
    const size_t n = rs.size();
    out[c].win_prob = prefix_wins(rs, n) / double(n);
    out[c].rollouts = int(n);
    if (c == leader) continue;
    const PairedWinDiff d = paired_win_diff(rs, rollouts[against[c]]);
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
