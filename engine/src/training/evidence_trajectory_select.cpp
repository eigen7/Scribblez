#include "training/evidence_trajectory_select.h"

#include <algorithm>
#include <numeric>

namespace scribblez::evidence {

size_t anchor_index(const std::vector<Move>& ranked) {
  // max_element keeps the first maximum: the equity-preferred instance.
  return std::ranges::max_element(ranked, {}, &Move::score) - ranked.begin();
}

std::vector<size_t> select_trajectory(const std::vector<Move>& ranked,
                                      const std::vector<float>& win_equity,
                                      const TrajectoryOptions& opt, std::mt19937_64& rng,
                                      util::SoftmaxSampler& sampler,
                                      std::vector<SimObsRole>* roles) {
  const size_t n = ranked.size();
  std::vector<size_t> chosen{anchor_index(ranked)};
  std::vector<char> taken(n, 0);
  taken[chosen[0]] = 1;
  roles->assign(1, SimObsRole::kAnchor);

  // The student's ranking, best first: proposals draw from a temperature
  // softmax over every unsimmed candidate (deployment's full support), so the
  // order is just the canonical iteration order the sampler maps its draw onto.
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return win_equity[a] > win_equity[b]; });

  std::uniform_int_distribution<int> length(opt.on_policy_min, opt.on_policy_max);
  const int proposals = length(rng);
  std::vector<double> pool_scores;
  std::vector<size_t> pool_index;
  for (int p = 0; p < proposals; ++p) {
    pool_scores.clear();
    pool_index.clear();
    for (size_t i : order) {
      if (taken[i]) continue;
      pool_scores.push_back(win_equity[i]);
      pool_index.push_back(i);
    }
    if (pool_index.empty()) break;
    const int j = sampler.sample(pool_scores, int(pool_scores.size()), opt.temperature, rng);
    chosen.push_back(pool_index[size_t(j)]);
    taken[pool_index[size_t(j)]] = 1;
    roles->push_back(SimObsRole::kOnPolicy);
  }

  // The off-policy floor: stratified + uniform draws over what the anchor and
  // on-policy picks have not taken. Labels-only -- never in an evidence set.
  const std::vector<size_t> off_policy = move_set_eval::off_policy_draws(
    ranked, opt.off_policy_quotas, opt.off_policy_uniform, &taken, rng);
  for (size_t idx : off_policy) {
    chosen.push_back(idx);
    roles->push_back(SimObsRole::kOffPolicy);
  }
  return chosen;
}

}  // namespace scribblez::evidence
