#include "training/evidence_trajectory_select.h"

#include "training/move_set_eval_candidates.h"

#include <algorithm>

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

  // Proposals draw from a temperature softmax over every unsimmed candidate
  // (deployment's full support). The softmax is permutation-invariant, so the
  // pool is assembled in `ranked`'s own (descending static-equity) order -- no
  // separate sort by win equity, which would change nothing but the wasted work.
  std::uniform_int_distribution<int> length(opt.on_policy_min, opt.on_policy_max);
  const int proposals = length(rng);
  std::vector<double> pool_scores;
  std::vector<size_t> pool_index;
  for (int p = 0; p < proposals; ++p) {
    pool_scores.clear();
    pool_index.clear();
    for (size_t i = 0; i < n; ++i) {
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

  // The off-policy floor: a uniform draw over what the anchor and on-policy
  // picks have not taken. Labels-only -- never in an evidence set.
  const std::vector<size_t> off_policy =
    move_set_eval::off_policy_draws(ranked, opt.off_policy_count, rng, &taken);
  for (size_t idx : off_policy) {
    chosen.push_back(idx);
    roles->push_back(SimObsRole::kOffPolicy);
  }
  return chosen;
}

}  // namespace scribblez::evidence
