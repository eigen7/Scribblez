#include "training/evidence_trajectory_select.h"

#include <algorithm>

namespace scribblez::evidence {

size_t anchor_index(const std::vector<Move>& ranked) {
  // max_element keeps the first maximum: the equity-preferred instance.
  return std::ranges::max_element(ranked, {}, &Move::score) - ranked.begin();
}

std::vector<size_t> select_trajectory(const std::vector<Move>& ranked,
                                      const std::vector<float>& win_equity,
                                      const TrajectoryOptions& opt, std::mt19937_64& rng,
                                      util::SoftmaxSampler& sampler, bool* uniform_tail) {
  const size_t n = ranked.size();
  std::vector<size_t> chosen{anchor_index(ranked)};
  std::vector<char> taken(n, 0);
  taken[chosen[0]] = 1;

  // Proposals draw from a temperature softmax over every unsimmed candidate
  // (deployment's full support). The softmax is permutation-invariant, so the
  // pool is built in the candidates' natural index order -- no ranking needed.
  std::uniform_int_distribution<int> length(opt.proposals_min, opt.proposals_max);
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
  }

  std::vector<size_t> unsimmed;
  for (size_t i = 0; i < n; ++i) {
    if (!taken[i]) unsimmed.push_back(i);
  }
  *uniform_tail = !unsimmed.empty();
  if (*uniform_tail) {
    std::uniform_int_distribution<size_t> pick(0, unsimmed.size() - 1);
    chosen.push_back(unsimmed[pick(rng)]);
  }
  return chosen;
}

}  // namespace scribblez::evidence
