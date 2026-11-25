#include "training/move_set_eval_candidates.h"

#include <algorithm>

namespace scribblez {
namespace move_set_eval {

namespace {

// Append `count` distinct picks from ranked[lo, hi) (uniform, without
// replacement) that are not already in *out.
void sample_range(const std::vector<Move>& ranked, int lo, int hi, int count, std::mt19937_64& rng,
                  std::vector<Move>* out) {
  std::vector<int> pool;
  for (int i = lo; i < hi && i < int(ranked.size()); ++i) {
    if (std::find(out->begin(), out->end(), ranked[i]) == out->end()) pool.push_back(i);
  }
  std::shuffle(pool.begin(), pool.end(), rng);
  for (int j = 0; j < count && j < int(pool.size()); ++j) {
    out->push_back(ranked[size_t(pool[j])]);
  }
}

}  // namespace

Selection stratified_candidates(const std::vector<Move>& ranked, const Move& played,
                                const StratumQuotas& quotas, std::mt19937_64& rng,
                                std::span<const Move> forced) {
  std::vector<Move> out;
  out.reserve(size_t(1 + forced.size() + quotas.top + quotas.mid + quotas.tail + quotas.exchange));
  out.push_back(played);
  for (const Move& m : forced) {
    if (std::find(out.begin(), out.end(), m) == out.end()) out.push_back(m);
  }
  const int n = ranked.size();

  // Top stratum: the head of the ranking, dense.
  const int head_target = int(out.size()) + quotas.top;
  for (int i = 0; i < n && int(out.size()) < head_target; ++i) {
    if (std::find(out.begin(), out.end(), ranked[i]) == out.end()) out.push_back(ranked[i]);
  }
  // Contention zone, then the tail, uniform within each.
  sample_range(ranked, quotas.top, quotas.mid_rank_limit, quotas.mid, rng, &out);
  sample_range(ranked, quotas.mid_rank_limit, n, quotas.tail, rng, &out);

  // Exchange stratum: uniform among the non-PLAY candidates.
  std::vector<Move> non_plays;
  for (const Move& m : ranked) {
    if (m.type() != MoveType::PLAY) non_plays.push_back(m);
  }
  std::shuffle(non_plays.begin(), non_plays.end(), rng);
  int taken = 0;
  for (const Move& m : non_plays) {
    if (taken >= quotas.exchange) break;
    if (std::find(out.begin(), out.end(), m) == out.end()) {
      out.push_back(m);
      ++taken;
    }
  }
  return {std::move(out), 0u};
}

}  // namespace move_set_eval
}  // namespace scribblez
