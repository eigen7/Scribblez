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
  for (int i = lo; i < hi && i < static_cast<int>(ranked.size()); ++i) {
    if (std::find(out->begin(), out->end(), ranked[i]) == out->end()) pool.push_back(i);
  }
  std::shuffle(pool.begin(), pool.end(), rng);
  for (int j = 0; j < count && j < static_cast<int>(pool.size()); ++j) {
    out->push_back(ranked[static_cast<size_t>(pool[j])]);
  }
}

}  // namespace

Selection stratified_candidates(const std::vector<Move>& ranked, const Move& played,
                                const StratumQuotas& quotas, std::mt19937_64& rng) {
  std::vector<Move> out;
  out.reserve(static_cast<size_t>(1 + quotas.top + quotas.mid + quotas.tail + quotas.exchange));
  out.push_back(played);
  const int n = static_cast<int>(ranked.size());

  // Top stratum: the head of the ranking, dense.
  for (int i = 0; i < n && static_cast<int>(out.size()) < 1 + quotas.top; ++i) {
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

Selection full_sweep_candidates(const std::vector<Move>& ranked, const Move& played, int cap) {
  std::vector<Move> out;
  bool played_kept = false;
  for (int i = 0; i < static_cast<int>(ranked.size()); ++i) {
    const Move& m = ranked[static_cast<size_t>(i)];
    const bool keep = i < cap || m.type() != MoveType::PLAY || m == played;
    if (!keep) continue;
    played_kept = played_kept || m == played;
    out.push_back(m);
  }
  // A played move the generator never enumerates -- a PASS chosen while other
  // moves were legal -- has no equity rank, so it can only go last; every other
  // candidate keeps its rank order. It counts toward the legal total all the
  // same: it was legal, and the sweep did reach it.
  uint32_t legal = static_cast<uint32_t>(ranked.size());
  if (!played_kept) {
    out.push_back(played);
    ++legal;
  }
  return {std::move(out), legal};
}

}  // namespace move_set_eval
}  // namespace scribblez
