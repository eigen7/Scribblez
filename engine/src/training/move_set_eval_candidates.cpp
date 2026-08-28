#include "training/move_set_eval_candidates.h"

#include <algorithm>

namespace scribblez {
namespace move_set_eval {

namespace {

// Uniformly draw up to `count` distinct indices i in [lo, min(hi, n)) with
// !(*taken)[i] (no replacement); append them to *out and mark each in *taken.
// The strata core: rank windows [lo, hi) select the contention zone, the tail,
// and (over [0, n)) the uniform draw.
void draw_rank_window(const std::vector<Move>& ranked, int lo, int hi, int count,
                      std::mt19937_64& rng, std::vector<char>* taken, std::vector<size_t>* out) {
  const int n = int(ranked.size());
  std::vector<size_t> pool;
  for (int i = std::max(lo, 0); i < hi && i < n; ++i) {
    if (!(*taken)[size_t(i)]) pool.push_back(size_t(i));
  }
  std::shuffle(pool.begin(), pool.end(), rng);
  for (int j = 0; j < count && j < int(pool.size()); ++j) {
    (*taken)[pool[size_t(j)]] = 1;
    out->push_back(pool[size_t(j)]);
  }
}

// The exchange stratum: up to `count` distinct non-PLAY indices, uniform over
// wherever they rank, excluding those already marked in *taken.
void draw_exchanges(const std::vector<Move>& ranked, int count, std::mt19937_64& rng,
                    std::vector<char>* taken, std::vector<size_t>* out) {
  std::vector<size_t> non_plays;
  for (size_t i = 0; i < ranked.size(); ++i) {
    if (ranked[i].type() != MoveType::PLAY) non_plays.push_back(i);
  }
  std::shuffle(non_plays.begin(), non_plays.end(), rng);
  int drawn = 0;
  for (size_t i : non_plays) {
    if (drawn >= count) break;
    if (!(*taken)[i]) {
      (*taken)[i] = 1;
      out->push_back(i);
      ++drawn;
    }
  }
}

// A ranked-index mask marking every index whose move already sits in `chosen`.
// `ranked` holds distinct legal moves, so value membership is index membership.
std::vector<char> mask_of(const std::vector<Move>& ranked, const std::vector<Move>& chosen) {
  std::vector<char> taken(ranked.size(), 0);
  for (const Move& m : chosen) {
    const auto it = std::find(ranked.begin(), ranked.end(), m);
    if (it != ranked.end()) taken[size_t(it - ranked.begin())] = 1;
  }
  return taken;
}

// The rank strata shared by the labeled sample and the off-policy floor: the
// contention window [quotas.top, quotas.mid_rank_limit), the tail
// [quotas.mid_rank_limit, n), then the exchanges. Appends the drawn indices to
// *out and marks each in *taken.
void draw_rank_strata(const std::vector<Move>& ranked, const StratumQuotas& quotas,
                      std::mt19937_64& rng, std::vector<char>* taken, std::vector<size_t>* out) {
  const int n = int(ranked.size());
  draw_rank_window(ranked, quotas.top, quotas.mid_rank_limit, quotas.mid, rng, taken, out);
  draw_rank_window(ranked, quotas.mid_rank_limit, n, quotas.tail, rng, taken, out);
  draw_exchanges(ranked, quotas.exchange, rng, taken, out);
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

  // Top stratum: the head of the ranking, dense. The bound is relative to
  // what the played move and the forced set already occupy, so forced
  // candidates add to the sample rather than stealing head slots from it.
  const int head_target = int(out.size()) + quotas.top;
  for (int i = 0; i < n && int(out.size()) < head_target; ++i) {
    if (std::find(out.begin(), out.end(), ranked[i]) == out.end()) out.push_back(ranked[i]);
  }

  // The sampled strata, over the indices the deterministic prefix has not taken.
  std::vector<char> taken = mask_of(ranked, out);
  std::vector<size_t> picks;
  draw_rank_strata(ranked, quotas, rng, &taken, &picks);
  for (size_t i : picks) out.push_back(ranked[i]);
  return {std::move(out), 0u};
}

std::vector<size_t> off_policy_draws(const std::vector<Move>& ranked, const StratumQuotas& quotas,
                                     int uniform, std::vector<char>* taken, std::mt19937_64& rng) {
  std::vector<size_t> picks;
  draw_rank_strata(ranked, quotas, rng, taken, &picks);
  draw_rank_window(ranked, 0, int(ranked.size()), uniform, rng, taken, &picks);
  return picks;
}

Selection full_sweep_candidates(const std::vector<Move>& ranked, const Move& played, int cap) {
  std::vector<Move> out;
  bool played_kept = false;
  for (int i = 0; i < int(ranked.size()); ++i) {
    const Move& m = ranked[size_t(i)];
    const bool keep = i < cap || m.type() != MoveType::PLAY || m == played;
    if (!keep) continue;
    played_kept = played_kept || m == played;
    out.push_back(m);
  }
  // A played move the generator never enumerates -- a PASS chosen while other
  // moves were legal -- has no equity rank, so it can only go last; every other
  // candidate keeps its rank order. It counts toward the legal total all the
  // same: it was legal, and the sweep did reach it.
  uint32_t legal = ranked.size();
  if (!played_kept) {
    out.push_back(played);
    ++legal;
  }
  return {std::move(out), legal};
}

}  // namespace move_set_eval
}  // namespace scribblez
