#include "belief/leave_prior.h"

#include "game/tile.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace scribblez::belief {

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

double log_binomial(int n, int k) {
  if (k < 0 || k > n) return kNegInf;
  return std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
}

// Walks the distinct size-`remaining` multisets drawable from `pool` using
// tile types `first`.. only, so each multiset is reached by exactly one
// (ascending) path. `visit` returns false to abandon the walk, which leaves
// `buf` in an unspecified state -- a caller that stops early discards it.
template <typename F>
bool walk_multisets(const TileCounts& pool, int first, int remaining, Rack& buf, F&& visit) {
  if (remaining == 0) return visit(buf);
  for (int i = first; i <= BLANK; ++i) {
    const Tile t = Tile::of(i);
    const int copies = std::min(pool.count(t), remaining);
    for (int c = 1; c <= copies; ++c) {
      buf.add(t);
      if (!walk_multisets(pool, i + 1, remaining - c, buf, visit)) return false;
    }
    for (int c = 0; c < copies; ++c) buf.remove(t);
  }
  return true;
}

}  // namespace

int64_t count_multisets(const TileCounts& pool, int k, int64_t cap) {
  if (k < 0 || k > RACK_SIZE || pool.size() < k) return 0;
  int64_t count = 0;
  Rack buf;
  walk_multisets(pool, 0, k, buf, [&](const Rack&) { return ++count <= cap; });
  return count;
}

std::vector<ScoredLeave> enumerate_leaves(const TileCounts& pool, int k) {
  std::vector<ScoredLeave> out;
  if (k < 0 || k > RACK_SIZE || pool.size() < k) return out;
  Rack buf;
  walk_multisets(pool, 0, k, buf, [&](const Rack& leave) {
    out.push_back({leave, log_hypergeometric_prior(leave, pool)});
    return true;
  });
  return out;
}

double log_hypergeometric_prior(const Rack& leave, const TileCounts& pool) {
  const int n = pool.size();
  const int k = leave.size();
  if (k > n) return kNegInf;
  // P = prod_t C(pool_t, leave_t) / C(n, k): the ways to choose this leave's
  // copies of each tile type, over the ways to choose any k tiles at all.
  const TileCounts want = leave.counts();
  double log_p = -log_binomial(n, k);
  for (int i = 0; i <= BLANK; ++i) {
    const Tile t = Tile::of(i);
    if (want.count(t) == 0) continue;
    if (want.count(t) > pool.count(t)) return kNegInf;
    log_p += log_binomial(pool.count(t), want.count(t));
  }
  return log_p;
}

Rack draw_leave(const TileCounts& pool, int k, std::mt19937_64& rng) {
  assert(pool.size() >= k);
  Rack leave;
  TileCounts left = pool;
  for (int drawn = 0; drawn < k; ++drawn) {
    std::uniform_int_distribution<int> pick(0, left.size() - 1);
    int index = pick(rng);
    for (int i = 0; i <= BLANK; ++i) {
      const Tile t = Tile::of(i);
      if (index < left.count(t)) {
        leave.add(t);
        left.remove(t);
        break;
      }
      index -= left.count(t);
    }
  }
  return leave;
}

}  // namespace scribblez::belief
