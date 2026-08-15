#include "data/slog_sampling.h"

#include "util/math.h"

#include <algorithm>
#include <numeric>
#include <random>

namespace scribblez {
namespace binlog {

void sample_eligible_turns(const GameMetadata& gm, uint32_t game_idx, uint64_t run_seed,
                           int positions_per_game, std::vector<GamePositionIndex>* out) {
  const int begin = gm.eligible_begin;
  const int end = gm.eligible_end;
  if (begin >= end) return;
  if (positions_per_game <= 0) {
    for (int t = begin; t < end; ++t) out->push_back({game_idx, static_cast<uint32_t>(t)});
    return;
  }
  std::vector<uint32_t> turns(static_cast<size_t>(end - begin));
  std::iota(turns.begin(), turns.end(), static_cast<uint32_t>(begin));
  std::mt19937_64 rng(util::splitmix64(run_seed ^ util::splitmix64(0xC0FFEEull + game_idx)));
  std::shuffle(turns.begin(), turns.end(), rng);
  const int take = std::min<int>(positions_per_game, static_cast<int>(turns.size()));
  for (int i = 0; i < take; ++i) out->push_back({game_idx, turns[i]});
}

int count_eligible_sample(const GameMetadata& gm, int positions_per_game) {
  const int eligible = std::max(0, gm.eligible_end - gm.eligible_begin);
  return positions_per_game <= 0 ? eligible : std::min(positions_per_game, eligible);
}

}  // namespace binlog
}  // namespace scribblez
