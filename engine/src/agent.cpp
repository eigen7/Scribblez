#include "scribblez/agent.h"

#include <algorithm>

namespace scribblez {

Move GreedyAgent::choose(const AgentContext& ctx, std::mt19937_64& rng) {
  if (!ctx.legal_plays.empty()) {
    int best = ctx.legal_plays.front().score;
    for (const auto& m : ctx.legal_plays) best = std::max(best, m.score);
    std::vector<const Move*> top;
    for (const auto& m : ctx.legal_plays)
      if (m.score == best) top.push_back(&m);
    std::uniform_int_distribution<size_t> d(0, top.size() - 1);
    return *top[d(rng)];
  }
  // No legal plays.
  if (ctx.bag_size >= RACK_SIZE) {
    Move m;
    m.type = MoveType::EXCHANGE;
    for (Letter L = 0; L <= BLANK; ++L) {
      for (int i = 0; i < ctx.my_rack.count(L); ++i) m.exchanged.push_back(L);
    }
    return m;
  }
  Move m;
  m.type = MoveType::PASS;
  return m;
}

}  // namespace scribblez
