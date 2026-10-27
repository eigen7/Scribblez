#include "sim/rollout_values.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace scribblez {

namespace {

// The synthetic leaf's spread scale: a 30-point lead is about a 73% favorite.
constexpr double kSyntheticSpreadScale = 30.0;

}  // namespace

Wld terminal_wld(int diff) {
  if (diff > 0) return {1.0f, 0.0f, 0.0f};
  if (diff < 0) return {0.0f, 0.0f, 1.0f};
  return {0.0f, 1.0f, 0.0f};
}

Wld wld_from_logits(const float* z) {
  const float m = std::max({z[0], z[1], z[2]});
  const double e0 = std::exp(z[0] - m), e1 = std::exp(z[1] - m), e2 = std::exp(z[2] - m);
  const double sum = e0 + e1 + e2;
  return {float(e0 / sum), float(e1 / sum), float(e2 / sum)};
}

CpuLeafFn synthetic_leaf() {
  return [](const CpuLeaf& leaf) {
    const double diff = leaf.scores[0] - leaf.scores[1];
    const float p = float(1.0 / (1.0 + std::exp(-diff / kSyntheticSpreadScale)));
    return Wld{p, 0.0f, 1.0f - p};
  };
}

CpuRolloutValues::CpuRolloutValues(const Dictionary& dict, CpuLeafFn leaf)
    : dict_(dict), leaf_(std::move(leaf)) {}

std::vector<RolloutOutcome> CpuRolloutValues::run(const RolloutBatch& batch) {
  std::vector<RolloutOutcome> out(batch.tasks.size());
  for_each_hasty_rollout_cpu(
    batch, dict_, [&](size_t k, const rollout_abi::RolloutResult& result, const CpuLeaf& leaf) {
      out[k].result = result;
      out[k].wld = result.game_over ? terminal_wld(result.diff) : leaf_(leaf);
    });
  return out;
}

}  // namespace scribblez
