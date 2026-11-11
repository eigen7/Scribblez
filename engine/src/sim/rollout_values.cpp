#include "sim/rollout_values.h"

#include "sim/sim_maps.h"
#include "util/work_pool.h"

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

// The synthetic leaf's continuation heads: every class, mean and variance
// varying by square, so a misplaced plane or square shows.
LeafPrediction synthetic_heads() {
  LeafPrediction p;
  for (int sq = 0; sq < rollout_abi::kSquares; ++sq) {
    for (int c = 0; c < rollout_abi::kFateClasses; ++c)
      p.ownership[size_t(c * rollout_abi::kSquares + sq)] = 0.5f * float((sq * 7 + c * 3) % 11) - 2;
    p.square_points[size_t(sq)] = 0.05f * float(sq % 13 - 6);
    p.square_points[size_t(rollout_abi::kSquares + sq)] = 0.01f * float(1 + sq % 5);
  }
  p.going_out = {0.4f, -0.3f, -1.5f};
  p.end_points = {0.05f, 0.02f};
  return p;
}

CpuLeafFn synthetic_leaf() {
  return [heads = synthetic_heads()](const CpuLeaf& leaf) {
    const double diff = leaf.scores[0] - leaf.scores[1];
    const float p = float(1.0 / (1.0 + std::exp(-diff / kSyntheticSpreadScale)));
    LeafPrediction out = heads;
    out.wld = {p, 0.0f, 1.0f - p};
    return out;
  };
}

CpuRolloutValues::CpuRolloutValues(const Dictionary& dict, CpuLeafFn leaf)
    : dict_(dict), leaf_(std::move(leaf)) {}

std::vector<LeafPrediction> CpuRolloutValues::predict(const std::vector<CpuLeaf>& leaves) {
  std::vector<LeafPrediction> out(leaves.size());
  util::parallel_for(leaves.size(), [&](size_t i) { out[i] = leaf_(leaves[i]); });
  return out;
}

BatchValues CpuRolloutValues::run(const RolloutBatch& batch) {
  BatchValues out;
  out.outcomes.resize(batch.tasks.size());
  // What the maps need, kept only for a batch that sums them.
  const bool sum_maps = batch.sum_maps;
  const size_t kept = sum_maps ? batch.tasks.size() : 0;
  std::vector<rollout_abi::RolloutFates> fates(kept);
  std::vector<LeafPrediction> predictions(kept);
  std::vector<rollout_abi::LeafHeadRows> heads(kept, rollout_abi::LeafHeadRows{});
  for_each_hasty_rollout_cpu(batch, dict_,
                             [&](size_t k, const rollout_abi::RolloutResult& result,
                                 const CpuLeaf& leaf, const rollout_abi::RolloutFates& f) {
                               out.outcomes[k].result = result;
                               if (sum_maps) fates[k] = f;
                               if (result.game_over) {
                                 out.outcomes[k].wld = terminal_wld(result.diff);
                                 return;
                               }
                               const LeafPrediction p = leaf_(leaf);
                               out.outcomes[k].wld = p.wld;
                               if (!sum_maps) return;
                               predictions[k] = p;
                               heads[k] = predictions[k].heads();
                             });
  if (!sum_maps) return out;
  const TasksByStart by_start = tasks_by_start(batch);
  out.maps = sum_sim_maps(fates, by_start);
  complete_sim_maps(out.maps, fates, by_start, heads);
  return out;
}

}  // namespace scribblez
