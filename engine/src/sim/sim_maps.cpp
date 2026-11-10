#include "sim/sim_maps.h"

#include <cstring>
#include <format>
#include <numeric>

namespace scribblez {

using rollout_abi::RolloutFates;
using rollout_abi::SimMaps;

TasksByStart tasks_by_start(const RolloutBatch& batch) {
  TasksByStart out;
  out.offsets.assign(batch.starts.size() + 1, 0);
  for (const rollout_abi::RolloutTask& t : batch.tasks) ++out.offsets[t.position + 1];
  std::partial_sum(out.offsets.begin(), out.offsets.end(), out.offsets.begin());
  std::vector<uint32_t> next(out.offsets.begin(), out.offsets.end() - 1);
  out.tasks.resize(batch.tasks.size());
  for (uint32_t k = 0; k < batch.tasks.size(); ++k) out.tasks[next[batch.tasks[k].position]++] = k;
  return out;
}

std::vector<SimMaps> sum_sim_maps(const std::vector<RolloutFates>& fates,
                                  const TasksByStart& by_start) {
  std::vector<SimMaps> maps(by_start.offsets.size() - 1);
  for (size_t s = 0; s < maps.size(); ++s) {
    SimMaps& m = maps[s];
    std::memset(&m, 0, sizeof(m));
    for (uint32_t i = by_start.offsets[s]; i < by_start.offsets[s + 1]; ++i) {
      const RolloutFates& f = fates[by_start.tasks[i]];
      for (int sq = 0; sq < rollout_abi::kSquares; ++sq) rollout_abi::add_square_fate(m, sq, f);
      rollout_abi::add_end_fate(m, f);
    }
  }
  return maps;
}

namespace {

// The arrays of a SimMaps, in order, as (name, elements).
constexpr std::pair<const char*, int> kSimMapsFields[] = {
  {"class_sum", rollout_abi::kFateClasses* rollout_abi::kSquares},
  {"points_sum", rollout_abi::kSquares},
  {"points_sq_sum", rollout_abi::kSquares},
  {"went_out_sum", rollout_abi::kWentOutClasses},
  {"end_points_sum", 1},
  {"end_points_sq_sum", 1},
  {"sims", 1},
};

}  // namespace

std::string sim_maps_difference(const SimMaps& a, const SimMaps& b) {
  const float* fa = reinterpret_cast<const float*>(&a);
  const float* fb = reinterpret_cast<const float*>(&b);
  int i = 0;
  for (const auto& [name, elements] : kSimMapsFields) {
    for (int j = 0; j < elements; ++j, ++i) {
      if (fa[i] != fb[i]) return std::format("{}[{}]: {} vs {}", name, j, fa[i], fb[i]);
    }
  }
  return "";
}

}  // namespace scribblez
