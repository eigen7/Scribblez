#include "sim/sim_maps.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <numeric>

namespace scribblez {

using rollout_abi::RolloutFates;
using rollout_abi::SimMaps;

TasksByStart bin_by_start(std::span<const uint32_t> start_of, size_t starts) {
  TasksByStart out;
  out.offsets.assign(starts + 1, 0);
  for (const uint32_t s : start_of) ++out.offsets[s + 1];
  std::partial_sum(out.offsets.begin(), out.offsets.end(), out.offsets.begin());
  std::vector<uint32_t> next(out.offsets.begin(), out.offsets.end() - 1);
  out.tasks.resize(start_of.size());
  for (uint32_t i = 0; i < start_of.size(); ++i) out.tasks[next[start_of[i]]++] = i;
  return out;
}

TasksByStart tasks_by_start(const RolloutBatch& batch) {
  std::vector<uint32_t> start_of(batch.tasks.size());
  for (size_t k = 0; k < batch.tasks.size(); ++k) start_of[k] = batch.tasks[k].position;
  return bin_by_start(start_of, batch.starts.size());
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

void complete_sim_maps(std::vector<SimMaps>& maps, const std::vector<RolloutFates>& fates,
                       const TasksByStart& by_start,
                       std::span<const rollout_abi::LeafHeadRows> heads) {
  for (size_t s = 0; s < maps.size(); ++s) {
    for (uint32_t i = by_start.offsets[s]; i < by_start.offsets[s + 1]; ++i) {
      const uint32_t k = by_start.tasks[i];
      if (!heads[k].ownership) continue;
      for (int sq = 0; sq < rollout_abi::kSquares; ++sq)
        rollout_abi::add_square_completion(maps[s], sq, fates[k], heads[k], 0);
      rollout_abi::add_end_completion(maps[s], fates[k], heads[k], 0);
    }
  }
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

constexpr int simmaps_table_floats() {
  int n = 0;
  for (const auto& field : kSimMapsFields) n += field.second;
  return n;
}

// The table covers SimMaps exactly, so the diff walks every field.
static_assert(sizeof(SimMaps) == sizeof(float) * simmaps_table_floats());

}  // namespace

std::string sim_maps_difference(const SimMaps& a, const SimMaps& b, float tolerance) {
  const float* fa = reinterpret_cast<const float*>(&a);
  const float* fb = reinterpret_cast<const float*>(&b);
  int i = 0;
  for (const auto& [name, elements] : kSimMapsFields) {
    for (int j = 0; j < elements; ++j, ++i) {
      const float scale = std::max({1.0f, std::abs(fa[i]), std::abs(fb[i])});
      if (std::abs(fa[i] - fb[i]) > tolerance * scale)
        return std::format("{}[{}]: {} vs {}", name, j, fa[i], fb[i]);
    }
  }
  return "";
}

float completion_gap(const SimMaps& m) {
  float gap = 0;
  for (int sq = 0; sq < rollout_abi::kSquares; ++sq) {
    float mass = 0;
    for (int c = 0; c < rollout_abi::kFateClasses; ++c) mass += m.class_sum[c][sq];
    gap = std::max(gap, std::min(std::abs(mass), std::abs(mass - m.sims)));
  }
  float end = 0;
  for (int c = 0; c < rollout_abi::kWentOutClasses; ++c) end += m.went_out_sum[c];
  return std::max(gap, std::abs(end - m.sims));
}

}  // namespace scribblez
