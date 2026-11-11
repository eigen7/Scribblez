#pragma once

// A start's rollouts summed into its maps (rollout_abi::SimMaps) on the CPU,
// then completed by the leaf network's predictions for what the rollouts that
// stopped at a leaf left open: the reference for the device reduction. The
// exact sums add the same rollouts in the same order as the device, so they
// are bit-identical; both bin a batch's tasks by start here, so the order is
// one definition. The completion adds the same terms but agrees only to
// rounding (gpu/rollout_kernel.cu's kSliceRows says why).

#include "gpu/rollout_abi.h"
#include "sim/hasty_rollout_task.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace scribblez {

// A batch's task indices grouped by start, each group ascending: start s's
// are tasks[offsets[s], offsets[s + 1]).
struct TasksByStart {
  std::vector<uint32_t> offsets;  // starts + 1
  std::vector<uint32_t> tasks;
};

// Items 0..start_of.size() binned by start: item i belongs to start_of[i],
// one of `starts`.
TasksByStart bin_by_start(std::span<const uint32_t> start_of, size_t starts);
TasksByStart tasks_by_start(const RolloutBatch& batch);

// SimMaps as a table of named float fields, in layout order, covering the
// struct exactly: what a file of maps records of its layout.
struct SimMapsField {
  const char* name;
  int rows;  // 1 for a vector or scalar
  int cols;
};
std::span<const SimMapsField> sim_maps_fields();

// Adds `more` to `maps`, field by field: a start's maps over several runs.
void add_sim_maps(rollout_abi::SimMaps& maps, const rollout_abi::SimMaps& more);

// Each start's maps from every task's fates (fates[k] for batch.tasks[k]).
std::vector<rollout_abi::SimMaps> sum_sim_maps(const std::vector<rollout_abi::RolloutFates>& fates,
                                               const TasksByStart& by_start);

// Adds to each start's maps the completion of its tasks that stopped at a
// leaf, in ascending task order: task k's heads are row 0 of heads[k], and a
// task whose game ended (heads[k].ownership null) adds nothing.
void complete_sim_maps(std::vector<rollout_abi::SimMaps>& maps,
                       const std::vector<rollout_abi::RolloutFates>& fates,
                       const TasksByStart& by_start,
                       std::span<const rollout_abi::LeafHeadRows> heads);

// The first field of `a` and `b` that differ by more than `tolerance`
// relative to the larger magnitude (at least 1), as "field[index]: a vs b",
// or empty. The default asks for bit-identity.
std::string sim_maps_difference(const rollout_abi::SimMaps& a, const rollout_abi::SimMaps& b,
                                float tolerance = 0.0f);

// How far completed maps are from counting every rollout once: the largest
// gap between a square's class counts and 0 (filled at the start) or the
// rollout count, and between the end's class counts and the rollout count.
float completion_gap(const rollout_abi::SimMaps& m);

}  // namespace scribblez
