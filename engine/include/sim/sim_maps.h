#pragma once

// A start's rollouts summed into its maps (rollout_abi::SimMaps) on the CPU:
// the reference for the device reduction, which adds the same rollouts in the
// same order. Both bin a batch's tasks by start here, so the order is one
// definition.

#include "gpu/rollout_abi.h"
#include "sim/hasty_rollout_task.h"

#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {

// A batch's task indices grouped by start, each group ascending: start s's
// are tasks[offsets[s], offsets[s + 1]).
struct TasksByStart {
  std::vector<uint32_t> offsets;  // starts + 1
  std::vector<uint32_t> tasks;
};

TasksByStart tasks_by_start(const RolloutBatch& batch);

// Each start's maps from every task's fates (fates[k] for batch.tasks[k]).
std::vector<rollout_abi::SimMaps> sum_sim_maps(const std::vector<rollout_abi::RolloutFates>& fates,
                                               const TasksByStart& by_start);

// The first field of `a` and `b` that differ, as "field[index]", or empty.
std::string sim_maps_difference(const rollout_abi::SimMaps& a, const rollout_abi::SimMaps& b);

}  // namespace scribblez
