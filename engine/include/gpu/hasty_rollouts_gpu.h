#pragma once

// Batches of the hasty-rollout task on the GPU: one thread block per rollout,
// every candidate play scored as it is found and reduced to the block's best,
// with the lexicon tables resident on the device. The design and the
// throughput model are docs/plans/gpu_hasty_rollouts.md; the CPU reference it
// must match rollout for rollout is sim/hasty_rollout_task.h.

#include "gpu/rollout_abi.h"
#include "sim/hasty_rollout_task.h"

#include <memory>
#include <vector>

namespace scribblez {

class Dictionary;
class HastyEquity;

class GpuHastyRollouts {
 public:
  // Uploads the word map, the leave table, the DAWG and the equity tables.
  GpuHastyRollouts(const Dictionary& dict, const HastyEquity& eq);
  ~GpuHastyRollouts();

  // The batch's tasks, results[k] for tasks[k], each drawn as the CPU
  // reference draws it. The end states and fates stay on the device, one per
  // task, and so, for a batch that sums its maps, does each start's sum of
  // its rollouts' fates, for complete_maps() and maps().
  std::vector<rollout_abi::RolloutResult> run(const RolloutBatch& batch, const Dictionary& dict);

  struct Timings {
    double upload_seconds = 0;    // the batch
    double kernel_seconds = 0;    // launch to completion
    double download_seconds = 0;  // the results
  };
  const Timings& last_timings() const { return timings_; }
  // The last run's kernel time by phase, as fractions of the whole, with the
  // phase names.
  struct PhaseShare {
    const char* name;
    double share;
  };
  std::vector<PhaseShare> last_phase_shares() const;
  // The last run's work counters (rollout_kernel.h's Count), with names.
  struct CountValue {
    const char* name;
    unsigned long long value;
  };
  std::vector<CountValue> last_counts() const;
  // The last run's end states, in device memory: what a leaf evaluator on
  // the device would read.
  const rollout_abi::LeafState* device_leaves() const;
  // The leaf eval model's input rows (LeafEvalInputEncoder's layout) for the
  // leaves of the last run's tasks tasks[0, count), row i for tasks[i],
  // built on the device into the given device buffers, e.g. a
  // NeuralNet<LeafEvalSpec>'s inputs.
  void encode_leaves(const uint32_t* tasks, int count, float* spatial, float* scalar);
  // encode_leaves, with the rows copied back to the host (for checks).
  void encode_leaves_to_host(const uint32_t* tasks, int count, float* spatial, float* scalar);
  // Adds to the last run's maps the leaf network's completion of tasks
  // [0, count), ascending, row i of the device buffers `heads` for tasks[i]:
  // what each of those rollouts left open (rollout_abi::add_square_completion).
  void complete_maps(const uint32_t* tasks, int count, const rollout_abi::LeafHeadRows& heads);
  // complete_maps, from heads in host memory (for checks).
  void complete_maps_from_host(const uint32_t* tasks, int count,
                               const rollout_abi::LeafHeadRows& heads);
  // The last run's maps, maps[s] for starts[s], as completed so far; empty
  // unless the batch sums its maps.
  std::vector<rollout_abi::SimMaps> maps() const;
  double table_upload_seconds() const { return table_upload_seconds_; }
  size_t table_bytes() const { return table_bytes_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Timings timings_;
  // The last run's task starts and start count, while it sums its maps.
  std::vector<uint32_t> task_starts_;
  size_t map_starts_ = 0;
  std::vector<unsigned long long> phase_cycles_;
  std::vector<unsigned long long> counts_;
  double table_upload_seconds_ = 0;
  size_t table_bytes_ = 0;
};

}  // namespace scribblez
