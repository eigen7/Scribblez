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

  // `n` rollouts of `pos`, rollout i drawn as the CPU reference draws it.
  std::vector<rollout_abi::RolloutResult> run(const HastyRolloutPosition& pos,
                                              const Dictionary& dict, int n);

  struct Timings {
    double upload_seconds = 0;    // the position
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
  double table_upload_seconds() const { return table_upload_seconds_; }
  size_t table_bytes() const { return table_bytes_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Timings timings_;
  std::vector<unsigned long long> phase_cycles_;
  std::vector<unsigned long long> counts_;
  double table_upload_seconds_ = 0;
  size_t table_bytes_ = 0;
};

}  // namespace scribblez
