#pragma once

// Rollout batches valued on the GPU (sim/rollout_values.h): the hasty-rollout
// kernel plays every task, a second kernel writes each end state's leaf eval
// row straight into the leaf network's TensorRT input buffers, and only the
// network's win/draw/loss logits come back to the host.

#include "gpu/hasty_rollouts_gpu.h"
#include "nn/model_specs.h"
#include "nn/neural_net.h"
#include "nn/trt_util.h"
#include "sim/rollout_values.h"

#include <string>
#include <vector>

namespace scribblez {

class Dictionary;
class HastyEquity;

class GpuRolloutValues : public RolloutValues {
 public:
  struct Params {
    std::string leaf_onnx;  // a deep_bot_leaf_eval export
    nn::Precision precision = nn::Precision::kFP16;
    int leaf_rows = nn::LeafEvalSpec::kDefaultMaxRows;  // rows per leaf network call
  };

  // Uploads the tables and loads the leaf network, refusing one trained
  // against other leave values than `eq`'s.
  GpuRolloutValues(const Dictionary& dict, const HastyEquity& eq, const Params& params);

  std::vector<RolloutOutcome> run(const RolloutBatch& batch) override;

  // Seconds in the rollout kernel (with the batch's conversion and
  // transfers) and in the leaf (encoding and the network): the last run's,
  // and the sums over every run.
  struct Timings {
    double rollout_seconds = 0;
    double kernel_seconds = 0;  // of rollout_seconds, the kernel alone
    double leaf_seconds = 0;
  };
  const Timings& last_timings() const { return timings_; }
  const Timings& total_timings() const { return totals_; }
  GpuHastyRollouts& rollouts() { return rollouts_; }
  nn::NeuralNet<nn::LeafEvalSpec>& net() { return net_; }

 private:
  // Fills outcomes[first, first + count) from the leaf network.
  void evaluate_leaves(int first, int count, std::vector<RolloutOutcome>* outcomes);

  const Dictionary& dict_;
  GpuHastyRollouts rollouts_;
  nn::NeuralNet<nn::LeafEvalSpec> net_;
  Timings timings_;
  Timings totals_;
};

}  // namespace scribblez
