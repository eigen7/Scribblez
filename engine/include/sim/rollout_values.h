#pragma once

// Rollout batches (sim/hasty_rollout_task.h) valued from the chooser's side:
// each task's win/draw/loss probabilities, from the leaf at its end state, or
// exact when the game ended inside the rollout. One interface over the
// backends: the CUDA kernel with the leaf network on the device
// (gpu/gpu_rollout_values.h), and the CPU reference with a leaf function,
// which the test suites use since they hide the GPU.

#include "gpu/rollout_abi.h"
#include "sim/hasty_rollout_task.h"

#include <array>
#include <functional>
#include <vector>

namespace scribblez {

class Dictionary;

// Win, draw and loss probabilities, from player 0's side.
using Wld = std::array<float, 3>;

// The exact outcome of a finished game with score differential `diff`.
Wld terminal_wld(int diff);

// The probabilities of a win/draw/loss logit triple.
Wld wld_from_logits(const float* z);

struct RolloutOutcome {
  rollout_abi::RolloutResult result;
  Wld wld{};

  double win_value() const { return wld[0] + 0.5 * wld[1]; }
};

class RolloutValues {
 public:
  virtual ~RolloutValues() = default;

  // Every task's outcome, outcomes[k] for batch.tasks[k].
  virtual std::vector<RolloutOutcome> run(const RolloutBatch& batch) = 0;
};

// A leaf's value from player 0's side (the player who moved last). Called
// from worker threads, so it must be thread-safe.
using CpuLeafFn = std::function<Wld(const CpuLeaf&)>;

// A stand-in leaf for tests: a logistic in the score differential, no draws.
CpuLeafFn synthetic_leaf();

class CpuRolloutValues : public RolloutValues {
 public:
  CpuRolloutValues(const Dictionary& dict, int threads, CpuLeafFn leaf);

  std::vector<RolloutOutcome> run(const RolloutBatch& batch) override;

 private:
  const Dictionary& dict_;
  int threads_;
  CpuLeafFn leaf_;
};

}  // namespace scribblez
