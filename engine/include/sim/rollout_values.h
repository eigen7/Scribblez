#pragma once

// Rollout batches (sim/hasty_rollout_task.h) valued from the chooser's side:
// each task's win/draw/loss probabilities, from the leaf at its end state, or
// exact when the game ended inside the rollout, and each start's rollouts
// summed into its maps (rollout_abi::SimMaps), with what a rollout that
// stopped at a leaf left open completed from the leaf's prediction. One
// interface over the backends: the CUDA kernel with the leaf network on the
// device (gpu/gpu_rollout_values.h), and the CPU reference with a leaf
// function, which the test suites use since they hide the GPU.

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

// P(win) + P(draw) / 2.
inline double win_value(const Wld& w) { return w[0] + 0.5 * w[1]; }

struct RolloutOutcome {
  rollout_abi::RolloutResult result;
  Wld wld{};

  double win_value() const { return scribblez::win_value(wld); }
};

struct LeafPrediction;

// A batch valued: outcomes[k] for batch.tasks[k], and maps[s] for
// batch.starts[s] when the batch sums its maps.
struct BatchValues {
  std::vector<RolloutOutcome> outcomes;
  std::vector<rollout_abi::SimMaps> maps;
};

class RolloutValues {
 public:
  virtual ~RolloutValues() = default;

  virtual BatchValues run(const RolloutBatch& batch) = 0;
  // The leaf's prediction at each of `leaves`, as the sims' leaf would be
  // valued there: the prior a sim's maps are read against when `leaves` are
  // the sims' starts (agent/bob_subtree.h).
  virtual std::vector<LeafPrediction> predict(const std::vector<CpuLeaf>& leaves) = 0;
};

// A leaf network's prediction at a rollout's leaf, from player 0's side (the
// player who moved last): the value, and the continuation heads as one row of
// the network's outputs.
struct LeafPrediction {
  Wld wld{};
  std::array<float, rollout_abi::kOwnershipFloats> ownership{};
  std::array<float, rollout_abi::kSquarePointsFloats> square_points{};
  std::array<float, rollout_abi::kWentOutClasses> going_out{};
  std::array<float, rollout_abi::kEndPointsFloats> end_points{};

  // A view of this object's arrays, valid while it lives where it is.
  rollout_abi::LeafHeadRows heads() const {
    return {ownership.data(), square_points.data(), going_out.data(), end_points.data()};
  }
};

// A leaf's prediction. Called from worker threads, so it must be thread-safe.
using CpuLeafFn = std::function<LeafPrediction(const CpuLeaf&)>;

// A stand-in leaf for tests: a logistic in the score differential, no draws,
// and continuation heads that are fixed functions of the square.
CpuLeafFn synthetic_leaf();

class CpuRolloutValues : public RolloutValues {
 public:
  CpuRolloutValues(const Dictionary& dict, CpuLeafFn leaf);

  BatchValues run(const RolloutBatch& batch) override;
  std::vector<LeafPrediction> predict(const std::vector<CpuLeaf>& leaves) override;

 private:
  const Dictionary& dict_;
  CpuLeafFn leaf_;
};

}  // namespace scribblez
