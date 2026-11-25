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
#include <span>
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

// A value at rollout leaves: the win/draw/loss probabilities at each, from
// player 0's side. The narrow face of RolloutValues that LeafBot needs, which
// a backend with no network (sim/monte_carlo_leaf_values.h) also offers.
class LeafValues {
 public:
  virtual ~LeafValues() = default;

  // out[i] for *leaves[i].
  virtual std::vector<Wld> predict_wld(std::span<const CpuLeaf* const> leaves) = 0;

  // Whether a leaf's value is one random outcome (a playout) rather than a
  // prediction, so that repeats of a leaf each need a value of their own.
  virtual bool noisy() const { return false; }
};

class RolloutValues : public LeafValues {
 public:
  virtual BatchValues run(const RolloutBatch& batch) = 0;
  // The leaf's prediction at each of `leaves`, as the sims' leaf would be
  // valued there: a candidate's value with no rollout at all
  // (agent/leaf_bot.h).
  std::vector<LeafPrediction> predict(const std::vector<CpuLeaf>& leaves);
  // predict() over leaves held elsewhere, out[i] for *leaves[i]: what a
  // caller that gathers several callers' leaves into one call passes.
  virtual std::vector<LeafPrediction> predict_leaves(std::span<const CpuLeaf* const> leaves) = 0;
  // The value alone of predict_leaves(): a backend may skip what it costs to
  // bring the continuation heads back.
  std::vector<Wld> predict_wld(std::span<const CpuLeaf* const> leaves) override;
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
  std::vector<LeafPrediction> predict_leaves(std::span<const CpuLeaf* const> leaves) override;

 private:
  const Dictionary& dict_;
  CpuLeafFn leaf_;
};

}  // namespace scribblez
