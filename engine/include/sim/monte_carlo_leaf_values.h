#pragma once

// The leaf eval model's target computed rather than predicted: a leaf's
// win/draw/loss as the outcome of one playout from it, both sides
// EndgameAgent<HastyBot> at its defaults under hidden leaves, the policy and
// rules of the self-play the leaf network trains on
// (py/scribblez/workloads/deep_bot_leaf_eval.py). Its mean over many leaves
// is, up to the playouts' noise, the mean of what a perfect leaf network would
// predict at them, so LeafBot over it, with each opponent leave drawn its own
// playout (noisy()), values each candidate as a perfect network would over the
// whole belief (the leaf_eval_truth workload).
//
// A playout seats both leaves and refills the opponent and then the mover from
// the pool (the tiles on neither the board nor a leave) in a shuffled order,
// then plays to the game's end, the opponent first. An exchange's tiles are in
// the pool from the start, as the network sees them. The playout of a call's
// leaf i is seeded by i + 1, so the values are deterministic and repeats of a
// leaf are played out apart.

#include "sim/rollout_values.h"

#include <span>
#include <vector>

namespace scribblez {

class Dictionary;

class MonteCarloLeafValues : public LeafValues {
 public:
  explicit MonteCarloLeafValues(const Dictionary& dict) : dict_(dict) {}

  // Plays every leaf out on the process's work pool.
  std::vector<Wld> predict_wld(std::span<const CpuLeaf* const> leaves) override;
  bool noisy() const override { return true; }

 private:
  const Dictionary& dict_;
};

}  // namespace scribblez
