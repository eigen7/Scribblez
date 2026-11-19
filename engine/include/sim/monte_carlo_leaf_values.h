#pragma once

// The leaf eval model's target computed rather than predicted: a leaf's
// win/draw/loss as the frequencies over `games` playouts from it, both sides
// EndgameAgent<HastyBot> at its defaults under hidden leaves, the policy and
// rules of the self-play the leaf network trains on
// (py/scribblez/workloads/deep_bot_leaf_eval.py). Up to the playouts' noise,
// it is what a perfect leaf network would predict, so LeafBot over it values
// each candidate as a perfect network would (the leaf_eval_truth workload).
//
// A playout seats both leaves and refills the opponent and then the mover from
// the pool (the tiles on neither the board nor a leave) in a shuffled order,
// then plays to the game's end, the opponent first. An exchange's tiles are in
// the pool from the start, as the network sees them. Playout g of every leaf
// is seeded by g + 1, so the values are deterministic.

#include "sim/rollout_values.h"

#include <span>
#include <vector>

namespace scribblez {

class Dictionary;

class MonteCarloLeafValues : public LeafValues {
 public:
  MonteCarloLeafValues(const Dictionary& dict, int games) : dict_(dict), games_(games) {}

  // Plays every leaf's games on the process's work pool.
  std::vector<Wld> predict_wld(std::span<const CpuLeaf* const> leaves) override;

 private:
  const Dictionary& dict_;
  int games_;
};

}  // namespace scribblez
