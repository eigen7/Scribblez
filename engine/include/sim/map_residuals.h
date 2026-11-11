#pragma once

// A start's sims against the leaf network's prior on the start itself
// (docs/plans/deep_bot.md, "Residuals"): what the sims' maps and end of game
// say beyond what the network predicted before any ply, and how much of that
// is sampling noise. Both sides are from player 0's (the start's last
// mover's); the consumer flips them to the evaluator's side if it is the
// other player.
//
// A map's residual at a square is the sims' mean minus the prior's mean, so
// its sampling noise is the spread of the rollouts' values (each exact or
// the leaf's expectation) over the rollout count: the unbiased sample
// variance, so one rollout gives no noise estimate (0). Summed over the squares
// empty at the start, a start's squared residual should then exceed its
// noise by its squared signal, which is what the summary lets a readout
// measure against the rollout count. The leaf's own variance of what it
// completed is kept beside, with the prior's, for their calibration.

#include "encoding/input_encoder.h"
#include "game/board.h"
#include "gpu/rollout_abi.h"
#include "sim/rollout_values.h"

namespace scribblez {

struct MapResidual {
  int sims = 0;
  int squares = 0;  // empty at the start: the squares with labels
  // The points map, in points, summed over those squares.
  double points_r2 = 0;         // (the sims' mean - the prior's mean)^2
  double points_noise = 0;      // the rollouts' spread / sims
  double points_var = 0;        // the rollouts' spread (per rollout)
  double points_leaf_var = 0;   // the leaf's variance, per rollout
  double points_prior_var = 0;  // the prior's variance
  // Ownership, summed over those squares and the classes.
  double ownership_r2 = 0;     // (the sims' class share - the prior's probability)^2
  double ownership_noise = 0;  // the shares' spread / sims
  // Where the ownership mass moved: per class, the sims' share minus the
  // prior's, summed over those squares (a side mix-up shows as mass moving
  // between the evaluator's and the opponent's classes).
  double ownership_shift[rollout_abi::kFateClasses] = {};
  // The end-of-game points, in points.
  double end_r2 = 0, end_noise = 0, end_var = 0, end_leaf_var = 0, end_prior_var = 0;
  // Who goes out, over the classes.
  double going_out_r2 = 0, going_out_noise = 0;
  double going_out_shift[rollout_abi::kWentOutClasses] = {};
  // The value: the sims' mean win value minus the prior's.
  double value_r = 0;
};

// The softmax of `classes` logits, logits[c * stride], into probs[c]: a
// per-square head's classes lie a plane apart (stride kBoardCells).
void class_probabilities(const float* logits, int classes, int stride, float* probs);

// `maps` are the start's sims summed (every rollout settled or completed, so
// each labelled square's class shares sum to 1); `prior` the network at the
// start; `start` its board; `sim_value` the sims' mean win value.
MapResidual map_residual(const rollout_abi::SimMaps& maps, const LeafPrediction& prior,
                         const Board& start, double sim_value);

}  // namespace scribblez
