// Evidence-trajectory candidate selection (docs/roadmap.md item 4): the pure,
// NN-free core of a trajectory -- which of a decision point's legal moves get
// simmed, in what order. The greedy anchor first, then a randomized-length
// sequence of proposals drawn from a temperature softmax over the move set
// evaluation model's win equities (over EVERY unsimmed candidate -- deployment's
// full support), then one uniform-random draw over the remaining legal moves.
// The uniform tail is appended LAST deliberately: training rows pair an evidence
// prefix with a held-out simmed candidate, so a last-slot sim yields proves-best
// labels at every prefix size while never entering an evidence set (the deployed
// loop's evidence contains only proposer picks). See
// docs/sim_residual_feedback.md, "Evidence-trajectory generation".
//
// This is selection only -- vectors in, indices out -- so it links without the
// TensorRT runtime the scoring front-end (evidence_trajectory.h) carries, and is
// unit-tested directly.
#pragma once

#include "game/move.h"
#include "util/math.h"

#include <cstdint>
#include <random>
#include <vector>

namespace scribblez::evidence {

struct TrajectoryOptions {
  int rollouts = 200;
  // Value truncation; see SimRunner::Params::horizon_plies for the full
  // semantics. The leaf service handed to TrajectoryRunner scores the horizon.
  int horizon = 0;
  int proposals_min = 2;
  int proposals_max = 8;
  double temperature = 0.05;  // win-equity units
};

// The anchor: the highest-raw-score candidate, taken off the move list by a
// rule no model can be wrong about. `ranked` is descending static equity, so
// ties resolve to the equity-preferred instance deterministically.
size_t anchor_index(const std::vector<Move>& ranked);

// The trajectory's candidate indices into `ranked`, in sim order: anchor, then
// up to a sampled count of temperature-softmax proposals over the student's win
// equities (drawn over every unsimmed candidate), then (when any move remains)
// one uniform draw. Sets *uniform_tail accordingly.
std::vector<size_t> select_trajectory(const std::vector<Move>& ranked,
                                      const std::vector<float>& win_equity,
                                      const TrajectoryOptions& opt, std::mt19937_64& rng,
                                      util::SoftmaxSampler& sampler, bool* uniform_tail);

}  // namespace scribblez::evidence
