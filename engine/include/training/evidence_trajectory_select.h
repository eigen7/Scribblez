// Evidence-trajectory candidate selection (docs/roadmap.md item 4): which of a
// decision point's legal moves get simmed, in what order, and in what evidence
// role. In sim order:
//
//   1. the anchor, the highest-raw-score move (SimObsRole::kAnchor);
//   2. a random number of on-policy proposals, each drawn from a temperature
//      softmax over the student's win equities across every unsimmed
//      candidate, the full support deployment chooses from (kOnPolicy);
//   3. a few off-policy candidates drawn uniformly from the rest (kOffPolicy).
//
// The anchor and on-policy picks are evidence. The off-policy draws are
// labels-only: they are simmed for their proves-best gain but never placed in
// an evidence set, because deployed evidence holds only the anchor and the
// proposer's picks. docs/plans/sim_residual_feedback.md, "Evidence-trajectory
// generation", has the rationale.
//
// Selection is NN-free (vectors in, indices and roles out) so it links without
// TensorRT and is unit-tested directly; evidence_trajectory.h does the scoring.
// The anchor rule is shared with the sequential evidence loop
// (agent/evidence_loop.h).
#pragma once

#include "data/sim_obs_role.h"
#include "game/move.h"
#include "util/math.h"

#include <random>
#include <vector>

namespace scribblez::evidence {

struct TrajectoryOptions {
  int rollouts = 200;
  // Value truncation, as SimRunner::Params::horizon_plies. The leaf service
  // handed to TrajectoryRunner scores the truncated rollouts.
  int horizon = 0;
  int on_policy_min = 2;
  int on_policy_max = 8;
  double temperature = 0.05;  // win-equity units
  // The off-policy floor: candidates drawn uniformly over the untaken legal
  // moves, all held out of every evidence set.
  int off_policy_count = 3;
};

// The anchor: the highest-raw-score candidate, chosen by a rule no model can get
// wrong. `ranked` is in descending static-equity order, so a score tie resolves
// deterministically to the equity-preferred move.
size_t anchor_index(const std::vector<Move>& ranked);

// The trajectory's candidate indices into `ranked`, in sim order, with *roles
// filled in parallel. `win_equity` is parallel to `ranked`.
std::vector<size_t> select_trajectory(const std::vector<Move>& ranked,
                                      const std::vector<float>& win_equity,
                                      const TrajectoryOptions& opt, std::mt19937_64& rng,
                                      util::SoftmaxSampler& sampler,
                                      std::vector<SimObsRole>* roles);

}  // namespace scribblez::evidence
