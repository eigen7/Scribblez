// Evidence-trajectory candidate selection (docs/roadmap.md item 4): the pure,
// NN-free core of a trajectory -- which of a decision point's legal moves get
// simmed, in what order, and each one's evidence role. The greedy anchor first,
// then a randomized-length sequence of on-policy proposals drawn from a
// temperature softmax over the move set evaluation model's win equities (over
// EVERY unsimmed candidate -- deployment's full support), then a uniform draw of
// off-policy candidates (the labels-only exploration floor).
//
// The anchor and the on-policy proposals are evidence (SimObsRole::kAnchor /
// kOnPolicy); the off-policy draws are labels-only (kOffPolicy) -- simmed for
// their proves-best gain but never placed in an evidence set, because deployed
// evidence holds only the anchor and the proposer's picks. See
// docs/sim_residual_feedback.md, "Evidence-trajectory generation".
//
// This is selection only -- vectors in, indices and roles out -- so it links
// without the TensorRT runtime the scoring front-end (evidence_trajectory.h)
// carries, and is unit-tested directly. The anchor rule is also the sequential
// evidence loop's (agent/evidence_loop.h, item 6): both sim the anchor first.
#pragma once

#include "data/sim_obs_role.h"
#include "game/move.h"
#include "util/math.h"

#include <random>
#include <vector>

namespace scribblez::evidence {

struct TrajectoryOptions {
  int rollouts = 200;
  // Value truncation; see SimRunner::Params::horizon_plies for the full
  // semantics. The leaf service handed to TrajectoryRunner scores the horizon.
  int horizon = 0;
  int on_policy_min = 2;
  int on_policy_max = 8;
  double temperature = 0.05;  // win-equity units
  // The off-policy floor (docs/roadmap.md item 4): this many candidates drawn
  // uniformly over the untaken legal moves, all held out of every evidence set.
  int off_policy_count = 3;
};

// The anchor: the highest-raw-score candidate, taken off the move list by a
// rule no model can be wrong about. `ranked` is descending static equity, so
// ties resolve to the equity-preferred instance deterministically.
size_t anchor_index(const std::vector<Move>& ranked);

// The trajectory's candidate indices into `ranked`, in sim order: the anchor,
// up to a sampled count of temperature-softmax proposals over the student's win
// equities, then the uniform off-policy draws. Fills *roles in parallel with
// the returned indices (kAnchor, kOnPolicy, kOffPolicy), so a reader recovers
// evidence-eligibility from the role.
std::vector<size_t> select_trajectory(const std::vector<Move>& ranked,
                                      const std::vector<float>& win_equity,
                                      const TrajectoryOptions& opt, std::mt19937_64& rng,
                                      util::SoftmaxSampler& sampler,
                                      std::vector<SimObsRole>* roles);

}  // namespace scribblez::evidence
