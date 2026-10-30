#pragma once

// Racing candidates on paired rollouts: at each checkpoint, a candidate
// clearly below the leader stops, keeping the rollouts it got. Rollout i is
// the same deal for every candidate that reaches it, so a survivor's
// rollouts are exactly those of an unraced sim. The CPU simmer
// (sim/slog_position_simmer.h) and the GPU pool tool race through here.

#include "sim/sim_runner.h"

#include <cstddef>
#include <vector>

namespace scribblez {

// Each candidate's rollouts, in rollout-index order.
using CandidateRollouts = std::vector<std::vector<Rollout>>;

// The checkpoints of a race to `rollouts`: 10%, 20%, 40% and 70% of it,
// those that round to at least one rollout. At three paired standard errors
// below the leader over four looks, a candidate truly equal to the leader is
// stopped well under once per hundred positions, and anything stopped was
// never going to be a pick.
std::vector<int> race_checkpoints(int rollouts);

// The `alive` candidate with the most wins so far, the first on ties.
size_t race_leader(const std::vector<size_t>& alive, const CandidateRollouts& rollouts);

// Drops from `alive` the candidates more than `sigmas` paired standard errors
// below the leader, except those `keep` marks (parallel to `rollouts`).
void stop_the_beaten(const CandidateRollouts& rollouts, double sigmas,
                     const std::vector<char>& keep, std::vector<size_t>* alive);

}  // namespace scribblez
