#pragma once

// PoolLabeler jobs (sim/pool_labeler.h) from replayed .slog positions: a
// position's group, with the opponent's leave per rollout index from the
// leave inference model's belief, and its candidates applied to the board.
// What the tools that label .slog positions on the GPU share
// (policy_set_pool_tool, blind_spot_confirm_tool).

#include "belief/leave_sampler.h"
#include "belief/slog_leave_beliefs.h"
#include "game/rack.h"
#include "sim/pool_labeler.h"
#include "sim/slog_position_simmer.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace scribblez {

class Dictionary;

// Rollout i's opponent leave from `belief`, at rollout_leave_quantile of the
// rollout's seed `seed + i`, for every rollout index below `rollouts`.
std::vector<Rack> belief_leaves(const belief::LeaveSampler& belief, uint64_t seed, int rollouts);

// A labeler job for the mover of `pos` choosing among `moves`: the group
// (the pool, `seed`, and rollout i's opponent leave from `leave`, or none for
// a null `leave`) and each move applied. The record holds only the moves.
PoolJob position_job(const SimPosition& pos, const std::vector<Move>& moves, uint64_t seed,
                     const belief::LeaveSampler* leave, const Dictionary& dict, int rollouts);

// The labeler's job for a selected position: its record but the labels, its
// group (pool, seed, per-rollout opponent leaves from `leave`, or none for a
// null `leave`) and its candidates applied.
PoolJob make_pool_job(const SimmedPosition& r, const belief::LeaveSampler* leave,
                      const Dictionary& dict, int rollouts);

// make_pool_job for each of `selected`, in parallel, the leaves from
// `beliefs` (null: the opponent's whole rack is drawn uniformly).
std::vector<PoolJob> make_pool_jobs(const std::vector<SimmedPosition>& selected,
                                    const belief::SlogLeaveBeliefs* beliefs, const Dictionary& dict,
                                    int rollouts);

// Throws util::CleanException naming the first of `slogs` without the .forgone
// sidecar the leave inference model's input reads, before any work starts.
void check_forgone_sidecars(const std::vector<std::filesystem::path>& slogs);

}  // namespace scribblez
