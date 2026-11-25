#pragma once

// The opponent's leaves a chooser values its candidates over: DeepBot's at
// play (agent/deep_bot.h), and the held-out draws a .belief sidecar stores
// (data/belief_log.h), which must be the same draws.

#include "belief/leave_query.h"
#include "belief/leave_sampler.h"
#include "game/rack.h"
#include "game/tile_counts.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace scribblez::belief {

// Whether opp_leave_draws reads a belief for `q`: a leave of a length the
// leave table holds.
inline bool draws_read_belief(const LeaveQuery& q) {
  return q.length > 0 && q.length <= kLeaveArraysMaxLength;
}

// `count` leaves for `q`:
//   - an empty leave (before the opponent's first move, or after a move that
//     kept nothing), `count` times;
//   - after a pass, which kept all seven tiles and so a leave the table does
//     not hold, racks drawn at random from the leave pool, as if the opponent
//     had just drawn the whole rack. That ignores what the pass says about the
//     rack and is wrong, but a pass is rare enough that it is accepted;
//   - otherwise `belief` at stratified quantiles jittered by `seed`: each draw
//     stands for an equal share of the belief's mass, so their mean has far
//     less spread than over independent draws.
// `belief` must be non-null where draws_read_belief(q).
std::vector<Rack> opp_leave_draws(const LeaveQuery& q, const LeaveSampler* belief, int count,
                                  uint64_t seed);

}  // namespace scribblez::belief
