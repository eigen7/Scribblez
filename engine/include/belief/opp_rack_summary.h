#pragma once

// The chooser's belief about the opponent's rack, summarized as the
// policy-set model reads it (docs/plans/deep_bot_v0.md, "The opponent-rack
// input"). It describes the rack the sims draw: the leave from the leave
// inference model's belief plus a uniform refill to a full rack, or, where no
// leave is modelled, a full rack drawn uniformly. A pool records it per
// position (data/pool_log.h), and the agent computes it from the same belief
// its sims draw from.

#include "belief/leave_sampler.h"
#include "game/tile.h"
#include "game/tile_counts.h"

#include <array>

namespace scribblez::belief {

struct OppRackSummary {
  // P(the rack holds at least one), by Tile code.
  std::array<float, TILE_KINDS> presence{};
  // Whether a leave inference belief drew the leave. The moments below are
  // over that leave's static value, in points, and 0 when it is not.
  bool modelled = false;
  float leave_value_mean = 0;
  float leave_value_sd = 0;
};

// The rack `leave` and a uniform refill from `unseen` less the leave make, or
// with no `leave` a uniform draw from `unseen`. `unseen` is the chooser's
// unseen pool, the bag plus the opponent's rack, and must exceed a full rack:
// the sims seat a full rack whenever the bag is not empty.
OppRackSummary summarize_opp_rack(const TileCounts& unseen, const LeaveSampler* leave);

}  // namespace scribblez::belief
