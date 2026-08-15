#pragma once

// Position-sampling arithmetic shared by the .slog sidecar generators
// (sim_obs_tool, move_set_eval_target_generator, evidence_trajectory_generator).
//
// The sampling is deterministic in (run seed, game, turn) alone -- never in
// thread count or visit order -- and every tool draws the SAME shuffle for
// equal seeds, so a tool sampling k positions per game visits a prefix of the
// positions a tool sampling k' > k (or all) visits. That subset relationship
// is load-bearing: it is what lets the target generator find a trajectory
// sidecar's positions among its own (roadmap item 4's forced-candidate
// labeling) without any cross-tool coordination beyond the seed.

#include "data/binary_log.h"

#include <compare>
#include <cstdint>
#include <vector>

namespace scribblez {
namespace binlog {

// One sampled decision point: (game, turn) within a .slog file. Ordering is
// the file's canonical position order.
struct GamePositionIndex {
  uint32_t game_idx;
  uint32_t turn_idx;

  auto operator<=>(const GamePositionIndex&) const = default;
};

// The deterministic per-position seed stream: drives a position's own
// randomness (stratum sampling, rollout seeds, trajectory draws) in every
// sidecar generator.
inline uint64_t position_seed(uint64_t run_seed, uint32_t game_idx, uint32_t turn_idx);

// Append `positions_per_game` of `gm`'s training-eligible turns to `out`,
// sampled without replacement (<= 0 takes every eligible turn, in order).
void sample_eligible_turns(const GameMetadata& gm, uint32_t game_idx, uint64_t run_seed,
                           int positions_per_game, std::vector<GamePositionIndex>* out);

// The number of turns sample_eligible_turns would append -- for sizing a
// progress bar before any work runs.
int count_eligible_sample(const GameMetadata& gm, int positions_per_game);

}  // namespace binlog
}  // namespace scribblez

#include "inlines/data/slog_sampling.inl"
