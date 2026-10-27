#pragma once

// Checks of the GPU rollout kernel (gpu/hasty_rollouts_gpu.h) against its CPU
// reference (sim/hasty_rollout_task.h), shared by gpu_kernel_tests and
// gpu_rollout_demo: a position reached by HastyBot self-play, a batch over the
// mover's candidates whose rollouts seat known opponent leaves of every length,
// and task-by-task comparisons of the results and of the leaf rows the device
// encodes.

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile_counts.h"
#include "gpu/rollout_abi.h"
#include "sim/hasty_rollout_task.h"

#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {

class Dictionary;
class GpuHastyRollouts;

// The decision point after `plies` plies of HastyBot self-play: the board,
// the mover's rack and the scores from the mover's side.
struct DecisionPoint {
  Board board;  // with its move-generation caches
  Rack rack;
  int my_score = 0, opp_score = 0;
};

// Throws std::runtime_error if the game ends before `plies` plies.
DecisionPoint self_play_position(const Dictionary& dict, uint64_t seed, int plies);

// The mover's top `count` moves by static equity, plus the best exchange if
// the cut has none.
std::vector<Move> rollout_candidates(const DecisionPoint& at, const Dictionary& dict, int count);

// Rollout i seats a known opponent leave of i % (RACK_SIZE + 1) tiles drawn
// from `pool`, a full rack included; with none, the opponent's whole rack is
// drawn. Leaves stop short of the pool's last RACK_SIZE tiles.
std::vector<Rack> known_leaves(const TileCounts& pool, int rollouts, uint64_t seed);

// One group over `moves`, `rollouts` each, seating known_leaves.
RolloutBatch candidate_batch(const DecisionPoint& at, const std::vector<Move>& moves, int rollouts,
                             uint64_t seed);

bool same_move(const rollout_abi::PackedMove& a, const rollout_abi::PackedMove& b);

// Task k's racks and both results ply by ply, the differing plies marked.
std::string describe_mismatch(const RolloutBatch& batch, size_t k,
                              const rollout_abi::RolloutResult& cpu,
                              const rollout_abi::RolloutResult& gpu);

// The CPU encoder's leaf eval rows for `leaves`, kInputFloats each.
std::vector<float> cpu_leaf_rows(const std::vector<CpuLeaf>& leaves, const Dictionary& dict);

// The device encoder's rows for the leaves of the last run's tasks [0, n),
// `batch` per call, laid out as the CPU writes them.
std::vector<float> device_leaf_rows(GpuHastyRollouts& gpu, int n, int batch);

struct RowCheck {
  int checked = 0, rows_identical = 0;
  float max_abs_diff = 0;
  std::string first_diffs;  // up to 8 elements of the first differing row
};

RowCheck compare_leaf_rows(const std::vector<float>& cpu, const std::vector<float>& gpu);

}  // namespace scribblez
