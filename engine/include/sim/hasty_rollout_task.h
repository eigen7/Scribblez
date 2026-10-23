#pragma once

// The hasty-rollout task (gpu/rollout_abi.h) on the CPU, through HastyBot:
// the reference every device implementation is checked against, rollout by
// rollout, and the CPU throughput it is measured against.
//
// A batch holds groups, starts and tasks. A group is what a chooser's
// candidates at one position share (the unseen pool, the seed, the
// opponent's leave per rollout index); a start is one candidate applied to
// the board; a task is one rollout of one start. Rollout i of every start in
// a group is the same deal, so their outcomes are paired, and a task's
// outcome depends only on its start and rollout index, never on the batch
// around it.

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile_counts.h"
#include "gpu/rollout_abi.h"
#include "lexicon/dictionary.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace scribblez {

// See rollout_abi::GroupAbi.
struct RolloutGroup {
  TileCounts pool;  // everything the chooser could not see before moving
  uint64_t seed = 0;
  // Rollout i's opponent leave for i < size(); empty means the opponent's
  // whole rack is drawn.
  std::vector<Rack> opp_leaves;
};

// See rollout_abi::PositionAbi: we (the chooser) have just moved.
struct RolloutStart {
  Board board;    // natural frame, after the move
  Rack leave;     // ours
  Rack returned;  // the tiles an exchange put back in the bag
  int my_score = 0;
  int opp_score = 0;
  uint32_t group = 0;
};

struct RolloutBatch {
  std::vector<RolloutGroup> groups;
  std::vector<RolloutStart> starts;
  std::vector<rollout_abi::RolloutTask> tasks;

  // Tasks for rollouts [first, first + count) of start `start`.
  void add_rollouts(uint32_t start, uint32_t first, uint32_t count);
};

// The group of a chooser holding `rack` at `board`, before moving.
RolloutGroup make_rollout_group(const Board& board, const Rack& rack, uint64_t seed);

// The start after the chooser, holding `rack` at `board` with scores
// (`my_score`, `opp_score`), makes `move`. `board` should carry its movegen
// caches, which the start's copy then updates incrementally.
RolloutStart make_rollout_start(const Board& board, const Rack& rack, int my_score, int opp_score,
                                const Move& move, uint32_t group);

// The device's layouts. A start's cross-checks are built here if its board
// lacks them. A group's opponent leaves go to the leave array from
// `leave_base` on.
rollout_abi::PositionAbi to_abi(const RolloutStart& start, const Dictionary& dict);
rollout_abi::GroupAbi to_abi(const RolloutGroup& group, uint32_t leave_base);
rollout_abi::OppLeaveAbi to_abi(const Rack& leave);

rollout_abi::PackedMove pack_move(const Move& m);
Move unpack_move(const rollout_abi::PackedMove& p);

// "8H ACROSS word 32", "exch ABC", or "pass"; `board` as before the move.
std::string describe_move(const Move& m, const Board& board);

// The racks a task draws, in the ABI's order: the opponent's, then ours.
struct RolloutRacks {
  Rack opp;
  Rack mine;
};
RolloutRacks rollout_racks(const RolloutBatch& batch, const rollout_abi::RolloutTask& task);

// A rollout's end state as the leaf eval model reads it (rollout_abi's
// LeafState on the host): the board, each player's rack after its last move
// before its draw, and the scores. Player 0 is us.
struct CpuLeaf {
  Board board;
  std::array<Rack, 2> leaves;
  std::array<int, 2> scores;
};

// Runs every task of the batch on `threads` threads, calling `each(k, result,
// leaf)` for task k from the thread that ran it.
using RolloutVisitor =
  std::function<void(size_t, const rollout_abi::RolloutResult&, const CpuLeaf&)>;
void for_each_hasty_rollout_cpu(const RolloutBatch& batch, const Dictionary& dict, int threads,
                                const RolloutVisitor& each);

// The batch's tasks on `threads` threads, results[k] for tasks[k]. A non-null
// `leaves_out` receives the leaves of the first leaves_out->size() tasks.
std::vector<rollout_abi::RolloutResult> run_hasty_rollouts_cpu(
  const RolloutBatch& batch, const Dictionary& dict, int threads,
  std::vector<CpuLeaf>* leaves_out = nullptr);

}  // namespace scribblez
