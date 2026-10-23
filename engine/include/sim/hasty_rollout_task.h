#pragma once

// The hasty-rollout task (gpu/rollout_abi.h) on the CPU, through HastyBot:
// the reference every device implementation is checked against, rollout by
// rollout, and the CPU throughput it is measured against.

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile_counts.h"
#include "gpu/rollout_abi.h"
#include "lexicon/dictionary.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {

// Where a batch of rollouts starts: we have just moved and hold `leave`; the
// opponent is to move. `pool` is everything we cannot see, the opponent's
// rack and the bag together.
struct HastyRolloutPosition {
  Board board;  // natural frame
  Rack leave;
  int my_score = 0;
  int opp_score = 0;
  TileCounts pool;
  uint64_t seed = 0;
};

HastyRolloutPosition make_rollout_position(const Board& board, const Rack& leave, int my_score,
                                           int opp_score, uint64_t seed);

// The position flattened for the device, cross-checks included (built here
// if the board lacks them).
rollout_abi::PositionAbi to_abi(const HastyRolloutPosition& pos, const Dictionary& dict);

rollout_abi::PackedMove pack_move(const Move& m);
Move unpack_move(const rollout_abi::PackedMove& p);

// "8H ACROSS word 32", "exch ABC", or "pass"; `board` as before the move.
std::string describe_move(const Move& m, const Board& board);

// The racks rollout `i` draws, in the ABI's order: the opponent's, then ours.
struct RolloutRacks {
  Rack opp;
  Rack mine;
};
RolloutRacks rollout_racks(const HastyRolloutPosition& pos, uint32_t rollout);

// A rollout's end state as the leaf eval model reads it (rollout_abi's
// LeafState on the host): the board, each player's rack after its last move
// before its draw, and the scores. Player 0 is us.
struct CpuLeaf {
  Board board;
  std::array<Rack, 2> leaves;
  std::array<int, 2> scores;
};

// `n` rollouts of `pos` on `threads` threads, rollout i under seed-derived
// draw i on whichever thread runs it. A non-null `leaves_out` receives the
// leaves of the first leaves_out->size() rollouts.
std::vector<rollout_abi::RolloutResult> run_hasty_rollouts_cpu(
  const HastyRolloutPosition& pos, const Dictionary& dict, int n, int threads,
  std::vector<CpuLeaf>* leaves_out = nullptr);

}  // namespace scribblez
