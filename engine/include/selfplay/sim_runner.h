#pragma once

// Monte-Carlo evaluation of candidate moves at one decision point, producing
// the per-candidate observations the sim-evidence loop consumes
// (docs/sim_residual_feedback.md): rollout counts, W/D/L tallies, final-score-
// delta moments, and per-square reply/outcome count maps mirroring the
// placement-mask training targets (training_targets.h).
//
// Common random numbers (CRN): rollout index i uses the same seed for every
// candidate. The unseen pool is a function of the pre-move board and the
// mover's full rack -- identical across candidates, since a candidate only
// moves tiles between that rack and the board -- so the opponent's sampled
// rack (drawn from the pool before the mover's replacement tiles) is
// identical across candidates for a given i. Shared rack luck cancels in
// candidate differences, making comparisons far sharper than independent
// sampling and making the observations a valid source of pairwise covariance
// estimates. Results are deterministic and independent of the thread count.

#include "game/bag.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <array>
#include <cstdint>
#include <vector>

namespace scribblez {

class Dictionary;
struct MoveRequest;  // agent.h

// The pre-move decision point candidates are simmed from: the board, both
// cumulative scores, and the mover's own rack. `opp_leave` carries the KNOWN
// part of the opponent's rack -- under the open-leaves information condition,
// the tiles they retained from their last move -- and every rollout starts
// the opponent from those tiles plus hidden replenishments drawn from the
// unseen pool (a full bag minus the board tiles, `rack`, and `opp_leave`).
// Empty means the whole rack is hidden and sampled. A full 7-tile `opp_leave`
// degenerates to a completely known rack, making the opponent's first reply
// to each candidate deterministic under the greedy rollout policy.
struct SimPosition {
  Board board;
  std::array<int, 2> scores{0, 0};
  int mover = 0;
  Rack rack;       // the mover's full pre-move rack
  Rack opp_leave;  // the known part of the opponent's rack; empty = all hidden
};

// Aggregate observations from `n` rollouts of one candidate, all from the
// mover's POV. The spatial planes are row-major 15x15 COUNTS (not
// frequencies, so the consumer can weigh them by sample size) with no
// symmetry flip, and mirror the placement-mask training targets: how often
// the opponent's reply / the mover's own next move placed a tile on the
// square, and how often it did so in a rollout that player went on to win.
// The layout is trivially copyable and packed; SimObsWriter/SimObsReader
// (sim_observation_log.h) serialize it verbatim.
struct SimObservation {
  static constexpr int kCells = BOARD_SIZE * BOARD_SIZE;

  uint32_t n = 0;
  uint32_t wins = 0;  // rollouts the mover won (draws are neither win nor loss)
  uint32_t draws = 0;
  uint32_t losses = 0;
  int64_t delta_sum = 0;     // sum over rollouts of (mover final - opp final)
  int64_t delta_sq_sum = 0;  // sum of squared deltas (yields the delta std)

  std::array<uint16_t, kCells> opp_next_count{};
  std::array<uint16_t, kCells> self_next_count{};
  std::array<uint16_t, kCells> opp_win_count{};
  std::array<uint16_t, kCells> self_win_count{};
};
static_assert(sizeof(SimObservation) == 32 + 4 * 2 * SimObservation::kCells,
              "SimObservation is serialized verbatim; its layout must stay packed");

class SimRunner {
 public:
  struct Params {
    int rollouts = 300;  // rollouts per candidate; capped at 65535 (u16 counts)
    int threads = 1;
  };

  SimRunner(const Dictionary& dict, const Params& params);

  // Sim every candidate (PLAY, EXCHANGE, or PASS) from `pos`. Rollout i of
  // every candidate is seeded by `base_seed + i` (the CRN scheme above; a
  // known opp_leave shrinks the shared randomness to the hidden draws), and
  // rollouts are HastyBot-vs-HastyBot played to a natural game end. Requires
  // a non-empty bag at the decision point -- the training-eligibility rule
  // (binary_log.h) -- so no candidate can end the game outright.
  std::vector<SimObservation> run(const SimPosition& pos, const std::vector<Move>& candidates,
                                  uint64_t base_seed) const;

 private:
  const Dictionary& dict_;
  Params params_;
};

// The unseen-tile pool from a player's POV: a full bag (with draw RNG seeded
// by `seed`) minus the tiles on the board and in the player's own rack.
Bag unseen_pool(const Board& board, const Rack& rack, uint64_t seed);

// The mover-POV candidate set at a decision point, ranked by HastyBot static
// equity (best first): every legal play and exchange, capped at `k`, or a
// lone PASS when nothing is legal. HastyEquity must be initialized. The
// opponent rack inside `req` should be empty mid-game (their tiles are hidden
// from the mover); it only influences equity near the endgame.
std::vector<Move> equity_top_k(const MoveRequest& req, int k);

}  // namespace scribblez
