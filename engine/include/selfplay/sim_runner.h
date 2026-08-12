#pragma once

// Monte-Carlo evaluation of candidate moves at one decision point, producing
// the per-candidate observations the sim-evidence loop consumes
// (docs/sim_residual_feedback.md).
//
// Common random numbers: rollout index i uses the same seed for every
// candidate. The unseen pool is a function of the pre-move board and the
// mover's full rack, identical across candidates since a candidate only moves
// tiles between the two, so for a given i the opponent's sampled rack is
// identical across candidates. Shared rack luck then cancels in candidate
// differences, sharpening comparisons far beyond independent sampling and
// making the observations a valid source of pairwise covariance estimates.
// Results are deterministic and independent of the thread count.

#include "game/bag.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {

class Dictionary;
struct MoveRequest;  // agent.h

// The pre-move decision point candidates are simmed from. `opp_leave` carries
// the KNOWN part of the opponent's rack -- under the open-leaves condition, the
// tiles they retained from their last move -- and every rollout starts them
// from those plus hidden replenishments drawn from the unseen pool. Empty means
// the whole rack is hidden and sampled; a full 7-tile leave degenerates to a
// completely known rack, making the opponent's first reply to each candidate
// deterministic under the greedy rollout policy.
struct SimPosition {
  Board board;
  std::array<int, 2> scores{0, 0};
  int mover = 0;
  Rack rack;       // the mover's full pre-move rack
  Rack opp_leave;  // the known part of the opponent's rack; empty = all hidden
};

// Aggregate observations from `n` rollouts of one candidate, all from the
// mover's POV. The spatial planes mirror the placement-mask training targets --
// how often the opponent's reply or the mover's own next move placed a tile on
// the square, and how often it did so in a rollout that player went on to win
// -- as row-major COUNTS rather than frequencies, so a consumer can weigh them
// by sample size. SimObsWriter/SimObsReader serialize the layout verbatim.
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

// Which simulated quantity ranks a candidate set: how often the rollouts were
// won, or the average final score differential they ended on.
enum class SimObjective { kWinRate, kSpread };

double sim_objective_value(const SimObservation& o, SimObjective objective);

// Index of the candidate `observations` rank highest under `objective`, ties
// going to the lower index -- so the caller's own candidate order (static
// equity for SimAgent, model rank for NeuralSimAgent) breaks them.
int best_observation_index(const std::vector<SimObservation>& observations, SimObjective objective);

// "winrate" or "spread"; anything else throws std::runtime_error naming `flag`
// as the offending option.
SimObjective parse_sim_objective(const std::string& name, const std::string& flag);

class SimRunner {
 public:
  // Rollouts per candidate are counted in u16 planes, so this bounds them.
  // Public because an agent validates its own --rollouts against it, to reject
  // the flag with a message rather than trip the constructor's assert.
  static constexpr int kMaxRollouts = 65535;

  struct Params {
    int rollouts = 300;  // per candidate; at most kMaxRollouts
    int threads = 1;
  };

  SimRunner(const Dictionary& dict, const Params& params);

  // Rollout i of every candidate is seeded by `base_seed + i` (the scheme
  // above), and rollouts are HastyBot-vs-HastyBot to a natural game end.
  // Requires a non-empty bag at the decision point -- the
  // training-eligibility rule -- so no candidate can end the game outright.
  std::vector<SimObservation> run(const SimPosition& pos, const std::vector<Move>& candidates,
                                  uint64_t base_seed) const;

 private:
  const Dictionary& dict_;
  Params params_;
};

// A full bag, its draw RNG seeded by `seed`, minus the tiles on the board and
// in the player's own rack.
Bag unseen_pool(const Board& board, const Rack& rack, uint64_t seed);

// Every legal play and exchange ranked by HastyBot static equity, best first
// and capped at `k`, or a lone PASS when nothing is legal. HastyEquity must be
// initialized. The opponent rack in `req` should be empty mid-game, their tiles
// being hidden; it only influences equity near the endgame.
std::vector<Move> equity_top_k(const MoveRequest& req, int k);

// `params`, checked against what SimRunner will accept, so a caller can reject
// a bad rollout count with a message naming `who`. SimRunner's constructor only
// ASSERTS the bound, which a Release build compiles out -- 0 rollouts then give
// every observation a 0/0 mean, whose NaN comparisons make best_observation_index
// return the first candidate every time, so the agent silently stops simulating.
// Returns `params` so a member-init list can validate on its way into SimRunner.
const SimRunner::Params& validated_sim_params(const SimRunner::Params& params,
                                              const std::string& who);

// The rollout position for an agent deciding `req`: the one place a turn is
// translated into a SimPosition, so the three simulating agents cannot drift on
// the seating convention or on what of the opponent's rack a rollout may seed
// from -- and so a new SimPosition field is filled for all of them at once.
SimPosition sim_position_from(const MoveRequest& req);

}  // namespace scribblez
