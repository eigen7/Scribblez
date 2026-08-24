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
//
// A terminal rollout contributes a {0,1} outcome and an integer delta; a
// value-truncated rollout (docs/roadmap.md item 2) contributes the leaf
// model's outcome probabilities and predicted final delta, so the outcome
// accumulators and the win-conjoined planes are fractional. The next-move
// planes stay integer counts: the plies they read are always simmed, never
// predicted. Every consumer normalizes by `n` -- the rollout count, integral
// in both configurations -- so the frequency semantics are unchanged.
struct SimObservation {
  static constexpr int kCells = BOARD_SIZE * BOARD_SIZE;

  // Doubles first, so the layout carries no alignment padding to serialize.
  double wins = 0;  // outcome weight for the mover winning (draws are neither)
  double draws = 0;
  double losses = 0;
  double delta_sum = 0;     // sum over rollouts of (mover final - opp final)
  double delta_sq_sum = 0;  // sum of squared deltas (yields the delta std)
  uint32_t n = 0;           // rollouts

  std::array<uint16_t, kCells> opp_next_count{};
  std::array<uint16_t, kCells> self_next_count{};
  std::array<float, kCells> opp_win_count{};
  std::array<float, kCells> self_win_count{};
};
static_assert(sizeof(SimObservation) == 44 + (2 + 2 + 4 + 4) * SimObservation::kCells,
              "SimObservation is serialized verbatim; its layout must stay packed");

// Which simulated quantity ranks a candidate set: how often the rollouts were
// won, or the average final score differential they ended on.
enum class SimObjective { kWinRate, kSpread };

double sim_objective_value(const SimObservation& o, SimObjective objective);

// Index of the candidate `observations` rank highest under `objective`, ties
// going to the lower index -- so the caller's own candidate order (static
// equity for SimAgent, model rank for NeuralSimAgent) breaks them.
int best_observation_index(const std::vector<SimObservation>& observations, SimObjective objective);

// "winrate" or "spread"; anything else throws util::CleanException naming
// `flag` as the offending option.
SimObjective parse_sim_objective(const std::string& name, const std::string& flag);

class SimRunner {
 public:
  // Rollouts per candidate are counted in u16 planes, so this bounds them.
  static constexpr int kMaxRollouts = 65535;

  struct Params {
    int rollouts = 300;  // per candidate; at most kMaxRollouts
    int threads = 1;
  };

  // Throws util::CleanException on params no SimRunner can honour. The
  // constructor calls it; an agent may call it earlier, to reject a bad flag
  // before spending seconds loading a model.
  static void validate(const Params& params);

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
//
// Throws util::Exception on k < 1. "All moves" is spelled as a large cap,
// not as 0: a caller wanting no cap passes INT_MAX.
std::vector<Move> equity_top_k(const MoveRequest& req, int k);

// The rollout position for an agent deciding `req`: the one place a turn is
// translated into a SimPosition, so the three simulating agents cannot drift on
// the seating convention or on what of the opponent's rack a rollout may seed
// from -- and so a new SimPosition field is filled for all of them at once.
SimPosition sim_position_from(const MoveRequest& req);

}  // namespace scribblez
