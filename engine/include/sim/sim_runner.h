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
// Results are deterministic and independent of the thread count (per-rollout
// results are reduced in a fixed order; under value truncation this also
// leans on the eval service's contract that a row's outputs do not depend on
// its batch, trt_eval_service.h).
//
// Value truncation (docs/roadmap.md item 2): with horizon_plies set, a
// rollout plays that many plies and the position evaluation model's readout
// at the horizon -- the post-move pre-draw state of the horizon ply's
// mover, the sample kind the model trains on -- stands for everything
// after: the rollout contributes the model's outcome probabilities and
// predicted final delta (root-mover POV) instead of a terminal {0,1}
// outcome, with the leaf Gaussian's variance folded into the delta second
// moment. A rollout whose game ends before the horizon, or whose horizon
// ply falls inside the endgame (the model's domain is the pre-endgame
// prefix), contributes its exact terminal outcome instead. Identical
// candidates reach identical horizon states under CRN, so their
// observations still cancel exactly in differences.

#include "game/bag.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "nn/eval_service.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
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
  double delta_sum = 0;  // sum over rollouts of (mover final - opp final)
  // Sum over rollouts of the final delta's second moment: delta^2 for a
  // terminal rollout, mean^2 + sigma^2 of the leaf Gaussian for a truncated
  // one. The recovered variance is therefore PREDICTIVE in both
  // configurations (law of total variance: across-rollout spread of the
  // means plus the mean leaf variance), distinguishing "win by 103 exactly"
  // from "win by 103 +/- 39".
  double delta_sq_sum = 0;
  uint32_t n = 0;  // rollouts

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

// What one rollout contributes to a SimObservation, root-mover POV: the two
// moves the placement maps read (a missing move is a default Move -- PASS --
// which places nothing), plus the outcome distribution. A terminal rollout
// contributes its exact result ({0,1} probabilities, integer delta, delta_sq
// = delta^2); a truncated one the leaf model's outcome probabilities and its
// Gaussian's moments -- delta_sq = mean^2 + sigma^2, so the leaf's own
// predictive uncertainty ("win by 103 +/- 39", not "win by exactly 103")
// reaches the aggregated delta moments.
struct RolloutResult {
  Move opp_reply{};
  Move self_next{};
  double p_win = 0;
  double p_draw = 0;
  double p_loss = 0;
  double delta = 0;     // (predicted) mean of the final delta
  double delta_sq = 0;  // (predicted) second moment of the final delta
};

// One rollout worker's staging for horizon leaf evaluations: encoded rows
// are buffered, flushed through the (shared) leaf service in chunks, and the
// decoded scoring heads written back into the pending slots' results,
// flipped to the root mover's POV. Buffering amortizes the service round
// trip while holding at most kRows encoded rows (~80 KB each).
class LeafBatcher {
 public:
  static constexpr int kRows = 64;

  LeafBatcher(nn::PositionEvalService* service, const InputEncodingSpec& spec,
              std::vector<RolloutResult>* results)
      : service_(service),
        results_(results),
        row_floats_(input_floats(spec)),
        rows_(size_t(kRows) * row_floats_),
        wld_(size_t(kRows) * nn::WldOutput::kRowElems),
        sd_(size_t(kRows) * nn::ScoreDiffOutput::kRowElems) {}

  // The destination for the next pending leaf's row; add() commits it.
  float* next_row() { return rows_.data() + pending_.size() * row_floats_; }

  // `root_pov`: whether the horizon state was encoded from the root mover's
  // own POV (the horizon ply was theirs) rather than the opponent's.
  void add(size_t slot, bool root_pov);

  void flush();

 private:
  struct Pending {
    size_t slot;
    bool root_pov;
  };

  nn::PositionEvalService* service_;
  std::vector<RolloutResult>* results_;
  size_t row_floats_;
  std::vector<float> rows_;
  std::vector<float> wld_;
  std::vector<float> sd_;
  std::vector<Pending> pending_;
};

class SimRunner {
 public:
  // Rollouts per candidate are counted in u16 planes, so this bounds them.
  static constexpr int kMaxRollouts = 65535;

  // Below this horizon the last two input plies at a leaf encode could
  // predate the rollout, and -- the real bound -- a contingent draw has not
  // had its draw-then-play plies to resolve, leaving the sim nothing to
  // observe that the leaf model did not already know (docs/roadmap.md
  // item 2).
  static constexpr int kMinHorizonPlies = 3;

  struct Params {
    int rollouts = 300;  // per candidate; at most kMaxRollouts
    int threads = 1;
    // Value truncation: 0 rolls every rollout to a natural game end (no
    // leaf service); otherwise rollouts stop after this many plies (at
    // least kMinHorizonPlies) and `leaf_service` -- a served position
    // evaluation model -- scores the horizon. May be shared freely across
    // workers and runners: EvalService serializes concurrent evaluate()
    // calls itself. Non-owning; must outlive the runner.
    int horizon_plies = 0;
    nn::PositionEvalService* leaf_service = nullptr;
  };

  // Throws util::CleanException on params no SimRunner can honour. The
  // constructor calls it; an agent may call it earlier, to reject a bad flag
  // before spending seconds loading a model.
  static void validate(const Params& params);

  // Checks the truncation-flag pairing (a horizon iff a leaf) and the horizon
  // lower bound for one CLI surface that knows both, throwing
  // util::CleanException prefixed with `context` (the agent or tool name) on a
  // bad combination. validate() re-checks these against the built Params; a
  // surface calls this earlier -- before loading the leaf model -- so a bad
  // flag fails fast rather than after seconds of model building.
  static void validate_horizon(std::string_view context, int horizon_plies, bool have_leaf_service);

  // The horizon lower bound alone, for a surface whose leaf presence is
  // decided (and whose pairing is thus checked) elsewhere -- an agent whose
  // factory already ran validate_horizon, or a library whose caller owns the
  // pairing. Still worth checking early, to fail before a model load.
  static void validate_min_horizon(std::string_view context, int horizon_plies);

  SimRunner(const Dictionary& dict, const Params& params);

  // Rollout i of every candidate is seeded by `base_seed + i` (the scheme
  // above), and rollouts are HastyBot-vs-HastyBot to a natural game end --
  // or to the truncation horizon, when the params set one. Requires a
  // non-empty bag at the decision point -- the training-eligibility rule --
  // so no candidate can end the game outright.
  std::vector<SimObservation> run(const SimPosition& pos, const std::vector<Move>& candidates,
                                  uint64_t base_seed) const;

 private:
  const Dictionary& dict_;
  Params params_;
  // The leaf model's input encoding, derived from the service's declared arm
  // at construction; meaningful only under truncation.
  InputEncodingSpec leaf_spec_{};
};

// Fills a SimRunner::Params from a simming agent's shared truncation knobs:
// its base sim params, its horizon, and its leaf evaluator. The one place the
// three simming agents translate their own Params into the runner's, so they
// cannot drift on this mapping. `leaf` is passed through as given (validate()
// then rejects a horizon/leaf mismatch); a caller whose leaf exists
// regardless of truncation -- NeuralSimAgent's own served model -- passes null
// itself when its horizon is 0.
SimRunner::Params make_runner_params(SimRunner::Params sim, int horizon_plies,
                                     nn::PositionEvalService* leaf);

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
