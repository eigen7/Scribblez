#pragma once

// Monte-Carlo simulation of candidate moves at one decision point: the
// simming agents' move choice, and the per-candidate observations the
// sim-evidence loop consumes (docs/plans/sim_residual_feedback.md).
//
// Common random numbers (CRN): rollout i of every candidate uses the same
// seed, and the unseen pool depends only on the pre-move board and the
// mover's full rack, so rollout i deals the opponent the same rack for every
// candidate. Rack luck then cancels in candidate differences, which sharpens
// comparisons well beyond independent sampling.
//
// Results are deterministic and independent of the thread count. Per-rollout
// results are reduced in a fixed order, and under value truncation the leaf
// service guarantees that a row's outputs do not depend on its batch
// (trt_eval_service.h).
//
// Value truncation (docs/roadmap.md item 2): with horizon_plies set, a rollout
// stops after that many plies, and the position-evaluation model's readout at
// the horizon stands in for the rest of the game. The rollout then contributes
// the model's outcome probabilities and predicted final delta instead of a
// terminal outcome. A rollout whose game ends before the horizon, or whose
// horizon falls in the endgame (outside the model's domain), contributes its
// exact terminal outcome instead.

#include "game/bag.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "nn/eval_service.h"
#include "training/footprint.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace scribblez {

class Dictionary;
struct MoveRequest;  // agent.h

// The pre-move decision point candidates are simmed from. `opp_leave` is the
// known part of the opponent's rack: under face-up leaves, the tiles they kept
// from their last move. Every rollout seats the opponent with it and draws the
// rest of their rack from the unseen pool. Empty means the whole rack is
// hidden.
struct SimPosition {
  Board board;
  std::array<int, 2> scores{0, 0};
  int mover = 0;
  Rack rack;  // the mover's full pre-move rack
  Rack opp_leave;
};

// Aggregate observations from `n` rollouts of one candidate, from the mover's
// point of view. Everything is a sum rather than a mean, so a consumer can
// weigh observations by sample size; consumers normalize by `n`.
// SimObsWriter/SimObsReader serialize the layout verbatim.
//
// Under value truncation the outcome sums and the `*_win_count` histograms are
// fractional, since they accumulate the leaf model's probabilities. The
// `*_next_count` histograms stay integral: the moves they record are always
// played, never predicted.
//
// The placement histograms are over footprint classes (training/footprint.h):
// each rollout credits the opponent's reply and the mover's next move to one
// class each. `*_win_count` weights that credit by the probability that the
// move's player won. Stored dense; the anchored classes reshape to
// (15, 15, slots).
struct SimObservation {
  static constexpr int kClasses = kFootprintClasses;

  // Doubles first, so the layout carries no alignment padding to serialize.
  double wins = 0;
  double draws = 0;
  double losses = 0;
  double delta_sum = 0;  // final delta: mover's score minus the opponent's
  // Sum of the final delta's second moment (see RolloutResult::delta_sq). The
  // variance recovered from it is predictive in both configurations: by the
  // law of total variance, the spread of the rollout means plus the mean leaf
  // variance.
  double delta_sq_sum = 0;
  uint32_t n = 0;  // rollouts

  std::array<uint16_t, kClasses> opp_next_count{};
  std::array<uint16_t, kClasses> self_next_count{};
  std::array<float, kClasses> opp_win_count{};
  std::array<float, kClasses> self_win_count{};
};
static_assert(sizeof(SimObservation) == 44 + (2 + 2 + 4 + 4) * SimObservation::kClasses,
              "SimObservation is serialized verbatim; its layout must stay packed");

// What ranks simulated candidates: win rate (draws count half) or mean final
// spread.
enum class SimObjective { kWinRate, kSpread };

double sim_objective_value(const SimObservation& o, SimObjective objective);

// Index of the best observation under `objective`. Ties go to the lower index,
// so the caller's own candidate order breaks them.
int best_observation_index(const std::vector<SimObservation>& observations, SimObjective objective);

// Parses "winrate" or "spread"; anything else throws util::CleanException
// naming `flag`.
SimObjective parse_sim_objective(const std::string& name, const std::string& flag);

// One rollout's outcome, from the root mover's point of view. A terminal
// rollout has 0/1 probabilities and an exact delta; a truncated one carries
// the leaf model's outcome probabilities and final-delta Gaussian.
struct RolloutResult {
  // The two moves the placement histograms read. A missing move is a default
  // Move (PASS), which places nothing.
  Move opp_reply{};
  Move self_next{};
  double p_win = 0;
  double p_draw = 0;
  double p_loss = 0;
  double delta = 0;  // mean of the final delta
  // Second moment of the final delta: delta^2 when terminal, mean^2 + sigma^2
  // when truncated, so the leaf's own uncertainty ("win by 103 +/- 39", not
  // "by exactly 103") reaches the aggregated moments.
  double delta_sq = 0;
  // The tile values each side was left holding at the end of a finished game
  // (0 for the side that played out; both 0 for a truncated rollout). Already
  // counted in `delta`; kept apart so an analysis can see how much of a
  // candidate's margin comes from an opponent stuck with the Q.
  int self_stranded = 0;
  int opp_stranded = 0;
  // Whether each side passed at any point after the candidate: typically a
  // side with no play left, stuck with the Q.
  bool self_passed = false;
  bool opp_passed = false;
};

// How much the end-of-game rack settlement moved the final delta, from the
// root mover's point of view: twice the other side's tiles to whoever played
// out, or each side docked its own when nobody did.
int end_rack_swing(const RolloutResult& r);

// Fold one rollout into the candidate's observation.
void accumulate_rollout(const RolloutResult& o, SimObservation* obs);

// One rollout worker's buffer of horizon leaves awaiting evaluation. Rows are
// flushed through the shared leaf service kRows at a time, and the readouts
// are written into the pending rollouts' results in the root mover's point of
// view. Batching amortizes the service round trip; each row is ~80 KB.
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

  // With a shorter horizon, a draw that depends on the candidate would not yet
  // have been drawn and played, so the sim would observe nothing the leaf
  // model did not already know (docs/roadmap.md item 2). It also guarantees
  // that the leaf's last-move input planes all come from the rollout.
  static constexpr int kMinHorizonPlies = 3;

  struct Params {
    int rollouts = 300;  // per candidate; at most kMaxRollouts
    int threads = 1;
    // Value truncation: 0 plays every rollout to the end of the game with no
    // leaf service. Otherwise rollouts stop after this many plies (at least
    // kMinHorizonPlies) and `leaf_service`, a served position-evaluation
    // model, scores the horizon. The service is non-owning, must outlive the
    // runner, and may be shared across runners: it serializes concurrent
    // evaluate() calls itself.
    int horizon_plies = 0;
    nn::PositionEvalService* leaf_service = nullptr;
    // Rollouts are HastyBot vs HastyBot. With this set, both sides hand the
    // endgame to the solver (EndgameHastyBotAgent) instead of playing it
    // greedily. Greedy endgames can misjudge a late-game candidate by tens of
    // percentage points of win rate; solving costs several times as much per
    // rollout.
    bool solve_endgames = false;
  };

  // Throws util::CleanException on invalid params. The constructor calls it;
  // an agent may call it earlier, to reject a bad flag before spending seconds
  // loading a model.
  static void validate(const Params& params);

  // The horizon checks of validate(), for a CLI surface to run before it loads
  // the leaf model: a horizon comes with a leaf model and vice versa, and the
  // horizon respects kMinHorizonPlies. Errors are prefixed with `context`, the
  // agent or tool name.
  static void validate_horizon(std::string_view context, int horizon_plies, bool have_leaf_service);

  // The kMinHorizonPlies check alone, for a surface whose horizon/leaf pairing
  // is checked elsewhere.
  static void validate_min_horizon(std::string_view context, int horizon_plies);

  SimRunner(const Dictionary& dict, const Params& params);

  // Rollout i of every candidate is seeded by `base_seed + i`. Requires a
  // non-empty bag at the decision point, so no candidate can end the game.
  std::vector<SimObservation> run(const SimPosition& pos, const std::vector<Move>& candidates,
                                  uint64_t base_seed) const;

  // The rollouts behind run(), unreduced, for a consumer that reduces them
  // differently (sim/rollout_summary.h). Candidate c's rollout i is at
  // [c * rollouts + i].
  std::vector<RolloutResult> run_rollouts(const SimPosition& pos,
                                          const std::vector<Move>& candidates,
                                          uint64_t base_seed) const;
  // As above with `rollouts` in place of the params' count, so a caller can sim
  // in instalments: rollouts [a, b) are run_rollouts(..., base_seed + a, b - a).
  std::vector<RolloutResult> run_rollouts(const SimPosition& pos,
                                          const std::vector<Move>& candidates, uint64_t base_seed,
                                          int rollouts) const;
  int rollouts() const { return params_.rollouts; }

 private:
  const Dictionary& dict_;
  Params params_;
  // The leaf model's input encoding; meaningful only under truncation.
  InputEncodingSpec leaf_spec_{};
};

// A simming agent's runner params: its base sim params plus its truncation
// horizon and leaf service. Shared so the simming agents cannot drift on this
// mapping. `leaf` passes through as given, and validate() rejects a
// horizon/leaf mismatch, so an agent whose model exists regardless of
// truncation must pass null when its horizon is 0.
SimRunner::Params make_runner_params(SimRunner::Params sim, int horizon_plies,
                                     nn::PositionEvalService* leaf);

// A full bag, its draw RNG seeded by `seed`, minus the tiles on the board and
// in the player's own rack.
Bag unseen_pool(const Board& board, const Rack& rack, uint64_t seed);

// The top `k` legal plays and exchanges by HastyBot static equity, best first,
// or a lone PASS when nothing is legal. HastyEquity must be initialized.
// Throws util::Exception on k < 1; for no cap, pass INT_MAX.
std::vector<Move> equity_top_k(const MoveRequest& req, int k);

// The rollout position for an agent deciding `req`. Shared so the simming
// agents cannot drift on the seating convention or on which of the opponent's
// tiles a rollout may treat as known.
SimPosition sim_position_from(const MoveRequest& req);

}  // namespace scribblez
