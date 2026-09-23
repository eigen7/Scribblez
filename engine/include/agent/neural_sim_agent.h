#pragma once

// The position-evaluation-top-K agent (--type=neural-sim): the position
// evaluation model evaluates every move on a generous static-equity
// shortlist, the model's top K are simmed, and the best sim result plays.
// Bag-empty turns go to the endgame solver.
//
// This is the reference for the move set evaluation model, whose job is to
// reproduce this ranking in one pass instead of one evaluation per candidate
// (MsetSimAgent). drop_best_prob exists for the sensitivity sweep of
// docs/evaluation_plan.md, which prices a recall miss in match play.

#include "agent/agent.h"
#include "agent/candidate_evaluator.h"
#include "agent/endgame_turn_policy.h"
#include "endgame/endgame_solver.h"
#include "nn/eval_service.h"
#include "nn/neural_net.h"
#include "sim/sim_runner.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace scribblez {

class Dictionary;

class NeuralSimAgent : public Agent {
 public:
  // `dict` is required and must outlive the agent. An `endgame` budget of 0
  // turns solving off, leaving the static-equity move to play the endgame.
  struct Params {
    int thread_id = 0;
    std::string name;
    const Dictionary* dict = nullptr;
    // The static-equity shortlist the model evaluates; 0 = every legal move.
    // Deliberately generous: it exists to cap blank-heavy racks (a two-blank
    // rack's ~20k moves are mostly redundant blank designations), not to
    // preempt the model's ranking.
    int shortlist = 50;
    int sim_top_k = 10;  // candidates simmed per turn, best by model rank
    EvalObjective rank_objective = EvalObjective::kWinProb;
    SimObjective sim_objective = SimObjective::kWinRate;
    // Per-turn probability of excluding the model's top-ranked candidate from
    // the sim set: a controlled recall miss. 0 plays the agent straight.
    double drop_best_prob = 0.0;
    // Rollouts per candidate, and their threading. SimAgent's default (see
    // there for why 400), so equal-budget comparisons need no configuration.
    // Leave the truncation fields alone; sim_horizon is the knob.
    SimRunner::Params sim = {400, 1};
    // Value truncation; see SimRunner::Params::horizon_plies. The agent's own
    // model scores the horizon.
    int sim_horizon = 0;
    uint64_t seed = 0;
    EndgameSolver::Params endgame = {};  // the solver's own defaults
  };

  using NetParams = nn::NeuralNetParams<nn::PositionEvaluationSpec>;

  // See CandidateEvaluator's constructor for `service` and `max_batch`. The
  // service doubles as the rollout leaf evaluator.
  NeuralSimAgent(const Params& params, std::shared_ptr<nn::PositionEvalService> service,
                 int max_batch = 256);

  MoveDecision make_move(const MoveRequest& req) override;
  void begin_game(const BeginGameRequest& req) override;
  void observe_move(const Move& move) override;
  bool supports_parallelism() const override { return true; }

  // Build from `--player "--type=neural-sim [options]"` tokens, with --type
  // and --name already stripped. Requires --model=<path.onnx>. Throws
  // util::CleanException on bad input.
  static std::unique_ptr<NeuralSimAgent> from_spec(const std::vector<std::string>& tokens,
                                                   int thread_id, const std::string& name);

  static std::string options_help();

  // The seed SimRunner::run is given on the turn after `ply` moves have been
  // observed. Public so a test can reproduce a decision's rollouts exactly.
  uint64_t sim_seed(int ply) const;

  // Whether this turn's sim set drops the model's top-ranked candidate. A
  // deterministic function of (seed, ply), so paired arms degrade
  // identically. Public for the same reason as sim_seed().
  bool drop_best(int ply) const;

 private:
  // Throws on out-of-range scalar params. from_spec also runs it before
  // loading the model, so a bad flag fails before the TensorRT engine build.
  static void validate(const Params& params);

  // Evaluate every candidate and fill rank_ with their indices in descending
  // objective order, ties keeping static-equity order.
  void rank_candidates(const MoveRequest& req, const std::vector<Move>& candidates);

  // The rank objective read off evaluated candidate `i`'s head rows.
  float objective(int i) const;

  int shortlist_;
  int sim_top_k_;
  EvalObjective rank_objective_;
  SimObjective sim_objective_;
  double drop_best_prob_;
  uint64_t seed_;
  CandidateEvaluator evaluator_;
  SimRunner runner_;
  EndgameTurnPolicy endgame_;
  int ply_ = 0;  // moves observed this game, by either seat

  // Reused across turns to avoid per-move allocation.
  std::vector<int> rank_;
  std::vector<Move> sim_moves_;
};

}  // namespace scribblez
