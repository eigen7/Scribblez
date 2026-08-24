#pragma once

// The position-evaluation-top-K agent (docs/roadmap.md, A4): exact
// per-candidate evaluation by the position evaluation model over a generous
// static-equity shortlist, then Monte-Carlo simulation of the model's top K.
// Once the bag empties the turn goes to the exact solver, as it does for every
// agent that plays the endgame properly.
//
// This agent is the reference the move set evaluation model is measured
// against: the model's job is to reproduce this agent's candidate ranking in
// one pass instead of one evaluation per candidate. It owns two of A4's
// measurements -- the sensitivity sweep that prices a recall miss in match-play
// terms (vary sim_top_k, or degrade the sim set with drop_best_prob), and the
// equal-budget baseline the learned filter must beat before it replaces exact
// evaluation anywhere.

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
  // turns endgame solving off, leaving the greedy static-equity move to play
  // the endgame out.
  struct Params {
    int thread_id = 0;
    std::string name;
    const Dictionary* dict = nullptr;
    // The static-equity shortlist the model evaluates exhaustively; 0 = every
    // legal move. Deliberately generous: its only job is to cap the blank
    // explosion (a two-blank rack's 20k moves are overwhelmingly redundant
    // designations), not to preempt the model's ranking.
    int shortlist = 50;
    int sim_top_k = 10;  // candidates simmed per turn, best by model rank
    EvalObjective rank_objective = EvalObjective::kWinProb;
    SimObjective sim_objective = SimObjective::kWinRate;
    // The A4 sensitivity knob: with this per-turn probability, the model's
    // top-ranked candidate is excluded from the sim set -- a controlled recall
    // miss, whose match-play cost prices the recall bar the move set
    // evaluation model has to clear. 0 plays the agent straight.
    double drop_best_prob = 0.0;
    // Rollouts per candidate, and their threading; see SimAgent::Params for
    // why 400. Sharing SimAgent's default keeps the equal-budget comparison
    // against it the configuration-free default. Leave the truncation fields
    // untouched here -- sim_horizon below is the one knob, and the agent's
    // own served model is the leaf evaluator.
    SimRunner::Params sim = {400, 1};
    // Value truncation (docs/roadmap.md item 2): 0 rolls out to a natural
    // end; otherwise rollouts stop after this many plies and the agent's own
    // position evaluation model scores the horizon.
    int sim_horizon = 0;
    uint64_t seed = 0;
    EndgameSolver::Params endgame = {};  // the solver's own defaults
  };

  using NetParams = nn::NeuralNetParams<nn::PositionEvaluationSpec>;

  NeuralSimAgent(const Params& params, const NetParams& net_params);

  // Takes an already-constructed evaluator (real or a scripted stub), loading
  // no model and touching no GPU. `max_batch` bounds one evaluate() call.
  NeuralSimAgent(const Params& params, std::unique_ptr<nn::PositionEvalService> service,
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

  // Whether this turn's sim set drops the model's top-ranked candidate: a
  // deterministic function of (seed, ply), so paired arms degrade identically.
  // Public for the same reproducibility reason as sim_seed().
  bool drop_best(int ply) const;

 private:
  // Throws on out-of-range scalar params. The constructors run it, and
  // from_spec additionally runs it BEFORE the production constructor, so a bad
  // flag fails fast instead of after the TensorRT engine build.
  static void validate(const Params& params);

  // Fills rank_ with indices into `candidates` in descending model-objective
  // order (ties keeping equity order), evaluating every candidate.
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
  // The agent's own model, serialized for the runner's sim threads.
  nn::SerializedEvalService<nn::PositionEvaluationSpec> leaf_;
  SimRunner runner_;
  EndgameTurnPolicy endgame_;
  int ply_ = 0;  // moves observed this game, by either seat

  // Scratch reused across turns to avoid per-move allocation.
  std::vector<int> rank_;
  std::vector<Move> sim_moves_;
};

}  // namespace scribblez
