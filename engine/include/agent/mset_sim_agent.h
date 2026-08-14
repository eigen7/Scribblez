#pragma once

// The move-set-evaluation agent (docs/roadmap.md, A4): the whole candidate set
// of a turn is scored in ONE model pass -- board trunk once, one cheap row per
// candidate -- and the model's top K are simulated, exactly as NeuralSimAgent
// simulates its own top K. Once the bag empties the turn goes to the exact
// solver, as it does for every agent that plays the endgame properly.
//
// NeuralSimAgent is the reference this agent is measured against: the two make
// the same decision from the same information and differ only in the cost of
// getting there (one position evaluation per candidate, versus one pass over
// the set). Because that cost no longer scales with the candidate count, the
// shortlist defaults to EVERY legal move -- the roadmap's whole objection to
// static-equity pre-filtering is that it decides, before the model is asked,
// which moves the model may consider. `--shortlist` exists to reimpose a cap
// for experiments, not because the agent needs one.
//
// The model reads a PRE-move board row (its candidates carry the move features
// that distinguish them), where the position evaluation model reads one
// post-move row per candidate. That is the training encoding on both sides.

#include "agent/agent.h"
#include "agent/candidate_evaluator.h"
#include "agent/endgame_turn_policy.h"
#include "encoding/game_state_encoder.h"
#include "endgame/endgame_solver.h"
#include "nn/eval_service.h"
#include "nn/neural_net.h"
#include "sim/sim_runner.h"
#include "training/move_set_encoder.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace scribblez {

class Dictionary;

class MsetSimAgent : public Agent {
 public:
  // `dict` is required and must outlive the agent. An `endgame` budget of 0
  // turns endgame solving off, leaving the greedy static-equity move to play
  // the endgame out.
  struct Params {
    int thread_id = 0;
    std::string name;
    const Dictionary* dict = nullptr;
    // The static-equity shortlist the model scores; 0 = every legal move, the
    // default (see the file comment on why this agent needs no cap).
    int shortlist = 0;
    int sim_top_k = 10;  // candidates simmed per turn, best by model rank
    EvalObjective rank_objective = EvalObjective::kWinProb;
    SimObjective sim_objective = SimObjective::kWinRate;
    // Rollouts per candidate, and their threading; see SimAgent::Params for
    // why 400. Sharing SimAgent's default keeps the equal-budget comparison
    // against it -- and against NeuralSimAgent -- the configuration-free
    // default.
    SimRunner::Params sim = {400, 1};
    uint64_t seed = 0;
    EndgameSolver::Params endgame = {};  // the solver's own defaults
  };

  MsetSimAgent(const Params& params,
               const nn::NeuralNetParams<nn::MoveSetEvaluationSpec>& net_params);

  // Takes an already-loaded service (real or a scripted stub), loading no model
  // and touching no GPU.
  MsetSimAgent(const Params& params, std::unique_ptr<nn::MoveSetEvalService> service);

  MoveDecision make_move(const MoveRequest& req) override;
  void begin_game(const BeginGameRequest& req) override;
  void observe_move(const Move& move) override;
  bool supports_parallelism() const override { return true; }

  // Build from `--player "--type=mset-sim [options]"` tokens, with --type and
  // --name already stripped. Requires --model=<path.onnx>. Throws
  // std::runtime_error on bad input.
  static std::unique_ptr<MsetSimAgent> from_spec(const std::vector<std::string>& tokens,
                                                 int thread_id, const std::string& name);

  static std::string options_help();

  // The seed SimRunner::run is given on the turn after `ply` moves have been
  // observed. Public so a test can reproduce a decision's rollouts exactly.
  uint64_t sim_seed(int ply) const;

  // The pre-move board row this agent hands the model for `req`, encoded
  // exactly as make_move() encodes it. Public so the row the model actually
  // sees can be checked against the training replay's row for the same
  // position -- the one drift the model itself could never reveal.
  void encode_board_row(const MoveRequest& req, float* dst) const;

 private:
  // Throws on out-of-range scalar params, the rollout ones included. from_spec
  // runs it BEFORE the production constructor, so a bad flag fails fast instead
  // of after the TensorRT engine build.
  static void validate(const Params& params);

  // Score `candidates` in one pass and fill rank_ with indices into them in
  // descending model-objective order. Ties keep equity order: the candidate
  // set arrives in static-equity rank, and a stable sort makes that the
  // agent's tie-break, so one seed gives one decision.
  void rank_candidates(const MoveRequest& req, const std::vector<Move>& candidates);

  int shortlist_;
  int sim_top_k_;
  EvalObjective rank_objective_;
  SimObjective sim_objective_;
  uint64_t seed_;
  std::unique_ptr<nn::MoveSetEvalService> service_;
  InputEncodingSpec spec_;
  GameStateEncoder encoder_;  // mirrors the live game, both seats' moves
  SimRunner runner_;
  EndgameTurnPolicy endgame_;
  int ply_ = 0;  // moves observed this game, by either seat

  // Scratch reused across turns to avoid per-move allocation.
  std::vector<float> board_row_;
  move_set::MoveFeatureArrays move_features_;
  std::vector<nn::Eval> evals_;
  std::vector<int> rank_;
  std::vector<Move> sim_moves_;
};

}  // namespace scribblez
