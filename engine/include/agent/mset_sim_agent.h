#pragma once

// The move-set agent (--type=mset-sim): the move set evaluation model scores a
// turn's whole candidate set in one pass (the board trunk once, one cheap row
// per candidate), the model's top K are simmed, and the best sim result plays.
// Bag-empty turns go to the endgame solver. It is UltimateBot without the
// evidence loop, and the baseline UltimateBot is measured against
// (docs/evaluation_plan.md).
//
// It makes NeuralSimAgent's decision from the same information, for one model
// pass instead of one position evaluation per candidate. Because that cost
// does not grow with the candidate count, the shortlist defaults to every
// legal move: static-equity pre-filtering would decide which moves the model
// may consider before the model is asked. --shortlist exists for experiments.
//
// The model reads one pre-move board row plus per-candidate move features,
// where the position evaluation model reads one post-move row per candidate;
// each is its model's training encoding.

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
    // The static-equity shortlist the model scores; 0 = every legal move.
    int shortlist = 0;
    int sim_top_k = 10;  // candidates simmed per turn, best by model rank
    EvalObjective rank_objective = EvalObjective::kWinProb;
    SimObjective sim_objective = SimObjective::kWinRate;
    // Rollouts per candidate, and their threading. SimAgent's default (see
    // there for why 400), so equal-budget comparisons need no configuration.
    SimRunner::Params sim = {400, 1};
    // Value truncation; see SimRunner::Params::horizon_plies for the full
    // semantics. The leaf service handed to the constructor scores the horizon.
    int sim_horizon = 0;
    uint64_t seed = 0;
    EndgameSolver::Params endgame = {};  // the solver's own defaults
  };

  using NetParams = nn::NeuralNetParams<nn::MoveSetEvaluationSpec>;

  MsetSimAgent(const Params& params, const NetParams& net_params);

  // Takes already-loaded services (real or scripted stubs). `leaf_service` is
  // the rollout leaf evaluator; give it iff params.sim_horizon is set.
  MsetSimAgent(const Params& params, std::unique_ptr<nn::MoveSetEvalService> service,
               std::shared_ptr<nn::PositionEvalService> leaf_service = nullptr);

  MoveDecision make_move(const MoveRequest& req) override;
  void begin_game(const BeginGameRequest& req) override;
  void observe_move(const Move& move) override;
  bool supports_parallelism() const override { return true; }

  // Build from `--player "--type=mset-sim [options]"` tokens, with --type and
  // --name already stripped. Requires --model=<path.onnx>. Throws
  // util::CleanException on bad input.
  static std::unique_ptr<MsetSimAgent> from_spec(const std::vector<std::string>& tokens,
                                                 int thread_id, const std::string& name);

  static std::string options_help();

  // The seed SimRunner::run is given on the turn after `ply` moves have been
  // observed. Public so a test can reproduce a decision's rollouts exactly.
  uint64_t sim_seed(int ply) const;

  // The pre-move board row make_move() hands the model for `req`. Public so a
  // test can check it against the training replay's row for the same
  // position, a drift the model itself could never reveal.
  void encode_board_row(const MoveRequest& req, float* dst) const;

 private:
  // Throws on out-of-range scalar params. from_spec runs it before loading the
  // model, so a bad flag fails before the TensorRT engine build.
  static void validate(const Params& params);

  // Score `candidates` in one pass and fill rank_ with their indices in
  // descending objective order. The stable sort keeps the candidates'
  // static-equity order among ties, so a seed determines the decision.
  void rank_candidates(const MoveRequest& req, const std::vector<Move>& candidates);

  // The rank objective read off scored candidate `i`'s head rows.
  float objective(int i) const;

  int shortlist_;
  int sim_top_k_;
  EvalObjective rank_objective_;
  SimObjective sim_objective_;
  uint64_t seed_;
  std::unique_ptr<nn::MoveSetEvalService> service_;
  InputEncodingSpec spec_;
  GameStateEncoder encoder_;  // mirrors the live game, both seats' moves
  std::shared_ptr<nn::PositionEvalService> leaf_service_;  // null = terminal sims
  SimRunner runner_;
  EndgameTurnPolicy endgame_;
  int ply_ = 0;  // moves observed this game, by either seat

  // Reused across turns to avoid per-move allocation.
  std::vector<float> board_row_;
  move_set::MoveFeatureArrays move_features_;
  std::vector<float> wld_buf_;
  std::vector<float> score_diff_buf_;
  std::vector<int> rank_;
  std::vector<Move> sim_moves_;
};

}  // namespace scribblez
