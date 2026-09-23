#pragma once

// UltimateBot (--type=ultimatebot; docs/roadmap.md item 6), the destination
// agent: the move proposal model driving a sequential sim loop
// (evidence_loop.h). Each turn it sims the greedy anchor, then repeatedly
// sims the candidate the model, conditioned on every sim so far, picks as
// most likely to prove best. It stops when the sim budget is spent or no
// candidate's predicted gain clears the threshold, and plays the simmed
// candidate with the best win rate. Bag-empty turns go to the endgame solver.
//
// Its baseline is MsetSimAgent (docs/evaluation_plan.md), with which it shares
// candidate generation, encoding, and the endgame handoff. MsetSimAgent picks
// all K sims from one unconditioned pass; this agent picks each sim from a
// pass conditioned on the previous ones, and may stop early.

#include "agent/agent.h"
#include "agent/candidate_evaluator.h"
#include "agent/endgame_turn_policy.h"
#include "agent/evidence_loop.h"
#include "agent/move_proposal_service.h"
#include "encoding/game_state_encoder.h"
#include "endgame/endgame_solver.h"
#include "nn/eval_service.h"
#include "sim/sim_runner.h"
#include "training/move_set_encoder.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace scribblez {

class Dictionary;

class UltimateBotAgent : public Agent {
 public:
  // `dict` is required and must outlive the agent. An `endgame` budget of 0
  // turns solving off, leaving the static-equity move to play the endgame.
  struct Params {
    int thread_id = 0;
    std::string name;
    const Dictionary* dict = nullptr;
    // The sim budget per turn, the anchor included; 1 plays the anchor
    // unsimmed. At most nn::kMaxEvidence. The default matches MsetSimAgent's
    // sim_top_k, so equal-budget comparisons need no configuration.
    int max_sims = 10;
    // Early stopping: no further sim once every unsimmed candidate's predicted
    // gain (in win probability) is below this. 0 never stops early.
    float gain_threshold = 0.0f;
    // Rollouts per candidate, and their threading; MsetSimAgent's defaults.
    SimRunner::Params sim = {400, 1};
    // Value truncation; see SimRunner::Params::horizon_plies. The leaf service
    // handed to the constructor scores the horizon.
    int sim_horizon = 0;
    uint64_t seed = 0;
    EndgameSolver::Params endgame = {};  // the solver's own defaults
  };

  // Takes an already-loaded service (a MoveProposalSession, or a scripted
  // stub). `leaf_service` is the rollout leaf evaluator; give it iff
  // params.sim_horizon is set.
  UltimateBotAgent(const Params& params, std::unique_ptr<agent::MoveProposalService> service,
                   std::shared_ptr<nn::PositionEvalService> leaf_service = nullptr);

  MoveDecision make_move(const MoveRequest& req) override;
  void begin_game(const BeginGameRequest& req) override;
  void observe_move(const Move& move) override;

  // Build from `--player "--type=ultimatebot [options]"` tokens, with --type
  // and --name already stripped. Requires --cache-model and --step-model.
  // Throws util::CleanException on bad input.
  static std::unique_ptr<UltimateBotAgent> from_spec(const std::vector<std::string>& tokens,
                                                     int thread_id, const std::string& name);

  static std::string options_help();

  // The seed every sim of the turn after `ply` observed moves is given, so
  // those sims pair. Public so a test can reproduce a decision's rollouts.
  uint64_t sim_seed(int ply) const;

  // The pre-move board row make_move() hands the model for `req`. Public so a
  // test can check it against the training replay's row for the same
  // position, a drift the model itself could never reveal.
  void encode_board_row(const MoveRequest& req, float* dst) const;

 private:
  // Throws on out-of-range scalar params. from_spec runs it before loading the
  // model, so a bad flag fails before the TensorRT engine build.
  static void validate(const Params& params);

  // Encode the board row and the whole candidate set, and run the model's
  // evidence-free pass over them.
  void encode_candidates(const MoveRequest& req, const std::vector<Move>& candidates);

  int max_sims_;
  agent::ArgmaxGainPolicy policy_;
  uint64_t seed_;
  std::unique_ptr<agent::MoveProposalService> service_;
  InputEncodingSpec spec_;
  GameStateEncoder encoder_;  // mirrors the live game, both seats' moves
  std::shared_ptr<nn::PositionEvalService> leaf_service_;  // null = terminal sims
  SimRunner runner_;
  EndgameTurnPolicy endgame_;
  int ply_ = 0;  // moves observed this game, by either seat

  // Reused across turns to avoid per-move allocation.
  std::vector<float> board_row_;
  move_set::MoveFeatureArrays move_features_;
};

}  // namespace scribblez
