#pragma once

// UltimateBot (docs/roadmap.md, item 6): the destination agent -- the move
// proposal model at the root of a sequential sim loop. Every turn: the greedy
// anchor (the highest-raw-score candidate) is simmed first, then the model,
// conditioned on every sim so far, re-scores the whole candidate set and the
// proves-best argmax is simmed next, until the sim budget is spent or no
// unsimmed candidate's predicted gain clears the stopping threshold; the best
// simmed candidate by simulation value plays. Once the bag empties the turn
// goes to the exact solver, as for every agent that plays the endgame
// properly.
//
// MsetSimAgent is the reference this agent is measured against (the same
// stack with the evidence loop removed, docs/evaluation_plan.md): both score
// the whole candidate set in one model pass and both sim K candidates under
// common random numbers, but that agent picks its K off the plain pass at once
// where this one picks each sim off a pass conditioned on the previous ones --
// and may stop early. Candidate generation, encoding, and the endgame handoff
// are theirs in common; the loop itself is agent/evidence_loop.h, shared with
// the conditioned trajectory generator to come.
//
// The model reads a PRE-move board row (its candidates carry the move features
// that distinguish them) -- the training encoding.

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
  // turns endgame solving off, leaving the greedy static-equity move to play
  // the endgame out.
  struct Params {
    int thread_id = 0;
    std::string name;
    const Dictionary* dict = nullptr;
    // The sim budget per turn, the anchor included; 1 plays the anchor
    // unsimmed. Bounded above by the padded evidence width nn::kMaxEvidence.
    // 10 matches
    // MsetSimAgent's --sim-top-k, so the equal-budget comparison against it
    // is the configuration-free default.
    int max_sims = 10;
    // Early stopping: no further sim once every unsimmed candidate's predicted
    // gain is below this, in win-probability units (the gain head's). 0 never
    // stops early; the budget curve of docs/evaluation_plan.md sweeps it.
    float gain_threshold = 0.0f;
    // Rollouts per candidate, and their threading; MsetSimAgent's defaults,
    // for the equal-budget comparison.
    SimRunner::Params sim = {400, 1};
    // Value truncation; see SimRunner::Params::horizon_plies for the full
    // semantics. The leaf service handed to the constructor scores the horizon.
    int sim_horizon = 0;
    uint64_t seed = 0;
    EndgameSolver::Params endgame = {};  // the solver's own defaults
  };

  // Takes an already-loaded service (a MoveProposalSession over the run's
  // shared nets, or a scripted stub), loading no model and touching no GPU.
  // `leaf_service` is the value-truncation leaf evaluator (the run's shared
  // service); give it iff params.sim_horizon is set.
  UltimateBotAgent(const Params& params, std::unique_ptr<agent::MoveProposalService> service,
                   std::shared_ptr<nn::PositionEvalService> leaf_service = nullptr);

  MoveDecision make_move(const MoveRequest& req) override;
  void begin_game(const BeginGameRequest& req) override;
  void observe_move(const Move& move) override;
  bool supports_parallelism() const override { return true; }

  // Build from `--player "--type=ultimatebot [options]"` tokens, with --type
  // and --name already stripped. Requires --cache-model and --step-model.
  // Throws util::CleanException on bad input.
  static std::unique_ptr<UltimateBotAgent> from_spec(const std::vector<std::string>& tokens,
                                                     int thread_id, const std::string& name);

  static std::string options_help();

  // The seed SimRunner::run is given on the turn after `ply` moves have been
  // observed -- every sim of that turn, so they pair. Public so a test can
  // reproduce a decision's rollouts exactly.
  uint64_t sim_seed(int ply) const;

  // The pre-move board row this agent hands the model for `req`, encoded
  // exactly as make_move() encodes it. Public so the row the model actually
  // sees can be checked against the training replay's row for the same
  // position -- the one drift the model itself could never reveal.
  void encode_board_row(const MoveRequest& req, float* dst) const;

 private:
  // Throws on out-of-range scalar params, the rollout ones and the evidence
  // width bound included. from_spec runs it BEFORE the production constructor,
  // so a bad flag fails fast instead of after the TensorRT engine build.
  static void validate(const Params& params);

  // The model's one pass of the turn: the board row and the whole candidate
  // set, encoded and handed to the service.
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

  // Scratch reused across turns to avoid per-move allocation.
  std::vector<float> board_row_;
  move_set::MoveFeatureArrays move_features_;
};

}  // namespace scribblez
