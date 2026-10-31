#pragma once

// PolicyBot (--type=policybot): plays the policy-set model's argmax
// (docs/plans/deep_bot.md, "Form 3: a policy"), with no search. Each turn it
// builds the model's training inputs (training/policy_set_inputs.h, through the
// same functions the pool labeler and the dataset's replay call): the pool of
// candidates, its belief about the opponent's rack from the leave inference
// generation the model's pools drew theirs from (belief::LeaveHistoryRecorder's
// query, the training row of its position), and each candidate's features and
// equity. One model call scores the pool, and it plays the highest logit, the
// earlier candidate (the better static equity) on ties.
//
// Hidden leaves only, before the endgame: EndgameAgent<PolicyBot> solves the
// turns with the bag empty, and PolicyBot alone plays HastyBot's move there,
// as the model never trained on them.

#include "agent/agent.h"
#include "agent/neural_service_options.h"
#include "belief/leave_history_recorder.h"
#include "belief/leave_params.h"
#include "belief/leave_query.h"
#include "belief/opp_rack_summary.h"
#include "encoding/game_state_encoder.h"
#include "encoding/input_encoder.h"
#include "lexicon/leave_value_arrays.h"
#include "nn/eval_service.h"
#include "training/policy_set_inputs.h"

#include <boost/program_options/options_description.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace scribblez {

class Dictionary;

class PolicyBot : public Agent {
 public:
  static constexpr const char* kType = "policybot";
  static constexpr bool kPreEndgameByDefault = false;

  // `dict` is required and must outlive the agent.
  struct Params {
    int thread_id = 0;
    std::string name;
    const Dictionary* dict = nullptr;
    // The policy-set model (a policy_set ONNX export) and how it is served.
    // Read by the factory.
    NeuralServiceOptions service{};
    // The leave inference model (ONNX) the model's pools drew their beliefs
    // from, which its export names; required iff it names one. Read by the
    // factory.
    std::string leave_inference{};
  };

  // What PolicyBot plays from: the served model, the blank prune its pools
  // were cut with (its export's metadata), and the leave inference model's
  // forward pass with the leave table its beliefs are over (an empty
  // `predict` for a model trained on uniform beliefs).
  struct Backend {
    std::shared_ptr<nn::PolicySetEvalService> policy;
    int blank_prune_points = 0;
    belief::LeaveParamsFn predict;
    std::shared_ptr<const LeaveValueArrays> table;
  };

  // Builds the backend Params names (policy_bot_factory.cpp, which reaches
  // CUDA and TensorRT).
  explicit PolicyBot(const Params& params);
  PolicyBot(const Params& params, Backend backend);

  MoveDecision make_move(const MoveRequest& req) override;
  void begin_game(const BeginGameRequest& req) override;
  void observe_move(const Move& move) override;

  // Build from `--player "--type=policybot [options]"` tokens, with --type and
  // --name already stripped. Throws on bad input.
  static std::unique_ptr<PolicyBot> from_spec(const std::vector<std::string>& tokens, int thread_id,
                                              const std::string& name);
  static std::string options_help();
  // Parse PolicyBot's options, plus any a derived agent registered in `extra`.
  static Params parse_params(const std::vector<std::string>& tokens, int thread_id,
                             const std::string& name,
                             boost::program_options::options_description& extra,
                             const char* type_label);

  // The model's inputs on the last turn it scored, and its candidates: public
  // so a test can check them against the training replay's for the same
  // position, a drift the model itself could never reveal.
  const std::vector<float>& board_row() const { return board_row_; }
  const std::array<float, policy_set::kOppRackFloats>& opp_rack() const { return opp_rack_; }
  const policy_set::CandidateInputs& candidate_inputs() const { return candidates_; }

 private:
  // The pool of the position: every legal move by static equity, cut as the
  // model's pools were.
  std::vector<Move> candidates(const MoveRequest& req) const;
  // The belief about the opponent's rack from the query `q` of this turn.
  belief::OppRackSummary opp_rack_belief(const MoveRequest& req, const belief::LeaveQuery& q) const;
  // Fill the model's inputs for `moves` at `req`'s position.
  void encode(const MoveRequest& req, const std::vector<Move>& moves,
              const belief::OppRackSummary& opp_rack);
  // The index of the highest logit the model gives the encoded candidates.
  size_t best_candidate();

  Backend backend_;
  const Dictionary& dict_;
  InputEncodingSpec spec_;
  GameStateEncoder encoder_;  // mirrors the live game, both seats' moves
  belief::LeaveHistoryRecorder recorder_;

  // Reused across turns to avoid per-move allocation.
  std::vector<float> board_row_;
  std::array<float, policy_set::kOppRackFloats> opp_rack_{};
  policy_set::CandidateInputs candidates_;
  std::vector<float> logits_;
};

}  // namespace scribblez
