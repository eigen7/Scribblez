#pragma once

// PolicyBot (--type=policybot): plays the policy-set model's argmax
// (docs/plans/deep_bot.md, "Form 3: a policy"), with no search. Each turn it
// scores the pool of candidates (agent/policy_scorer.h, the model's training
// inputs) with its belief about the opponent's rack from the leave inference
// generation the model's pools drew theirs from (belief::LeaveHistoryRecorder's
// query, the training row of its position), and plays the highest logit, the
// earlier candidate (the better static equity) on ties.
//
// Hidden leaves only, before the endgame: EndgameAgent<PolicyBot> solves the
// turns with the bag empty, and PolicyBot alone plays HastyBot's move there,
// as the model never trained on them.

#include "agent/agent.h"
#include "agent/neural_service_options.h"
#include "agent/policy_scorer.h"
#include "belief/leave_history_recorder.h"
#include "belief/leave_params.h"
#include "belief/leave_query.h"
#include "belief/opp_rack_summary.h"
#include "encoding/game_state_encoder.h"
#include "lexicon/leave_value_arrays.h"
#include "nn/eval_service.h"
#include "nn/onnx_metadata.h"
#include "training/cross_check_delta.h"

#include <boost/program_options/options_description.hpp>

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
  // were cut with (its export's metadata), the leave inference model's
  // forward pass with the leave table its beliefs are over (an empty
  // `predict` for a model trained on uniform beliefs), and where its
  // candidates' cross-check entries come from (null: the CPU reference; the
  // factory's is the GPU kernel, which matches it).
  struct Backend {
    std::shared_ptr<nn::PolicySetEvalService> policy;
    int blank_prune_points = 0;
    belief::LeaveParamsFn predict;
    std::shared_ptr<const LeaveValueArrays> table;
    std::shared_ptr<move_set::CrossCheckDeltaSource> cross_checks;
  };

  // What the model's export metadata says its pools were built with: the
  // blank prune they were cut with (-1 for an export without it), and the
  // leave inference generation they drew their beliefs from (an empty tag for
  // uniform beliefs).
  struct TrainedOn {
    int blank_prune_points = -1;
    nn::ModelIdentity leave_model;
  };

  // Throws util::CleanException unless `p` can play the model `trained`
  // describes: the export records a blank prune, and `leave`, the identity of
  // p.leave_inference's export (nullptr when p names none), is the leave
  // inference generation the model trained on. The factory runs it before
  // building either TensorRT engine.
  static void check_trained_on(const Params& p, const TrainedOn& trained,
                               const nn::ModelIdentity* leave);

  // The backend Params names, checked against the model's export metadata
  // (policy_bot_factory.cpp, which reaches CUDA and TensorRT). Public so an
  // agent serving the model beside its own (DeepBot's dump) builds the same.
  static Backend make_backend(const Params& params);

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

 private:
  // The pool of the position: every legal move by static equity, cut as the
  // model's pools were.
  std::vector<Move> candidates(const MoveRequest& req) const;
  // The belief about the opponent's rack from the query `q` of this turn.
  belief::OppRackSummary opp_rack_belief(const MoveRequest& req, const belief::LeaveQuery& q) const;

  Backend backend_;
  PolicyScorer scorer_;
  GameStateEncoder encoder_;  // mirrors the live game, both seats' moves
  belief::LeaveHistoryRecorder recorder_;
};

}  // namespace scribblez
