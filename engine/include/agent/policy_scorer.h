#pragma once

// Scores a chooser's candidates with the policy-set model, as PolicyBot plays
// it and DeepBot's subtree dump reads it: builds the model's training inputs
// (training/policy_set_inputs.h, through the functions the pool labeler and
// the dataset's replay call) for the chooser's position, and one model call
// gives each candidate's logit.

#include "belief/opp_rack_summary.h"
#include "encoding/game_state_encoder.h"
#include "encoding/input_encoder.h"
#include "game/move.h"
#include "game/rack.h"
#include "nn/eval_service.h"
#include "training/cross_check_delta.h"
#include "training/policy_set_inputs.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace scribblez {

class Dictionary;

class PolicyScorer {
 public:
  // `dict` must outlive the scorer. `cross_checks` is where the candidates'
  // cross-check entries come from (null: the CPU reference; the factories'
  // is the GPU kernel, which matches it). Throws, naming `who`, unless the
  // model reads hidden leaves, the only pools there are.
  PolicyScorer(std::shared_ptr<nn::PolicySetEvalService> policy,
               std::shared_ptr<move_set::CrossCheckDeltaSource> cross_checks,
               const Dictionary& dict, const std::string& who);

  // The input arm the model declares, for an encoder mirroring the game.
  const InputEncodingSpec& spec() const { return spec_; }

  // The logits of `moves` for the chooser to move at `position` (its active
  // player, every prior move applied), holding `rack` with `bag` tiles in the
  // bag and `opp_rack` its belief about the opponent's rack. Valid until the
  // next call. Throws on a non-finite logit.
  const std::vector<float>& score(const GameStateEncoder& position, const Rack& rack, int bag,
                                  const belief::OppRackSummary& opp_rack,
                                  const std::vector<Move>& moves);

 private:
  std::shared_ptr<nn::PolicySetEvalService> policy_;
  std::shared_ptr<move_set::CrossCheckDeltaSource> cross_checks_;
  const Dictionary& dict_;
  InputEncodingSpec spec_;

  // Reused across calls to avoid per-move allocation.
  std::vector<float> board_row_;
  std::array<float, policy_set::kOppRackFloats> opp_rack_{};
  policy_set::CandidateInputs candidates_;
  std::vector<float> logits_;
};

// The index of the largest of `logits`, the earliest on ties (the better
// static equity, in a pool's order).
size_t best_logit(const std::vector<float>& logits);

}  // namespace scribblez
