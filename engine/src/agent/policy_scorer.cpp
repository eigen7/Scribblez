#include "agent/policy_scorer.h"

#include "agent/candidate_evaluator.h"
#include "util/exception.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace scribblez {

namespace {

InputEncodingSpec hidden_leaves_spec(const Dictionary& dict, const nn::PolicySetEvalService& policy,
                                     const std::string& who) {
  InputEncodingSpec spec = derive_input_spec(dict, policy, who);
  if (spec.opp_leave_input) {
    throw util::CleanException("{}: the model reads face-up leaves; its pools are hidden", who);
  }
  return spec;
}

const nn::PolicySetEvalService& require_policy(const nn::PolicySetEvalService* policy,
                                               const std::string& who) {
  if (policy == nullptr) throw util::Exception("{}: a policy-set model is required", who);
  return *policy;
}

}  // namespace

PolicyScorer::PolicyScorer(std::shared_ptr<nn::PolicySetEvalService> policy,
                           std::shared_ptr<move_set::CrossCheckDeltaSource> cross_checks,
                           const Dictionary& dict, const std::string& who)
    : policy_(std::move(policy)),
      cross_checks_(std::move(cross_checks)),
      dict_(dict),
      spec_(hidden_leaves_spec(dict, require_policy(policy_.get(), who), who)) {
  board_row_.resize(size_t(input_floats(spec_)));
}

const std::vector<float>& PolicyScorer::score(const GameStateEncoder& position, const Rack& rack,
                                              int bag, const belief::OppRackSummary& opp_rack,
                                              const std::vector<Move>& moves) {
  // The cross-check input planes read the board's move-generation caches;
  // building them here is a no-op once they are valid.
  position.board().ensure_movegen_caches(dict_);
  const Rack hidden;
  position.encode_input(position.active_player(), rack, hidden, board_row_.data());
  opp_rack_ = policy_set::encode_opp_rack(opp_rack);
  candidates_.encode(position, dict_, rack, bag, moves,
                     cross_checks_ ? *cross_checks_ : move_set::cpu_cross_check_deltas());
  logits_.resize(moves.size());
  float* const head_out[] = {logits_.data()};
  policy_->evaluate({board_row_.data(), opp_rack_.data(), &candidates_}, head_out);
  if (!std::ranges::all_of(logits_, [](float x) { return std::isfinite(x); }))
    throw util::Exception("the policy-set model gave a non-finite logit");
  return logits_;
}

size_t best_logit(const std::vector<float>& logits) {
  return size_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

}  // namespace scribblez
