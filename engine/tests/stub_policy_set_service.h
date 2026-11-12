#pragma once

// A scripted policy-set model for the agents' tests (PolicyBot, DeepBot's
// dump): keeps its inputs, and its logits favour candidate `favourite` (the
// first when the pool is shorter), or follow `logits` when set; it declares
// the hidden-leaves layout, or the face-up one with `face_up`.

#include "encoding/input_encoder.h"
#include "nn/eval_service.h"
#include "training/policy_set_inputs.h"

#include <algorithm>
#include <array>
#include <span>
#include <vector>

namespace scribblez::testing {

// One scored position, as the stub saw it.
struct ScoredPosition {
  std::vector<float> board_row;
  std::array<float, policy_set::kOppRackFloats> opp_rack;
  policy_set::CandidateInputs candidates;
};

class StubPolicySetService : public nn::PolicySetEvalService {
 public:
  int favourite = 1;
  bool face_up = false;
  std::vector<float> logits;
  std::vector<ScoredPosition> calls;

  bool opp_leave_input() const override { return face_up; }
  int spatial_planes() const override { return scribblez::spatial_planes(); }
  int scalar_floats() const override { return scribblez::scalar_floats({nullptr, face_up}); }

 protected:
  void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) override;
};

inline void StubPolicySetService::do_evaluate(const SpecBatch& batch,
                                              std::span<float* const> head_out) {
  const int n = batch.candidates->count();
  ScoredPosition& seen = calls.emplace_back();
  seen.board_row.assign(batch.board_row, batch.board_row + input_floats(InputEncodingSpec{}));
  std::copy(batch.opp_rack, batch.opp_rack + policy_set::kOppRackFloats, seen.opp_rack.begin());
  seen.candidates = *batch.candidates;
  for (int i = 0; i < n; ++i) {
    head_out[0][i] = logits.empty() ? float(i == std::min(favourite, n - 1)) : logits[size_t(i)];
  }
}

}  // namespace scribblez::testing
