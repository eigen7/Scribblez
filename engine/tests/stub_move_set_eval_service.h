#pragma once

// A scripted nn::MoveSetEvalService stub for agent unit tests -- no ONNX, no
// TensorRT, no GPU. It declares the full contingent-features input layout (no
// opponent-leave block) and, unlike the position-model stubs, keeps what it was
// handed: the board row and the encoded candidate set are what a test checks
// the agent's encoding against.

#include "encoding/input_encoder.h"
#include "nn/eval_service.h"
#include "stub_eval_service.h"
#include "training/move_set_encoder.h"

#include <span>
#include <vector>

namespace scribblez {
namespace testing {

class StubMoveSetEvalService : public nn::MoveSetEvalService {
 public:
  // One scripted row per candidate, in candidate order; candidates past its
  // end score a default (all-zero) row.
  std::vector<ScriptedEval> scripted;

  // What the last evaluate() saw, and how many times it was called. One call
  // per turn is the architecture's whole claim, so a test asserts on it.
  int calls = 0;
  int total_moves = 0;
  std::vector<float> last_board_row;
  move_set::MoveFeatureArrays last_moves;

  bool contingent_features() const override { return true; }
  bool opp_leave_input() const override { return false; }
  int spatial_planes() const override { return scribblez::spatial_planes({nullptr, true}); }
  int scalar_floats() const override { return scribblez::scalar_floats({nullptr, true}); }

  void evaluate(const SpecBatch& batch, std::span<float* const> head_out) override {
    const move_set::MoveFeatureArrays& moves = *batch.moves;
    ++calls;
    total_moves += moves.count;
    last_board_row.assign(batch.board_row,
                          batch.board_row + input_floats(InputEncodingSpec{nullptr, true}));
    last_moves = moves;
    for (int i = 0; i < moves.count; ++i) {
      write_scripted(
        (i < static_cast<int>(scripted.size())) ? scripted[static_cast<size_t>(i)] : ScriptedEval{},
        i, head_out);
    }
  }
};

}  // namespace testing
}  // namespace scribblez
