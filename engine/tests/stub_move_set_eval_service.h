#pragma once

// A scripted move-set evaluation service, so agent tests run without a model
// or GPU. Unlike the position-evaluation stubs it keeps its inputs, so a test
// can check the agent's board row and candidate encoding. Declares the base
// input layout (no opponent-leave block).

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
  // One row per candidate, in order; candidates past the end get zeros.
  std::vector<ScriptedEval> scripted;

  // Call counts, and the inputs of the last call. The move-set model exists to
  // score a whole turn in one call, so tests assert on `calls`.
  int calls = 0;
  int total_moves = 0;
  std::vector<float> last_board_row;
  move_set::MoveFeatureArrays last_moves;

  bool opp_leave_input() const override { return false; }
  int spatial_planes() const override { return scribblez::spatial_planes(); }
  int scalar_floats() const override { return scribblez::scalar_floats({nullptr}); }

  void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) override {
    const move_set::MoveFeatureArrays& moves = *batch.moves;
    ++calls;
    total_moves += moves.count;
    last_board_row.assign(batch.board_row,
                          batch.board_row + input_floats(InputEncodingSpec{nullptr}));
    last_moves = moves;
    for (int i = 0; i < moves.count; ++i) {
      write_scripted((i < int(scripted.size())) ? scripted[size_t(i)] : ScriptedEval{}, i,
                     head_out);
    }
  }
};

}  // namespace testing
}  // namespace scribblez
