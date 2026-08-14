#pragma once

// Scripted nn::EvalService stubs for agent unit tests -- no ONNX, no TensorRT,
// no GPU. Both declare the full contingent-features input layout (no
// opponent-leave block) and ignore the input rows entirely.

#include "encoding/input_encoder.h"
#include "nn/eval_service.h"

#include <algorithm>
#include <vector>

namespace scribblez {
namespace testing {

// Returns pre-set evals, one per candidate in *per-call* order. Suits tests
// that re-script and call make_move repeatedly on the same agent (the scripted
// index resets every call).
class StubEvalService : public nn::PositionEvalService {
 public:
  std::vector<nn::Eval> scripted;
  bool contingent_features() const override { return true; }
  bool opp_leave_input() const override { return false; }
  int spatial_planes() const override { return scribblez::spatial_planes({nullptr, true}); }
  int scalar_floats() const override { return scribblez::scalar_floats({nullptr, true}); }
  void evaluate(const nn::PositionEvaluationSpec::Batch& batch, nn::Eval* out) override {
    for (int i = 0; i < batch.count; ++i) {
      out[i] = (i < static_cast<int>(scripted.size())) ? scripted[i] : nn::Eval{};
    }
  }
};

// Returns pre-set evals indexed in *global* candidate order across however
// many chunked evaluate() calls a single make_move() makes, and records the
// total rows seen, the largest chunk, and the call count. Suits all-moves and
// chunking tests (one make_move per agent).
class CountingStubEvalService : public nn::PositionEvalService {
 public:
  std::vector<nn::Eval> scripted;
  bool contingent_features() const override { return true; }
  bool opp_leave_input() const override { return false; }
  int spatial_planes() const override { return scribblez::spatial_planes({nullptr, true}); }
  int scalar_floats() const override { return scribblez::scalar_floats({nullptr, true}); }
  int total_rows = 0;
  int max_chunk = 0;
  int calls = 0;

  void evaluate(const nn::PositionEvaluationSpec::Batch& batch, nn::Eval* out) override {
    ++calls;
    max_chunk = std::max(max_chunk, batch.count);
    for (int i = 0; i < batch.count; ++i) {
      const int g = total_rows + i;
      out[i] = (g < static_cast<int>(scripted.size())) ? scripted[g] : nn::Eval{};
    }
    total_rows += batch.count;
  }
};

}  // namespace testing
}  // namespace scribblez
