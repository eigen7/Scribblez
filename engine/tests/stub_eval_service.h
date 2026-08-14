#pragma once

// Scripted nn::EvalService stubs for agent unit tests -- no ONNX, no TensorRT,
// no GPU. Both declare the full contingent-features input layout (no
// opponent-leave block) and ignore the input rows entirely.

#include "encoding/input_encoder.h"
#include "nn/eval_service.h"

#include <algorithm>
#include <array>
#include <span>
#include <vector>

namespace scribblez {
namespace testing {

// One scripted row of the two scoring heads, exactly as the service interface
// transports them: [P(win), P(draw), P(loss)] and [mean, std]. A test-side
// convenience -- production consumers read decoded head rows directly. The
// win-prob ranking objective reads wld[0] + 0.5 * wld[1].
struct ScriptedEval {
  std::array<float, 3> wld{};
  std::array<float, 2> score_diff{};
};

inline void write_scripted(const ScriptedEval& e, int row, std::span<float* const> head_out) {
  float* wld = head_out[0] + static_cast<size_t>(row) * nn::WldOutput::kRowElems;
  std::copy(e.wld.begin(), e.wld.end(), wld);
  float* sd = head_out[1] + static_cast<size_t>(row) * nn::ScoreDiffOutput::kRowElems;
  std::copy(e.score_diff.begin(), e.score_diff.end(), sd);
}

// Returns pre-set evals, one per candidate in *per-call* order. Suits tests
// that re-script and call make_move repeatedly on the same agent (the scripted
// index resets every call).
class StubEvalService : public nn::PositionEvalService {
 public:
  std::vector<ScriptedEval> scripted;
  bool contingent_features() const override { return true; }
  bool opp_leave_input() const override { return false; }
  int spatial_planes() const override { return scribblez::spatial_planes({nullptr, true}); }
  int scalar_floats() const override { return scribblez::scalar_floats({nullptr, true}); }
  void evaluate(const SpecBatch& batch, std::span<float* const> head_out) override {
    for (int i = 0; i < batch.count; ++i) {
      write_scripted((i < static_cast<int>(scripted.size())) ? scripted[i] : ScriptedEval{}, i,
                     head_out);
    }
  }
};

// Returns pre-set evals indexed in *global* candidate order across however
// many chunked evaluate() calls a single make_move() makes, and records the
// total rows seen, the largest chunk, and the call count. Suits all-moves and
// chunking tests (one make_move per agent).
class CountingStubEvalService : public nn::PositionEvalService {
 public:
  std::vector<ScriptedEval> scripted;
  bool contingent_features() const override { return true; }
  bool opp_leave_input() const override { return false; }
  int spatial_planes() const override { return scribblez::spatial_planes({nullptr, true}); }
  int scalar_floats() const override { return scribblez::scalar_floats({nullptr, true}); }
  int total_rows = 0;
  int max_chunk = 0;
  int calls = 0;

  void evaluate(const SpecBatch& batch, std::span<float* const> head_out) override {
    ++calls;
    max_chunk = std::max(max_chunk, batch.count);
    for (int i = 0; i < batch.count; ++i) {
      const int g = total_rows + i;
      write_scripted((g < static_cast<int>(scripted.size())) ? scripted[g] : ScriptedEval{}, i,
                     head_out);
    }
    total_rows += batch.count;
  }
};

}  // namespace testing
}  // namespace scribblez
