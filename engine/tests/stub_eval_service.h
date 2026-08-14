#pragma once

// Scripted nn::EvalService stubs for agent unit tests -- no ONNX, no TensorRT,
// no GPU. Both declare the full contingent-features input layout (no
// opponent-leave block) and ignore the input rows entirely.

#include "encoding/input_encoder.h"
#include "nn/eval_service.h"

#include <algorithm>
#include <span>
#include <vector>

namespace scribblez {
namespace testing {

// The inverse of nn::make_evals, so a test scripts whole Evals while the
// service interface transports decoded head rows. win_prob does not survive
// the trip on its own -- consumers recompute it as p_win + 0.5 * p_draw -- so
// scripted evals must carry their ranking signal in p_win (the helpers in the
// test files do).
inline void write_eval_heads(const nn::Eval& e, int row, std::span<float* const> head_out) {
  float* wld = head_out[0] + static_cast<size_t>(row) * nn::WldOutput::kRowElems;
  wld[0] = e.p_win;
  wld[1] = e.p_draw;
  wld[2] = e.p_loss;
  float* sd = head_out[1] + static_cast<size_t>(row) * nn::ScoreDiffOutput::kRowElems;
  sd[0] = e.score_diff_mean;
  sd[1] = e.score_diff_std;
}

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
  void evaluate(const SpecBatch& batch, std::span<float* const> head_out) override {
    for (int i = 0; i < batch.count; ++i) {
      write_eval_heads((i < static_cast<int>(scripted.size())) ? scripted[i] : nn::Eval{}, i,
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
  std::vector<nn::Eval> scripted;
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
      write_eval_heads((g < static_cast<int>(scripted.size())) ? scripted[g] : nn::Eval{}, i,
                       head_out);
    }
    total_rows += batch.count;
  }
};

}  // namespace testing
}  // namespace scribblez
