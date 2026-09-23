#pragma once

// Scripted position-evaluation services, so agent tests run without a model or
// GPU. Both declare the base input layout (no opponent-leave block) and ignore
// the input rows.

#include "encoding/input_encoder.h"
#include "nn/eval_service.h"

#include <algorithm>
#include <array>
#include <span>
#include <vector>

namespace scribblez {
namespace testing {

// One scripted row of the two scoring heads: [P(win), P(draw), P(loss)] and
// the score difference's [mean, std].
struct ScriptedEval {
  std::array<float, 3> wld{};
  std::array<float, 2> score_diff{};
};

inline void write_scripted(const ScriptedEval& e, int row, std::span<float* const> head_out) {
  float* wld = head_out[0] + size_t(row) * nn::WldOutput::kRowElems;
  std::copy(e.wld.begin(), e.wld.end(), wld);
  float* sd = head_out[1] + size_t(row) * nn::ScoreDiffOutput::kRowElems;
  std::copy(e.score_diff.begin(), e.score_diff.end(), sd);
}

// Row i of every evaluate() call gets scripted[i]; rows past the end get
// zeros. For tests that re-script between make_move calls on one agent.
class StubEvalService : public nn::PositionEvalService {
 public:
  std::vector<ScriptedEval> scripted;
  bool opp_leave_input() const override { return false; }
  int spatial_planes() const override { return scribblez::spatial_planes(); }
  int scalar_floats() const override { return scribblez::scalar_floats({nullptr}); }
  void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) override {
    for (int i = 0; i < batch.count; ++i) {
      write_scripted((i < int(scripted.size())) ? scripted[i] : ScriptedEval{}, i, head_out);
    }
  }
};

// Indexes `scripted` across all evaluate() calls, so a make_move() split into
// chunks still sees one row per candidate, and records the rows, largest
// chunk and calls it saw. Use one per make_move().
class CountingStubEvalService : public nn::PositionEvalService {
 public:
  std::vector<ScriptedEval> scripted;
  bool opp_leave_input() const override { return false; }
  int spatial_planes() const override { return scribblez::spatial_planes(); }
  int scalar_floats() const override { return scribblez::scalar_floats({nullptr}); }
  int total_rows = 0;
  int max_chunk = 0;
  int calls = 0;

  void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) override {
    ++calls;
    max_chunk = std::max(max_chunk, batch.count);
    for (int i = 0; i < batch.count; ++i) {
      const int g = total_rows + i;
      write_scripted((g < int(scripted.size())) ? scripted[g] : ScriptedEval{}, i, head_out);
    }
    total_rows += batch.count;
  }
};

}  // namespace testing
}  // namespace scribblez
