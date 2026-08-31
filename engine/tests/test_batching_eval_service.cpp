// BatchingPositionEvalService: the caller-led cooperative batching decorator.
// A scripted stub echoes each row's first input float into both scoring heads,
// so a caller can assert it got back exactly the rows it submitted -- any
// mis-gather or mis-scatter across coalesced requests shows up as a wrong or
// swapped marker. No ONNX, no TensorRT, no GPU.

#include "encoding/input_encoder.h"
#include "nn/batching_position_eval_service.h"
#include "nn/model_specs.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using scribblez::kBoardCells;
using scribblez::scalar_floats;
using scribblez::spatial_planes;
using scribblez::nn::BatchingPositionEvalService;
using scribblez::nn::PositionEvalService;
using scribblez::nn::ScoreDiffOutput;
using scribblez::nn::WldOutput;
using SpecBatch = scribblez::nn::PositionEvaluationSpec::Batch;

int row_floats() { return spatial_planes() * kBoardCells + scalar_floats({nullptr}); }

// Echoes row i's first input float m into head 0 (as m) and head 1 (as m+0.5),
// so a caller recognizes its own rows in the output.
class EchoStub : public PositionEvalService {
 public:
  std::atomic<int> calls{0};
  bool opp_leave_input() const override { return false; }
  int spatial_planes() const override { return ::spatial_planes(); }
  int scalar_floats() const override { return ::scalar_floats({nullptr}); }
  void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) override {
    ++calls;
    const int rf = row_floats();
    for (int i = 0; i < batch.count; ++i) {
      const float m = batch.rows[size_t(i) * rf];
      head_out[0][size_t(i) * WldOutput::kRowElems] = m;
      head_out[1][size_t(i) * ScoreDiffOutput::kRowElems] = m + 0.5f;
    }
  }
};

class ThrowingStub : public PositionEvalService {
 public:
  bool opp_leave_input() const override { return false; }
  int spatial_planes() const override { return ::spatial_planes(); }
  int scalar_floats() const override { return ::scalar_floats({nullptr}); }
  void do_evaluate(const SpecBatch&, std::span<float* const>) override {
    throw std::runtime_error("boom");
  }
};

// One row's input block, all zero but for the marker in its first float.
std::vector<float> rows_with_markers(const std::vector<float>& markers) {
  std::vector<float> in(markers.size() * row_floats(), 0.0f);
  for (size_t r = 0; r < markers.size(); ++r) in[r * row_floats()] = markers[r];
  return in;
}

TEST(BatchingPositionEvalService, ServesOneCallerCorrectly) {
  BatchingPositionEvalService svc(std::make_unique<EchoStub>());
  const std::vector<float> markers = {3.0f, 7.0f, 11.0f};
  std::vector<float> in = rows_with_markers(markers);
  std::vector<float> wld(markers.size() * WldOutput::kRowElems, -1.0f);
  std::vector<float> sd(markers.size() * ScoreDiffOutput::kRowElems, -1.0f);
  float* const head_out[] = {wld.data(), sd.data()};

  svc.evaluate(SpecBatch{in.data(), int(markers.size())}, head_out);

  for (size_t r = 0; r < markers.size(); ++r) {
    EXPECT_EQ(wld[r * WldOutput::kRowElems], markers[r]);
    EXPECT_EQ(sd[r * ScoreDiffOutput::kRowElems], markers[r] + 0.5f);
  }
}

TEST(BatchingPositionEvalService, ConcurrentCallersGetTheirOwnRows) {
  auto stub = std::make_unique<EchoStub>();
  EchoStub* raw = stub.get();
  BatchingPositionEvalService svc(std::move(stub));

  constexpr int kThreads = 8;
  constexpr int kRows = 5;
  constexpr int kCalls = 300;

  auto worker = [&](int tid) {
    std::vector<float> markers(kRows);
    for (int r = 0; r < kRows; ++r) markers[r] = float(tid * 100 + r);  // unique per (tid, row)
    std::vector<float> in = rows_with_markers(markers);
    std::vector<float> wld(kRows * WldOutput::kRowElems);
    std::vector<float> sd(kRows * ScoreDiffOutput::kRowElems);
    for (int c = 0; c < kCalls; ++c) {
      std::fill(wld.begin(), wld.end(), -1.0f);
      std::fill(sd.begin(), sd.end(), -1.0f);
      float* const head_out[] = {wld.data(), sd.data()};
      svc.evaluate(SpecBatch{in.data(), kRows}, head_out);
      for (int r = 0; r < kRows; ++r) {
        ASSERT_EQ(wld[size_t(r) * WldOutput::kRowElems], markers[r])
          << "thread " << tid << " got another caller's row " << r;
        ASSERT_EQ(sd[size_t(r) * ScoreDiffOutput::kRowElems], markers[r] + 0.5f);
      }
    }
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) threads.emplace_back(worker, t);
  for (std::thread& t : threads) t.join();

  // Every request was served, and coalescing never manufactures work: the
  // wrapped service saw at most one call per request (fewer when it coalesced).
  EXPECT_GT(raw->calls.load(), 0);
  EXPECT_LE(raw->calls.load(), kThreads * kCalls);
}

TEST(BatchingPositionEvalService, PropagatesInnerFailure) {
  BatchingPositionEvalService svc(std::make_unique<ThrowingStub>());
  std::vector<float> in = rows_with_markers({1.0f});
  std::vector<float> wld(WldOutput::kRowElems);
  std::vector<float> sd(ScoreDiffOutput::kRowElems);
  float* const head_out[] = {wld.data(), sd.data()};
  EXPECT_THROW(svc.evaluate(SpecBatch{in.data(), 1}, head_out), std::runtime_error);
}

TEST(BatchingPositionEvalService, FailureReachesEveryCoalescedCaller) {
  // Many threads contend, so drains coalesce more than one request; serve()'s
  // catch must set the exception on every request in the pack, not just one --
  // otherwise a co-batched caller returns garbage (or hangs) instead of
  // throwing. Every call must observe the failure.
  BatchingPositionEvalService svc(std::make_unique<ThrowingStub>());
  constexpr int kThreads = 8;
  constexpr int kCalls = 200;
  std::atomic<int> threw{0};
  std::atomic<int> returned{0};

  auto worker = [&] {
    std::vector<float> in = rows_with_markers({1.0f, 2.0f});
    std::vector<float> wld(2 * WldOutput::kRowElems);
    std::vector<float> sd(2 * ScoreDiffOutput::kRowElems);
    float* const head_out[] = {wld.data(), sd.data()};
    for (int c = 0; c < kCalls; ++c) {
      try {
        svc.evaluate(SpecBatch{in.data(), 2}, head_out);
        ++returned;
      } catch (const std::runtime_error&) {
        ++threw;
      }
    }
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) threads.emplace_back(worker);
  for (std::thread& t : threads) t.join();

  EXPECT_EQ(returned.load(), 0);
  EXPECT_EQ(threw.load(), kThreads * kCalls);
}

}  // namespace
