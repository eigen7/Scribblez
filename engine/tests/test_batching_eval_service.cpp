// BatchingPositionEvalService, the decorator that coalesces concurrent callers'
// requests into one inner evaluate(). The stub echoes each row's first input
// float into every element of every output head, offset per head and element,
// so a gather or scatter error across coalesced requests (a wrong row, head or
// width) shows up as a caller receiving a wrong value. No GPU needed.

#include "encoding/input_encoder.h"
#include "nn/batching_position_eval_service.h"
#include "nn/model_specs.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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
using Outputs = PositionEvalService::Outputs;
using SpecBatch = PositionEvalService::SpecBatch;

int row_floats() { return spatial_planes() * kBoardCells + scalar_floats({nullptr}); }

// The value the echo stub writes at element k of head h for a row whose marker
// is m. Every term is exactly representable, so callers compare with ==.
float echoed(float m, size_t h, int k) { return m + 0.5f * float(h) + 0.125f * float(k); }

// Writes echoed(m, h, k) across every head's full row, m being row i's first
// input float.
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
      for (size_t h = 0; h < Outputs::size; ++h) {
        const int width = Outputs::row_elems[h];
        for (int k = 0; k < width; ++k) head_out[h][size_t(i) * width + k] = echoed(m, h, k);
      }
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

// One caller's output buffers: one per Outputs entry, sized for `rows` rows.
class HeadBuffers {
 public:
  explicit HeadBuffers(int rows);

  // Overwrite every buffer with a value the stub never writes.
  void poison();

  std::span<float* const> ptrs() const { return ptrs_; }
  const std::vector<float>& head(size_t h) const { return bufs_[h]; }

 private:
  std::array<std::vector<float>, Outputs::size> bufs_;
  std::array<float*, Outputs::size> ptrs_;
};

HeadBuffers::HeadBuffers(int rows) {
  for (size_t h = 0; h < Outputs::size; ++h) {
    bufs_[h].resize(size_t(rows) * Outputs::row_elems[h]);
    ptrs_[h] = bufs_[h].data();
  }
  poison();
}

void HeadBuffers::poison() {
  for (std::vector<float>& b : bufs_) std::fill(b.begin(), b.end(), -1.0f);
}

// How many elements of `out` differ from what the stub echoes for `markers`.
int echo_mismatches(const HeadBuffers& out, const std::vector<float>& markers) {
  int bad = 0;
  for (size_t h = 0; h < Outputs::size; ++h) {
    const int width = Outputs::row_elems[h];
    for (size_t r = 0; r < markers.size(); ++r) {
      for (int k = 0; k < width; ++k) {
        bad += out.head(h)[r * width + k] != echoed(markers[r], h, k);
      }
    }
  }
  return bad;
}

// One input row per marker, all zero but for the marker in its first float.
std::vector<float> rows_with_markers(const std::vector<float>& markers) {
  std::vector<float> in(markers.size() * row_floats(), 0.0f);
  for (size_t r = 0; r < markers.size(); ++r) in[r * row_floats()] = markers[r];
  return in;
}

TEST(BatchingPositionEvalService, ServesOneCallerCorrectly) {
  BatchingPositionEvalService svc(std::make_unique<EchoStub>());
  const std::vector<float> markers = {3.0f, 7.0f, 11.0f};
  std::vector<float> in = rows_with_markers(markers);
  HeadBuffers out(int(markers.size()));

  svc.evaluate(SpecBatch{in.data(), int(markers.size())}, out.ptrs());

  EXPECT_EQ(echo_mismatches(out, markers), 0);
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
    HeadBuffers out(kRows);
    for (int c = 0; c < kCalls; ++c) {
      out.poison();
      svc.evaluate(SpecBatch{in.data(), kRows}, out.ptrs());
      ASSERT_EQ(echo_mismatches(out, markers), 0) << "thread " << tid << " got wrong rows";
    }
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) threads.emplace_back(worker, t);
  for (std::thread& t : threads) t.join();

  // Coalescing may merge requests but never adds inner calls.
  EXPECT_GT(raw->calls.load(), 0);
  EXPECT_LE(raw->calls.load(), kThreads * kCalls);
}

TEST(BatchingPositionEvalService, PropagatesInnerFailure) {
  BatchingPositionEvalService svc(std::make_unique<ThrowingStub>());
  std::vector<float> in = rows_with_markers({1.0f});
  HeadBuffers out(1);
  EXPECT_THROW(svc.evaluate(SpecBatch{in.data(), 1}, out.ptrs()), std::runtime_error);
}

TEST(BatchingPositionEvalService, FailureReachesEveryCoalescedCaller) {
  // Contention makes drains coalesce several requests. The inner failure must
  // reach every request in the pack; a co-batched caller that missed it would
  // return garbage or hang instead of throwing.
  BatchingPositionEvalService svc(std::make_unique<ThrowingStub>());
  constexpr int kThreads = 8;
  constexpr int kCalls = 200;
  std::atomic<int> threw{0};
  std::atomic<int> returned{0};

  auto worker = [&] {
    std::vector<float> in = rows_with_markers({1.0f, 2.0f});
    HeadBuffers out(2);
    for (int c = 0; c < kCalls; ++c) {
      try {
        svc.evaluate(SpecBatch{in.data(), 2}, out.ptrs());
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
