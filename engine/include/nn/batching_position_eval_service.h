#pragma once

#include "nn/eval_service.h"

#include <array>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace scribblez {
namespace nn {

// Wraps a position-evaluation service and coalesces the concurrent evaluate()
// calls of many game threads into larger GPU batches: fewer, fuller inference
// calls than one per thread.
//
// The batching is caller-led, with no background thread. A caller enqueues its
// request and, if no dispatcher is running, becomes the dispatcher; otherwise
// it blocks until its request is served. The dispatcher repeatedly takes every
// queued request, gathers their rows into one batch for the wrapped service,
// and scatters the results back to each caller, until the queue is empty.
// Requests that arrive during an inference go into the next round.
//
// Consequences worth knowing:
//   - Results are unchanged, since rows are scored independently.
//   - There is no owner thread to start, stop, or outlive the callers.
//   - A failed inference is rethrown in exactly the callers it was serving.
//   - Only the dispatcher calls the wrapped service, so the inner service's
//     one-call-at-a-time contract holds.
//
// Position family only: a move-set call is one board with its candidates, so
// rows from different requests cannot share a batch.
class BatchingPositionEvalService : public PositionEvalService {
 public:
  // Decorates `inner`, which must already be loaded. Takes ownership.
  explicit BatchingPositionEvalService(std::unique_ptr<PositionEvalService> inner);

  bool opp_leave_input() const override { return inner_->opp_leave_input(); }
  int spatial_planes() const override { return inner_->spatial_planes(); }
  int scalar_floats() const override { return inner_->scalar_floats(); }

  void evaluate(const SpecBatch& batch, std::span<float* const> head_out) override;

 protected:
  // Unreachable, since evaluate() is overridden, but the interface requires it.
  // Forwards to the wrapped service so that a direct call is still correct.
  void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) override {
    inner_->evaluate(batch, head_out);
  }

 private:
  // One caller's pending work and its completion state, which the dispatcher
  // sets (with the exception, if inference failed).
  struct Request {
    const SpecBatch* batch;
    std::span<float* const> head_out;
    bool done = false;
    std::exception_ptr error = nullptr;
  };

  // Evaluate every request in `pack` into its own buffers. Never throws; see
  // try_evaluate().
  void serve(const std::vector<Request*>& pack);

  // Size the staging buffers for `pack` and copy its input rows into in_rows_,
  // in pack order. Returns the combined row count.
  int gather(const std::vector<Request*>& pack);

  // Copy each request's slice of every head's combined output back into the
  // request's own head_out.
  void scatter(const std::vector<Request*>& pack) const;

  // Evaluate `batch` through inner_ into `head_out`. On failure, record the
  // exception on every request in `blame` and return false rather than throw:
  // an exception escaping the dispatcher loop would strand the other waiters.
  bool try_evaluate(const SpecBatch& batch, std::span<float* const> head_out,
                    const std::vector<Request*>& blame);

  std::unique_ptr<PositionEvalService> inner_;

  std::mutex mutex_;
  std::condition_variable served_;
  std::deque<Request*> queue_;
  bool dispatching_ = false;

  // Dispatcher-only staging for combined batches, reused across rounds: the
  // input rows, and one output buffer per Outputs entry, in list order.
  std::vector<float> in_rows_;
  std::array<std::vector<float>, Outputs::size> out_rows_;
};

}  // namespace nn
}  // namespace scribblez
