#pragma once

#include "nn/eval_service.h"

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
// calls of many game threads into larger GPU batches, so a shared model runs
// fuller, fewer inference calls than one-per-thread would.
//
// Caller-led cooperative batching, not a background thread: a caller enqueues
// its request and, if no dispatcher is running, becomes the dispatcher itself;
// otherwise it blocks until its request is served. The dispatcher drains the
// queue -- gathering every waiting request's rows into one combined batch it
// hands to the wrapped service (which chunks to its own max_rows), then
// scattering each request's decoded rows back to that caller's buffers -- and
// keeps going until the queue empties before standing down. Rows are scored
// independently, so a combined batch changes no result; a request that arrives
// mid-inference is simply served by the next drain. There is no owner thread to
// start, stop, or outlive the callers, and a failed inference propagates to
// exactly the requests it was serving.
//
// The wrapped service must itself be serialized-one-call-at-a-time (the default
// EvalService contract); this decorator guarantees only the dispatcher calls it,
// so that holds. Position only: the move-set model stages one board per call and
// cannot merge rows across requests.
class BatchingPositionEvalService : public PositionEvalService {
 public:
  // Decorates `inner`, which must already be loaded. Takes ownership.
  explicit BatchingPositionEvalService(std::unique_ptr<PositionEvalService> inner);

  bool opp_leave_input() const override { return inner_->opp_leave_input(); }
  int spatial_planes() const override { return inner_->spatial_planes(); }
  int scalar_floats() const override { return inner_->scalar_floats(); }

  void evaluate(const SpecBatch& batch, std::span<float* const> head_out) override;

 protected:
  // Never reached -- evaluate() is overridden -- but the interface demands it;
  // forward to the wrapped service so a direct call would still be correct.
  void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) override {
    inner_->evaluate(batch, head_out);
  }

 private:
  // One blocked caller's work: its inputs, its output destinations, and the
  // completion the dispatcher signals (with an exception if inference failed).
  struct Request {
    const SpecBatch* batch;
    std::span<float* const> head_out;
    bool done = false;
    std::exception_ptr error;
  };

  // Gather every request's rows into one combined batch, evaluate it through
  // inner_, and scatter each request's decoded rows back. Records a per-request
  // exception instead of throwing, so the dispatcher loop cannot strand waiting
  // callers.
  void serve(const std::vector<Request*>& pack);

  std::unique_ptr<PositionEvalService> inner_;

  std::mutex mutex_;
  std::condition_variable served_;
  std::deque<Request*> queue_;
  bool dispatching_ = false;

  // Dispatcher-only staging, reused across drains: the gathered input rows and
  // the two scoring heads' combined outputs.
  std::vector<float> in_rows_;
  std::vector<float> wld_out_;
  std::vector<float> score_diff_out_;
};

}  // namespace nn
}  // namespace scribblez
