#pragma once

#include "nn/eval_service.h"
#include "util/caller_led_batcher.h"

#include <array>
#include <exception>
#include <memory>
#include <span>
#include <vector>

namespace scribblez {
namespace nn {

// Wraps a position-evaluation service and coalesces the concurrent evaluate()
// calls of many game threads into larger GPU batches (util::CallerLedBatcher):
// each pack's rows are gathered into one batch for the wrapped service and the
// results scattered back to each caller.
//
// Consequences worth knowing:
//   - Results are unchanged, since rows are scored independently.
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
  // One caller's pending work, and the exception if inference failed.
  struct Request {
    const SpecBatch* batch;
    std::span<float* const> head_out;
    std::exception_ptr error = nullptr;
  };
  using Pack = std::span<Request* const>;

  // Evaluate every request in `pack` into its own buffers. Never throws; see
  // try_evaluate().
  void serve(Pack pack);

  // Size the staging buffers for `pack` and copy its input rows into in_rows_,
  // in pack order. Returns the combined row count.
  int gather(Pack pack);

  // Copy each request's slice of every head's combined output back into the
  // request's own head_out.
  void scatter(Pack pack) const;

  // Evaluate `batch` through inner_ into `head_out`. On failure, record the
  // exception on every request in `blame` and return false rather than throw:
  // an exception escaping the dispatcher would strand the other waiters.
  bool try_evaluate(const SpecBatch& batch, std::span<float* const> head_out, Pack blame);

  std::unique_ptr<PositionEvalService> inner_;
  util::CallerLedBatcher<Request> batcher_;

  // Dispatcher-only staging for combined batches, reused across rounds: the
  // input rows, and one output buffer per Outputs entry, in list order.
  std::vector<float> in_rows_;
  std::array<std::vector<float>, Outputs::size> out_rows_;
};

}  // namespace nn
}  // namespace scribblez
