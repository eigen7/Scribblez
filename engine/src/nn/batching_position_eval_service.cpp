#include "nn/batching_position_eval_service.h"

#include "encoding/input_encoder.h"
#include "nn/model_specs.h"

#include <cstring>

namespace scribblez {
namespace nn {

BatchingPositionEvalService::BatchingPositionEvalService(std::unique_ptr<PositionEvalService> inner)
    : inner_(std::move(inner)) {}

void BatchingPositionEvalService::evaluate(const SpecBatch& batch,
                                           std::span<float* const> head_out) {
  Request req{&batch, head_out};
  std::unique_lock<std::mutex> lock(mutex_);
  queue_.push_back(&req);

  if (dispatching_) {
    // Another caller is draining; it will serve this request.
    served_.wait(lock, [&req] { return req.done; });
  } else {
    // Become the dispatcher until the queue empties. Requests that arrive while
    // serve() runs (lock released) are picked up by the next iteration, so none
    // is stranded and a still-active dispatcher blocks new callers from racing
    // in as a second dispatcher.
    dispatching_ = true;
    while (!queue_.empty()) {
      const std::vector<Request*> pack(queue_.begin(), queue_.end());
      queue_.clear();
      lock.unlock();
      serve(pack);
      lock.lock();
      for (Request* r : pack) r->done = true;
      served_.notify_all();
    }
    dispatching_ = false;
  }

  lock.unlock();
  if (req.error) std::rethrow_exception(req.error);
}

bool BatchingPositionEvalService::try_evaluate(const SpecBatch& batch,
                                               std::span<float* const> head_out,
                                               const std::vector<Request*>& blame) {
  try {
    inner_->evaluate(batch, head_out);
    return true;
  } catch (...) {
    const std::exception_ptr error = std::current_exception();
    for (Request* r : blame) r->error = error;
    return false;
  }
}

void BatchingPositionEvalService::serve(const std::vector<Request*>& pack) {
  // Single-request drain -- the common case under low or bursty concurrency,
  // where a caller finds the queue empty and serves only itself. Evaluate
  // straight into the caller's buffers, skipping the gather/scatter copies that
  // earn their keep only when coalescing more than one request.
  if (pack.size() == 1) {
    try_evaluate(*pack.front()->batch, pack.front()->head_out, pack);
    return;
  }

  const int row_floats = spatial_planes() * kBoardCells + scalar_floats();
  int total = 0;
  for (const Request* r : pack) total += r->batch->count;

  in_rows_.resize(size_t(total) * row_floats);
  wld_out_.resize(size_t(total) * WldOutput::kRowElems);
  score_diff_out_.resize(size_t(total) * ScoreDiffOutput::kRowElems);

  int offset = 0;
  for (const Request* r : pack) {
    std::memcpy(in_rows_.data() + size_t(offset) * row_floats, r->batch->rows,
                sizeof(float) * size_t(r->batch->count) * row_floats);
    offset += r->batch->count;
  }

  // The wrapped service chunks this to its own max_rows; the combined batch is
  // in general larger than -- and unaligned with -- any single request's.
  float* const combined[] = {wld_out_.data(), score_diff_out_.data()};
  if (!try_evaluate(SpecBatch{in_rows_.data(), total}, combined, pack)) return;

  offset = 0;
  for (const Request* r : pack) {
    std::memcpy(r->head_out[0], wld_out_.data() + size_t(offset) * WldOutput::kRowElems,
                sizeof(float) * size_t(r->batch->count) * WldOutput::kRowElems);
    std::memcpy(r->head_out[1],
                score_diff_out_.data() + size_t(offset) * ScoreDiffOutput::kRowElems,
                sizeof(float) * size_t(r->batch->count) * ScoreDiffOutput::kRowElems);
    offset += r->batch->count;
  }
}

}  // namespace nn
}  // namespace scribblez
