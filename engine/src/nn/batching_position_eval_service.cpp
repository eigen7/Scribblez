#include "nn/batching_position_eval_service.h"

#include "encoding/input_encoder.h"

#include <array>
#include <cstring>

namespace scribblez {
namespace nn {

BatchingPositionEvalService::BatchingPositionEvalService(std::unique_ptr<PositionEvalService> inner)
    : inner_(std::move(inner)), batcher_([this](Pack pack) { serve(pack); }) {}

void BatchingPositionEvalService::evaluate(const SpecBatch& batch,
                                           std::span<float* const> head_out) {
  Request req{&batch, head_out};
  batcher_.submit(req);
  if (req.error) std::rethrow_exception(req.error);
}

bool BatchingPositionEvalService::try_evaluate(const SpecBatch& batch,
                                               std::span<float* const> head_out, Pack blame) {
  try {
    inner_->evaluate(batch, head_out);
    return true;
  } catch (...) {
    const std::exception_ptr error = std::current_exception();
    for (Request* r : blame) r->error = error;
    return false;
  }
}

void BatchingPositionEvalService::serve(Pack pack) {
  // A lone request, the common case under light load, is evaluated straight
  // into its caller's buffers with no gather/scatter copies.
  if (pack.size() == 1) {
    try_evaluate(*pack.front()->batch, pack.front()->head_out, pack);
    return;
  }

  const int total = gather(pack);
  std::array<float*, Outputs::size> combined;
  for (size_t h = 0; h < Outputs::size; ++h) combined[h] = out_rows_[h].data();

  // The combined batch may exceed max_rows; the wrapped service chunks it.
  if (!try_evaluate(SpecBatch{in_rows_.data(), total}, combined, pack)) return;
  scatter(pack);
}

int BatchingPositionEvalService::gather(Pack pack) {
  const int row_floats = spatial_planes() * kBoardCells + scalar_floats();
  int total = 0;
  for (const Request* r : pack) total += r->batch->count;

  in_rows_.resize(size_t(total) * row_floats);
  for (size_t h = 0; h < Outputs::size; ++h) {
    out_rows_[h].resize(size_t(total) * Outputs::row_elems[h]);
  }

  int offset = 0;
  for (const Request* r : pack) {
    std::memcpy(in_rows_.data() + size_t(offset) * row_floats, r->batch->rows,
                sizeof(float) * size_t(r->batch->count) * row_floats);
    offset += r->batch->count;
  }
  return total;
}

void BatchingPositionEvalService::scatter(Pack pack) const {
  int offset = 0;
  for (const Request* r : pack) {
    for (size_t h = 0; h < Outputs::size; ++h) {
      const int width = Outputs::row_elems[h];
      std::memcpy(r->head_out[h], out_rows_[h].data() + size_t(offset) * width,
                  sizeof(float) * size_t(r->batch->count) * width);
    }
    offset += r->batch->count;
  }
}

}  // namespace nn
}  // namespace scribblez
