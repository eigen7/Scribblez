#include "nn/trt_eval_service.h"

#include "encoding/input_encoder.h"

#include <Eigen/Core>

#include <algorithm>
#include <cstring>

namespace scribblez {
namespace nn {

namespace {

// The one genuinely family-specific part of serving -- how a Batch's rows
// reach the engine's staging buffers -- as free functions overloaded on the
// spec's Batch type. Everything else (chunking, decode, the aux path) is the
// shared driver below.

int batch_rows(const PositionEvaluationSpec::Batch& batch) { return batch.count; }
int batch_rows(const MoveSetEvaluationSpec::Batch& batch) { return batch.moves->count; }

// Split one encoder row's [spatial | scalar] block into the engine's two
// separate, densely packed board buffers, at the model's own widths.
template <typename Spec>
void stage_board_row(NeuralNet<Spec>& net, const float* row, int dst_row) {
  const size_t spatial_floats = static_cast<size_t>(net.spatial_planes()) * kBoardCells;
  const size_t scalar_floats = net.scalar_floats();
  std::memcpy(net.template host<SpatialInput>() + dst_row * spatial_floats, row,
              sizeof(float) * spatial_floats);
  std::memcpy(net.template host<ScalarInput>() + dst_row * scalar_floats, row + spatial_floats,
              sizeof(float) * scalar_floats);
}

// Once per call: nothing for the position model (every tensor is per-chunk);
// the board row for the move set model, staged once and re-sent with every
// chunk, the engine holding one position per call by construction.
void stage_call(NeuralNet<PositionEvaluationSpec>&, const PositionEvaluationSpec::Batch&) {}
void stage_call(NeuralNet<MoveSetEvaluationSpec>& net, const MoveSetEvaluationSpec::Batch& batch) {
  stage_board_row(net, batch.board_row, 0);
}

// One per-move tensor's chunk, from the MoveFeatureArrays field its descriptor
// names.
template <typename Tensor>
void stage_move_rows(NeuralNet<MoveSetEvaluationSpec>& net,
                     const move_set::MoveFeatureArrays& moves, int start, int rows) {
  using Elem = typename Tensor::Elem;
  const std::vector<Elem>& src = moves.*Tensor::kBatchSource;
  std::memcpy(net.host<Tensor>(), src.data() + static_cast<size_t>(start) * Tensor::kRowElems,
              sizeof(Elem) * rows * Tensor::kRowElems);
}

template <typename... Ts>
void stage_move_tensors(NeuralNet<MoveSetEvaluationSpec>& net,
                        const move_set::MoveFeatureArrays& moves, int start, int rows,
                        TensorList<Ts...>) {
  (stage_move_rows<Ts>(net, moves, start, rows), ...);
}

// The chunk's rows: de-interleaved encoder rows for the position model, the
// spec's per-move tensor list for the move set model.
void stage_chunk(NeuralNet<PositionEvaluationSpec>& net, const PositionEvaluationSpec::Batch& batch,
                 int start, int chunk) {
  const size_t row_floats =
    static_cast<size_t>(net.spatial_planes()) * kBoardCells + net.scalar_floats();
  for (int r = 0; r < chunk; ++r) {
    stage_board_row(net, batch.rows + (static_cast<size_t>(start) + r) * row_floats, r);
  }
}
void stage_chunk(NeuralNet<MoveSetEvaluationSpec>& net, const MoveSetEvaluationSpec::Batch& batch,
                 int start, int chunk) {
  stage_move_tensors(net, *batch.moves, start, chunk, MoveSetEvaluationSpec::MoveInputs{});
}

// `rows` rows of one head's raw output into its decoded form, per the head's
// declared RowDecode (model_specs.h): each row `width` floats, landing at
// `dst_stride`-float steps.
void decode_head_rows(RowDecode decode, const float* raw, int rows, int width, float* dst,
                      int dst_stride) {
  for (int r = 0; r < rows; ++r) {
    Eigen::Map<const Eigen::ArrayXf> in(raw + static_cast<size_t>(r) * width, width);
    Eigen::Map<Eigen::ArrayXf> out(dst + static_cast<size_t>(r) * dst_stride, width);
    switch (decode) {
      case RowDecode::kIdentity:
        out = in;
        break;
      case RowDecode::kSoftmax:
        // Numerically stable: subtract the max before exponentiating.
        out = (in - in.maxCoeff()).exp();
        out /= out.sum();
        break;
      case RowDecode::kSigmoid:
        out = 1.0f / (1.0f + (-in).exp());
        break;
    }
  }
}

// One chunk of every scoring head into the caller's per-head destinations, in
// list order, `start` rows in.
template <typename Spec, TensorDescriptor... Ts>
void decode_outputs(const NeuralNet<Spec>& net, int start, int chunk,
                    std::span<float* const> head_out, TensorList<Ts...>) {
  int i = 0;
  ((decode_head_rows(Ts::kDecode, net.template host<Ts>(), chunk, Ts::kRowElems,
                     head_out[i] + static_cast<size_t>(start) * Ts::kRowElems, Ts::kRowElems),
    ++i),
   ...);
}

// Every aux head into its slot of the caller's per-row aux block, in list
// (head) order.
template <typename Spec, TensorDescriptor... Ts>
void copy_aux_outputs(const NeuralNet<Spec>& net, int chunk, float* aux_out, TensorList<Ts...>) {
  constexpr int row_stride = TensorList<Ts...>::total_row_elems;
  int offset = 0;
  ((decode_head_rows(Ts::kDecode, net.template host<Ts>(), chunk, Ts::kRowElems, aux_out + offset,
                     row_stride),
    offset += Ts::kRowElems),
   ...);
}

}  // namespace

template <typename Spec>
void TrtEvalService<Spec>::evaluate_batch(const SpecBatch& batch, std::span<float* const> head_out,
                                          float* aux_out) {
  stage_call(net_, batch);

  const int rows = batch_rows(batch);
  const int max_rows = net_.max_rows();
  for (int start = 0; start < rows; start += max_rows) {
    const int chunk = std::min(max_rows, rows - start);
    stage_chunk(net_, batch, start, chunk);
    net_.predict(chunk);

    decode_outputs(net_, start, chunk, head_out, Outputs{});
    if constexpr (AuxOutputs::size > 0) {
      constexpr int aux_row_floats = AuxOutputs::total_row_elems;
      if (aux_out) {
        copy_aux_outputs(net_, chunk, aux_out + static_cast<size_t>(start) * aux_row_floats,
                         AuxOutputs{});
      }
    }
  }
}

template <typename Spec>
void TrtEvalService<Spec>::evaluate(const SpecBatch& batch, std::span<float* const> head_out) {
  evaluate_batch(batch, head_out, nullptr);
}

template <typename Spec>
void TrtEvalService<Spec>::evaluate(const SpecBatch& batch, std::span<float* const> head_out,
                                    float* aux_out)
  requires(AuxOutputs::size > 0)
{
  evaluate_batch(batch, head_out, aux_out);
}

template <typename Spec>
std::unique_ptr<EvalService<Spec>> make_loaded_service(const NeuralNetParams<Spec>& params) {
  auto svc = std::make_unique<TrtEvalService<Spec>>(params);
  svc->load();
  return svc;
}

template std::unique_ptr<EvalService<PositionEvaluationSpec>> make_loaded_service(
  const NeuralNetParams<PositionEvaluationSpec>& params);
template std::unique_ptr<EvalService<MoveSetEvaluationSpec>> make_loaded_service(
  const NeuralNetParams<MoveSetEvaluationSpec>& params);

template class TrtEvalService<PositionEvaluationSpec>;
template class TrtEvalService<MoveSetEvaluationSpec>;

}  // namespace nn
}  // namespace scribblez
