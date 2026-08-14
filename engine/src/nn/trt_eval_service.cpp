#include "nn/trt_eval_service.h"

#include "encoding/input_encoder.h"
#include "nn/eval_decode.h"

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
              sizeof(Elem) * static_cast<size_t>(rows) * Tensor::kRowElems);
}

template <typename... Ts>
void stage_move_tensors(NeuralNet<MoveSetEvaluationSpec>& net,
                        const move_set::MoveFeatureArrays& moves, int start, int rows,
                        TensorList<Ts...>) {
  (stage_move_rows<Ts>(net, moves, start, rows), ...);
}

// The chunk's rows: de-interleaved encoder rows for the position model, the
// spec's per-move tensor list for the move set model.
void stage_chunk(NeuralNet<PositionEvaluationSpec>& net,
                 const PositionEvaluationSpec::Batch& batch, int start, int chunk) {
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

// `chunk` rows of one aux head's logits into probabilities, each row `width`
// floats landing inside a `row_stride`-float output row. The aux heads emit
// logits; consumers get probabilities, mirroring the WLD softmax in
// decode_eval.
void sigmoid_rows(const float* logits, int chunk, int width, int row_stride, float* dst) {
  for (int r = 0; r < chunk; ++r) {
    Eigen::Map<const Eigen::ArrayXf> in(logits + static_cast<size_t>(r) * width, width);
    Eigen::Map<Eigen::ArrayXf> out(dst + static_cast<size_t>(r) * row_stride, width);
    out = 1.0f / (1.0f + (-in).exp());
  }
}

// Every aux head into its slot of the caller's per-row aux block, in list
// (head) order.
template <typename... Ts>
void copy_aux_outputs(const NeuralNet<PositionEvaluationSpec>& net, int chunk, float* aux_out,
                      TensorList<Ts...>) {
  constexpr int row_stride = TensorList<Ts...>::total_row_elems;
  int offset = 0;
  ((sigmoid_rows(net.host<Ts>(), chunk, Ts::kRowElems, row_stride, aux_out + offset),
    offset += Ts::kRowElems),
   ...);
}

}  // namespace

template <typename Spec>
void TrtEvalService<Spec>::evaluate_batch(const typename Spec::Batch& batch, Eval* out,
                                          float* aux_out) {
  stage_call(net_, batch);

  const int rows = batch_rows(batch);
  const int max_rows = net_.max_rows();
  for (int start = 0; start < rows; start += max_rows) {
    const int chunk = std::min(max_rows, rows - start);
    stage_chunk(net_, batch, start, chunk);
    net_.predict(chunk);

    const float* wld = net_.template host<WldOutput>();
    const float* sd = net_.template host<ScoreDiffOutput>();
    for (int r = 0; r < chunk; ++r) {
      out[start + r] = decode_eval(wld + static_cast<size_t>(r) * WldOutput::kRowElems,
                                   sd + static_cast<size_t>(r) * ScoreDiffOutput::kRowElems);
    }
    if constexpr (Spec::AuxOutputs::size > 0) {
      constexpr int aux_row_floats = Spec::AuxOutputs::total_row_elems;
      if (aux_out) {
        copy_aux_outputs(net_, chunk, aux_out + static_cast<size_t>(start) * aux_row_floats,
                         typename Spec::AuxOutputs{});
      }
    }
  }
}

template <typename Spec>
void TrtEvalService<Spec>::evaluate(const typename Spec::Batch& batch, Eval* out) {
  evaluate_batch(batch, out, nullptr);
}

template <typename Spec>
std::vector<Eval> TrtEvalService<Spec>::evaluate(const typename Spec::Batch& batch) {
  std::vector<Eval> out(batch_rows(batch));
  if (!out.empty()) evaluate(batch, out.data());
  return out;
}

template <typename Spec>
void TrtEvalService<Spec>::evaluate(const typename Spec::Batch& batch, Eval* out, float* aux_out)
  requires(Spec::AuxOutputs::size > 0)
{
  evaluate_batch(batch, out, aux_out);
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
