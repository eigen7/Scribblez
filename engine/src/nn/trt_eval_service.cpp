#include "nn/trt_eval_service.h"

#include "encoding/input_encoder.h"
#include "nn/batching_position_eval_service.h"
#include "nn/shared_registry.h"

#include <Eigen/Core>

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace scribblez {
namespace nn {

namespace {

// The family-specific part of serving, how a Batch's rows reach the engine's
// staging buffers, as free functions overloaded on the spec's Batch type.
// Chunking, decoding, and the aux path are shared (evaluate_batch()).

int batch_rows(const PositionEvaluationSpec::Batch& batch) { return batch.count; }
int batch_rows(const MoveSetEvaluationSpec::Batch& batch) { return batch.moves->count; }

// Split one encoder row, [spatial | scalar], across the engine's two board
// input buffers.
template <typename Spec>
void stage_board_row(NeuralNet<Spec>& net, const float* row, int dst_row) {
  const size_t spatial_floats = size_t(net.spatial_planes()) * kBoardCells;
  const size_t scalar_floats = net.scalar_floats();
  std::memcpy(net.template host<SpatialInput>() + dst_row * spatial_floats, row,
              sizeof(float) * spatial_floats);
  std::memcpy(net.template host<ScalarInput>() + dst_row * scalar_floats, row + spatial_floats,
              sizeof(float) * scalar_floats);
}

// Staging done once per call rather than per chunk. The move-set board input
// is static, so it is staged once and re-sent with every chunk.
void stage_call(NeuralNet<PositionEvaluationSpec>&, const PositionEvaluationSpec::Batch&) {}
void stage_call(NeuralNet<MoveSetEvaluationSpec>& net, const MoveSetEvaluationSpec::Batch& batch) {
  stage_board_row(net, batch.board_row, 0);
}

// One chunk of a per-move tensor, copied from the MoveFeatureArrays field its
// descriptor names (kBatchSource).
template <typename Tensor>
void stage_move_rows(NeuralNet<MoveSetEvaluationSpec>& net,
                     const move_set::MoveFeatureArrays& moves, int start, int rows) {
  using Elem = typename Tensor::Elem;
  const std::vector<Elem>& src = moves.*Tensor::kBatchSource;
  std::memcpy(net.host<Tensor>(), src.data() + size_t(start) * Tensor::kRowElems,
              sizeof(Elem) * rows * Tensor::kRowElems);
}

template <typename... Ts>
void stage_move_tensors(NeuralNet<MoveSetEvaluationSpec>& net,
                        const move_set::MoveFeatureArrays& moves, int start, int rows,
                        TensorList<Ts...>) {
  (stage_move_rows<Ts>(net, moves, start, rows), ...);
}

// Stage rows [start, start + chunk) of the batch.
void stage_chunk(NeuralNet<PositionEvaluationSpec>& net, const PositionEvaluationSpec::Batch& batch,
                 int start, int chunk) {
  const size_t row_floats = size_t(net.spatial_planes()) * kBoardCells + net.scalar_floats();
  for (int r = 0; r < chunk; ++r) {
    stage_board_row(net, batch.rows + (size_t(start) + r) * row_floats, r);
  }
}
void stage_chunk(NeuralNet<MoveSetEvaluationSpec>& net, const MoveSetEvaluationSpec::Batch& batch,
                 int start, int chunk) {
  stage_move_tensors(net, *batch.moves, start, chunk, MoveSetEvaluationSpec::MoveInputs{});
}

// Decode `rows` rows of one head's raw output per its RowDecode. Each row is
// `width` floats; output rows start `dst_stride` floats apart.
void decode_head_rows(RowDecode decode, const float* raw, int rows, int width, float* dst,
                      int dst_stride) {
  for (int r = 0; r < rows; ++r) {
    Eigen::Map<const Eigen::ArrayXf> in(raw + size_t(r) * width, width);
    Eigen::Map<Eigen::ArrayXf> out(dst + size_t(r) * dst_stride, width);
    switch (decode) {
      case RowDecode::kIdentity:
        out = in;
        break;
      case RowDecode::kSoftmax:
        // Subtracting the max keeps exp() from overflowing.
        out = (in - in.maxCoeff()).exp();
        out /= out.sum();
        break;
      case RowDecode::kSigmoid:
        out = 1.0f / (1.0f + (-in).exp());
        break;
    }
  }
}

// Decode one chunk of every output head into the caller's per-head
// destinations, `start` rows in.
template <typename Spec, TensorDescriptor... Ts>
void decode_outputs(const NeuralNet<Spec>& net, int start, int chunk,
                    std::span<float* const> head_out, TensorList<Ts...>) {
  int i = 0;
  ((decode_head_rows(Ts::kDecode, net.template host<Ts>(), chunk, Ts::kRowElems,
                     head_out[i] + size_t(start) * Ts::kRowElems, Ts::kRowElems),
    ++i),
   ...);
}

// Decode one chunk of every aux head into the caller's aux block, where each
// row holds all aux heads side by side in list order.
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
        copy_aux_outputs(net_, chunk, aux_out + size_t(start) * aux_row_floats, AuxOutputs{});
      }
    }
  }
}

template <typename Spec>
void TrtEvalService<Spec>::do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) {
  evaluate_batch(batch, head_out, nullptr);
}

template <typename Spec>
void TrtEvalService<Spec>::evaluate(const SpecBatch& batch, std::span<float* const> head_out,
                                    float* aux_out)
  requires(AuxOutputs::size > 0)
{
  std::lock_guard<std::mutex> lock(this->eval_mutex());
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

// The shared engine is wrapped in the batching decorator, so the threads
// sharing it also coalesce their requests. The move-set family has no create():
// its agents each load their own service through make_loaded_service().
template <>
std::shared_ptr<PositionEvalService> EvalService<PositionEvaluationSpec>::create(
  const NeuralNetParams<PositionEvaluationSpec>& params) {
  // Keyed on every param field; see NeuralNetParamsBase::operator==.
  static SharedRegistry<NeuralNetParamsBase, PositionEvalService> registry;
  return registry.get_or_create(params, [&] {
    return std::make_shared<BatchingPositionEvalService>(
      make_loaded_service<PositionEvaluationSpec>(params));
  });
}

std::shared_ptr<PositionEvalService> load_leaf_position_service(const std::string& onnx_path,
                                                                int cuda_device_id) {
  if (onnx_path.empty()) return nullptr;
  NeuralNetParams<PositionEvaluationSpec> params;
  params.onnx_path = onnx_path;
  params.cuda_device_id = cuda_device_id;
  // Rollout leaves include extreme-advantage positions whose activations
  // overflow FP16. BF16 has FP32's exponent range, and its mantissa loss is far
  // below the model's own error against Monte-Carlo truth.
  params.precision = Precision::kBF16;
  return PositionEvalService::create(params);
}

template class TrtEvalService<PositionEvaluationSpec>;
template class TrtEvalService<MoveSetEvaluationSpec>;

}  // namespace nn
}  // namespace scribblez
