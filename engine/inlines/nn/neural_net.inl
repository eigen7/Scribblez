#include "nn/neural_net.h"

#include <cstring>
#include <vector>

namespace scribblez {
namespace nn {

template <typename Spec>
void stage_board_row(NeuralNet<Spec>& net, const float* row, int dst_row) {
  const size_t spatial_floats = size_t(net.spatial_planes()) * kBoardCells;
  const size_t scalar_floats = net.scalar_floats();
  std::memcpy(net.template host<SpatialInput>() + dst_row * spatial_floats, row,
              sizeof(float) * spatial_floats);
  std::memcpy(net.template host<ScalarInput>() + dst_row * scalar_floats, row + spatial_floats,
              sizeof(float) * scalar_floats);
}

namespace detail {

template <typename Tensor, typename Spec>
void stage_move_rows(NeuralNet<Spec>& net, const move_set::MoveFeatureArrays& moves, int start,
                     int rows) {
  using Elem = typename Tensor::Elem;
  const std::vector<Elem>& src = moves.*Tensor::kBatchSource;
  std::memcpy(net.template host<Tensor>(), src.data() + size_t(start) * Tensor::kRowElems,
              sizeof(Elem) * size_t(rows) * Tensor::kRowElems);
}

}  // namespace detail

template <typename Spec, typename... Ts>
void stage_move_tensors(NeuralNet<Spec>& net, const move_set::MoveFeatureArrays& moves, int start,
                        int rows, TensorList<Ts...>) {
  (detail::stage_move_rows<Ts>(net, moves, start, rows), ...);
}

}  // namespace nn
}  // namespace scribblez
