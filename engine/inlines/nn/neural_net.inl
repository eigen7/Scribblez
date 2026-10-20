#include "nn/neural_net.h"

#include <cstring>

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

}  // namespace nn
}  // namespace scribblez
