#pragma once

#include <array>
#include <utility>

namespace util {

// Linear index of cell (r, c) in a side x side grid stored row-major. When
// transpose is true the grid is reflected across the main diagonal, i.e. the
// roles of row and column are swapped.
constexpr int plane_index(int r, int c, int side, bool transpose) {
  return transpose ? (c * side + r) : (r * side + c);
}

// Row/column deltas of the four orthogonal neighbors: up, down, left, right.
constexpr std::array<std::pair<int, int>, 4> kFourNeighborDeltas = {
  {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};

}  // namespace util
