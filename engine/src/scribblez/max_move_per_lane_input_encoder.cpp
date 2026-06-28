#include "scribblez/max_move_per_lane_input_encoder.h"

#include "scribblez/board_planes.h"
#include "scribblez/tile.h"

#include <cstring>

namespace scribblez {

void MaxMovePerLaneInputEncoder::encode(const Board& board, const Rack& rack, bool flip,
                                        float* out) {
  std::memset(out, 0, sizeof(float) * static_cast<size_t>(kInputFloats));

  BoardPlanes::encode(board, flip, out);

  // Rack as raw per-tile counts; Tile::index() maps the blank to slot 26.
  float* counts = out + kSpatialFloats;
  for (Tile t : rack.tiles()) {
    if (!t.is_empty()) counts[t.index()] += 1.0f;
  }
}

}  // namespace scribblez
