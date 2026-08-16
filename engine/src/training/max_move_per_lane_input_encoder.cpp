#include "training/max_move_per_lane_input_encoder.h"

#include "encoding/board_planes.h"
#include "game/tile.h"

#include <cstring>

namespace scribblez {

void MaxMovePerLaneInputEncoder::encode(const Board& board, const Rack& rack, bool flip,
                                        float* out) {
  std::memset(out, 0, sizeof(float) * size_t(kInputFloats));

  BoardPlanes::encode(board, flip, out);

  // Rack as raw per-tile counts; Tile::index() maps the blank to slot 26.
  float* counts = out + kSpatialFloats;
  for (Tile t : rack.tiles()) {
    if (!t.is_empty()) counts[t.index()] += 1.0f;
  }
}

}  // namespace scribblez
