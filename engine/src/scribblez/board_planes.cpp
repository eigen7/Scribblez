#include "scribblez/board_planes.h"

#include "scribblez/glyph.h"
#include "util/grid.h"

namespace scribblez {

namespace {

constexpr int kCells = BOARD_SIZE * BOARD_SIZE;

// Plane offset for premium kind `p`, or -1 for a non-premium square. Order
// matches the canonical block: DLS, TLS, DWS, TWS.
int premium_offset(Premium p) {
  if (p == Premium::DLS) return 0;
  if (p == Premium::TLS) return 1;
  if (p == Premium::DWS) return 2;
  if (p == Premium::TWS) return 3;
  return -1;
}

}  // namespace

void BoardPlanes::encode(const Board& board, bool flip, float* planes_out) {
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const int idx = util::plane_index(r, c, BOARD_SIZE, flip);

      const Glyph g = board.at(r, c);
      if (!g.is_empty()) {
        planes_out[g.letter().index() * kCells + idx] = 1.0f;  // letter plane 0..25
        if (g.is_blank()) planes_out[kBlankMarkerPlane * kCells + idx] = 1.0f;
      }

      const int off = premium_offset(board.premium_at(r, c));
      if (off >= 0) planes_out[(kPremiumPlane0 + off) * kCells + idx] = 1.0f;
    }
  }
}

}  // namespace scribblez
