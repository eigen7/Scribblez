#include "scribblez/label_encoder.h"

#include "scribblez/binary_log.h"  // for PositionKind (referenced via GameLogView)

#include <algorithm>

namespace scribblez {
namespace binlog {

namespace {

// Mirror of the helper in input_encoder.cpp; flipping is a transpose across
// the main diagonal so cell (r,c) goes to (c,r).
inline int plane_idx(int r, int c, bool flip) {
  return flip ? (c * kOppNextPlacementSide + r) : (r * kOppNextPlacementSide + c);
}

void encode_opp_next_placement(const GameLogView& view, float* out) {
  std::fill_n(out, kOppNextPlacementFloats, 0.0f);
  if (!view.has_next_move()) return;
  const Move& m = view.next_move();
  if (m.type != MoveType::PLAY) return;
  const int dr = m.horizontal ? 0 : 1;
  const int dc = m.horizontal ? 1 : 0;
  int r = m.start_row;
  int c = m.start_col;
  uint16_t mask = m.square_mask;
  for (int i = 0; i < kOppNextPlacementSide; ++i) {
    if (r < 0 || r >= kOppNextPlacementSide || c < 0 || c >= kOppNextPlacementSide) break;
    if (mask == 0) break;
    if (mask & 1u) {
      out[plane_idx(r, c, view.apply_flip)] = 1.0f;
    }
    mask = static_cast<uint16_t>(mask >> 1);
    r += dr;
    c += dc;
  }
}

}  // namespace

void encode_labels(const GameLogView& view, float** out) {
  // Head 0: WLD from the active player's POV.
  const int active = view.final_active();
  const int opp = view.final_opp();
  float* wld = out[0];
  if (active > opp) {
    wld[0] = 1.0f;
    wld[1] = 0.0f;
    wld[2] = 0.0f;
  } else if (active == opp) {
    wld[0] = 0.0f;
    wld[1] = 1.0f;
    wld[2] = 0.0f;
  } else {
    wld[0] = 0.0f;
    wld[1] = 0.0f;
    wld[2] = 1.0f;
  }

  // Head 1: signed score difference.
  out[1][0] = static_cast<float>(active - opp);

  // Head 2: opponent's next placement (15x15 binary mask).
  encode_opp_next_placement(view, out[2]);
}

}  // namespace binlog
}  // namespace scribblez
