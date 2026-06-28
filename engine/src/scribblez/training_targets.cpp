#include "scribblez/training_targets.h"

#include "util/grid.h"

#include <algorithm>

namespace scribblez {

// ---------- WldTarget ---------------------------------------------------

void WldTarget::encode(const EncodeContext& v, float* out) {
  const int a = v.final_active();
  const int o = v.final_opp();
  out[0] = (a > o) ? 1.0f : 0.0f;
  out[1] = (a == o) ? 1.0f : 0.0f;
  out[2] = (a < o) ? 1.0f : 0.0f;
}

// ---------- ScoreDiffTarget --------------------------------------------

void ScoreDiffTarget::encode(const EncodeContext& v, float* out) {
  const int diff = v.final_active() - v.final_opp();
  out[0] = static_cast<float>(std::clamp(diff, -kClip, kClip));
}

// ---------- OppNextPlacementTarget -------------------------------------

namespace {
// Flipping is a transpose across the main diagonal: cell (r,c) -> (c,r).
inline int plane_idx(int r, int c, bool flip) {
  return util::plane_index(r, c, OppNextPlacementTarget::kSide, flip);
}
}  // namespace

void OppNextPlacementTarget::encode(const EncodeContext& v, float* out) {
  std::fill_n(out, kSide * kSide, 0.0f);
  if (!v.has_next_move) return;
  const Move& m = v.next_move;
  if (m.type() != MoveType::PLAY) return;
  const bool horizontal = m.horizontal();
  const int start = m.start();
  uint16_t mask = m.square_mask();
  for (int along = 0; mask; ++along, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    const int r = horizontal ? start : along;
    const int c = horizontal ? along : start;
    if (r < 0 || r >= kSide || c < 0 || c >= kSide) break;
    out[plane_idx(r, c, v.apply_flip)] = 1.0f;
  }
}

}  // namespace scribblez
