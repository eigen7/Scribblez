#include "scribblez/training_targets.h"

#include <algorithm>

namespace scribblez {
namespace binlog {

// ---------- WldTarget ---------------------------------------------------

void WldTarget::encode(const GameLogView& v, float* out) {
  const int a = v.final_active();
  const int o = v.final_opp();
  out[0] = (a > o) ? 1.0f : 0.0f;
  out[1] = (a == o) ? 1.0f : 0.0f;
  out[2] = (a < o) ? 1.0f : 0.0f;
}

// ---------- ScoreDiffTarget --------------------------------------------

void ScoreDiffTarget::encode(const GameLogView& v, float* out) {
  std::fill_n(out, kBins, 0.0f);
  const int diff = v.final_active() - v.final_opp();
  const int clipped = std::clamp(diff, -kClip, kClip);
  out[clipped + kClip] = 1.0f;
}

// ---------- OppNextPlacementTarget -------------------------------------

namespace {
// Flipping is a transpose across the main diagonal: cell (r,c) -> (c,r).
inline int plane_idx(int r, int c, bool flip) {
  return flip ? (c * OppNextPlacementTarget::kSide + r) : (r * OppNextPlacementTarget::kSide + c);
}
}  // namespace

void OppNextPlacementTarget::encode(const GameLogView& v, float* out) {
  std::fill_n(out, kSide * kSide, 0.0f);
  if (!v.has_next_move) return;
  const Move& m = v.next_move;
  if (m.type != MoveType::PLAY) return;
  const int dr = m.horizontal ? 0 : 1;
  const int dc = m.horizontal ? 1 : 0;
  int r = m.start_row;
  int c = m.start_col;
  uint16_t mask = m.square_mask;
  for (int i = 0; i < kSide; ++i) {
    if (r < 0 || r >= kSide || c < 0 || c >= kSide) break;
    if (mask == 0) break;
    if (mask & 1u) out[plane_idx(r, c, v.apply_flip)] = 1.0f;
    mask = static_cast<uint16_t>(mask >> 1);
    r += dr;
    c += dc;
  }
}

}  // namespace binlog
}  // namespace scribblez
