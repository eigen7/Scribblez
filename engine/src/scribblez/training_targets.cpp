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

// ---------- placement-plane targets -------------------------------------

namespace {

constexpr int kPlacementSide = OppNextPlacementTarget::kSide;
static_assert(SelfNextPlacementTarget::kSide == kPlacementSide);
static_assert(OppWinPlacementTarget::kSide == kPlacementSide);
static_assert(SelfWinPlacementTarget::kSide == kPlacementSide);

// Writes a kPlacementSide^2 plane: 1.0 on each square `m` places a tile on,
// 0.0 elsewhere. The plane stays all zeros when `enabled` is false, when the
// move is absent (`has_move` false), or when it is an EXCHANGE or PASS (no
// tiles placed). Flipping transposes across the main diagonal: (r,c) -> (c,r).
void encode_placement_plane(const Move& m, bool has_move, bool enabled, bool flip, float* out) {
  std::fill_n(out, kPlacementSide * kPlacementSide, 0.0f);
  if (!enabled || !has_move) return;
  if (m.type() != MoveType::PLAY) return;
  const bool horizontal = m.horizontal();
  const int start = m.start();
  uint16_t mask = m.square_mask();
  for (int along = 0; mask; ++along, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    const int r = horizontal ? start : along;
    const int c = horizontal ? along : start;
    if (r < 0 || r >= kPlacementSide || c < 0 || c >= kPlacementSide) break;
    out[util::plane_index(r, c, kPlacementSide, flip)] = 1.0f;
  }
}

}  // namespace

void OppNextPlacementTarget::encode(const EncodeContext& v, float* out) {
  encode_placement_plane(v.opp_next_move, v.has_opp_next_move, /*enabled=*/true, v.apply_flip, out);
}

void SelfNextPlacementTarget::encode(const EncodeContext& v, float* out) {
  encode_placement_plane(v.self_next_move, v.has_self_next_move, /*enabled=*/true, v.apply_flip,
                         out);
}

void OppWinPlacementTarget::encode(const EncodeContext& v, float* out) {
  const bool opp_won = v.final_opp() > v.final_active();
  encode_placement_plane(v.opp_next_move, v.has_opp_next_move, opp_won, v.apply_flip, out);
}

void SelfWinPlacementTarget::encode(const EncodeContext& v, float* out) {
  const bool self_won = v.final_active() > v.final_opp();
  encode_placement_plane(v.self_next_move, v.has_self_next_move, self_won, v.apply_flip, out);
}

}  // namespace scribblez
