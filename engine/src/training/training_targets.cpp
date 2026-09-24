#include "training/training_targets.h"

#include "encoding/game_state_encoder.h"
#include "training/footprint_mask.h"
#include "util/assert.h"

#include <cstdint>

namespace scribblez {

void WldTarget::encode(const EncodeContext& v, float* out) {
  const int a = v.final_active();
  const int o = v.final_opp();
  out[0] = (a > o) ? 1.0f : 0.0f;
  out[1] = (a == o) ? 1.0f : 0.0f;
  out[2] = (a < o) ? 1.0f : 0.0f;
}

void ScoreDiffTarget::encode(const EncodeContext& v, float* out) {
  out[0] = float(v.final_active() - v.final_opp());
}

namespace {

// `m` must be in the sampled board's frame, or the class would name the
// transposed square.
float plays_class(const EncodeContext& v, const Move& m, bool has_move) {
  if (!has_move) return float(kPassClass);
  DEBUG_ASSERT(m.transposed() == v.enc->board().transposed());
  return float(footprint_class(m));
}

float win_class(const EncodeContext& v, const Move& m, bool has_move, bool seat_won) {
  return seat_won ? plays_class(v, m, has_move) : float(kExtraClass);
}

}  // namespace

void OppNextPlacementTarget::encode(const EncodeContext& v, float* out) {
  out[0] = plays_class(v, v.opp_next_move, v.has_opp_next_move);
}

void SelfNextPlacementTarget::encode(const EncodeContext& v, float* out) {
  out[0] = plays_class(v, v.self_next_move, v.has_self_next_move);
}

void OppWinPlacementTarget::encode(const EncodeContext& v, float* out) {
  const bool opp_won = v.final_opp() > v.final_active();
  out[0] = win_class(v, v.opp_next_move, v.has_opp_next_move, opp_won);
}

void SelfWinPlacementTarget::encode(const EncodeContext& v, float* out) {
  const bool self_won = v.final_active() > v.final_opp();
  out[0] = win_class(v, v.self_next_move, v.has_self_next_move, self_won);
}

namespace {

void write_mask(const FootprintMask& mask, float* out) {
  for (int c = 0; c < kFootprintClasses; ++c) out[c] = mask[c] ? 1.0f : 0.0f;
}

// Readies the sampled board for the footprint masks and returns the tile
// availability that gates them, written into `pool`; nullptr (ungated) when the
// context has no POV rack. Availability is the unseen pool (100 - board -
// mover's rack), the same pool the input encoder feeds the model, so the mask
// agrees with the belief the model can form. ensure_movegen_caches is a no-op
// when the input encoder has already built the caches.
const uint8_t* prepare_mask_inputs(const EncodeContext& v, uint8_t (&pool)[TILE_KINDS]) {
  const Board& board = v.enc->board();
  board.ensure_movegen_caches(*v.spec.dict);
  if (v.pov_rack == nullptr) return nullptr;
  compute_unseen_pool(pool, board, *v.pov_rack);
  return pool;
}

}  // namespace

void OppPlacementMaskTarget::encode(const EncodeContext& v, float* out) {
  uint8_t pool[TILE_KINDS];
  const uint8_t* available = prepare_mask_inputs(v, pool);
  FootprintMask mask;
  opp_footprint_mask(v.enc->board(), available, kMaskTileBudget, /*win_head=*/false, mask);
  write_mask(mask, out);
}

// The opponent's ply is gated by the same unseen pool as the opp mask.
void SelfPlacementMaskTarget::encode(const EncodeContext& v, float* out) {
  uint8_t pool[TILE_KINDS];
  const uint8_t* available = prepare_mask_inputs(v, pool);
  FootprintMask mask;
  self_footprint_mask(v.enc->board(), kMaskTileBudget, kMaskTileBudget, available,
                      /*win_head=*/false, mask);
  write_mask(mask, out);
}

}  // namespace scribblez
