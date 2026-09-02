#include "training/training_targets.h"

#include "encoding/game_state_encoder.h"
#include "training/footprint_mask.h"
#include "util/assert.h"

#include <cstdint>

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
  out[0] = float(v.final_active() - v.final_opp());
}

// ---------- placement footprint-class targets ---------------------------

namespace {

// A plays head's target: the footprint class of `m`, or kPassClass when the
// move is absent (footprint_class already maps EXCHANGE / PASS to kPassClass).
// `m` must be in the sampled board's frame, or the class would name the
// transposed square.
float plays_class(const EncodeContext& v, const Move& m, bool has_move) {
  if (!has_move) return float(kPassClass);
  DEBUG_ASSERT(m.transposed() == v.enc->board().transposed());
  return float(footprint_class(m));
}

// A win head's target: the played footprint class if `seat_won` (the same class
// the plays head would emit), else kExtraClass -- the "not-win" outcome that
// makes {footprints} u {pass} u {not-win} a proper distribution.
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

// ---------- placement legality-mask targets -----------------------------

namespace {

// TODO(sharpen self-mask): kMaskTileBudget (footprint_mask.h) caps k at a full
// rack, the loosest sound bound. It could be tightened by the tiles actually
// left in the endgame (fewer remaining -> a tighter, more precise reachable set)
// -- precision only, 7 is already sound, and it needs care: two plies out the
// mover has re-drawn, so a sound tight bound is not simply today's rack size,
// and masking a real target would hit -log(0) (the loss's force-keep-target
// guard backstops that). Bag count is not directly in the observer's information
// set (bag + opp rack are lumped as the unseen pool); total unseen = 100 -
// tiles-on-board - my-rack is what is knowable.

void write_mask(const FootprintMask& mask, float* out) {
  for (int c = 0; c < kFootprintClasses; ++c) out[c] = mask[c] ? 1.0f : 0.0f;
}

}  // namespace

// The opp side's plays-head legality: the opponent moves next on the sampled
// board, so the mask is exact-ish there. Availability comes from the unseen pool
// (100 - board - mover's rack), exactly the pool the input encoder feeds the
// model, so the mask agrees with the belief the model can form. kExtraClass stays
// illegal (win_head false) -- the loss sets it legal for the win head. Binds the
// dictionary on demand (idempotent if the input encoder already built the caches).
void OppPlacementMaskTarget::encode(const EncodeContext& v, float* out) {
  const Board& board = v.enc->board();
  board.ensure_movegen_caches(*v.spec.dict);
  uint8_t available_counts[27];
  const uint8_t* available_ptr = nullptr;
  if (v.pov_rack != nullptr) {
    compute_unseen_pool(available_counts, board, *v.pov_rack);
    available_ptr = available_counts;
  }
  FootprintMask mask;
  opp_footprint_mask(board, available_ptr, kMaskTileBudget, /*win_head=*/false, mask);
  write_mask(mask, out);
}

// The self side's plays-head legality: the mover plays after the opponent, so
// the opponent's this-turn reach (under the same unseen pool the opp mask uses)
// seeds a cross-check-free expansion for the mover -- an over-approximation
// invariant to whichever move the opponent actually makes.
void SelfPlacementMaskTarget::encode(const EncodeContext& v, float* out) {
  const Board& board = v.enc->board();
  board.ensure_movegen_caches(*v.spec.dict);
  uint8_t available_counts[27];
  const uint8_t* available_ptr = nullptr;
  if (v.pov_rack != nullptr) {
    compute_unseen_pool(available_counts, board, *v.pov_rack);
    available_ptr = available_counts;
  }
  FootprintMask mask;
  self_footprint_mask(board, kMaskTileBudget, kMaskTileBudget, available_ptr, /*win_head=*/false,
                      mask);
  write_mask(mask, out);
}

}  // namespace scribblez
