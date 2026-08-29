#include "training/training_targets.h"

#include "encoding/game_state_encoder.h"
#include "training/footprint_mask.h"

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

// A plays head's target: the footprint class of `m` in the flip frame, or
// kPassClass when the move is absent (footprint_class already maps EXCHANGE /
// PASS to kPassClass).
float plays_class(const Move& m, bool has_move, bool flip) {
  return float(has_move ? footprint_class(m, flip) : kPassClass);
}

// A win head's target: the played footprint class if `seat_won` (the same class
// the plays head would emit), else kExtraClass -- the "not-win" outcome that
// makes {footprints} u {pass} u {not-win} a proper distribution.
float win_class(const Move& m, bool has_move, bool seat_won, bool flip) {
  return seat_won ? plays_class(m, has_move, flip) : float(kExtraClass);
}

}  // namespace

void OppNextPlacementTarget::encode(const EncodeContext& v, float* out) {
  out[0] = plays_class(v.opp_next_move, v.has_opp_next_move, v.apply_flip);
}

void SelfNextPlacementTarget::encode(const EncodeContext& v, float* out) {
  out[0] = plays_class(v.self_next_move, v.has_self_next_move, v.apply_flip);
}

void OppWinPlacementTarget::encode(const EncodeContext& v, float* out) {
  const bool opp_won = v.final_opp() > v.final_active();
  out[0] = win_class(v.opp_next_move, v.has_opp_next_move, opp_won, v.apply_flip);
}

void SelfWinPlacementTarget::encode(const EncodeContext& v, float* out) {
  const bool self_won = v.final_active() > v.final_opp();
  out[0] = win_class(v.self_next_move, v.has_self_next_move, self_won, v.apply_flip);
}

// ---------- placement legality-mask targets -----------------------------

namespace {

// The mover holds at most RACK_SIZE tiles, so a full rack is the sound tile
// budget for every mask -- an over-approximation never masks a real move.
// TODO(sharpen self-mask): cap this by the tiles actually left in the endgame
// (fewer remaining -> a tighter, more precise reachable set) rather than always
// assuming a full rack. Precision only -- 7 is already sound -- and it needs
// care: two plies out the mover has re-drawn, so a sound tight bound is not
// simply today's rack size, and masking a real target would hit -log(0) (the
// loss's force-keep-target guard backstops that). Bag count is not directly in
// the observer's information set (bag + opp rack are lumped as the unseen pool);
// total unseen = 100 - tiles-on-board - my-rack is what is knowable.
constexpr int kMaskTileBudget = RACK_SIZE;

// Writes an opp-head legality mask over the footprint classes. The opponent
// moves next on the sampled board, so the mask is computed there; opp_win adds
// the not-win outcome via `win_head`. Reads cross-checks off the board, binding
// the dictionary on demand (idempotent if the input encoder already built them).
void encode_opp_mask(const EncodeContext& v, bool win_head, float* out) {
  const Board& board = v.enc->board();
  board.ensure_movegen_caches(*v.spec.dict);
  FootprintMask mask;
  opp_footprint_mask(board, kMaskTileBudget, v.apply_flip, win_head, mask);
  for (int c = 0; c < kFootprintClasses; ++c) out[c] = mask[c] ? 1.0f : 0.0f;
}

// Writes a self-head legality mask. The mover plays two plies out on an unknown
// board, so the mask is the opp-move-invariant (cross-check-oblivious)
// over-approximation from the sampled board; self_win adds not-win via
// `win_head`.
void encode_self_mask(const EncodeContext& v, bool win_head, float* out) {
  FootprintMask mask;
  self_footprint_mask(v.enc->board(), kMaskTileBudget, kMaskTileBudget, v.apply_flip, win_head,
                      mask);
  for (int c = 0; c < kFootprintClasses; ++c) out[c] = mask[c] ? 1.0f : 0.0f;
}

}  // namespace

void OppNextPlacementMaskTarget::encode(const EncodeContext& v, float* out) {
  encode_opp_mask(v, /*win_head=*/false, out);
}

void SelfNextPlacementMaskTarget::encode(const EncodeContext& v, float* out) {
  encode_self_mask(v, /*win_head=*/false, out);
}

void OppWinPlacementMaskTarget::encode(const EncodeContext& v, float* out) {
  encode_opp_mask(v, /*win_head=*/true, out);
}

void SelfWinPlacementMaskTarget::encode(const EncodeContext& v, float* out) {
  encode_self_mask(v, /*win_head=*/true, out);
}

}  // namespace scribblez
