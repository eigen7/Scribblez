#include "encoding/leave_inference_input_encoder.h"

#include "encoding/game_state_encoder.h"
#include "util/assert.h"

#include <cstring>

namespace scribblez {

namespace {

using Encoder = LeaveInferenceInputEncoder;

float* encode_current_planes(const LeaveInferenceInput& in, float* out) {
  BoardPlanes::encode(*in.board, out);
  out += BoardPlanes::kPlanes * kBoardCells;
  return out + encode_cross_check_planes(*in.board, out) * kBoardCells;
}

float* encode_slot_planes(const LeaveHistorySlot& slot, float* out) {
  if (slot.present) {
    encode_placement_plane(slot.move, out);
    encode_cross_check_planes(slot.cross_checks, out + kBoardCells);
  }
  return out + Encoder::kSlotPlanes * kBoardCells;
}

float* encode_current_scalars(const LeaveInferenceInput& in, float* out) {
  out += encode_rack_counts(in.rack, out);
  uint8_t unseen[TILE_KINDS];
  compute_unseen_pool(unseen, *in.board, in.rack);
  out += encode_unseen_pool_thermometer(unseen, out);
  out += encode_score_diff_scalar(in.score_diff, out);
  *out++ = float(in.bag_size) / Encoder::kCountScale;
  if (in.leave_length >= 1 && in.leave_length <= Encoder::kMaxLeaveLength) {
    out[in.leave_length - 1] = 1.0f;
  }
  return out + Encoder::kMaxLeaveLength;
}

// The tiles a move placed or exchanged, as per-tile counts. An opponent's
// exchange hides which tiles went back, so it writes none.
void encode_move_tiles(const Move& m, bool opponent, float* out) {
  if (opponent && m.type() == MoveType::EXCHANGE) return;
  for (int i = 0; i < m.num_glyphs(); ++i) out[m.glyph(i).rack_tile().index()] += 1.0f;
}

float* encode_slot_scalars(const LeaveHistorySlot& slot, bool opponent, float* out) {
  float* end = out + Encoder::kSlotScalars;
  if (!slot.present) return end;
  *out++ = 1.0f;
  out[int(slot.move.type())] = 1.0f;
  out += kMoveMetaTypeFloats;
  encode_move_tiles(slot.move, opponent, out);
  out += TILE_KINDS;
  *out++ = float(slot.move.num_glyphs());
  *out++ = float(slot.score) / kScoreDiffInputScale;
  *out++ = float(slot.bag_before) / Encoder::kCountScale;
  DEBUG_ASSERT(out == end);
  return end;
}

// An opponent slot's forgone-play scalars: per tile kind, whether its margin
// applies, then the margin.
float* encode_forgone_scalars(const LeaveHistorySlot& slot, float* out) {
  for (int t = 0; t < TILE_KINDS; ++t, out += 2) {
    const int8_t margin = slot.forgone.margin[t];
    if (margin == belief::ForgonePlays::kNone || !((slot.unseen_by_other >> t) & 1u)) continue;
    out[0] = 1.0f;
    out[1] = float(margin) / kScoreDiffInputScale;
  }
  return out;
}

}  // namespace

LeaveHistory LeaveHistory::transpose() const {
  LeaveHistory out = *this;
  for (LeaveHistorySlot& slot : out.slots) {
    slot.move = slot.move.transpose();
    slot.cross_checks = slot.cross_checks.transpose();
  }
  return out;
}

void LeaveInferenceInputEncoder::encode(const LeaveInferenceInput& in, const Dictionary& dict,
                                        float* out) {
  std::memset(out, 0, sizeof(float) * size_t(kInputFloats));
  in.board->ensure_movegen_caches(dict);
  float* cursor = encode_current_planes(in, out);
  for (const LeaveHistorySlot& slot : in.history->slots) cursor = encode_slot_planes(slot, cursor);
  RELEASE_ASSERT(cursor == out + kSpatialFloats);
  cursor = encode_current_scalars(in, cursor);
  for (int s = 0; s < LeaveHistory::kSlots; ++s) {
    cursor = encode_slot_scalars(in.history->slots[s], LeaveHistory::is_opponent_slot(s), cursor);
  }
  for (int s = 0; s < LeaveHistory::kSlots; ++s) {
    if (LeaveHistory::is_opponent_slot(s))
      cursor = encode_forgone_scalars(in.history->slots[s], cursor);
  }
  RELEASE_ASSERT(cursor == out + kInputFloats);
}

}  // namespace scribblez
