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

bool is_public(const LeaveInferenceInput& in) { return in.view == LeaveInferenceView::kPublic; }

float* encode_current_scalars(const LeaveInferenceInput& in, float* out) {
  // A public row's observer holds no tiles anyone else can't see.
  const Rack rack = is_public(in) ? Rack{} : in.rack;
  out += encode_rack_counts(rack, out);
  uint8_t unseen[TILE_KINDS];
  compute_unseen_pool(unseen, *in.board, rack);
  out += encode_unseen_pool_thermometer(unseen, out);
  out += encode_score_diff_scalar(in.score_diff, out);
  *out++ = float(in.bag_size) / Encoder::kCountScale;
  if (in.leave_length >= 1 && in.leave_length <= Encoder::kMaxLeaveLength) {
    out[in.leave_length - 1] = 1.0f;
  }
  out += Encoder::kMaxLeaveLength;
  if (is_public(in)) return out + LeaveHistory::kObserverExchanges * TILE_KINDS;
  for (const Rack& exchange : in.history->observer_exchanges)
    out += encode_rack_counts(exchange, out);
  return out;
}

// The tiles a play placed, as per-tile counts. An exchange hides which tiles
// went back, so it writes none.
void encode_played_tiles(const Move& m, float* out) {
  if (m.type() != MoveType::PLAY) return;
  for (int i = 0; i < m.num_glyphs(); ++i) out[m.glyph(i).rack_tile().index()] += 1.0f;
}

// Per tile kind, whether the slot's forgone-play margin applies, then the
// margin. A public row skips the observer's leave_kinds: the board alone
// already withholds a margin from a tile kind it holds every copy of.
float* encode_forgone_scalars(const LeaveHistorySlot& slot, LeaveInferenceView view, float* out) {
  const uint32_t kinds = view == LeaveInferenceView::kPublic ? ~0u : slot.leave_kinds;
  for (int t = 0; t < TILE_KINDS; ++t, out += 2) {
    const int8_t margin = slot.forgone.margin[t];
    if (margin == belief::ForgonePlays::kNone || !((kinds >> t) & 1u)) continue;
    out[0] = 1.0f;
    out[1] = float(margin) / kScoreDiffInputScale;
  }
  return out;
}

float* encode_slot_scalars(const LeaveHistorySlot& slot, LeaveInferenceView view, float* out) {
  float* end = out + Encoder::kSlotScalars;
  if (!slot.present) return end;
  *out++ = 1.0f;
  out[int(slot.move.type())] = 1.0f;
  out += kMoveMetaTypeFloats;
  encode_played_tiles(slot.move, out);
  out += TILE_KINDS;
  *out++ = float(slot.move.num_glyphs());
  *out++ = float(slot.score) / kScoreDiffInputScale;
  *out++ = float(slot.bag_before) / Encoder::kCountScale;
  out = encode_forgone_scalars(slot, view, out);
  DEBUG_ASSERT(out == end);
  return end;
}

}  // namespace

TileCounts leave_pool(TileCounts unseen, const Rack& observer_exchange) {
  for (int i = 0; i < observer_exchange.size(); ++i) {
    const bool ok = unseen.remove(observer_exchange.tiles()[i]);
    RELEASE_ASSERT(ok, "the observer's exchanged tiles are not all unseen");
  }
  return unseen;
}

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
  for (const LeaveHistorySlot& slot : in.history->slots)
    cursor = encode_slot_scalars(slot, in.view, cursor);
  RELEASE_ASSERT(cursor == out + kInputFloats);
}

}  // namespace scribblez
