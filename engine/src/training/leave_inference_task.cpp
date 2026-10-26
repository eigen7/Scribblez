#include "training/leave_inference_task.h"

#include "encoding/game_state_encoder.h"
#include "util/assert.h"

#include <algorithm>
#include <array>

namespace scribblez {

LeaveRowCause OppLeaveTarget::cause(const EncodeContext& ctx) {
  const LeaveHistorySlot& last = ctx.leave_history->slots[0];
  if (!last.present) return LeaveRowCause::kNoOpponentMove;
  if (last.move.type() == MoveType::PASS) return LeaveRowCause::kPass;
  if (ctx.opp_known_leave.size() == 0) return LeaveRowCause::kEmptyLeave;
  return LeaveRowCause::kValid;
}

void OppLeaveTarget::encode(const EncodeContext& ctx, float* out) {
  const Rack& leave = ctx.opp_known_leave;
  // Unsigned, so the -1 padding sorts after the tiles and the whole fixed-size
  // array can be sorted (a sort of a variable-length prefix draws a false
  // -Warray-bounds from gcc).
  std::array<unsigned, kTiles> tiles;
  tiles.fill(unsigned(-1));
  for (int i = 0; i < leave.size(); ++i) tiles[i] = unsigned(leave.tiles()[i].index());
  std::sort(tiles.begin(), tiles.end());
  for (int i = 0; i < kTiles; ++i) out[i] = float(int(tiles[i]));
  out[kLengthOffset] = float(leave.size());
  const LeaveRowCause c = cause(ctx);
  out[kValidOffset] = c == LeaveRowCause::kValid ? 1.0f : 0.0f;
  out[kCauseOffset] = float(c);
  const TileCounts pool = leave_pool(ctx.enc->board().unseen_tiles(*ctx.pov_rack),
                                     ctx.leave_history->observer_exchanges[0]);
  for (Tile t = Tile::of(0); t < TILE_KINDS; ++t)
    out[kPoolOffset + t.index()] = float(pool.count(t));
  out[kBagOffset] = float(ctx.bag_size);
}

void encode_leave_inference_row(const EncodeContext& ctx, LeaveInferenceView view, float* out_row) {
  DEBUG_ASSERT(ctx.leave_history != nullptr);
  const GameStateEncoder& enc = *ctx.enc;
  const int me = ctx.active_player;
  const LeaveInferenceInput in{&enc.board(),
                               *ctx.pov_rack,
                               enc.score(me) - enc.score(1 - me),
                               ctx.bag_size,
                               ctx.opp_known_leave.size(),
                               ctx.leave_history,
                               view};
  LeaveInferenceInputEncoder::encode(in, *ctx.spec.dict, out_row);
  OppLeaveTarget::encode(ctx, out_row + LeaveInferenceInputEncoder::kInputFloats);
}

}  // namespace scribblez
