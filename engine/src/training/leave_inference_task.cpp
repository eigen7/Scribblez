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
  std::array<int, kTiles> tiles;
  tiles.fill(-1);
  for (int i = 0; i < leave.size(); ++i) tiles[i] = leave.tiles()[i].index();
  std::sort(tiles.begin(), tiles.begin() + leave.size());
  for (int i = 0; i < kTiles; ++i) out[i] = float(tiles[i]);
  out[kLengthOffset] = float(leave.size());
  const LeaveRowCause c = cause(ctx);
  out[kValidOffset] = c == LeaveRowCause::kValid ? 1.0f : 0.0f;
  out[kCauseOffset] = float(c);
  const TileCounts pool =
    leave_pool(ctx.enc->board().unseen_tiles(*ctx.pov_rack), ctx.leave_history->observer_exchange);
  for (Tile t = Tile::of(0); t < TILE_KINDS; ++t)
    out[kPoolOffset + t.index()] = float(pool.count(t));
  out[kBagOffset] = float(ctx.bag_size);
}

void LeaveInferenceTask::encode_row(const EncodeContext& ctx, float* out_row) {
  DEBUG_ASSERT(ctx.leave_history != nullptr);
  const GameStateEncoder& enc = *ctx.enc;
  const int me = ctx.active_player;
  const LeaveInferenceInput in{&enc.board(),
                               *ctx.pov_rack,
                               enc.score(me) - enc.score(1 - me),
                               ctx.bag_size,
                               ctx.opp_known_leave.size(),
                               ctx.leave_history};
  LeaveInferenceInputEncoder::encode(in, *ctx.spec.dict, out_row);
  OppLeaveTarget::encode(ctx, out_row + kInputFloats);
}

}  // namespace scribblez
