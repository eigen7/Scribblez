#include "scribblez/max_move_per_lane_task.h"

#include "scribblez/game_state_encoder.h"

namespace scribblez {

void MaxMovePerLaneTask::encode_row(const EncodeContext& ctx, float* out_row) {
  const Board& board = ctx.enc->board();
  MaxMovePerLaneInputEncoder::encode(board, *ctx.pov_rack, ctx.apply_flip, out_row);

  const LaneTargets targets = compute_lane_targets(board, *ctx.pov_rack, *ctx.dict);
  encode_lane_targets(targets, ctx.apply_flip, out_row + kInputFloats);
}

}  // namespace scribblez
