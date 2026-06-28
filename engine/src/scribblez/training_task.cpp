#include "scribblez/training_task.h"

#include "scribblez/game_state_encoder.h"

namespace scribblez {

void PostMoveTask::encode_row(const EncodeContext& ctx, float* out_row) {
  ctx.enc->encode_input(ctx.active_player, *ctx.pov_rack, ctx.apply_flip, out_row);
  AllTargets::encode_all(ctx, out_row + kInputFloats);
}

}  // namespace scribblez
