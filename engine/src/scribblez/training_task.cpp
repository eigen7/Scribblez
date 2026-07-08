#include "scribblez/training_task.h"

#include "scribblez/game_state_encoder.h"

namespace scribblez {

int PositionEvalTask::row_floats(const InputEncodingSpec& spec) {
  return input_floats(spec) + kLabelFloats;
}

void PositionEvalTask::encode_row(const EncodeContext& ctx, float* out_row) {
  if (ctx.spec.opp_leave_input) {
    ctx.enc->encode_input(ctx.active_player, *ctx.pov_rack, ctx.opp_known_leave, ctx.apply_flip,
                          out_row);
  } else {
    ctx.enc->encode_input(ctx.active_player, *ctx.pov_rack, ctx.apply_flip, out_row);
  }
  AllTargets::encode_all(ctx, out_row + input_floats(ctx.spec));
}

}  // namespace scribblez
