#include "training/leaf_eval_task.h"

#include "encoding/game_state_encoder.h"
#include "lexicon/hasty_equity.h"

namespace scribblez {

void LeafEvalTask::encode_row(const EncodeContext& ctx, float* out_row) {
  const GameStateEncoder& enc = *ctx.enc;
  const int me = ctx.active_player;
  const LeafEvalInput in{&enc.board(), *ctx.pov_rack, ctx.opp_known_leave,
                         enc.score(me) - enc.score(1 - me)};
  LeafEvalInputEncoder::encode(in, *ctx.spec.dict, HastyEquity::instance().leave_values(), out_row);
  float* labels = out_row + kInputFloats;
  WldTarget::encode(ctx, labels + kWldOffset);
  ScoreDiffTarget::encode(ctx, labels + kScoreDiffOffset);
}

}  // namespace scribblez
