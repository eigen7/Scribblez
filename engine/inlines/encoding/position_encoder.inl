#pragma once

#include "encoding/position_encoder.h"

namespace scribblez {
namespace binlog {

template <typename Task>
void PositionEncoder::encode_row(const GameLog& g, int sampled_turn, bool post_move, bool transpose,
                                 float* out_row) {
  const int mover = replay_to_sampled(g, sampled_turn, post_move);
  if (transpose) enc_ = enc_.transpose();
  const EncodeContext ctx = make_context(g, sampled_turn, mover, post_move);
  Task::encode_row(ctx, out_row);
}

}  // namespace binlog
}  // namespace scribblez
