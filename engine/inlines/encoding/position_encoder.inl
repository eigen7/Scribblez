#pragma once

#include "encoding/position_encoder.h"

namespace scribblez {
namespace binlog {

template <typename Task>
void PositionEncoder::encode_row(const GameLog& g, int sampled_turn, bool post_move, bool flip,
                                 float* out_row) {
  const int mover = replay_to_sampled(g, sampled_turn, post_move);
  const EncodeContext ctx = make_context(g, sampled_turn, mover, post_move, flip);
  Task::encode_row(ctx, out_row);
}

}  // namespace binlog
}  // namespace scribblez
