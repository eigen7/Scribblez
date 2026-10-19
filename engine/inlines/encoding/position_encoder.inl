#pragma once

#include "encoding/position_encoder.h"

namespace scribblez {
namespace binlog {

// Whether `Task` reads the move history (EncodeContext::leave_history).
template <typename Task>
concept RecordsLeaveHistory = requires { requires Task::kRecordsLeaveHistory; };

template <typename Task>
void PositionEncoder::encode_row(const GameLog& g, int sampled_turn, bool post_move, bool transpose,
                                 float* out_row, std::span<const belief::ForgonePlays> forgone) {
  LeaveHistory* history = RecordsLeaveHistory<Task> ? &history_ : nullptr;
  const int mover = replay_to_sampled(g, sampled_turn, post_move, history);
  if (history) attach_forgone(sampled_turn, forgone);
  if (transpose) {
    enc_ = enc_.transpose();
    if (history) history_ = history_.transpose();
  }
  EncodeContext ctx = make_context(g, sampled_turn, mover, post_move);
  ctx.leave_history = history;
  Task::encode_row(ctx, out_row);
}

}  // namespace binlog
}  // namespace scribblez
