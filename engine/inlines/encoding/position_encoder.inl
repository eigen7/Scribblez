#pragma once

#include "encoding/position_encoder.h"

#include "util/assert.h"

namespace scribblez {
namespace binlog {

// Whether `Task` reads the move history (EncodeContext::leave_history).
template <typename Task>
concept RecordsLeaveHistory = requires { requires Task::kRecordsLeaveHistory; };

// Whether `Task` reads the squares' fates (EncodeContext::square_fates).
template <typename Task>
concept RecordsSquareFates = requires { requires Task::kRecordsSquareFates; };

template <typename Task>
void PositionEncoder::encode_row(const GameLog& g, int sampled_turn, bool post_move,
                                 RowAugment augment, float* out_row,
                                 std::span<const belief::ForgonePlays> forgone) {
  LeaveHistory* history = RecordsLeaveHistory<Task> ? &history_ : nullptr;
  const int mover = replay_to_sampled(g, sampled_turn, post_move, history);
  if (history) {
    attach_forgone(sampled_turn, forgone);
    if (augment.hide_private) history_.hide_private();
  }
  if (augment.transpose) {
    enc_ = enc_.transpose();
    if (history) history_ = history_.transpose();
  }
  EncodeContext ctx = make_context(g, sampled_turn, mover, post_move);
  ctx.leave_history = history;
  if constexpr (RecordsSquareFates<Task>) {
    RELEASE_ASSERT(post_move, "square fates label a post-move position");
    fates_ = square_fates(g, sampled_turn, enc_.board());
    ctx.square_fates = &fates_;
  }
  Task::encode_row(ctx, out_row);
}

}  // namespace binlog
}  // namespace scribblez
